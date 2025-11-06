import torch
import os
import os.path as osp
import sys

import tempfile
import matplotlib.pyplot as pl
import copy
import pow3r2.tools.path_to_dust3r

from dust3r.utils.device import todevice, to_numpy
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images as li
from dust3r.utils.image import rgb

from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo

from dust3r.cloud_opt.commons import i_j_ij, compute_edge_scores, edge_str


import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images
from inference.utils import *
import os, shutil, json

from voxel.latent_voxel import *



sys.path.append("Thesis/")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
POINTS = "world_points"
CONF = "world_points_conf"
VIZ = False
threshold = 1.0     
coverage_threshold = 0.7       
z_clip_map = (-0.1, 0.3)   

R_w2m = np.array([[0, 0, -1],
                [-1, 0, 0],
                [0, -1, 0]], dtype=np.float32)

# R_w2m = np.array([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, -1]], dtype=np.float32)

t_w2m = np.zeros(3, dtype=np.float32)

R_w2m = to_torch(R_w2m, device=device)
t_w2m = to_torch(t_w2m, device=device)
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
print("Loading DUST3R")


weights_path = "naver/" + "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)

model.eval()
model = model.to(device)
# model = model.to(dtype)


voxel_size = 0.01
feature_dim = 32
vox = LatentVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float16),
            params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
            device=device, dtype=torch.bfloat16, feature_dim=feature_dim
        )
vox = vox.to(device)
vox.eval()

projector = FeatureProjector(out_dim=feature_dim)
projector = projector.to(device)
projector.eval()
save_root = "kitchen_habitat_dust3r_fast_2/"
os.makedirs(save_root, exist_ok=True)

# target_dir = "/Users/marvin/Documents/Thesis/vggt/examples/fishbowl1/"
# sub_dirs = ["images","images", "images1", "images2", "images3"]
# sub_dirs = ["00000000", "00000050","00000100", "00000150", "00000200", "00000250"]
# sub_dirs = ["00000000", "00000300",  "00000350", "00000400"]

# target_dir = "/Users/marvin/Documents/Thesis/repo/dataset_generation/habitat/frames_bedroom/"  # folder that contains intrinsics.json and time_XXXXX/
target_dir = "../data/frames_bedroom/"  # folder that contains intrinsics.json and time_XXXXX/

sub_dirs = sorted([d for d in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, d))])


# sub_dirs = [d for d in os.listdir(target_dir) 
#             if os.path.isdir(os.path.join(target_dir, d))]

keyframes = []
changed_idx = []
last_save_paths = None  # global-ish tracker for previous iter’s artifact paths

for i, images in enumerate(sub_dirs):

    print(f"Iteration {i}")
    start_full = time.time()
    torch.cuda.synchronize()

    startc = torch.cuda.Event(enable_timing=True)
    endc = torch.cuda.Event(enable_timing=True)


    startc.record()
    

    start = time.time()
    imgs = li(target_dir + images, size=512, verbose=False)

    

    #image_tensors = torch.stack([d["img"] for d in imgs])

    image_tensors = []
    for d in imgs:
        t = d["img"]                        # (1,3,H,W), likely float in [0,1]
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]                        # -> (3,H,W)
        t = t.detach().cpu()
        if not t.dtype.is_floating_point:
            t = t.float()
        if t.max() > 1.0:                   # in case values are 0..255
            t = t / 255.0
        image_tensors.append(t.clamp(0,1))
        
    image_tensors = torch.stack(image_tensors, dim=0).to(device)

    end = time.time()
    length = end - start

    print("tensors images took", length, "seconds!")
        


    if i < 1:
        
        start = time.time()
   
        predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, target_dir + images, "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=projector)
        
        keyframes = image_tensors.clone()
        
        end = time.time()
        length = end - start

        print("Running inference took", length, "seconds!")
        
        stacked_predictions = [predictions]

    else:
    
        start = time.time()

        changed_idx = changed_images(image_tensors, keyframes, thresh=0.000005)
        
        print(changed_idx)

        end = time.time()
        length = end - start
        
        if len(changed_idx) < 2:
            if last_save_paths is None:
                print("No previous outputs to clone (i==0 case). Running normally.")
            else:
                print("No images have changed; cloning previous outputs for this iter.")
                # Build this iter's destination paths
                dst_paths = {
                    "bev_png":  os.path.join(save_root, f"bev_{i}.png"),
                    "bev_npy":  os.path.join(save_root, f"bev_{i}_np.npy"),
                    "bev_meta": os.path.join(save_root, f"bev_{i}_meta.json"),
                    "vox_ply":  os.path.join(save_root, f"voxels{i}.ply"),
                    "vox_ijk":  os.path.join(save_root, f"voxels{i}_ijk.npy"),
                    "vox_meta": os.path.join(save_root, f"voxels{i}_meta.json"),
                }
                # Clone each previous artifact to the new iter name
                clone(last_save_paths["bev_png"],  dst_paths["bev_png"])
                clone(last_save_paths["bev_npy"],  dst_paths["bev_npy"])
                clone(last_save_paths["bev_meta"], dst_paths["bev_meta"])
                clone(last_save_paths["vox_ply"],  dst_paths["vox_ply"])
                clone(last_save_paths["vox_ijk"],  dst_paths["vox_ijk"])
                clone(last_save_paths["vox_meta"], dst_paths["vox_meta"])

                # Optional: write a tiny provenance note
                with open(os.path.join(save_root, f"iter_{i}_provenance.json"), "w") as f:
                    json.dump({"copied_from_iter": i-1,
                               "reason": "no image change"}, f)

                # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                vox.next_epoch()

                # Update last_save_paths to the newly written copies (so a run of no-op iters still works)
                last_save_paths = dst_paths
            continue

        print("Finding changed images took", length, "seconds!")

        changed_idx = [0] + [x for x in changed_idx if x != 0]
        
        index_map = {new: old for new, old in enumerate(changed_idx)}
                
        idx_t = torch.tensor(changed_idx, device=device, dtype=torch.long)
        keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
        
        
        print("final changed idx:", changed_idx)
        new_vggt_input = [[imgs[g] for g in changed_idx]]

        start = time.time()

        # stacked_predictions = []
        # for input_frames in [imgs]:
        predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, target_dir + images, "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=projector)
        
        # predictions = run_model(model, vggt_input, attn_mask=adj)s
        torch.cuda.synchronize()
   
        end = time.time()
        length = end - start


        print("Running inference took", length, "seconds!")


    # Keep tensors; only extract what we need later.
    # If you truly need NumPy later, convert specific keys then.
    needed = {
        "images","extrinsic", POINTS, CONF, "view_feats"
    }
    for k in list(predictions.keys()):
        if k not in needed:
            del predictions[k]  # drop unneeded heavy stuff early



    start = time.time()
    
    WPTS_m = torch.from_numpy(predictions[POINTS]).to(device=device)

    WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
    Rmw, tmw, info = align_pointcloud_torch_fast(WPTS_m, inlier_dist=voxel_size*0.75, ransac_iters=200, point_chunk=5_000_000, cand_chunk=4096)
    WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
    predictions[POINTS] = WPTS_m
    
    
    torch.cuda.synchronize()

    end = time.time()
    length = end - start

    print("Aligning frames took", length, "seconds!")
    start = time.time()

    # camera_R = R_w2m @ Rmw
    # camera_t = t_w2m + tmw
    # frames_map, cam_centers_map, conf_map, images_map, features_map, (S,H,W), frame_ids = build_frames_and_centers_vectorized_torch(
    #     predictions,
    #     POINTS=POINTS,
    #     CONF=CONF,
    #     FEAT="view_feats",
    #     threshold=threshold,
    #     Rmw=camera_R, tmw=camera_t,
    #     z_clip_map=z_clip_map,   # or None
    #     return_flat=True

    # )   
    # feats = torch.from_numpy(to_numpy(predictions["view_feats"]))
    # feats = feats.to(device=device, dtype=torch.float32)  # (S, H, W,)
    
    
    # with torch.no_grad():
    #     predictions["view_feats"] = projector(feats)
    
    # del feats
    torch.cuda.synchronize()

    end = time.time()
    length = end - start

    print("Projecting view feats", length, "seconds!")
    
    start = time.time()

    frames_map, conf_map, images_map, features_map, (S,H,W), frame_ids = filter_frames(
        predictions,
        POINTS=POINTS,
        CONF=CONF,
        FEAT="view_feats",
        threshold=threshold,
        z_clip_map=z_clip_map,   # or None
    )  
    
    torch.cuda.synchronize()

    end = time.time()
    length = end - start

    print("Building frames/camera centers took", length, "seconds!")

    start = time.time()

    align_to_voxel = False #(i > 0)
        

    #features_map = [vf_t[i] for i in range(vf.shape[0])]  # one vector per image

    # features_map = [pointnext_inference(preprocess_points(f,i)) for f, i in zip(frames_map, images_map)]

    with torch.no_grad():
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            vox, bev, meta = build_maps_from_latent_features(
                i,
                frames_map,
                conf_map,
                features_map,
                vox,
                voxel_size=voxel_size,           # 10 cm
                bev_window_m=(5.0, 5.0), # local 20x20 m
                bev_origin_xy=(-2.0, -2.0),
                z_clip_vox=(-np.inf, np.inf),
                z_band_bev=(0.02, 0.5),
                frame_ids=frame_ids
            )

    vox = vox

    torch.cuda.synchronize()
     
    end = time.time()
    length = end - start

    print("Building Voxel and BEV took", length, "seconds!")

        
            


    # if i == 0:
    #     vox.end_bootstrap()   # promotes current occupied to LT and clears ST
    #     vox.lock_long_term()  # optional: forbids any future LT change
        
    end = time.time()
    length = end - start

    print("Building Voxel and BEV took", length, "seconds!")

    # print("BEV:", bev.shape, meta)    

    num_cells = vox.keys.numel() if hasattr(vox, "keys") else len(vox.logodds)
    num_occ = int((vox.vals > vox.p.occ_thresh).sum().item()) if hasattr(vox, "vals") else \
            sum(1 for L in vox.logodds.values() if L > vox.p.occ_thresh)

    # print("Voxels (stored cells):", num_cells)
    # print("Voxels (occupied):", num_occ)

    # On-screen plots
    if VIZ:
        visualize_bev(bev, meta)
        # visualize_voxels(vox, z_band=(-0.5, 3), max_dim=200)
        visualize_points_and_voxels_open3d(frames_map, vox,cam_centers_map,max_points=150_000, max_voxels=25_000)

    # Files
    # save_bev_png(bev, meta, f"bev_{i}_{j}.png")
    # export_occupied_voxels_as_ply(vox, "voxels.ply")
        
    end_full = time.time()
    
    endc.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(startc.elapsed_time(endc))
    print("Full iter took: ",end_full -  start_full)

    bev = bev.detach().float().cpu().numpy()
    save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
    export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

    bev_png  = save_root + f"bev_{i}.png"
    bev_npy  = save_root + f"bev_{i}_np.npy"
    bev_meta = save_root + f"bev_{i}_meta.json"
    vox_ply  = save_root + f"voxels{i}.ply"
    vox_ijk  = save_root + f"voxels{i}_ijk.npy"
    vox_meta = save_root + f"voxels{i}_meta.json"
    
    for p in [bev_png, bev_npy, bev_meta, vox_ply, vox_ijk, vox_meta]:
        ensure_dir_for_file(p)

    save_bev(bev, meta, bev_png, bev_npy, bev_meta)
    export_occupied_voxels(vox, vox_ply, vox_ijk, vox_meta, z_clip_map)

    # >>> after-save bookkeeping so we can clone next time
    last_save_paths = {
        "bev_png": bev_png, "bev_npy": bev_npy, "bev_meta": bev_meta,
        "vox_ply": vox_ply, "vox_ijk": vox_ijk, "vox_meta": vox_meta,
    }

    
    vox.next_epoch()

