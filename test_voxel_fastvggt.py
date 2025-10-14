import torch
from fastvggt.models.vggt import VGGT
import sys
import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from preprocess_images.filter_images import changed_images
import os, shutil, json

def _clone(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

def ensure_dir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
def decide_grouping_vs_all(groups, changed_indices, rho=0.8):
    """
    Decide whether to run VGGT on groups or on all changed frames at once.

    groups: list[list[int]]   # each group is a list of frame indices (already chosen)
    changed_indices: list[int]  # the set you would process in one-shot mode
    ref_id: int               # the reference frame id
    include_ref_in_each: bool # if True, ref is duplicated in EVERY group (as your code does)
    rho: float                # need Σ n_i^2 <= rho * S^2 to justify grouping
    min_group: int            # avoid tiny groups (overhead dominates)
    max_groups: int           # avoid too many groups (launcher/overheads)

    Returns dict with decision and scores.
    """
    uniq = sorted(set(changed_indices))
    S = len(uniq)

    # Effective group sizes (count ref if you actually prepend it to each group)
    n = []
    for g in groups:
        g_set = set(g)
        n_i = len(g_set)
        n.append(n_i)


    score_all = S**2
    score_groups = sum(k*k for k in n)

    use_groups = (score_groups <= rho * score_all)
    return {
        "use_groups": use_groups,
        "reason": "quadratic saving" if use_groups else "not enough saving",
        "S": S,
        "n": n,
        "score_all": score_all,
        "score_groups": score_groups,
        "saving_ratio": (score_groups / score_all) if score_all > 0 else 1.0,
        "num_groups": len(n),
    }


sys.path.append("vggt/")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
POINTS = "world_points_from_depth"
CONF = "depth_conf"
VIZ = True
threshold = 62.0     
coverage_threshold = 0.7       
z_clip_map = (-0.5, 0.3)   
R_w2m = np.array([[1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]], dtype=np.float32)
t_w2m = np.zeros(3, dtype=np.float32)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
print("Loading VGGT")

model = VGGT(merging=0, vis_attn_map=False)
ckpt = torch.load("model_tracker_fixed_e20.pt", map_location="cpu")
incompat = model.load_state_dict(ckpt, strict=False)

model = model.eval()
model = model.to(device)
model = model.to(dtype)


voxel_size = 0.01
vox = TorchSparseVoxelGrid(
    origin_xyz=np.zeros(3, dtype=np.float32),
    params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
    device=device, dtype=torch.float32
)

save_root = "pred_cafe2/"
target_dir = "/Users/marvin/Documents/Thesis/vggt/examples/cafe2/"
# sub_dirs = ["images","images", "images1", "images2", "images3"]
# sub_dirs = ["00000000", "00000050","00000100", "00000150", "00000200", "00000250"]
sub_dirs = ["00000000", "00000300",  "00000350", "00000400"]


# sub_dirs = [d for d in os.listdir(target_dir) 
#             if os.path.isdir(os.path.join(target_dir, d))]

keyframes = []
changed_idx = []
last_save_paths = None  # global-ish tracker for previous iter’s artifact paths

for i, images in enumerate(sub_dirs):
    print(f"Iteration {i}")
    # print("Loading and filtering images")
    #run inference
    image_tensors = load_images(target_dir + images, device=device)
    image_tensors = image_tensors.to(dtype=dtype)

    if i < 1:
        
        keyframes = image_tensors.clone()
        keyframes = keyframes.to(dtype=dtype)

        start = time.time()
    
        predictions = run_model(model, keyframes, dtype=dtype)

        
        end = time.time()
        length = end - start

        print("Running VGGT inference took", length, "seconds!")
        
        start = time.time()

        S = predictions[POINTS].shape[0]
        # T_cw = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
        # extr = predictions["extrinsic"]
        # T_cw[:,:3,:3] = extr[:,:3,:3]
        # T_cw[:,:3, 3] = extr[:,:3, 3]
        
        T_cw = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))

        # Fill with extrinsic [R|t]
        extr = predictions["extrinsic"]  # (S,3,4)
        T_cw[:, :3, :4] = extr

        covisibility_graph = covisibility_from_world_proximity(
            P_world=predictions[POINTS],
            conf_map=predictions[CONF],
            stride=4,           # try 8 or 4 for denser sampling
            max_points=50000,   # keep computation manageable
            eps=0.05,           # 5 cm if units are meters; tune to your data
            sym="mean",
            normalize=True,
            chunk_size=10000,
            dtype=dtype,
            conf_percentile=threshold
        )
        
        print(covisibility_graph)
        
        end = time.time()
        length = end - start

        # Turn extrinsic into homogeneous (S,4,4)
  

        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(6,6))
        # plt.imshow(covisibility_graph, cmap="viridis")
        # plt.colorbar(label="Covisibility weight")
        # plt.title("Covisibility matrix")
        # plt.xlabel("Frame index")
        # plt.ylabel("Frame index")
        # plt.show()
        print("Building covisiblity graph took", length, "seconds!")
        
        stacked_predictions = [predictions]

    else:
        start = time.time()

        changed_idx = changed_images(image_tensors, keyframes, cos_sim_thresh=0.97)
        
        print(changed_idx)

        end = time.time()
        length = end - start
        
        if len(changed_idx) == 0:
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
                _clone(last_save_paths["bev_png"],  dst_paths["bev_png"])
                _clone(last_save_paths["bev_npy"],  dst_paths["bev_npy"])
                _clone(last_save_paths["bev_meta"], dst_paths["bev_meta"])
                _clone(last_save_paths["vox_ply"],  dst_paths["vox_ply"])
                _clone(last_save_paths["vox_ijk"],  dst_paths["vox_ijk"])
                _clone(last_save_paths["vox_meta"], dst_paths["vox_meta"])

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
                
        idx_t = torch.tensor(changed_idx, device=image_tensors.device, dtype=torch.long)
        keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
        
        
        idx_t = torch.tensor(changed_idx, device=covisibility_graph.device, dtype=torch.long)
        covisibility_changed = covisibility_graph.index_select(0, idx_t).index_select(1, idx_t)
    
        #TEMPOROARY
        # covisibility_changed = torch.ones_like(covisibility_changed)

        # tau = 0.5
        # adj = (covisibility_changed >= tau)    # torch.bool or 0/1
        # adj = adj | adj.T   # (optional) symmetrize
        # adj.fill_diagonal_(True)
        # # covisibility_input = covisibility_graph[np.ix_(changed_idx, changed_idx)] if i > 1 else covisibility_graph
   
        adj = covis_to_adj(covisibility_changed, tau=0.3, kmin=0)
        
        print(adj)
        
        

        groups = compute_tight_components(adj,          
                    mutual=True,        # require i<->j
                    window=None,           # keep only neighbors within ±2 frames (tune or None)
                    jaccard_tau=False,    # drop weak overlaps (0.1–0.4 typical), or None
                    k_core_k=False       # e.g., 2 or 3 to peel fringes, or None)
        )
        
        groups = merge_singletons_to_next(groups, adj.shape[0])
        
        
        # groups = [[0] + [x for x in group if x != 0] for group in groups]

        groups = [[index_map[x] for x in group] for group in groups]
        print(groups)
        
        
        new_vggt_input = []
        grown_groups = []
        for group in groups:
            # _, path,_ = dijkstra_to_any(covisibility_graph, group)
            if group == [0]:
                continue
            
            group, _ ,_ = grow_set_until_coverage(covisibility_graph, thresh=coverage_threshold, start=group)
            group = [0] + [x for x in group if x != 0]
            grown_groups.append(group)
            print(group)
            new_vggt_input.append(image_tensors[group])
        
        
        grown_changed_idx,_,_ = grow_set_until_coverage(covisibility_graph, thresh=coverage_threshold, start=changed_idx)
        
        print(grown_changed_idx)
    
        decision = decide_grouping_vs_all(grown_groups, grown_changed_idx, rho=0.9)

        print(decision)
        
        new_vggt_input = new_vggt_input if decision["use_groups"] else [image_tensors[grown_changed_idx]]
        
        start = time.time()

        stacked_predictions = []
        for input_frames in new_vggt_input:
            print(input_frames.size())
            stacked_predictions.append(run_model(model, input_frames))
        
        # predictions = run_model(model, vggt_input, attn_mask=adj)

            
        end = time.time()
        length = end - start

        print("Running VGGT inference took", length, "seconds!")
   

    for j, predictions in enumerate(stacked_predictions):
        # Keep tensors; only extract what we need later.
        # If you truly need NumPy later, convert specific keys then.
        needed = {
            "images","extrinsic", "intrinsic", POINTS, CONF
        }
        for k in list(predictions.keys()):
            if k not in needed:
                del predictions[k]  # drop unneeded heavy stuff early


        # print("Align Point Cloud")

        start = time.time()
        
        WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
        Rmw, tmw, info = align_pointcloud(WPTS_m, inlier_dist=voxel_size*0.75)
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
        predictions[POINTS] = WPTS_m

        if VIZ:
            visualize_vggt_pointcloud(predictions, key=POINTS, conf_key=CONF, threshold=threshold)

        # print("Building Voxels and BEV")

        camera_R = R_w2m @ Rmw
        camera_t = t_w2m + tmw
        frames_map, cam_centers_map, conf_map, (S,H,W) = build_frames_and_centers_vectorized(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            threshold=threshold,
            Rmw=camera_R, tmw=camera_t,
            z_clip_map=z_clip_map,   # or None
        )   
        end = time.time()
        length = end - start

        print("Aligning and building frames/camera centers took", length, "seconds!")

        start = time.time()

        align_to_voxel = (i > 0)
        
        if i == 0:
            vox.begin_bootstrap()

        vox, bev, meta = build_maps_from_points_and_centers_torch(
            frames_map,
            cam_centers_map,
            conf_map,
            vox,
            align_to_voxel=align_to_voxel,
            voxel_size=voxel_size,           # 10 cm
            bev_window_m=(5.0, 5.0), # local 20x20 m
            bev_origin_xy=(-2.0, -2.0),
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(0.05, 0.3),
            max_range_m=None,
            carve_free=True,
            samples_per_voxel=1,
            ray_stride=2
        )

        if i == 0:
            vox.end_bootstrap()   # promotes current occupied to LT and clears ST
            vox.lock_long_term()  # optional: forbids any future LT change
            
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
            visualize_points_and_voxels_open3d(frames_map, vox,cam_centers_map,
                                            max_points=150_000, max_voxels=25_000)

        # Files
        # save_bev_png(bev, meta, f"bev_{i}_{j}.png")
        # export_occupied_voxels_as_ply(vox, "voxels.ply")
        

    # save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
    # export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

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
