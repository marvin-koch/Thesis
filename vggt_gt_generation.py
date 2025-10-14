import torch
from vggt.models.vggt import VGGT
import sys
import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from preprocess_images.filter_images import changed_images


sys.path.append("vggt/")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
POINTS = "world_points"
CONF = "world_points_conf"
VIZ = False
threshold = 62.0     
z_clip_map = (-0.5, 0.3)   
R_w2m = np.array([[1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]], dtype=np.float32)
t_w2m = np.zeros(3, dtype=np.float32)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
print("Loading VGGT")

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)


voxel_size = 0.01
vox = TorchSparseVoxelGrid(
    origin_xyz=np.zeros(3, dtype=np.float32),
    params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
    device=device, dtype=torch.float32
)

# target_dir = "/Users/marvin/Documents/Thesis/vggt/examples/cafe2/"
# sub_dirs = ["images","images", "images1", "images2", "images3"]
# sub_dirs = ["00000000", "00000050","00000100", "00000150", "00000200", "00000250"]
# sub_dirs = ["00000000", "00000300",  "00000350", "00000400"]

target_dir = "/Users/marvin/Documents/Thesis/repo/dataset_generation/habitat/frames_bedroom/"  # folder that contains intrinsics.json and time_XXXXX/

sub_dirs = sorted([d for d in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, d))])




for i, images in enumerate(sub_dirs):
    print(f"Iteration {i}")
    #run inference

    image_tensors = load_images(target_dir + images, device=device)

    start = time.time()

    predictions = run_model(model, image_tensors, dtype=dtype)

    
    end = time.time()
    length = end - start

    print("Running VGGT inference took", length, "seconds!")
    

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


    align_to_voxel = (i > 0)

    start = time.time()

    # if i == 0:
    #     vox.begin_bootstrap()
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
        z_band_bev=(-0.02, 0.5),
        max_range_m=None,
        carve_free=True,
        samples_per_voxel=1,
        ray_stride=2
    )

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
        visualize_points_and_voxels_open3d(frames_map, vox,cam_centers_map,
                                        max_points=150_000, max_voxels=25_000)


    save_root = "bedroom_habitat_vggt_gt/"
    # Files
    save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
    export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

    vox.next_epoch()
