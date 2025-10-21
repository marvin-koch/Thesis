import torch
from vggt.models.vggt import VGGT
import sys
import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images

import os, json, numpy as np, torch
import imageio.v2 as imageio

def load_habitat_time_as_predictions(
    time_dir,
    intrinsics_json_path,
    device="cpu",
    dtype=torch.float16,
    frame_stride=1,          # e.g., 2 to take every other frame
    sample_stride=2,         # e.g., 2 or 4 to subsample pixels
    keep_images=False,       # set True only if you really need (S,3,H,W)
    points_dtype=np.float16, # NaNs work fine in float16
    conf_dtype=np.uint8      # 0 or 100 -> uint8 saves 4x memory
):
    """
    Builds VGGT-shaped predictions, optionally subsampled:

      images:   (S,3,H,W) on 'device' if keep_images else None
      intrinsic:(S,3,3)
      extrinsic:(S,3,4)  OpenCV w2c (R|t)
      world_points: (S,H,W,3) world coords; invalid pixels = NaN   [points_dtype]
      world_points_conf: (S,H,W) confidences (0 if invalid, 100)   [conf_dtype]

    Subsampling:
      - frame_stride picks every k-th pose.
      - sample_stride downsamples H,W by k using [::k] and rescales intrinsics.
    """
    with open(intrinsics_json_path, "r") as f:
        intr = json.load(f)
    fx, fy, cx, cy = float(intr["fl_x"]), float(intr["fl_y"]), float(intr["cx"]), float(intr["cy"])
    H_full, W_full = int(intr["h"]), int(intr["w"])

    # Pixel subsampling
    s = int(sample_stride)
    if s < 1: s = 1
    H = H_full // s
    W = W_full // s

    # Rescale intrinsics for [::s] sampling
    fx_s, fy_s = fx / s, fy / s
    cx_s, cy_s = cx / s, cy / s
    K_np = np.array([[fx_s, 0,    cx_s],
                     [0,    fy_s, cy_s],
                     [0,    0,    1.0]], dtype=np.float32)

    with open(os.path.join(time_dir, "poses.json"), "r") as f:
        poses_pack = json.load(f)
    poses_all = sorted(poses_pack["poses"], key=lambda p: int(p["camera_index"]))
    poses = poses_all[::max(1, int(frame_stride))]
    S = len(poses)

    # Allocate compact outputs
    images   = None
    if keep_images:
        images = torch.empty((S, 3, H, W), dtype=torch.float32, device=device)
    K_mats   = torch.empty((S, 3, 3),     dtype=torch.float32, device=device)
    w2c_mats = torch.empty((S, 3, 4),     dtype=torch.float32, device=device)

    P_world_all = np.full((S, H, W, 3), np.nan, dtype=points_dtype)
    Conf_all    = np.zeros((S, H, W),    dtype=conf_dtype)

    # Precompute subsampled pixel grid in downsampled coordinates
    # After subsampling, pixel centers are (x_ds, y_ds) = (x/s, y/s)
    ys_full, xs_full = np.meshgrid(np.arange(H_full, dtype=np.float32),
                                   np.arange(W_full, dtype=np.float32), indexing="ij")
    ys = ys_full[::s, ::s] / s  # or just np.arange(H) if you prefer; both consistent with fx_s/cx_s
    xs = xs_full[::s, ::s] / s

    for i, p in enumerate(poses):
        # --- image (optional) ---
        if keep_images:
            rgb = imageio.imread(os.path.join(time_dir, p["file_path"]))[..., :3]
            if rgb.shape[:2] != (H_full, W_full):
                raise ValueError(f"Image size mismatch: got {rgb.shape[:2]}, expected {(H_full,W_full)}")
            # simple stride subsample
            rgb_ds = rgb[::s, ::s]
            images[i] = torch.from_numpy((rgb_ds.astype(np.float32)/255.0).transpose(2,0,1)).to(device)

        # --- intrinsics / extrinsics ---
        K_mats[i] = torch.from_numpy(K_np).to(device)

        c2w = np.array(p["transform_matrix"], dtype=np.float32)   # 4x4 camera_to_world
        Rcw = c2w[:3, :3]
        tcw = c2w[:3,  3]
        Rwc = Rcw.T
        twc = -Rwc @ tcw
        w2c = np.concatenate([Rwc, twc[:,None]], axis=1).astype(np.float32)  # 3x4 OpenCV
        w2c_mats[i] = torch.from_numpy(w2c).to(device)

        # --- depth -> dense world points (subsampled) ---
        dfile = p.get("depth_file", None)
        if dfile is None:
            continue
        D = np.load(os.path.join(time_dir, dfile)).astype(np.float32)  # H_full x W_full meters
        if D.shape != (H_full, W_full):
            raise ValueError(f"Depth size mismatch: got {D.shape}, expected {(H_full, W_full)}")

        D_ds = D[::s, ::s]  # subsample depth to (H,W)
        valid = np.isfinite(D_ds) & (D_ds > 0.0)
        if not np.any(valid):
            continue

        Z = D_ds
        # Backproject in *downsampled* pixel coordinates with rescaled intrinsics
        X = (xs - cx_s) * Z / fx_s
        Y = (ys - cy_s) * Z / fy_s
        P_cam = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (H,W,3), float32 to preserve mult
        # cam -> world: Xw = Rcw @ Xc + tcw
        P_world = (P_cam.reshape(-1,3) @ Rcw.T) + tcw[None,:]
        P_world = P_world.reshape(H, W, 3).astype(points_dtype)

        # fill outputs
        if points_dtype == np.float16:
            # preserve NaNs after cast (float16 supports NaN)
            Pw = P_world
        else:
            Pw = P_world
        P_world_all[i] = np.where(valid[...,None], Pw, np.array(np.nan, dtype=points_dtype))
        Conf_all[i]    = np.where(valid, np.array(100, dtype=conf_dtype), np.array(0, dtype=conf_dtype))

    predictions = {
        "images":    images.to(dtype) if keep_images else None,  # (S,3,H,W) or None
        "intrinsic": K_mats,               # (S,3,3)
        "extrinsic": w2c_mats,             # (S,3,4)
        "world_points":      P_world_all,  # (S,H,W,3) [points_dtype] with NaNs
        "world_points_conf": Conf_all,     # (S,H,W)   [conf_dtype]
    }
    return predictions, (H, W)


sys.path.append("vggt/")


from voxel.utils import *
from voxel.voxel import *
from voxel.align import *

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
POINTS = "world_points"
CONF   = "world_points_conf"
threshold = 62.0     # our GT points use conf=100 so they pass
voxel_size = 0.01
VIZ = False
z_clip_map = [-50, 20]

# Map/world alignment you used before (keep as-is if your voxel code expects it)
R_w2m = np.array([[1,0,0],[0,0,1],[0,1,0]], dtype=np.float32)
t_w2m = np.zeros(3, dtype=np.float32)

vox = TorchSparseVoxelGrid(
    origin_xyz=np.zeros(3, dtype=np.float32),
    params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
    device=device, dtype=torch.float32
)

root = "/Users/marvin/Documents/Thesis/repo/dataset_generation/habitat/frames_kitchen"  # folder that contains intrinsics.json and time_XXXXX/
intrinsics_json = os.path.join(root, "intrinsics.json")
time_dirs = sorted([d for d in os.listdir(root) if d.startswith("time_")])

for i, td in enumerate(time_dirs):
    print("Timeframe:", td)
    # predictions, (H, W) = load_habitat_time_as_predictions(
    #     os.path.join(root, td),
    #     intrinsics_json,
    #     device=device,
    #     dtype=dtype,
    #     # sample_stride=2,     # tune for density/speed
    # )
    
    predictions, (H, W) = load_habitat_time_as_predictions(
        os.path.join(root, td),
        intrinsics_json,
        device=device,                 # keep CPU to save VRAM
        dtype=dtype,
        frame_stride=1,              
        sample_stride=2,              # every 4th pixel -> 16x fewer points
        keep_images=True,            # skip images if you don't need them
        points_dtype=np.float32,      # 2x smaller
        conf_dtype=np.float32
    )


    # Align / rotate into your "map" frame (same as you do with VGGT)
    WPTS = predictions[POINTS]
    # list → list (in-place rotation)
      
    WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
    Rmw, tmw, info = align_pointcloud(WPTS_m, inlier_dist=voxel_size*0.75)
    WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
    predictions[POINTS] = WPTS_m

    if VIZ:
        visualize_vggt_pointcloud(predictions, key=POINTS, conf_key=CONF, threshold=threshold)

    # And proceed exactly as you did:
    frames_map, cam_centers_map, conf_map, (S,H,W) = build_frames_and_centers_vectorized(
        predictions,
        POINTS=POINTS,
        CONF=CONF,
        threshold=threshold,
        Rmw=R_w2m @ Rmw, tmw=t_w2m + tmw,
        z_clip_map=z_clip_map,
    )

    align_to_voxel = False #(i > 0)
    # if i == 0:
    #     vox.begin_bootstrap()

    vox, bev, meta = build_maps_from_points_and_centers_torch(
        frames_map, cam_centers_map, conf_map, vox,
        align_to_voxel=align_to_voxel,
        voxel_size=voxel_size,
        bev_window_m=(15.0, 15.0),
        bev_origin_xy=(-15.0, -7.0),#(-4.0, -10.0),
        z_clip_vox=(-np.inf, np.inf),
        z_band_bev=(0.1, 1),
        max_range_m=None,
        carve_free=True,
        samples_per_voxel=1,
        ray_stride=2
    )

    # if i == 0:
    #     vox.end_bootstrap()
    #     vox.lock_long_term()

    # save if you want (same as your VGGT code)
    
    save_root = "kitchen_habitat/"
    # Files
    # Files
    save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
    export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

    vox.next_epoch()
