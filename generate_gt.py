import os
import numpy as np
import torch
from train import HabitatSeqDataset
import pow3r2.tools.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as li



from voxel.voxel import TorchSparseVoxelGrid, VoxelParams
from voxel.utils import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images


from tqdm import tqdm

from inference.utils import *

@torch.no_grad()
def save_sparse_voxel_grid(grid: TorchSparseVoxelGrid, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # origin
    origin = getattr(grid, "origin_xyz", None)
    if origin is None:
        origin = getattr(grid, "origin", None)
    if origin is None:
        raise ValueError("Grid has no origin_xyz/origin attribute")

    origin = torch.as_tensor(origin, dtype=torch.float32).detach().cpu().numpy()
    voxel_size = np.array([grid.p.voxel_size], dtype=np.float32)

    keys = grid.keys.detach().cpu().numpy()      # int64
    vals = grid.vals_st.detach().cpu().numpy()   # float32

    # --- DEBUGGING STATS ---
    # We use the threshold 0.0 or the grid's own threshold to decide what is "Occupied"
    # Usually vals > 0 means occupied log-odds.
    n_total = keys.shape[0]
    n_occupied = (vals > 0.0).sum()
    n_empty = n_total - n_occupied
    
    print(f"\n[GT CHECK] {os.path.basename(path)}")
    print(f"  Total Voxels:    {n_total}")
    print(f"  Occupied Walls:  {n_occupied}")
    print(f"  Empty Air:       {n_empty}")
    
    if n_total > 0:
        ratio = n_occupied / n_total
        print(f"  Wall Ratio:      {ratio:.2%}")
        
        if ratio > 0.50:
            print("  ⚠️ WARNING: >50% Walls! You are likely missing free space (Ray Tracing failed).")
        elif n_total < 1000:
            print("  ⚠️ WARNING: Extremely low voxel count! Points likely clipped.")
    else:
        print("  ⚠️ ERROR: Grid is empty!")
    # -----------------------

    np.savez_compressed(
        path,
        origin=origin,
        voxel_size=voxel_size,
        keys=keys,
        vals=vals,
    )
    # print(f"[GT] Saved voxel grid to {path}")

def build_gt_voxel_for_timestep(
    imgs,
    model: AsymmetricCroCo3DStereo,
    device: torch.device,
    voxel_size: float,
) -> TorchSparseVoxelGrid:
    """
    Compute GT voxel grid for a single timestep (one list of imgs).
    This is a per-timestep version of your inference_gt().
    """
    POINTS = "world_points"
    CONF   = "world_points_conf"
    threshold = 1.0
    z_clip_map = (-1.0, 2.0)

    # rotation to map world->metric frame (same as in your code)
    R_w2m_np = np.array([[0, 0, -1],
                         [-1, 0, 0],
                         [0, -1, 0]], dtype=np.float32)
    t_w2m_np = np.zeros(3, dtype=np.float32)
    R_w2m = torch.from_numpy(R_w2m_np).to(device=device, dtype=torch.float32)
    t_w2m = torch.from_numpy(t_w2m_np).to(device=device, dtype=torch.float32)

    # --- DUSt3R prediction (Accelerated) ---
    # We use inference_mode for speed. Autocast is helpful but explicit casting above handles the hard crash.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = get_reconstructed_scene_no_opt(0, ".", imgs, model, device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0)

    # --- normalize images like in your inference_gt() ---
#    for d in imgs:
#        t = d["img"]  # (1,3,H,W) or (3,H,W)

#        t = t.float()
#        if t.max() > 1.0:
#            t = t / 255.0
#        d["img"] = t.clamp(0, 1)

    # --- DUSt3R prediction ---
    #predictions = get_reconstructed_scene_no_opt(0, ".", imgs, model, device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0)


    with torch.no_grad():
        # ==============================
        # 🛑 SCALE FIX: Dollhouse -> Real House
        # ==============================
        raw_pts = predictions["world_points"]
        # Calculate current scale (how big is the scene?)
        current_size = torch.median(torch.norm(raw_pts, dim=1))

        # Target 5.0 meters (typical room depth)
        target_size = 5.0

        scale_factor = target_size / (current_size + 1e-6)
        print(f"[GT] Scaling Scene: {current_size:.2f}m -> 3.00m (Factor: {scale_factor:.2f}x)")

        # 1. Scale Points
        predictions["world_points"] = raw_pts * scale_factor

        # 2. Scale Camera Positions (Translations)
        # Iterate over the batch of extrinsics to scale the translation vector
        # Extrinsic is typically [R | t]. Scaling t moves cameras apart.
        # Check shape: usually (N, 4, 4)
        if isinstance(predictions["extrinsic"], torch.Tensor):
            predictions["extrinsic"][:, :3, 3] *= scale_factor
        elif isinstance(predictions["extrinsic"], list):
            for i in range(len(predictions["extrinsic"])):
                predictions["extrinsic"][i][:3, 3] *= scale_factor
        # ==============================




        # keep only needed keys
        needed = {"images", "extrinsic", POINTS, CONF}
        for k in list(predictions.keys()):
            if k not in needed:
                del predictions[k]

        # --- align points ---
        WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)


        Rmw, tmw, _ = align_pointcloud_torch_fast(
            WPTS_m,
            inlier_dist=voxel_size * 0.75,
        )
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
        predictions[POINTS] = WPTS_m

        camera_R = R_w2m @ Rmw
        camera_t = t_w2m + tmw

        frames_map, cam_centers_map, conf_map, images_map, _, (S, H, W), frame_ids = \
            build_frames_and_centers_vectorized_torch(
                predictions,
                POINTS=POINTS,
                CONF=CONF,
                threshold=threshold,
                Rmw=camera_R,
                tmw=camera_t,
                z_clip_map=z_clip_map,
                return_flat=True,
            )
            
        total_points = sum(f.shape[0] for f in frames_map)
        print(f"  [Points] Survivors after filtering/clipping: {total_points}")
        if total_points < 100:
            print("  🔴 CRITICAL: Almost no points left! Check your z_clip_map or threshold.")
            
        # --- make a *fresh* GT grid for this timestep only ---
        vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
            device=device,
        )

        vox_gt, bev, meta = build_maps_from_points_and_centers_torch(
            frames_map,
            cam_centers_map,
            conf_map,
            vox_gt,
            align_to_voxel=False,
            voxel_size=voxel_size,
            bev_window_m=(5.0, 5.0),
            bev_origin_xy=(-2.0, -2.0),
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(0.02, 0.5),
            samples_per_voxel=2.0,
            ray_stride=4,
            max_free_rays=10000,
            frame_ids=frame_ids,
        )

    # vox_gt.next_epoch()  # optional for bookkeeping

    # free some stuff
    del predictions, frames_map, cam_centers_map, conf_map, images_map
    return vox_gt


def main():
    dataset_root = "/cluster/scratch/kochmar/renders/"   # same as in your TrainConfig
    voxel_size = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DUSt3R weights path – same as in your VoxelUpdaterSystem __init__
    weights_path = "/cluster/home/kochmar/Thesis/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Use your HabitatSeqDataset defined in train.py
    dataset = HabitatSeqDataset(
        dataset_root=dataset_root,
        size=512,
        verbose=False,
    )

    seqs = dataset.seq_paths
    print(f"[GT] Found {len(seqs)} sequences.")

    out_root = os.path.join(dataset_root, "gt_voxels_per_timestep_005_v2")
    os.makedirs(out_root, exist_ok=True)

    #for seq_idx in range(len(seqs)):
    for seq_idx in range(len(seqs), -1, -1):
        print(seq_idx)
        batch = dataset[seq_idx]        # __getitem__ returns dict with seq info
        seq_id = batch["seq_id"]
        imgs_t = batch["imgs_t"]
        print(seq_id)
        print(batch["seq_path"])
        T = batch["timesteps"]

        print(f"\n[GT] Sequence {seq_idx+1}/{len(seqs)}: {seq_id} (T={T})")


        for t, imgs in enumerate(imgs_t):
            if t % 10 != 0:
                continue

            out_path = os.path.join(out_root, f"{seq_id}_t{t:04d}_gt.npz")
            if os.path.exists(out_path):
                print(f"[GT]   skip t={t} (exists)")
                continue


            print(f"[GT]   computing t={t}/{T-1}")
            vox_gt = build_gt_voxel_for_timestep(imgs, model, device, voxel_size)
            save_sparse_voxel_grid(vox_gt, out_path)

    print("\n[GT] Done.")




if __name__ == "__main__":
    main()
