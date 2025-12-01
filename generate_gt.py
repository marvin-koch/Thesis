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




def save_sparse_voxel_grid(grid: TorchSparseVoxelGrid, path: str):
    """
    Save a sparse voxel grid as compressed npz:
      origin: (3,) float32
      voxel_size: (1,) float32
      keys: (N,) int64
      vals: (N,) float32  (occupancy probs or log-odds)
    """
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

    np.savez_compressed(
        path,
        origin=origin,
        voxel_size=voxel_size,
        keys=keys,
        vals=vals,
    )
    print(f"[GT] Saved voxel grid with {keys.shape[0]} voxels to {path}")
    
############################################
# 1. GT VOXEL GENERATOR (NO LIGHTNING)
############################################

# class GTGenerator:
#     def __init__(self, voxel_size, device, dust3r_weights):
#         self.voxel_size = voxel_size
#         self.device = device

#         # DUSt3R model (fp32 for stability)
#         self.model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_weights).eval()
#         self.model = self.model.to(device)
#         for p in self.model.parameters():
#             p.requires_grad = False

#     @torch.no_grad()
#     def run_sequence(self, imgs_t):
#         """
#         imgs_t : list[timestep][image_dict]
#         Returns the final sparse GT voxel grid.
#         """

#         # Initialize empty GT grid
#         vox_gt = TorchSparseVoxelGrid(
#             origin_xyz=np.zeros(3, dtype=np.float32),
#             params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
#             device=self.device,
#         )

#         for i, imgs in enumerate(tqdm(imgs_t, desc="Timesteps")):
#             # --- convert images to proper format ---
#             POINTS = "world_points"
#             CONF   = "world_points_conf"
#             z_clip = (-0.1, 0.3)

#             # convert image tensors
#             for d in imgs:
#                 timg = d["img"]
#                 if timg.ndim == 4 and timg.shape[0] == 1:
#                     timg = timg[0]
#                 d["img"] = (timg.float() / 255.0).clamp(0,1)

#             # 1) DUSt3R inference
#             preds = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=self.projector)

#             # 2) Drop unwanted keys
#             for k in list(preds.keys()):
#                 if k not in {POINTS, CONF, "extrinsic"}:
#                     del preds[k]

#             # 3) Align points
#             pts = torch.from_numpy(preds[POINTS]).to(self.device)
#             R_w2m = torch.tensor([[0,0,-1],[-1,0,0],[0,-1,0]], dtype=torch.float32, device=self.device)
#             t_w2m = torch.zeros(3, device=self.device)

#             pts = rotate_points(pts, R_w2m, t_w2m)
#             Rmw, tmw, _ = align_pointcloud_torch_fast(
#                 pts,
#                 inlier_dist=self.voxel_size * 0.75,
#                 ransac_iters=200
#             )
#             pts = rotate_points(pts, Rmw, tmw)
#             preds[POINTS] = pts

#             # 4) Build GT voxel update for this timestep
#             frames_map, cam_centers_map, conf_map, images_map, _, (S,H,W), frame_ids = \
#                 build_frames_and_centers_vectorized_torch(
#                     preds,
#                     POINTS=POINTS, CONF=CONF,
#                     threshold=1.0,
#                     Rmw=R_w2m @ Rmw,
#                     tmw=t_w2m + tmw,
#                     z_clip_map=z_clip,
#                     return_flat=True
#                 )

#             vox_gt, _, _ = build_maps_from_points_and_centers_torch(
#                 frames_map,
#                 cam_centers_map,
#                 conf_map,
#                 vox_gt,
#                 align_to_voxel=False,
#                 voxel_size=self.voxel_size,
#                 bev_window_m=(5.0, 5.0),
#                 bev_origin_xy=(-2.0, -2.0),
#                 z_clip_vox=(-np.inf, np.inf),
#                 z_band_bev=(0.02, 0.5),
#                 samples_per_voxel=0.7,
#                 ray_stride=6,
#                 max_free_rays=10000,
#                 frame_ids=frame_ids,
#             )

#             vox_gt.next_epoch()

#         return vox_gt



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
    z_clip_map = (-0.1, 0.3)

    # rotation to map world->metric frame (same as in your code)
    R_w2m_np = np.array([[0, 0, -1],
                         [-1, 0, 0],
                         [0, -1, 0]], dtype=np.float32)
    t_w2m_np = np.zeros(3, dtype=np.float32)
    R_w2m = torch.from_numpy(R_w2m_np).to(device=device, dtype=torch.float32)
    t_w2m = torch.from_numpy(t_w2m_np).to(device=device, dtype=torch.float32)

    # --- normalize images like in your inference_gt() ---
#    for d in imgs:
#        t = d["img"]  # (1,3,H,W) or (3,H,W)

#        t = t.float()
#        if t.max() > 1.0:
#            t = t / 255.0
#        d["img"] = t.clamp(0, 1)

    # --- DUSt3R prediction ---
    predictions = get_reconstructed_scene_no_opt(0, ".", imgs, model, device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0)


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
        samples_per_voxel=0.7,
        ray_stride=6,
        max_free_rays=10000,
        frame_ids=frame_ids,
    )

    vox_gt.next_epoch()  # optional for bookkeeping

    # free some stuff
    del predictions, frames_map, cam_centers_map, conf_map, images_map
    return vox_gt


############################################
# 2. MAIN SCRIPT
############################################

# def main():
#     dataset_root = "/cluster/scratch/kochmar/renders/"
#     dust3r_weights = "/cluster/home/kochmar/Thesis/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
#     voxel_size = 0.10
#     device = torch.device("cuda")

#     out_root = os.path.join(dataset_root, "gt_voxels")
#     os.makedirs(out_root, exist_ok=True)

#     # load dataset (no lightning)
#     dataset = HabitatSeqDataset(dataset_root, size=512)
#     print(f"Found {len(dataset)} sequences.")

#     gtgen = GTGenerator(voxel_size, device, dust3r_weights)

#     for idx in range(len(dataset)):
#         batch = dataset[idx]
#         seq_id = batch["seq_id"]
#         out_file = os.path.join(out_root, f"{seq_id}_gt_vox.npz")

#         if os.path.exists(out_file):
#             print(f"[SKIP] {seq_id} already computed.")
#             continue

#         print(f"[GT] Processing {seq_id}")

#         vox_gt = gtgen.run_sequence(batch["imgs_t"])
#         save_sparse_voxel_grid(vox_gt, out_file)

#         print(f"[GT] Saved {seq_id} → {out_file}\n")




def main():
    dataset_root = "/cluster/scratch/kochmar/renders/"   # same as in your TrainConfig
    voxel_size = 0.10
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

    out_root = os.path.join(dataset_root, "gt_voxels_per_timestep")
    os.makedirs(out_root, exist_ok=True)

    for seq_idx in reversed(range(len(seqs))):
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
