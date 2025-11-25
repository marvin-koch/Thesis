import os
import numpy as np
import torch

from voxel.voxel import TorchSparseVoxelGrid, VoxelParams
from voxel.utils import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as li

from tqdm import tqdm
from train import HabitatSeqDataset
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

class GTGenerator:
    def __init__(self, voxel_size, device, dust3r_weights):
        self.voxel_size = voxel_size
        self.device = device

        # DUSt3R model (fp32 for stability)
        self.model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_weights).eval()
        self.model = self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def run_sequence(self, imgs_t):
        """
        imgs_t : list[timestep][image_dict]
        Returns the final sparse GT voxel grid.
        """

        # Initialize empty GT grid
        vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device,
        )

        for i, imgs in enumerate(tqdm(imgs_t, desc="Timesteps")):
            # --- convert images to proper format ---
            POINTS = "world_points"
            CONF   = "world_points_conf"
            z_clip = (-0.1, 0.3)

            # convert image tensors
            for d in imgs:
                timg = d["img"]
                if timg.ndim == 4 and timg.shape[0] == 1:
                    timg = timg[0]
                d["img"] = (timg.float() / 255.0).clamp(0,1)

            # 1) DUSt3R inference
            preds = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=self.projector)

            # 2) Drop unwanted keys
            for k in list(preds.keys()):
                if k not in {POINTS, CONF, "extrinsic"}:
                    del preds[k]

            # 3) Align points
            pts = torch.from_numpy(preds[POINTS]).to(self.device)
            R_w2m = torch.tensor([[0,0,-1],[-1,0,0],[0,-1,0]], dtype=torch.float32, device=self.device)
            t_w2m = torch.zeros(3, device=self.device)

            pts = rotate_points(pts, R_w2m, t_w2m)
            Rmw, tmw, _ = align_pointcloud_torch_fast(
                pts,
                inlier_dist=self.voxel_size * 0.75,
                ransac_iters=200
            )
            pts = rotate_points(pts, Rmw, tmw)
            preds[POINTS] = pts

            # 4) Build GT voxel update for this timestep
            frames_map, cam_centers_map, conf_map, images_map, _, (S,H,W), frame_ids = \
                build_frames_and_centers_vectorized_torch(
                    preds,
                    POINTS=POINTS, CONF=CONF,
                    threshold=1.0,
                    Rmw=R_w2m @ Rmw,
                    tmw=t_w2m + tmw,
                    z_clip_map=z_clip,
                    return_flat=True
                )

            vox_gt, _, _ = build_maps_from_points_and_centers_torch(
                frames_map,
                cam_centers_map,
                conf_map,
                vox_gt,
                align_to_voxel=False,
                voxel_size=self.voxel_size,
                bev_window_m=(5.0, 5.0),
                bev_origin_xy=(-2.0, -2.0),
                z_clip_vox=(-np.inf, np.inf),
                z_band_bev=(0.02, 0.5),
                samples_per_voxel=0.7,
                ray_stride=6,
                max_free_rays=10000,
                frame_ids=frame_ids,
            )

            vox_gt.next_epoch()

        return vox_gt



############################################
# 2. MAIN SCRIPT
############################################

def main():
    dataset_root = "/cluster/scratch/kochmar/renders/"
    dust3r_weights = "/cluster/home/kochmar/Thesis/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    voxel_size = 0.10
    device = torch.device("cuda")

    out_root = os.path.join(dataset_root, "gt_voxels")
    os.makedirs(out_root, exist_ok=True)

    # load dataset (no lightning)
    dataset = HabitatSeqDataset(dataset_root, size=512)
    print(f"Found {len(dataset)} sequences.")

    gtgen = GTGenerator(voxel_size, device, dust3r_weights)

    for idx in range(len(dataset)):
        batch = dataset[idx]
        seq_id = batch["seq_id"]
        out_file = os.path.join(out_root, f"{seq_id}_gt_vox.npz")

        if os.path.exists(out_file):
            print(f"[SKIP] {seq_id} already computed.")
            continue

        print(f"[GT] Processing {seq_id}")

        vox_gt = gtgen.run_sequence(batch["imgs_t"])
        save_sparse_voxel_grid(vox_gt, out_file)

        print(f"[GT] Saved {seq_id} → {out_file}\n")


if __name__ == "__main__":
    main()
