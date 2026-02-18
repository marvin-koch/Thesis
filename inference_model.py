# run_predict.py
import os
import torch
import pytorch_lightning as pl


from train import TrainConfig, HabitatDataModule, VoxelUpdaterSystem
from voxel.utils import *

import matplotlib.patches as mpatches

def save_bev_comparison(bev_pred, bev_gt, meta, path="bev_comparison.png"):
    """
    Overlays Pred and GT BEVs to visualize errors.
    Handles both 2D (H,W) occupancy maps and 3D (H,W,3) RGB visualizations.
    """
    # 1. Helper to convert any input to a 2D Boolean Mask
    def to_binary(bev):
        # Handle PyTorch Tensors
        if torch.is_tensor(bev):
            bev = bev.detach().cpu().numpy()

        # Handle 3D RGB (H, W, 3) -> 2D Mask (H, W)
        if bev.ndim == 3:
            # If it's an image, assume non-black pixels = occupied
            return np.any(bev > 0, axis=-1)

        # Handle 2D Maps (H, W)
        if bev.dtype.kind == 'f':
            return bev > 0.5  # Probability > 0.5
        else:
            return bev == 100 # Int class 100 (from standard occupancy)

    # 2. Get Binary Masks
    pred_wall = to_binary(bev_pred)
    gt_wall   = to_binary(bev_gt)

    # 3. Get Dimensions (Robust Unpacking)
    H, W = pred_wall.shape  # Unpack from the 2D mask, not the original input

    # 4. Create RGB Error Map
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # --- Logic ---
    # Green:  Match (TP)
    # Red:    Ghost (FP)
    # Blue:   Missed (FN)
    # Grey:   Both Empty (TN) - Optional background color

    tp_mask = pred_wall & gt_wall
    fp_mask = pred_wall & (~gt_wall)
    fn_mask = (~pred_wall) & gt_wall

    # Apply Colors
    img[tp_mask] = [0, 255, 0]      # Green
    img[fp_mask] = [255, 0, 0]      # Red
    img[fn_mask] = [0, 100, 255]    # Blue

    # Optional: Draw faint grey for empty space if you want context (requires known free space)
    # For now, black background is standard for error maps.

    # 5. Plot
    res = float(meta.get("resolution", 0.05))
    ox, oy = meta.get("origin_xy", (0.0, 0.0))
    extent = [ox, ox + W * res, oy, oy + H * res]

    plt.figure(figsize=(10, 10))
    plt.imshow(img, origin="lower", extent=extent)

    patches = [
        mpatches.Patch(color='#00FF00', label='Correct (TP)'),
        mpatches.Patch(color='#FF0000', label='Ghost (FP)'),
        mpatches.Patch(color='#0064FF', label='Missed (FN)'),
    ]
    plt.legend(handles=patches, loc='upper right', fontsize='small', framealpha=0.9)
    plt.title(f"2D Occupancy Comparison")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()

    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Viz] Saved comparison to {path}")

def main():
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full/voxup-epoch=07-val_loss_total=12.8810.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full/voxup-epoch=07-val_loss_total=10.9045.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full2/voxup-epoch=09-val_loss_total=8.8623.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full3/voxup-epoch=07-val_loss_total=8.8636.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full5/voxup-epoch=05-val_loss_total=9.1113.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full7/voxup-epoch=05-val_loss_total=8.7155.ckpt"

    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full7/voxup-epoch=08-val_loss_total=8.0448.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full8/voxup-epoch=11-val_loss_total=8.2162.ckpt"

    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full9/voxup-epoch=03-val_loss_total=7.4241.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full9/voxup-epoch=05-val_loss_total=7.4378.ckpt"

    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=05-val_loss_total=7.3701.ckpt"

    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=09-val_loss_total=7.5282.ckpt"



    #sigma 3
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=09-val_loss_total=7.6451.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full_finetune/voxup-epoch=01-val_loss_total=7.9398.ckpt"

    #sigam 2
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=07-val_loss_total=7.9987.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=03-val_loss_total=7.8870.ckpt"


    #sigma 1
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=13-val_loss_total=8.0316.ckpt"



    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full_finetune/voxup-epoch=01-val_loss_total=20.6306.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full_finetune/voxup-epoch=09-val_loss_total=21.1281.ckpt"



    out_dir = "/cluster/scratch/kochmar/predict_outputs_eval/"
    os.makedirs(out_dir, exist_ok=True)

    STEP = 1

    # IMPORTANT: cfg must match what the checkpoint expects (feature_dim, voxel_size, etc.)
    cfg = TrainConfig(
        dataset_root="/cluster/scratch/kochmar/renders3/",
        #dataset_root="/cluster/scratch/kochmar/eval/",

        #gt_voxels_file="gt_voxels_per_timestep_new_3",
        gt_voxels_file="gt_voxels_per_timestep_new_2",
        precomputed_cache_file="precomputed_cache",
        #precomputed_cache_file="precomputed_cache_2",
        #pose_file="gt_poses_new_3",
        pose_file="gt_poses_new_2",
        seq_file="seq_manifest.json",
        voxel_size=0.2,
        #voxel_size=0.01,
        radius_m=1,
        topk=8,
        temp=0.5,
        #feature_dim=32,
        feature_dim=16,
        occ_decoder_hidden=64,
        lr=3e-4,
        max_epochs=200,
        batch_size=1,
        num_workers=0,
        precision="bf16",
        skip=True,
        weight_decay=0.05,
        lambda_occ=1.0,
        lambda_temp=0.05,
        lambda_ent=1e-3,
        lambda_tv=1e-4,
    )

    dm = HabitatDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=1,
        num_workers=cfg.num_workers,
        size=512,
        verbose=False,
        #train_val_split=0.3,  # doesn't matter if you pass your own loader below
        train_val_split=1.0,  # doesn't matter if you pass your own loader below
        skip=True,
        seq_list=os.path.join(cfg.dataset_root, cfg.seq_file),
        step=STEP
    )
    dm.setup("predict")

    # Load model from ckpt
    model = VoxelUpdaterSystem.load_from_checkpoint(
        ckpt_path,
        strict=False,
        cfg=cfg,   # <- you already suppor5 this pattern
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.precision,
        logger=False,
        enable_checkpointing=False,
    )

    # Use val_dataloader() or train_dataloader() depending on what you want to run on
    outputs = trainer.predict(model, dataloaders=dm.val_dataloader())


    meta = {}
    # outputs is a list: one entry per sequence (batch). Your predict_step returns (bev, bev_gt).
    for i, out in enumerate(outputs):
        if out is None:
            print("out is None")
            continue
        bevs, bevs_gt, bevs_baseline = out
        for j, (bev, bev_gt, bev_baseline) in enumerate(zip(bevs, bevs_gt, bevs_baseline)):
            print(j)
            #save_bev(bev, meta, out_dir + f"bev_{i}_{j}.png", out_dir + f"bev_{i}_{j}_np.npy", out_dir + f"bev_{i}_{j}_meta.json")
            save_bev(bev, meta, out_dir + f"bev_{i}_{j}.png", None, None)
            #save_bev(bev_gt, meta, out_dir + f"bev_{i}_{j}_gt.png", out_dir + f"bev_{i}_{j}_gt_np.npy", out_dir + f"bev_{i}_{j}_gt_meta.json")
            save_bev(bev_gt, meta, out_dir + f"gt_bev_{i}_{j}.png", None, None)
            #save_bev(bev_baseline, meta, out_dir + f"bev_{i}_{j}_baseline.png", out_dir + f"bev_{i}_{j}_baseline_np.npy", out_dir + f"bev_{i}_{j}_baseline_meta.json")
            save_bev(bev_baseline, meta, out_dir + f"baseline_bev_{i}_{j}.png", None, None)
            save_bev_comparison(
                bev, 
                bev_gt, 
                meta, 
                path=f"{out_dir}compare_step_{i}_{j}.png"
            )
      


if __name__ == "__main__":
    main()
