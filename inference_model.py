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
        #dataset_root="/cluster/scratch/kochmar/renders3/",
        #dataset_root="/cluster/scratch/kochmar/eval/",
        dataset_root="/cluster/scratch/kochmar/hm3d_gt/",

        real_gt_voxels_file="hm3d_voxels",

        #gt_voxels_file="gt_voxels_per_timestep_new_3",
        #gt_voxels_file="gt_voxels_per_timestep_new_2",
        gt_voxels_file="gt_voxels_per_timestep_new",


        precomputed_cache_file="precomputed_cache",
        #precomputed_cache_file="precomputed_cache_2",

        #pose_file="gt_poses_new_3",
        #pose_file="gt_poses_new_2",
        pose_file="gt_poses_new",

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

    # --- Aggregate metrics across all sequences ---
    all_metrics = {}   # key -> list of all per-step values across dataset

    meta = {}
    # outputs is a list: one entry per sequence (batch). Your predict_step returns (bev, bev_gt, bev_baseline, metrics_buffer).
    for i, out in enumerate(outputs):
        if out is None:
            print("out is None")
            continue
        bevs, bevs_gt, bevs_baseline, seq_metrics = out

        # Accumulate per-step values into global buffer
        for key, vals in seq_metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].extend(vals)

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

    # --- Print dataset-wide average metrics ---
    def avg(vals):
        return sum(vals) / len(vals) if len(vals) > 0 else 0.0

    print("\n" + "=" * 70)
    print("DATASET-WIDE AVERAGE METRICS (across all sequences)")
    print("=" * 70)

    if "gt_occ_iou" in all_metrics and len(all_metrics["gt_occ_iou"]) > 0:
        print(f"  GT Metrics:")
        print(f"    IoU:                 {avg(all_metrics['gt_occ_iou']):.4f}")
        print(f"    Recall:              {avg(all_metrics['gt_occ_recall']):.4f}")
        print(f"    Precision:           {avg(all_metrics['gt_occ_precision']):.4f}")
        print("-" * 70)

    print(f"  MODEL Metrics:")
    print(f"    IoU:                 {avg(all_metrics.get('occ_iou', [])):.4f}")
    print(f"    Recall:              {avg(all_metrics.get('occ_recall', [])):.4f}")
    print(f"    Precision:           {avg(all_metrics.get('occ_precision', [])):.4f}")
    print("-" * 70)
    print(f"  BASELINE Metrics:")
    print(f"    IoU:                 {avg(all_metrics.get('baseline_occ_iou', [])):.4f}")
    print(f"    Recall:              {avg(all_metrics.get('baseline_occ_recall', [])):.4f}")
    print(f"    Precision:           {avg(all_metrics.get('baseline_occ_precision', [])):.4f}")
    print("-" * 70)
    print(f"  STATIC Metrics:")
    print(f"    IoU:                 {avg(all_metrics.get('static_occ_iou', [])):.4f}")
    print(f"    Recall:              {avg(all_metrics.get('static_occ_recall', [])):.4f}")
    print(f"    Precision:           {avg(all_metrics.get('static_occ_precision', [])):.4f}")
    print("-" * 70)

    # --- Extra baselines (TSDF, EMA, LastFrame, ConfWt, MonoDepth) ---
    extra_baseline_names = ["tsdf", "ema", "lastframe", "confwt", "monodepth"]
    extra_baseline_labels = {
        "tsdf":      "TSDF Fusion",
        "ema":       "EMA Fusion",
        "lastframe": "Last-Frame",
        "confwt":    "Conf-Weighted",
        "monodepth": "MONODEPTH (DepthAnythingV2 + OctoMap)",
    }
    for bname in extra_baseline_names:
        key = f"{bname}_occ_iou"
        if key in all_metrics and len(all_metrics[key]) > 0:
            label = extra_baseline_labels.get(bname, bname.upper())
            print(f"  {label} Metrics:")
            print(f"    IoU:                 {avg(all_metrics.get(f'{bname}_occ_iou', [])):.4f}")
            print(f"    Recall:              {avg(all_metrics.get(f'{bname}_occ_recall', [])):.4f}")
            print(f"    Precision:           {avg(all_metrics.get(f'{bname}_occ_precision', [])):.4f}")
            print("-" * 70)

    if "gt_dyn_iou" in all_metrics and len(all_metrics["gt_dyn_iou"]) > 0:
        print(f"  Dynamic Metrics GT (Changes Only):")
        print(f"    Dynamic IoU:         {avg(all_metrics['gt_dyn_iou']):.4f}")
        print(f"    Appearing Recall:    {avg(all_metrics.get('gt_dyn_recall_appearing', [])):.4f}")
        print(f"    Disappearing Recall: {avg(all_metrics.get('gt_dyn_recall_disappearing', [])):.4f}")
        print(f"    Ghost Rate:          {avg(all_metrics.get('gt_dyn_ghost_rate', [])):.4f}")
        print("-" * 70)

    print(f"  Dynamic Metrics MODEL (Changes Only):")
    print(f"    Dynamic IoU:         {avg(all_metrics.get('dyn_iou', [])):.4f}")
    print(f"    Appearing Recall:    {avg(all_metrics.get('dyn_recall_appearing', [])):.4f}")
    print(f"    Disappearing Recall: {avg(all_metrics.get('dyn_recall_disappearing', [])):.4f}")
    print(f"    Ghost Rate:          {avg(all_metrics.get('dyn_ghost_rate', [])):.4f}")
    print("-" * 70)
    print(f"  Dynamic Metrics BASELINE (Changes Only):")
    print(f"    Dynamic IoU:         {avg(all_metrics.get('baseline_dyn_iou', [])):.4f}")
    print(f"    Appearing Recall:    {avg(all_metrics.get('baseline_dyn_recall_appearing', [])):.4f}")
    print(f"    Disappearing Recall: {avg(all_metrics.get('baseline_dyn_recall_disappearing', [])):.4f}")
    print(f"    Ghost Rate:          {avg(all_metrics.get('baseline_dyn_ghost_rate', [])):.4f}")
    print("-" * 70)
    print(f"  Dynamic Metrics STATIC (Changes Only):")
    print(f"    Dynamic IoU:         {avg(all_metrics.get('static_dyn_iou', [])):.4f}")
    print(f"    Appearing Recall:    {avg(all_metrics.get('static_dyn_recall_appearing', [])):.4f}")
    print(f"    Disappearing Recall: {avg(all_metrics.get('static_dyn_recall_disappearing', [])):.4f}")
    print(f"    Ghost Rate:          {avg(all_metrics.get('static_dyn_ghost_rate', [])):.4f}")
    print("-" * 70)

    # --- Extra baseline dynamic metrics ---
    for bname in extra_baseline_names:
        key = f"{bname}_dyn_iou"
        if key in all_metrics and len(all_metrics[key]) > 0:
            label = extra_baseline_labels.get(bname, bname.upper())
            print(f"  Dynamic Metrics {label} (Changes Only):")
            print(f"    Dynamic IoU:         {avg(all_metrics.get(f'{bname}_dyn_iou', [])):.4f}")
            print(f"    Appearing Recall:    {avg(all_metrics.get(f'{bname}_dyn_recall_appearing', [])):.4f}")
            print(f"    Disappearing Recall: {avg(all_metrics.get(f'{bname}_dyn_recall_disappearing', [])):.4f}")
            print(f"    Ghost Rate:          {avg(all_metrics.get(f'{bname}_dyn_ghost_rate', [])):.4f}")
            print("-" * 70)

    if "gt_chamfer_dist" in all_metrics and len(all_metrics["gt_chamfer_dist"]) > 0:
        print(f"    GT Chamfer Dist:       {avg(all_metrics['gt_chamfer_dist']):.4f}")
    print(f"    Chamfer Dist:          {avg(all_metrics.get('chamfer_dist', [])):.4f}")
    print(f"    Baseline Chamfer Dist: {avg(all_metrics.get('baseline_chamfer_dist', [])):.4f}")
    print(f"    Static Chamfer Dist:   {avg(all_metrics.get('static_chamfer_dist', [])):.4f}")
    print("-" * 70)
    print(f"  Temporal Flicker Score (Lower = More Stable):")
    print(f"    GT TFS:              {avg(all_metrics.get('tfs_gt', [])):.4f}")
    print(f"    Model TFS:           {avg(all_metrics.get('tfs_model', [])):.4f}")
    print(f"    Baseline TFS:        {avg(all_metrics.get('tfs_baseline', [])):.4f}")
    for bname in extra_baseline_names:
        key = f"{bname}_tfs"
        if key in all_metrics and len(all_metrics[key]) > 0:
            label = extra_baseline_labels.get(bname, bname.upper())
            print(f"    {label} TFS: {avg(all_metrics[key]):.4f}")
    print("=" * 70)
      


if __name__ == "__main__":
    main()