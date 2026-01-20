# run_predict.py
import os
import torch
import pytorch_lightning as pl


from train import TrainConfig, HabitatDataModule, VoxelUpdaterSystem
from voxel.utils import *


def main():
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full/voxup-epoch=07-val_loss_total=12.8810.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full/voxup-epoch=07-val_loss_total=10.9045.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full2/voxup-epoch=09-val_loss_total=8.8623.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full3/voxup-epoch=07-val_loss_total=8.8636.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full5/voxup-epoch=02-val_loss_total=10.3785.ckpt"
    out_dir = "/cluster/scratch/kochmar/predict_outputs_one/"
    os.makedirs(out_dir, exist_ok=True)

    STEP = 20

    # IMPORTANT: cfg must match what the checkpoint expects (feature_dim, voxel_size, etc.)
    cfg = TrainConfig(
        dataset_root="/cluster/scratch/kochmar/renders/",
        gt_voxels_file="gt_voxels_per_timestep_new",
        precomputed_cache_file="precomputed_cache",
        pose_file="gt_poses_new",
        seq_file="seq_manifest.json",
        voxel_size=0.2,
        #voxel_size=0.01,
        radius_m=1,
        topk=8,
        temp=0.5,
        feature_dim=32,
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
        train_val_split=0.1,  # doesn't matter if you pass your own loader below
        skip=True,
        seq_list=os.path.join(cfg.dataset_root, cfg.seq_file),
        step=STEP
    )
    dm.setup("predict")

    # Load model from ckpt
    model = VoxelUpdaterSystem.load_from_checkpoint(
        ckpt_path,
        strict=False,
        cfg=cfg,   # <- you already support this pattern
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
        bevs, bevs_gt = out
        for j, (bev, bev_gt) in enumerate(zip(bevs, bevs_gt)):
            print(j)
            save_bev(bev, meta, out_dir + f"bev_{i}_{j}.png", out_dir + f"bev_{i}_{j}_np.npy", out_dir + f"bev_{i}_{j}_meta.json")
            save_bev(bev_gt, meta, out_dir + f"bev_{i}_{j}_gt.png", out_dir + f"bev_{i}_{j}_gt_np.npy", out_dir + f"bev_{i}_{j}_gt_meta.json")

      


if __name__ == "__main__":
    main()
