import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from voxel.latent_voxel import *

import pow3r2.tools.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo
import torch
import os
from dust3r.utils.image import load_images as li
from inference.utils import *
import numpy as np
import time
from voxel.utils import *
from voxel.latent_voxel import *
from voxel.voxel import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images
import os, shutil, json

# --- crash-safe profiling helpers ---

import torch.serialization as serialization
import argparse

from pytorch_lightning.loggers import WandbLogger

serialization.add_safe_globals([argparse.Namespace])

import logging
from torch.cuda.amp import autocast

logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)


import numpy as np
from voxel.voxel import TorchSparseVoxelGrid, VoxelParams
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import PIL
import copy

# Add this to your imports in train.py
from pytorch3d.ops import knn_points

from sklearn.manifold import TSNE
from pytorch3d.ops import knn_points
from voxel.utils import build_maps_from_points_and_centers_torch, rotate_points


try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


STEP = 1

def load_sparse_voxel_grid(path, device):
    data = np.load(path)
    origin = data["origin"].astype(np.float32)
    voxel_size = float(data["voxel_size"][0])
    keys = torch.from_numpy(data["keys"]).to(device)
    vals = torch.from_numpy(data["vals"]).to(device, dtype=torch.float32)


    n_total = keys.shape[0]
    n_occupied = (vals > 0.0).sum()
    n_empty = n_total - n_occupied
    
    #print(f"  Total Voxels:    {n_total}")
    #print(f"  Occupied Walls:  {n_occupied}")
    #print(f"  Empty Air:       {n_empty}")
    
    vox_gt = TorchSparseVoxelGrid(
        origin_xyz=origin,
        params=VoxelParams(voxel_size=0.2, promote_hits=2),
        device=device,
    )
    vox_gt.keys = keys
    vox_gt.vals_st = vals
    vox_gt.vals_lt = vals
    vox_gt.vals = vals


    return vox_gt

# --------------------------
# 1) Your modules (import these from your codebase)
# --------------------------
# from your_voxel_impl import TorchSparseVoxelGrid, VoxelParams, LatentToOccupancyDecoder
# from your_point_feats import PointNeXtExtractor  # or your feature extractor
# from your_dust3r_wrapper import Dust3RTeacher     # wraps full DUSt3R inference -> teacher supervision

# --------------------------
# 2) Config
# --------------------------
@dataclass
class TrainConfig:
    # data
    dataset_root: str
    voxel_size: float = 0.10
    real_gt_voxels_file: str = None,
    gt_voxels_file: str = "gt_voxels_per_timestep_005_v2",
    precomputed_cache_file: str ="precomputed_cache",
    pose_file: str ="gt_pose",
    seq_file: str ="seq_manifest.json",
    radius_m: float = 0.25
    topk: int = 8
    temp: float = 0.5

    # model
    feature_dim: int = 64         # latent/feature size (match PointNeXt output)
    occ_decoder_hidden: int = 64
    ema_to_st: float = 0.4        # how strongly decoded prob refreshes ST log-odds during training updates

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    batch_size: int = 1           # 1 sequence per batch (we iterate over timesteps inside)
    num_workers: int = 4
    precision: str = "16-mixed"

    # losses
    lambda_occ: float = 1.0
    lambda_temp: float = 0.2      # temporal consistency weight
    lambda_ent: float = 1e-3      # routing entropy reg
    lambda_tv: float = 1e-4       # (optional) spatial TV on occupancy

    # teacher supervision
    teacher_beam_every_t: bool = True  # run teacher for every timestep (offline precomputed if possible)
    skip: bool = False
    
    #n_accum: int = 4               # gradient accumulation steps
    n_accum: int = 1               # gradient accumulation steps
    stride: int = 4               # ray stride for voxel supervision



# --------------------------
# 5) LightningModule
# --------------------------
class VoxelUpdaterSystem(pl.LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg
        self.feature_dim = self.cfg.feature_dim
        # ---- core components (replace with your actual imports) ----
        #self.voxel_size = 0.01
        self.voxel_size = self.cfg.voxel_size
        self.vox = LatentVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device, feature_dim=self.feature_dim
        )
        
        self.vox = self.vox.to(self.device)
        
        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device 
        )
        
        weights_path = "naver/" + "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        weights_path = "/cluster/home/kochmar/Thesis/" + "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)

        self.model.eval()

        self.model = self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.keyframes = []
        # convenience buffer for device transfers
        self.register_buffer("_origin", torch.zeros(3), persistent=False)
        
        #self.projector = FeatureProjector(in_dim=768, out_dim=self.feature_dim)

        self.projector = FeatureProjector(
            in_dim=768,
            out_dim=self.feature_dim,
            hidden_dim=64,
            activation="gelu"   
        )


        self.automatic_optimization = False   # <<< add this
        
        self.bev_window_m=(50.0, 50.0)
        #self.bev_window_m=(5.0, 5.0)
        self.bev_origin_xy=(-25.0, -25.0)
        #self.bev_origin_xy=(-10.0, -10.0)

        #self.z_band_bev=(-0.75, 2.0)
        self.z_band_bev=(-2.0, 1.0)
        self.z_band_bev=(-1.0, 1.0)
        #self.z_band_bev=(-3.5, 4.5)

        self.fp_weight = 0.0


    def configure_optimizers(self):
    
        params = list(self.vox.parameters()) + list(self.projector.parameters())
        # if your feature extractor is finetuned, extend params with extractor params
        
        # opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # return opt
    
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.max_epochs, eta_min=self.cfg.lr * 0.1
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # CosineAnnealingLR is usually per-epoch (with T_max in epochs)
                "frequency": 1,
            },
        }



    def apply_projector_to_map(self, feat_map_raw, target_hw=(512, 512)):
        """
        Args:
            feat_map_raw: Tensor (C_in, H_small, W_small) e.g. (1024, 32, 32)
            target_hw: Tuple (H, W) target size e.g. (512, 512)
        """
        if feat_map_raw is None:
            return None
        
        if feat_map_raw.shape[-1] in [768, 1024]: 
            # Permute (H, W, C) -> (C, H, W)
            feat_map_raw = feat_map_raw.permute(2, 0, 1)

        # 1. Project at low resolution (Computationally cheap!)
        C_in, h, w = feat_map_raw.shape
        flat = feat_map_raw.flatten(1).permute(1, 0) # (h*w, 1024)
        
        proj = self.projector(flat) # (h*w, 32)
        
        # Reshape back to small spatial map
        proj = proj.permute(1, 0).view(self.feature_dim, h, w) # (32, 32, 32)
        
        # 2. Upsample to full resolution for the voxel grid
        # Use bilinear interpolation
        proj = F.interpolate(
            proj.unsqueeze(0),       # Add batch dim: (1, 32, 32, 32)
            size=target_hw,          # Target size: (512, 512)
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)                 # Remove batch dim
        
        proj = proj.permute(1, 2, 0) 
        return proj
    

    def inference(self, i, imgs, mst, Rmw=None, tmw=None, predictions=None, scale_factor=None, threshold=1.0):
        # --- GPU Timer Setup ---
        def get_event():
            return torch.cuda.Event(enable_timing=True)

        events = {k: (get_event(), get_event()) for k in [
            "prep_images", "dust3r_recon", "change_detection",
            "alignment", "voxel_building"
        ]}

        POINTS = "world_points"
        CONF = "world_points_conf"
        z_clip_map = (-3.0, 3.0)

        R_w2m = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        t_w2m = np.zeros(3, dtype=np.float32)
        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)

        if predictions is None:
            events["prep_images"][0].record() # Start Prep
            image_tensors = []
            for d in imgs:
                t = d["img"]
                if t.ndim == 4 and t.shape[0] == 1: t = t[0]
                t = t.detach().cpu().float()
                if t.max() > 1.0: t = t / 255.0
                image_tensors.append(t.clamp(0,1))
            image_tensors = torch.stack(image_tensors, dim=0).to(self.device)
            events["prep_images"][1].record() # End Prep

            if i < 1:
                events["dust3r_recon"][0].record()
                predictions = get_reconstructed_scene_no_opt(
                    i, ".", imgs, self.model, self.device, False, 512, "", "linear",
                    50, 1, True, False, True, False, 0.05, "oneref", 1, 0,
                    projector=self.projector
                )
                self.keyframes = image_tensors.clone()
                events["dust3r_recon"][1].record()
            else:
                events["change_detection"][0].record()
                changed_idx = changed_images(image_tensors, self.keyframes, thresh=0.000005)
                events["change_detection"][1].record()

                if len(changed_idx) < 2:
                    return None, None, None, None

                changed_idx = [0] + [x for x in changed_idx if x != 0]
                idx_t = torch.tensor(changed_idx, device=self.device, dtype=torch.long)
                self.keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))

                events["dust3r_recon"][0].record()
                mst = True
                predictions = get_reconstructed_scene_no_opt(
                    i, ".", imgs, self.model, self.device, False, 512, "", "linear",
                    50, 1, True, False, True, False, 0.05, "oneref", 1, 0,
                    changed_gids=changed_idx, projector=self.projector
                )
                events["dust3r_recon"][1].record()

            # Early cleanup of heavy prediction keys
            needed = {"images", "extrinsic", POINTS, CONF, "view_feats"}
            for k in list(predictions.keys()):
                if k not in needed: del predictions[k]

        # --- Transformation & Filtering ---
        events["alignment"][0].record()
        camera_R = Rmw @ R_w2m
        camera_t = t_w2m + tmw
        z_clip_map = (scale_factor * z_clip_map[0], scale_factor * z_clip_map[1])

        frames_map, conf_map, images_map, features_map, camera_centers, (S,H,W), frame_ids = filter_frames(
            predictions, POINTS=POINTS, CONF=CONF, FEAT="view_feats",
            threshold=threshold, Rmw=camera_R, tmw=camera_t, z_clip_map=z_clip_map,
        )
        events["alignment"][1].record()

        # --- Map Building (Deep Dive) ---
        v_events = {k: (get_event(), get_event()) for k in [
            "latent_aggregation", "sparse_insertion", "bev_generation"
        ]}
        v_events["latent_aggregation"][0].record()
        # This function likely does the heavy lifting:
        # project-and-pool features from 2D maps into 3D space
        vox, bev, meta = build_maps_from_latent_features(
            i,
            frames_map,
            conf_map,
            features_map,
            camera_centers,
            self.vox,
            voxel_size=self.voxel_size,
            bev_window_m=self.bev_window_m,
            bev_origin_xy=self.bev_origin_xy,
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(self.z_band_bev[0], self.z_band_bev[1]),
            frame_ids=frame_ids,
            radius=self.cfg.radius_m,
        )
        v_events["latent_aggregation"][1].record()

        self.vox = vox
        events["voxel_building"][1].record()

        # --- Report Granular Voxel Stats ---
        torch.cuda.synchronize()
        print(f"  [Voxel Breakdown]")
        for name, (start, end) in v_events.items():
            try:
                ms = start.elapsed_time(end)
                if ms > 0: print(f"    {name:20}: {ms:8.2f} ms")
            except: pass

        # --- Summary Timing ---
        torch.cuda.synchronize() # Wait for all GPU work to finish
        print(f"\n[Inference {i} Profiling]")
        for name, (start, end) in events.items():
            # Check if events were actually recorded (e.g. change detection might be skipped at i=0)
            try:
                ms = start.elapsed_time(end)
                if ms > 0:
                    print(f"  {name:20}: {ms:8.2f} ms")
            except: pass
        print("-" * 35)

        del predictions, frames_map, conf_map, images_map, features_map
        return bev, mst, Rmw, tmw

    def inferenceOG(self, i, imgs, mst, Rmw=None, tmw=None, predictions=None, scale_factor=None, threshold=1.0):


        POINTS = "world_points"
        CONF = "world_points_conf"
        #threshold = 2.0 
        #threshold = 50.0 

        z_clip_map = (-3.0, 3.0)  

        R_w2m = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)

        t_w2m = np.zeros(3, dtype=np.float32)

        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)
        
        if predictions is None:
            image_tensors = torch.stack([d["img"] for d in imgs])

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
                
            image_tensors = torch.stack(image_tensors, dim=0)  
            image_tensors = image_tensors.to(self.device)

            if i < 1:
                
                start = time.time()

                predictions = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=self.projector)

                self.keyframes = image_tensors.clone()
                
                end = time.time()
                length = end - start

                #print("Running inference took", length, "seconds!")
                

            else:
            
                start = time.time()

                changed_idx = changed_images(image_tensors, self.keyframes, thresh=0.000005)
                
                #print(changed_idx)

                end = time.time()
                length = end - start
                
                if len(changed_idx) < 2:
                        # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                        # self.vox.next_epoch()
                        return None, None, None, None
    
                #print("Finding changed images took", length, "seconds!")

                changed_idx = [0] + [x for x in changed_idx if x != 0]
                
                index_map = {new: old for new, old in enumerate(changed_idx)}
                        
                idx_t = torch.tensor(changed_idx, device=self.device, dtype=torch.long)
                self.keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
                
                
                #print("final changed idx:", changed_idx)

                start = time.time()
    
                #print("inference pred")
                mst = True
                predictions = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=self.projector)
                    
                end = time.time()
                length = end - start

                #print("Running inference took", length, "seconds!")


            # Keep tensors; only extract what we need later.
            # If you truly need NumPy later, convert specific keys then.
            needed = {
                "images","extrinsic", POINTS, CONF, "view_feats"
            }
            for k in list(predictions.keys()):
                if k not in needed:
                    del predictions[k]  # drop unneeded heavy stuff early



        
        
               
                
 
        start = time.time()
       
        WPTS_m = predictions[POINTS]

       



        predictions[POINTS] = WPTS_m


        end = time.time()
        length = end - start

        #print("Aligning frames took", length, "seconds!")
        start = time.time()



        end = time.time()
        length = end - start

        #print("Projecting view feats", length, "seconds!")
        
        start = time.time()


        #camera_R = R_w2m @ Rmw
        camera_R = Rmw @ R_w2m
        camera_t = t_w2m + tmw
        z_clip_map = (scale_factor*z_clip_map[0], scale_factor*z_clip_map[1])
        
        frames_map, conf_map, images_map, features_map, camera_centers, (S,H,W), frame_ids = filter_frames(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            FEAT="view_feats",
            threshold=threshold,
            Rmw=camera_R, tmw=camera_t,
            z_clip_map=z_clip_map,   # or None
        )  
        
        
        print(f"Features require grad: {features_map[0].requires_grad}")
        
        end = time.time()
        length = end - start

        #print("Building frames/camera centers took", length, "seconds!")

        start = time.time()

        align_to_voxel = False #(i > 0)
         

        #features_map = [vf_t[i] for i in range(vf.shape[0])]  # one vector per image

        # features_map = [pointnext_inference(preprocess_points(f,i)) for f, i in zip(frames_map, images_map)]
        vox, bev, meta = build_maps_from_latent_features(
            i,
            frames_map,
            conf_map,
            features_map,
            camera_centers,
            self.vox,
            voxel_size=self.voxel_size,           # 10 cm
            bev_window_m=self.bev_window_m, # local 20x20 m
            bev_origin_xy=self.bev_origin_xy,
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(self.z_band_bev[0], self.z_band_bev[1]),
            #z_band_bev=(self.z_band_bev[0] - 0.4, self.z_band_bev[1]),
            frame_ids=frame_ids,
            radius= self.cfg.radius_m,
        )

        self.vox = vox

        
        end = time.time()
        length = end - start

        #print("Building Voxel and BEV took", length, "seconds!")

        
            
        # self.vox.next_epoch()
        
        # after build_frames_and_centers_vectorized(...)
        del predictions  # drops images, view_feats, etc. all at once

        # after build_maps_from_latent_features(...)
        del frames_map, conf_map, images_map, features_map

        return bev, mst, Rmw, tmw
    
    def run_baseline_inference(self, predictions, Rmw=None, tmw=None, scale_factor=1.0, threshold=50.0):
        """
        Runs the baseline reconstruction pipeline (Ray carving/Integration) on the provided predictions.
        Matches the logic in test_voxel_dust3r_fast_no_opt.py
        """
        POINTS = "world_points"
        CONF = "world_points_conf"
        z_clip_map = (-3.0 * scale_factor, 3.0 * scale_factor)

        # Standard rotation to camera frame
        R_w2m = np.array([[0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)
        t_w2m = np.zeros(3, dtype=np.float32)

        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)

        # Combine transforms: World -> Model -> Aligned -> Scaled
        # Note: predictions['world_points'] is already aligned and scaled in predict_step loop
        # But for build_frames..., we pass Rmw/tmw to transform the camera centers.

        camera_R = Rmw @ R_w2m
        camera_t = t_w2m + tmw

        # 1. Extract Frames
        # Note: We skip 'view_feats' as baseline doesn't use them
        frames_map, cam_centers_map, conf_map, images_map, _, (S,H,W), frame_ids = build_frames_and_centers_vectorized_torch(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            threshold=threshold,
            Rmw=camera_R, tmw=camera_t,
            z_clip_map=z_clip_map,
            return_flat=True
        )

        # 2. Integrate into Baseline Voxel Grid
        vox, bev, meta = build_maps_from_points_and_centers_torch(
            frames_xyz=frames_map,
            cam_centers=cam_centers_map,
            conf_map=conf_map,
            tvox=self.vox_baseline,
            align_to_voxel=False,
            voxel_size=self.voxel_size,
            bev_window_m=self.bev_window_m,
            bev_origin_xy=self.bev_origin_xy,
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(self.z_band_bev[0], self.z_band_bev[1]),
            max_range_m=None,
            carve_free=True,
            samples_per_voxel=0.7, #0.7,
            ray_stride=4,
            max_free_rays=10000,
            frame_ids=frame_ids,
            device=self.device
        )

        self.vox_baseline = vox

        self.vox_baseline.next_epoch()

        return bev, meta
  


    def align_probs_to_keys_soft(self, vox_gt, gt_probs, vox_pred, r_vox=3, default=0.0):
        """
        r_vox=1 -> 27 neighbors. r_vox=2 -> 125 neighbors.
        Returns:
          tgt_soft: (M_pred,) float
          valid:    (M_pred,) bool
        """
        dev = vox_pred.keys.device
        M = int(vox_pred.keys.numel())
        if M == 0 or vox_gt.keys.numel() == 0:
            return (torch.full((M,), default, device=dev, dtype=gt_probs.dtype),
                    torch.zeros((M,), device=dev, dtype=torch.bool))

        # pred ijk
        ijk_p = vox_pred._unhash_keys(vox_pred.keys).to(torch.int32)  # (M,3)

        # neighbor offsets
        rng = torch.arange(-r_vox, r_vox + 1, device=dev, dtype=torch.int32)
        off = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).view(-1, 3)  # (Nn,3)
        Nn = off.shape[0]

        # candidate neighbor ijk and keys
        ijk_n = (ijk_p[:, None, :] + off[None, :, :]).to(torch.int64)        # (M,Nn,3)
        keys_n = vox_pred._hash_ijk(ijk_n.reshape(-1, 3)).view(M, Nn)        # (M,Nn)

        # sort GT keys once
        gt_keys = vox_gt.keys.view(-1)
        gt_probs = gt_probs.view(-1)
        gt_sorted, perm = torch.sort(gt_keys)
        gt_probs_sorted = gt_probs[perm]

        # lookup each neighbor key via searchsorted
        pos = torch.searchsorted(gt_sorted, keys_n)                           # (M,Nn)
        inb = pos < gt_sorted.numel()
        pos_safe = pos.clamp(max=gt_sorted.numel() - 1)
        hit = inb & (gt_sorted[pos_safe] == keys_n)

        # gather probs for hits
        p = torch.zeros((M, Nn), device=dev, dtype=gt_probs.dtype)
        p[hit] = gt_probs_sorted[pos_safe[hit]]

        # simple weighting: closer neighbors higher weight (in voxel units)
        d2 = (off.to(torch.float32) ** 2).sum(dim=1)                          # (Nn,)
        w = torch.exp(-d2 / (2.0 * (0.8 ** 2)))                                # (Nn,) sigma~0.8 vox
        w = w.to(dev, p.dtype)[None, :] * hit.to(p.dtype)                     # (M,Nn)

        valid = hit.any(dim=1)
        tgt_soft = (w * p).sum(dim=1) / (w.sum(dim=1) + 1e-8)
        tgt_soft = torch.where(valid, tgt_soft, torch.full_like(tgt_soft, default))

        return tgt_soft, valid

    def align_probs_to_keys(self, src_keys, src_probs, dst_keys, default=0.5):
        """
        Align (src_keys, src_probs) to dst_keys.

        Returns:
            out_probs:  (len(dst_keys),)  aligned probs (default for unknowns)
            valid_mask: (len(dst_keys),)  bool, True where dst_keys[i] exists in src_keys
        """

        # --- handle degenerate case: no src keys at all ---
        if src_keys.numel() == 0:
            out = torch.full(
                (dst_keys.numel(),),
                default,
                device=dst_keys.device,
                dtype=src_probs.dtype,
            )
            valid = torch.zeros(dst_keys.numel(), dtype=torch.bool, device=dst_keys.device)
            return out, valid

        # Force everything to 1D (defensive)
        src_keys = src_keys.view(-1)
        dst_keys = dst_keys.view(-1)
        src_probs = src_probs.view(-1)

        # Sanity check: they *must* correspond 1:1
        assert src_keys.shape[0] == src_probs.shape[0], \
            f"src_keys ({src_keys.shape}) and src_probs ({src_probs.shape}) length mismatch"

        # 1) sort src and carry probs with it
        src_sorted, src_perm = torch.sort(src_keys)
        src_probs_sorted = src_probs[src_perm]


        #print(src_sorted[-100:])

        # 2) sort dst
        dst_sorted, dst_sort_idx = torch.sort(dst_keys)

        #print(dst_sorted[-100:])

        # 3) positions where dst_sorted would be inserted in src_sorted
        pos = torch.searchsorted(src_sorted, dst_sorted)

        n_src = src_sorted.shape[0]

        # Candidate matches must be strictly within [0, n_src)
        in_bounds = pos < n_src          # shape: (len(dst_sorted),)
        pos_in   = pos[in_bounds]        # indices into src_sorted / src_probs_sorted
        dst_in   = dst_sorted[in_bounds]

        # Values at those positions in src
        src_at_pos = src_sorted[pos_in]
        is_eq      = (src_at_pos == dst_in)      # only these are true matches

        # Build match mask in *sorted-dst* space
        match_sorted = torch.zeros_like(dst_sorted, dtype=torch.bool, device=dst_keys.device)
        match_sorted[in_bounds] = is_eq

        # 4) fill out_sorted in sorted-dst space
        out_sorted = torch.full(
            (dst_sorted.numel(),),
            default,
            device=dst_keys.device,
            dtype=src_probs.dtype,
        )

        # src_probs_sorted[pos_in[is_eq]] is guaranteed in-bounds because pos_in < n_src
        out_sorted[match_sorted] = src_probs_sorted[pos_in[is_eq]]

        # 5) map both out and mask back to original dst order
        orig_to_sorted = torch.argsort(dst_sort_idx)
        out        = out_sorted[orig_to_sorted]
        valid_mask = match_sorted[orig_to_sorted]

        return out, valid_mask

    def compute_grad_norm(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        return total_norm ** 0.5
    

    
    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Encourages high confidence (0 or 1).
        """
        if logits.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        probs = torch.sigmoid(logits)
        eps = 1e-6
        # H(p) = -[p * log(p) + (1-p) * log(1-p)]
        entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        return entropy.mean()

    def compute_tv_loss(self, logits: torch.Tensor, K: int = 5) -> torch.Tensor:
        """
        Calculates TV loss by finding spatial neighbors for sparse voxels.
        """
        if logits.numel() < 2:
            return torch.tensor(0.0, device=self.device)

        # 1. Get centers (Detached from graph for geometry, but probs map to them)
        # Note: We rely on self.vox to get coordinates
        centers = self.vox.voxel_centers().unsqueeze(0)  # (1, N, 3)

        # 2. Find K Nearest Neighbors
        # idx: (1, N, K), dists: (1, N, K) squared distances
        res = knn_points(centers, centers, K=K, return_nn=False)
        idx = res.idx[0]
        dists = res.dists[0]

        # 3. Filter for immediate neighbors (approx 1 voxel size away)
        # dists are squared euclidean. Neighbor dist^2 should be close to voxel_size^2.
        # We exclude 0 (self) and diagonals (> 1.5 * voxel_size^2).
        vs2 = self.voxel_size ** 2
        mask = (dists > 1e-6) & (dists < (1.3 * vs2))

        # 4. Get probabilities (differentiable)
        probs = torch.sigmoid(logits)

        p_center = probs.unsqueeze(-1)      # (N, 1)
        p_neighbors = probs[idx]            # (N, K)

        # 5. Compute L1 difference
        # We only count differences where valid spatial neighbors exist
        diff = torch.abs(p_center - p_neighbors) * mask.float()

        # Normalize by number of valid neighbors found
        loss_tv = diff.sum() / (mask.sum() + 1e-8)

        return loss_tv

    def training_step(self, batch: Dict, batch_idx: int):
        """
        One batch = one sequence.
        Flow:
          t=0: init latents from full cloud
          t>0: extract point features, learned update, decode occupancy, compute loss vs teacher
        """
        cfg = self.cfg
        device = self.device
        opt = self.optimizers()
        
        
        self.vox.reset_state()
        self.vox = self.vox.to(self.device)

        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device 
        )
        
        self._prev_keys = None
        self._prev_probs = None

        # ---- unpack sequence ----
        T = batch["timesteps"]
        loss_total_seq = torch.zeros([], device=device) # Renamed to avoid confusion

        # ---- Accumulators for averaging metrics over the sequence ----
        metrics_buffer = {
            "loss_occ": [],
            "loss_temp": [],
            "loss_ent": [],
            "loss_tv": [],
            "occ_iou": [],
            "occ_precision":[],
            "occ_precision_inter":[],
            "occ_recall": []

        }

        # ---- iterate timesteps ----
        seq_id = batch["seq_id"]
        gt_root = os.path.join(self.cfg.dataset_root, self.cfg.gt_voxels_file)
        precomputed_root = os.path.join(self.cfg.dataset_root, self.cfg.precomputed_cache_file)
        pose_root = os.path.join(self.cfg.dataset_root, self.cfg.pose_file)


        # Preload GT (optional optimization you had)
        gt_seq = []
        if not os.path.exists(os.path.join(gt_root, f"{seq_id}_t0000_gt.npz")):
            print("No gt_root")
            return loss_total_seq



        for t in range(T):
            if self.cfg.skip:
                t = t*STEP # Adjust indexing if skipping
            
            gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
            print(gt_path)
            if os.path.exists(gt_path):
                vox_gt_t = load_sparse_voxel_grid(gt_path, device)
            else:
                vox_gt_t = None
            gt_seq.append(vox_gt_t)

        d = np.load(os.path.join(pose_root, f"{seq_id}_t0000_align.npz"), allow_pickle=True)
        Rmw = d["Rmw"]   # (3,3) float32
        tmw = d["tmw"]   # (3,)  float32
        Rmw = to_torch(Rmw, device=self.device)
        tmw = to_torch(tmw, device=self.device)
        
        scale_factor = float(d["scale"])
        tmw_scaled = tmw * scale_factor


        mst = False
        


        self.vox_gt_prev = None
        self.tgt_prev = None




        for t in range(T):
            #print(t)
            imgs = batch["imgs_t"][t]

            self.vox_gt = gt_seq[t]
            if self.vox_gt is None:
                print("Missing GT voxel at time", t)
                continue
            
            
           
            p = t * STEP
            cache_path = os.path.join(precomputed_root, seq_id, f"t{p:04d}.pt")
            if not os.path.exists(cache_path):
                print("Missing cache:", cache_path)
                continue
        
    
            predictions = torch.load(cache_path, map_location=self.device)


            # predictions["world_points"] = predictions["world_points"].to(dtype=torch.float32)
            # predictions["world_points_conf"] = predictions["world_points_conf"].to(dtype=torch.float32)
            # predictions["view_feats"] = [f.to(dtype=torch.float32) for f in predictions["view_feats"]]



            
            R_w2m = np.array([[0, 0, -1],
                            [1, 0, 0],
                            [0, -1, 0]], dtype=np.float32)

            t_w2m = np.zeros(3, dtype=np.float32)

            R_w2m = to_torch(R_w2m, device=self.device)
            t_w2m = to_torch(t_w2m, device=self.device)
            WPTS_m = rotate_points(predictions["world_points"], R_w2m, t_w2m)
            #Rmw, tmw, info = align_pointcloud_torch_fast(WPTS_m, inlier_dist=self.voxel_size*0.75, ransac_iters=500, point_chunk=5_000_000, cand_chunk=4096)
            #Rmw, tmw, _ = align_pointcloud_torch_fast(
            #    WPTS_m,
            #    inlier_dist=self.voxel_size * 0.75,
            #)

            predictions["world_points"] = rotate_points(WPTS_m, Rmw, tmw)


            raw_pts = predictions["world_points"]

        
            
            
            # 1. Scale Points
            predictions["world_points"] = raw_pts * scale_factor

            #predictions[POINTS] = self.align_icp(predictions[POINTS], self.vox_gt, self.device)

            # 2. Scale Camera Positions (Translations)
            # Iterate over the batch of extrinsics to scale the translation vector
            # Extrinsic is typically [R | t]. Scaling t moves cameras apart.
            # Check shape: usually (N, 4, 4)
            if isinstance(predictions["extrinsic"], torch.Tensor):
                predictions["extrinsic"][:, :3, 3] *= scale_factor
            elif isinstance(predictions["extrinsic"], list):
                for i in range(len(predictions["extrinsic"])):
                    predictions["extrinsic"][i][:3, 3] *= scale_factor
     
            #predictions["world_points"] = self.align_to_gt_centroid(predictions["world_points"], self.vox_gt, self.device)



                
                
            stride = 1 if t == 0 else self.cfg.stride

            if "world_points_conf" in predictions:
                predictions["world_points_conf"] = predictions["world_points_conf"][..., ::stride, ::stride]
                
                # Get NEW smaller target size (e.g. 128x128)
                conf_tensor = predictions["world_points_conf"]
                if isinstance(conf_tensor, list):
                    ref = next((x for x in conf_tensor if x is not None), None)
                    target_hw = ref.shape[-2:] if ref is not None else (128, 128)
                else:
                    target_hw = conf_tensor.shape[-2:]
            else:
                target_hw = (512 // stride, 512 // stride)

            if "world_points" in predictions:
                predictions["world_points"] = predictions["world_points"][..., ::stride, ::stride, :]

            if "images" in predictions:
                img = predictions["images"]
                if img.shape[-3] == 3 and img.shape[-1] != 3:
                     img = img.permute(0, 2, 3, 1) # (S, 3, H, W) -> (S, H, W, 3)
                predictions["images"] = img[..., ::stride, ::stride, :]
                del img

            # --- PROCESS FEATURES ---
            raw_feats_list = predictions["view_feats"]
           

            projected_feats_map = []

            for f_raw in raw_feats_list:
                # Pass the dynamically inferred size
                if torch.isnan(f_raw).any() or torch.isinf(f_raw).any():
                    f_raw = torch.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0)
                f_proj = self.apply_projector_to_map(f_raw, target_hw=target_hw)

                # Inside the loop where you process features
                #print(f"Feat Stats: Min={f_proj.min()}, Max={f_proj.max()}, Mean={f_proj.mean()}")
                #print(f"Non-Zero Ratio: {(f_proj.abs() > 1e-5).float().mean()}")
                
                projected_feats_map.append(f_proj)
                
            predictions["view_feats"] = projected_feats_map
            del projected_feats_map
        
            threshold = 50.0
            if not mst and t != 0:
                bev, mst, _, _ = self.inference(1, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)
            else:
                bev, mst, _, _ = self.inference(t, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)

        
            #with autocast(enabled=False):
            # (D) decode current occupancy
            logit_gt   = self.vox_gt.vals_st.clamp(-10.0, 10.0)
            # p_occ_tgt  = torch.sigmoid(logit_gt)
            p_occ_tgt = torch.sigmoid(logit_gt * 10.0)
            
            logit_pred_before = self.vox.decode_occupancy(with_xyz_cond=False)

            if not torch.isfinite(logit_pred_before).all():
                logit_pred_before = torch.nan_to_num(logit_pred_before, nan=0.001)
            
            # -------------------------------------------------------------
            # DUAL LOSS LOGIC START
            # -------------------------------------------------------------
            
            # 1. Align GT to Prediction Keys
            # valid_mask is TRUE where prediction keys exist in GT

            assert self.vox_gt.keys.dtype == torch.int64, "Keys GT must be int64! Float keys will cause hash collisions."
            assert self.vox.keys.dtype == torch.int64, "Keys must be int64! Float keys will cause hash collisions."


            #p_occ_tgt_gt_aligned, valid_mask = self.align_probs_to_keys(
            #    self.vox_gt.keys, p_occ_tgt, self.vox.keys, default=0.0
            #)      

            p_occ_tgt_gt_aligned, valid_mask = self.align_probs_to_keys_soft(
                self.vox_gt, p_occ_tgt, self.vox, default=0.0
            )      
           
            # ---------------------------------------------------------
            # PART A: Loss on Intersection (Pred & GT)
            # ---------------------------------------------------------

            total_pred = logit_pred_before.numel()
            total_gt = p_occ_tgt.numel()
            mask_pred = sum(valid_mask)
            #print("Pred len", total_pred)
            #print("GT len", total_gt)
            #print("Mask len", mask_pred)
            #print("Mask/Pred", (mask_pred/total_pred))
            #print("Mask/GT", (mask_pred/total_gt))

            #_, valid_gt = self.align_probs_to_keys(self.vox.keys, torch.ones_like(logit_pred_before), self.vox_gt.keys, default=0.0)

            _, valid_gt = self.align_probs_to_keys_soft(
                self.vox, torch.ones_like(logit_pred_before), self.vox_gt, default=0.0
            )      
           
            gt_covered = valid_gt.sum()
            #print("GT coverage", gt_covered / total_gt)


            pred_intersect = logit_pred_before[valid_mask]
            tgt_intersect  = p_occ_tgt_gt_aligned[valid_mask]
            
            loss_intersect = torch.tensor(0.0, device=self.device)
            pos_weight = torch.tensor(1.0, device=self.device)





            # ---------------------------------------------------------
            # NEW: TEMPORAL DELTA LOGIC
            # ---------------------------------------------------------


            if pred_intersect.numel() > 0:
                # Calculate weight for positives just like before
                pos_mask = (tgt_intersect > 0.5)
                num_pos = pos_mask.sum()
                num_neg = (~pos_mask).sum()
                
                if num_pos > 0:
                    pos_weight = (num_neg.float() / (num_pos.float() + 1e-8)).to(self.device)
                    pos_weight = pos_weight.clamp(min=1.0, max=20.0).to(self.device)
                else:
                    pos_weight = torch.tensor(1.0, device=self.device)


                weights = torch.ones_like(tgt_intersect, device=self.device)
                weights[pos_mask] = pos_weight
                
              
                if self.vox_gt_prev is not None:
                    # 1. Align Past to Present
                    p_occ_tgt_gt_aligned_prev, _ = self.align_probs_to_keys_soft(
                        self.vox_gt_prev, self.tgt_prev, self.vox, default=0.0
                    )
                    tgt_intersect_prev = p_occ_tgt_gt_aligned_prev[valid_mask]

                    # 2. Identify the specific types of change
                    # curr_bool: True if Wall NOW
                    # prev_bool: True if Wall BEFORE
                    curr_bool = (tgt_intersect > 0.5)
                    prev_bool = (tgt_intersect_prev > 0.5)

                    # Case A: Object Jumped IN (Air -> Wall)
                    appearing_mask = curr_bool & (~prev_bool)

                    # Case B: Object Jumped OUT (Wall -> Air) -> THIS CAUSES GHOSTING
                    disappearing_mask = (~curr_bool) & prev_bool

                    # 3. Apply the Weights
                    weights = torch.ones_like(tgt_intersect)

                    # Base weight for occupied voxels (Static Walls + Appearing Objects)
                    # If pos_weight is huge (e.g. 50), this ensures we capture walls.
                    weights[curr_bool] = pos_weight

                    # --- THE BOUNTY FIX ---

                    # Boost "Appearing" slightly more to ensure we catch the jump
                    #weights[appearing_mask] *= 20.0
                    weights[appearing_mask] = 20.0

                    # (Total weight = pos_weight * 5.0 = 250ish)

                    # Boost "Disappearing" MASSIVELY to fix Precision/Ghosting
                    # Since the base weight was 1.0, we need to multiply it by pos_weight * 5
                    # to match the importance of the appearing objects.
                    weights[disappearing_mask] = 20.0
                    # (Total weight = 250ish)

                    # 4. Calculate Loss
                    loss_intersect = F.binary_cross_entropy_with_logits(
                        pred_intersect, tgt_intersect, weight=weights, reduction='mean'
                    )
                else:
                    loss_intersect = F.binary_cross_entropy_with_logits(
                        pred_intersect, 
                        tgt_intersect, 
                        weight=weights,
                        reduction='mean'

                    )




            self.vox_gt_prev = self.vox_gt
            self.tgt_prev = p_occ_tgt.detach()




            # ---------------------------------------------------------
            # PART B: Loss on False Positives (Pred - GT)
            # ---------------------------------------------------------
            # These are voxels in your prediction that DO NOT exist in GT.
            # Since GT is truth, these must be empty (0.0).
            pred_fp = logit_pred_before[~valid_mask]
            loss_fp = torch.tensor(0.0, device=self.device)

        
            if pred_fp.numel() > 0:
                # Target is all zeros
                tgt_fp = torch.zeros_like(pred_fp)
                
                # Weighting: You might want to weigh this less than intersection
                # but here we start with 1.0 (strict precision).
                # with autocast(enabled=False):

            
                    
                loss_fp = F.binary_cross_entropy_with_logits(
                    pred_fp, 
                    tgt_fp, 
                    reduction='mean'
                )

            # ---------------------------------------------------------
            # TOTAL OCCUPANCY LOSS & IoU
            # ---------------------------------------------------------
            #self.fp_weight = 0.1

            loss_occ = loss_intersect + (self.fp_weight * loss_fp)


            # print("Loss Occ", loss_occ)
            # print("Loss Intersect", loss_intersect)
            # print("Loss Fp", loss_fp)

            # --- Metrics: Global IoU (Including FP Hallucinations) ---
            # Valid/Intersect Part
            pred_bin_int = (pred_intersect > 0.0)
            tgt_bin_int  = (tgt_intersect  > 0.5)
            
            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()
            
            # Hallucination Part (Preds outside GT are all FPs if > 0.5)
            fp_hallucination = (pred_fp > 0.0).sum()
            
            total_fp = fp_int + fp_hallucination
            
            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            occ_recall = tp/ (tp + fn + 1e-8)
            occ_precision = tp / (tp + total_fp + 1e-8)
            occ_precision_inter = tp / (tp + fp_int + 1e-8)
            metrics_buffer["occ_iou"].append(occ_iou.item())
            metrics_buffer["occ_recall"].append(occ_recall.item())
            metrics_buffer["occ_precision"].append(occ_precision.item())
            metrics_buffer["occ_precision_inter"].append(occ_precision_inter.item())

            # -------------------------------------------------------------
            # DUAL LOSS LOGIC END
            # -------------------------------------------------------------

            # --- Loss: Temporal ---
            if (t == 0) or (self._prev_keys is None):
                loss_temp = torch.tensor(0.0, device=self.device)
            else:
                prev_aligned, valid_mask_temp = self.align_probs_to_keys(
                    self._prev_keys, self._prev_probs, self.vox.keys, default=-10.0
                )
                
                # logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
                # logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                
                logit_now  = logit_pred_before[valid_mask_temp]
                logit_prev = prev_aligned[valid_mask_temp]
                
                if logit_now.numel() == 0:
                    loss_temp = torch.tensor(0.0, device=self.device)
                else:
                    loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

            # update buffers for next step
            self._prev_keys  = self.vox.keys.detach().clone()
            self._prev_probs = logit_pred_before.detach().clone()

            # --- Loss: Others (Entropy / TV) ---
            loss_ent = torch.tensor(0.0, device=device)
            if hasattr(self.vox, "_last_entropy") and self.vox._last_entropy is not None:
                loss_ent = self.vox._last_entropy
            
            loss_tv = torch.tensor(0.0, device=device)
            if cfg.lambda_tv > 0.0:
                 loss_tv = self.compute_tv_loss(logit_pred_before)


            loss_ent = self.compute_entropy_loss(logit_pred_before)

            # Combine Losses
            loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp \
                        + cfg.lambda_ent * loss_ent + cfg.lambda_tv * loss_tv

            # Manual Backward (per timestep)
            self.manual_backward(loss_t)

            # Accumulate for logging (detach to save memory)
            loss_total_seq += loss_t.detach()
            
            metrics_buffer["loss_occ"].append(loss_occ.detach().item())
            metrics_buffer["loss_temp"].append(loss_temp.detach().item())
            metrics_buffer["loss_ent"].append(loss_ent.detach().item())
            metrics_buffer["loss_tv"].append(loss_tv.detach().item())

            self.vox.z_latent = self.vox.z_latent.detach()

            # 1. GROUND TRUTH STATS (Is my batch empty?)
            num_pos_gt = (tgt_intersect > 0.5).sum().float()
            num_neg_gt = (~(tgt_intersect > 0.5)).sum().float()
            pos_ratio = num_pos_gt / (num_pos_gt + num_neg_gt + 1e-8)

            # 2. PREDICTION STATS (Is my model confident or scared?)
            # "Active" means prediction > 0.5 (model thinks it's a wall)
            num_pred_active = (pred_intersect > 0.0).sum().float()
            avg_prob_on_walls = pred_intersect[tgt_intersect > 0.5].mean() if num_pos_gt > 0 else torch.tensor(0.0)
            avg_prob_on_empty = pred_intersect[tgt_intersect < 0.5].mean()

            # 3. OVERLAP DIAGNOSTICS (Why is IoU low?)
            intersection = ((pred_intersect > 0.0) & (tgt_intersect > 0.5)).sum().float()
            union = ((pred_intersect > 0.0) | (tgt_intersect > 0.5)).sum().float()

            # 4. WEIGHT CHECK (What is my dynamic weight actually doing?)
            # If you used the dynamic formula, log what it calculated
            current_pos_weight = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight)

            # --- PRINT TO TERMINAL (Every 100 steps or on specific batch) ---
            #print(f"\n[Step {batch_idx} Analysis]")
            #print(f"  GT Walls: {int(num_pos_gt)} voxels ({pos_ratio:.4%} of volume)")
            #print(f"  Pred Walls: {int(num_pred_active)} voxels")
            #print(f"  Confidence: Walls={avg_prob_on_walls:.4f}, Empty={avg_prob_on_empty:.4f}")
            #print(f"  Pos Weight Used: {current_pos_weight.item():.2f}")
            #print(f"  IoU Components: Intersect={int(intersection)} / Union={int(union)}")
            #print("-" * 30)


            torch.cuda.empty_cache()
            
    
        # ---- End of Sequence Loop ----

        # # 1. Clip Gradients
        grad_norm = self.compute_grad_norm()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float(self.cfg.n_accum))            

        
        # # 2. Optimizer Step
        # opt.step()            
        # opt.zero_grad(set_to_none=True)
        
        
        if (batch_idx + 1) % self.cfg.n_accum == 0:
            
            # 2. Step Optimizer
            opt.step()            
            
            # 3. Zero Gradients (Clear buffer for next accumulation cycle)
            opt.zero_grad(set_to_none=True)
   

                # 3. Aggregate Metrics (Mean over sequence)


        avg_loss_total = loss_total_seq# / max(T, 1)

        # 4. Log Averaged Metrics
        self.log_dict({
            "loss/occ": self.get_avg(metrics_buffer, "loss_occ"),
            "loss/temp": self.get_avg(metrics_buffer,"loss_temp"),
            "loss/ent": self.get_avg(metrics_buffer,"loss_ent"),
            "loss/tv": self.get_avg(metrics_buffer,"loss_tv"),
            "loss/total_avg": avg_loss_total,
            "metric/occ_iou": self.get_avg(metrics_buffer,"occ_iou"),
            "metric/occ_recall": self.get_avg(metrics_buffer,"occ_recall"),
            "metric/occ_precision": self.get_avg(metrics_buffer,"occ_precision"),           
            "metric/occ_precision_inter": self.get_avg(metrics_buffer,"occ_precision_inter"),
            "stats/num_voxels": float(self.vox.keys.numel()),
            "grad_norm": grad_norm
        }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)

        return loss_total_seq

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
            
    
    def validation_step(self, batch: Dict, batch_idx: int):
        device = self.device
        cfg = self.cfg

        self.vox.reset_state()
        self.vox = self.vox.to(self.device)

        # Initialize an empty GT grid structure
        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device 
        )
        
        # Buffers for temporal consistency
        self._prev_keys = None
        self._prev_probs = None

        T = batch["timesteps"]
        val_loss_total_seq = torch.zeros([], device=device)

        # Metrics buffer for averaging
        metrics_buffer = {
            "loss_occ": [],
            "loss_temp": [],
            "occ_iou": [],
            "occ_precision":[],
            "occ_precision_inter":[],
            "occ_recall": []
        }

        seq_id = batch["seq_id"]
        gt_root = os.path.join(self.cfg.dataset_root, self.cfg.gt_voxels_file)
        precomputed_root = os.path.join(self.cfg.dataset_root, self.cfg.precomputed_cache_file)
        pose_root = os.path.join(self.cfg.dataset_root, self.cfg.pose_file)

        # Preload GT
        gt_seq = []
        for t in range(T):
            if self.cfg.skip:
                t = t * STEP
            gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
            if os.path.exists(gt_path):
                vox_gt_t = load_sparse_voxel_grid(gt_path, device)
            else:
                vox_gt_t = None
            gt_seq.append(vox_gt_t)
            

        d = np.load(os.path.join(pose_root, f"{seq_id}_t0000_align.npz"), allow_pickle=True)

        
        Rmw = d["Rmw"]   # (3,3) float32
        tmw = d["tmw"]   # (3,)  float32
        scale_factor = float(d["scale"])
        Rmw = to_torch(Rmw, device=self.device)
        tmw = to_torch(tmw, device=self.device)
        tmw_scaled = tmw * scale_factor




        mst = False
        for t in range(T):
            # print("Val ", t)
            # #print(f"== Val Step {t} ==")
            imgs = batch["imgs_t"][t]
            
            self.vox_gt = gt_seq[t]
            if self.vox_gt is None:
                continue
            
            p = t * STEP
            cache_path = os.path.join(precomputed_root, seq_id, f"t{p:04d}.pt")
            if not os.path.exists(cache_path):
                print("No cache path")
                continue
       
    


            predictions = torch.load(cache_path, map_location=self.device)

            """
            predictions["world_points"] = predictions["world_points"].to(dtype=torch.float32)
            predictions["world_points_conf"] = predictions["world_points_conf"].to(dtype=torch.float32)
            predictions["view_feats"] = [f.to(dtype=torch.float32) for f in predictions["view_feats"]]
            """


            R_w2m = np.array([[0, 0, -1],
                            [1, 0, 0],
                            [0, -1, 0]], dtype=np.float32)

            t_w2m = np.zeros(3, dtype=np.float32)

            R_w2m = to_torch(R_w2m, device=self.device)
            t_w2m = to_torch(t_w2m, device=self.device)

            WPTS_m = rotate_points(predictions["world_points"], R_w2m, t_w2m)
            #Rmw, tmw, info = align_pointcloud_torch_fast(WPTS_m, inlier_dist=self.voxel_size*0.75, ransac_iters=500, point_chunk=5_000_000, cand_chunk=4096)
            #Rmw, tmw, _ = align_pointcloud_torch_fast(
            #    WPTS_m,
            #    inlier_dist=self.voxel_size * 0.75,
            #)
            predictions["world_points"] = rotate_points(WPTS_m, Rmw, tmw)






            raw_pts = predictions["world_points"]
           
           
            # 1. Scale Points
            predictions["world_points"] = raw_pts * scale_factor
            #tmw = tmw * scale_factor
            
            #predictions[POINTS] = self.align_to_gt_centroid(predictions[POINTS], self.vox_gt, self.device)
            #predictions[POINTS] = self.align_icp(predictions[POINTS], self.vox_gt, self.device)

            # 2. Scale Camera Positions (Translations)
            # Iterate over the batch of extrinsics to scale the translation vector
            # Extrinsic is typically [R | t]. Scaling t moves cameras apart.
            # Check shape: usually (N, 4, 4)
            if isinstance(predictions["extrinsic"], torch.Tensor):
                predictions["extrinsic"][:, :3, 3] *= scale_factor
            elif isinstance(predictions["extrinsic"], list):
                for i in range(len(predictions["extrinsic"])):
                    predictions["extrinsic"][i][:3, 3] *= scale_factor


            stride = 1 if t == 0 else self.cfg.stride

            if "world_points_conf" in predictions:
                predictions["world_points_conf"] = predictions["world_points_conf"][..., ::stride, ::stride]

                # Get NEW smaller target size (e.g. 128x128)
                conf_tensor = predictions["world_points_conf"]
                if isinstance(conf_tensor, list):
                    ref = next((x for x in conf_tensor if x is not None), None)
                    target_hw = ref.shape[-2:] if ref is not None else (128, 128)
                else:
                    target_hw = conf_tensor.shape[-2:]
            else:
                target_hw = (512 // stride, 512 // stride)

            if "world_points" in predictions:
                predictions["world_points"] = predictions["world_points"][..., ::stride, ::stride, :]

            if "images" in predictions:
                img = predictions["images"]
                if img.shape[-3] == 3 and img.shape[-1] != 3:
                     img = img.permute(0, 2, 3, 1) # (S, 3, H, W) -> (S, H, W, 3)
                predictions["images"] = img[..., ::stride, ::stride, :]
                del img

            # --- PROCESS FEATURES ---
            raw_feats_list = predictions["view_feats"]
            projected_feats_map = []

            for f_raw in raw_feats_list:
                # Pass the dynamically inferred size
                f_proj = self.apply_projector_to_map(f_raw, target_hw=target_hw)
                projected_feats_map.append(f_proj)
                
            predictions["view_feats"] = projected_feats_map
            del projected_feats_map
            


            threshold = 50.0
            with torch.enable_grad(): # (Keep grad enabled for inference/update parts if needed by model)
                if not mst and t != 0:
                    bev, mst, _, _ = self.inference(1, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)
                else:
                    bev, mst, _, _ = self.inference(t, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)

        
            # Validation Loss Calculation (No Autocast needed strictly, but good for consistency)
            # with autocast(enabled=False):
            # p_occ_tgt = torch.sigmoid(self.vox_gt.vals_st)
            logit_gt   = self.vox_gt.vals_st.clamp(-10.0, 10.0)

            p_occ_tgt = torch.sigmoid(logit_gt * 10.0)

            logit_pred_before = self.vox.decode_occupancy(with_xyz_cond=False)

            if not torch.isfinite(logit_pred_before).all():
                logit_pred_before = torch.nan_to_num(logit_pred_before, nan=0.001)
            
            # 1. Align GT to Prediction
               
            p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                self.vox_gt, p_occ_tgt, self.vox, default=0.0
            )      

            # ---------------------------------------------------------
            # PART A: Intersection Loss
            # ---------------------------------------------------------
            pred_intersect = logit_pred_before[valid_mask]
            tgt_intersect  = p_occ_tgt_aligned[valid_mask]
            
            loss_intersect = torch.tensor(0.0, device=self.device)


            if pred_intersect.numel() > 0:
               
                loss_intersect = F.binary_cross_entropy_with_logits(
                    pred_intersect, 
                    tgt_intersect, 
                    reduction='mean'
                )

            # ---------------------------------------------------------
            # PART B: Loss on False Positives (Pred - GT)
            # ---------------------------------------------------------
            # These are voxels in your prediction that DO NOT exist in GT.
            # Since GT is truth, these must be empty (0.0).
            pred_fp = logit_pred_before[~valid_mask]
            loss_fp = torch.tensor(0.0, device=self.device)

            if pred_fp.numel() > 0:
                # Target is all zeros
                tgt_fp = torch.zeros_like(pred_fp)
           
                loss_fp = F.binary_cross_entropy_with_logits(
                    pred_fp, 
                    tgt_fp, 
                    reduction='mean'
                )

            # ---------------------------------------------------------
            # TOTAL OCCUPANCY LOSS & IoU
            # ---------------------------------------------------------
            #self.fp_weight = 0.1
            loss_occ = loss_intersect + (self.fp_weight * loss_fp)

            # --- Metrics: Global IoU (Including FP Hallucinations) ---
            # Valid/Intersect Part
            pred_bin_int = (pred_intersect > 0.0)
            tgt_bin_int  = (tgt_intersect  > 0.5)
            
            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()
            
            # Hallucination Part (Preds outside GT are all FPs if > 0.5)
            fp_hallucination = (pred_fp > 0.0).sum()
            
            total_fp = fp_int + fp_hallucination
        


            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            occ_recall = tp/ (tp + fn + 1e-8)
            occ_precision = tp / (tp + total_fp + 1e-8)
            occ_precision_inter = tp / (tp + fp_int + 1e-8)
            metrics_buffer["occ_iou"].append(occ_iou.item())
            metrics_buffer["occ_recall"].append(occ_recall.item())
            metrics_buffer["occ_precision"].append(occ_precision.item())
            metrics_buffer["occ_precision_inter"].append(occ_precision_inter.item())
            
            # -------------------------------------------------------------
            # DUAL LOSS LOGIC END
            # -------------------------------------------------------------

            # --- Loss: Temporal ---
            if (t == 0) or (self._prev_keys is None):
                loss_temp = torch.tensor(0.0, device=self.device)
            else:
                prev_aligned, valid_mask_temp = self.align_probs_to_keys(
                    self._prev_keys, self._prev_probs, self.vox.keys, default=-10.0
                )
                
                # logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
                # logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                
                logit_now  = logit_pred_before[valid_mask_temp]
                logit_prev = prev_aligned[valid_mask_temp]
                
                if logit_now.numel() == 0:
                    loss_temp = torch.tensor(0.0, device=self.device)
                else:
                    loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)
                    


            # update buffers for next step
            self._prev_keys  = self.vox.keys.detach().clone()
            self._prev_probs = logit_pred_before.detach().clone()

        
            loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp
            val_loss_total_seq += loss_t.detach()
            
            metrics_buffer["loss_occ"].append(loss_occ.item())
            metrics_buffer["loss_temp"].append(loss_temp.item())
        
            self.vox.z_latent = self.vox.z_latent.detach()
            torch.cuda.empty_cache()
            
            """
            save_dir = "debug_viz_val"
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/step_{t}.ply"
            self.export_debug_ply(fname, t)
            self.export_separated_ply(t, save_dir="debug_viz_val")
            """


        # Average over sequence
        val_loss_total = val_loss_total_seq #/ max(T, 1)

        self.log("val_loss_total", val_loss_total, prog_bar=True, on_epoch=True, sync_dist=False)
        
        
        
        # 4. Log Averaged Metrics
        self.log_dict({
            "val_metric/occ_iou": self.get_avg(metrics_buffer,"occ_iou"),
            "val_metric/occ_recall": self.get_avg(metrics_buffer,"occ_recall"),
            "val_metric/occ_precision": self.get_avg(metrics_buffer,"occ_precision"),
            "val_metric/occ_precision_inter": self.get_avg(metrics_buffer,"occ_precision_inter"),
        }, prog_bar=True, on_epoch=True, sync_dist=False)
        
        return val_loss_total
    

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0, step=1):
        def build_voxel_from_gt_direct(sim_data, voxel_size=0.2, device="cuda", scale_factor=1.0):
            """
            Build GT voxel grid directly from Habitat depth — no filtering,
            no confidence thresholds, no subsampling.
            """
            pts = sim_data["world_points"].to(device, dtype=torch.float32)   # (S, H, W, 3)
            extr = sim_data["extrinsic"].to(device, dtype=torch.float32)     # (S, 4, 4)

            S, H, W, _ = pts.shape
            cam_centers = extr[:, :3, 3]  # (S, 3)

            # Flatten and filter only NaN/inf (from invalid depth)
            pts_flat = pts.reshape(-1, 3)
            # Repeat each camera center for H*W points
            cams_flat = cam_centers[:, None, None, :].expand(S, H, W, 3).reshape(-1, 3)

            valid = torch.isfinite(pts_flat).all(dim=-1)
            pts_valid = pts_flat[valid]
            cams_valid = cams_flat[valid]

            vox = TorchSparseVoxelGrid(
                origin_xyz=[0, 0, 0],
                params=VoxelParams(voxel_size=voxel_size, promote_hits=1),
                device=device,
            )

            vox.integrate_points_with_cameras(
                pts_valid,
                cams_valid,
                carve_free=True,
                max_range=20.0,
                z_clip=None,
                #samples_per_voxel=1.5,
                #ray_stride=1,
                #max_free_rays=500_000,
                samples_per_voxel=0.7, #0.7,
                ray_stride=4,
                max_free_rays=10000,
            )

            return vox
        def build_voxel_from_sim_data(sim_data, voxel_size=0.2, device="cuda", scale_factor=1.0):
            """
            Converts loaded sim 'tensors.pt' data into a TorchSparseVoxelGrid.

            Args:
                sim_data: Dict loaded via torch.load("tensors.pt")
                voxel_size: Size of voxel side in meters
            """
            # 1. Coordinate Setup (Matches generate_gt.py)
            # Rotation to map Habitat world -> your model's metric frame
            R_w2m = torch.tensor([[0, 0, -1],
                                  [1, 0, 0],
                                  [0, -1, 0]], dtype=torch.float32, device=device)
            t_w2m = torch.zeros(3, device=device)

            # 2. Extract and Move Data to Device
            pts = sim_data["world_points"].to(device)
            conf = sim_data["world_points_conf"].to(device)
            # Extrinsics are (N, 4, 4) [R | t] where t is camera position
            extrinsics = sim_data["extrinsic"].to(device)

            """
            # 3. Apply Frame Alignment
            # Rotate raw points into the metric frame
            pts_m = rotate_points(pts, R_w2m, t_w2m)

            # 4. Apply Scaling Logic (Matches your 5.0m target size logic)
            valid_mask = torch.isfinite(pts_m).all(dim=-1)
            centroid = pts_m[valid_mask].median(dim=0).values
            centered_pts = pts_m - centroid
            current_size = torch.median(torch.norm(centered_pts[valid_mask], dim=1))

            target_size = 5.0
            scale_factor = (target_size / (current_size + 1e-6)).item()

            # Scale points and camera positions
            pts_scaled = pts_m * scale_factor
            cam_centers_world = extrinsics[:, :3, 3] # (N, 3)
            cam_centers_m = (cam_centers_world @ R_w2m.T) + t_w2m
            cam_centers_scaled = cam_centers_m * scale_factor
            """

            z_clip_map = (scale_factor * z_clip_map[0], scale_factor * z_clip_map[1])


            renders_map, cam_centers_map, conf_map, images_map, _, (S, H, W), frame_ids = \
                build_frames_and_centers_vectorized_torch(
                    sim_data,
                    POINTS="world_points",
                    CONF="world_points_conf",
                    threshold=1.0,      # Matches your generate_gt.py logic
                    Rmw=None,        # Your aligned Rotation
                    tmw=None,        # Your aligned Translation
                    z_clip_map=z_clip_map,
                    return_flat=True,
                )
            # 5. Prepare for Voxelization
            # We need to treat the points as coming from different "renders" (cameras)
            # If sim_data saved points as one big cloud, we must split them or
            # assign them to the nearest camera for ray-casting.
            # (Assuming world_points is already the concatenated cloud from N cameras)


            # Create fresh grid
            vox_real = TorchSparseVoxelGrid(
                origin_xyz=np.zeros(3, dtype=np.float32),
                params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
                device=device,
            )

            # Note: build_maps_from_points_and_centers_torch expects lists of tensors per view
            # Since this is GT, we can pass them as single-item lists
            vox_real, bev, meta = build_maps_from_points_and_centers_torch(
                renders_map,          # List of point tensors
                cam_centers_map,
                conf_map,
                vox_real,
                align_to_voxel=False,
                voxel_size=voxel_size,
                bev_window_m=(5.0, 5.0),
                bev_origin_xy=(-2.0, -2.0),
                z_clip_vox=(-np.inf, np.inf),
                z_band_bev=(0.02, 0.5),
                samples_per_voxel=2.0,
                ray_stride=1,
                max_free_rays=500_000,  # was 10,000
                #ray_stride=4,          # Speed up integration
                #max_free_rays=10000,
                frame_ids=frame_ids
            )

            return vox_real

        def kabsch_umeyama_sim3(src, dst):
            """
            Computes the optimal Sim(3) transform that aligns src to dst.
            src, dst: (N, 3) tensors
            Returns: R (3,3), t (3,), s (float)
            Equation: dst = s * (src @ R.T) + t
            """
            # 1. Centroid subtraction
            mu_src = src.mean(dim=0)
            mu_dst = dst.mean(dim=0)
            
            src_c = src - mu_src
            dst_c = dst - mu_dst

            # 2. Compute Scale (s)
            # Scale is the ratio of root-mean-square deviations from centroids
            var_src = (src_c**2).sum(dim=-1).mean()
            var_dst = (dst_c**2).sum(dim=-1).mean()
            scale = torch.sqrt(var_dst / var_src)

            # 3. Compute Rotation (R) using SVD
            # Correlation matrix
            H = src_c.T @ dst_c
            U, S, Vh = torch.linalg.svd(H)
            
            # Correction for reflection
            d = torch.sign(torch.linalg.det(U @ Vh))
            diag = torch.ones(3, device=src.device)
            diag[2] = d
            
            R = Vh.T @ torch.diag(diag) @ U.T

            # 4. Compute Translation (t)
            # t = mu_dst - s * (R @ mu_src)
            translation = mu_dst - scale * (R @ mu_src)

            return R, translation, scale

        def icp_sim3(src, dst, init_R, init_t, init_s, max_iters=20, tolerance=1e-5):
            """
            Iterative Closest Point with Sim(3) transformation.
            Uses PyTorch3D knn_points to find true spatial neighbors instead of array indices.
            """
            # Apply initial Kabsch alignment
            src_curr = init_s * (src @ init_R.T) + init_t

            # Accumulators for the total transformation
            R_acc = init_R.clone()
            t_acc = init_t.clone()
            s_acc = init_s.clone()

            prev_loss = float('inf')

            for i in range(max_iters):
                # 1. Find true nearest neighbors in 3D space
                # knn_points expects batched tensors: (1, N, 3)
                res = knn_points(src_curr.unsqueeze(0), dst.unsqueeze(0), K=1)

                # Squeeze out the batch dimension
                idx = res.idx.squeeze(0).squeeze(-1)      # (N,)
                dists = res.dists.squeeze(0).squeeze(-1)  # (N,)

                # 2. Filter Outliers (Crucial for Dust3R noise!)
                # Only keep the 85% closest points so floating artifacts don't warp the room
                q85 = torch.quantile(dists, 0.85)
                valid = dists < q85

                src_matched = src_curr[valid]
                dst_matched = dst[idx[valid]]

                # 3. Estimate new transformation on the matched pairs
                R, t, s = kabsch_umeyama_sim3(src_matched, dst_matched)

                # 4. Apply transformation for the next iteration
                src_curr = s * (src_curr @ R.T) + t

                # 5. Update global accumulators
                # (Sim3 Composition: S2 * R2 * (S1 * R1 * X + T1) + T2)
                t_acc = s * (R @ t_acc) + t
                R_acc = R @ R_acc
                s_acc = s * s_acc

                # Check for convergence
                mean_dist = dists[valid].mean().item()
                if abs(prev_loss - mean_dist) < tolerance:
                    break
                prev_loss = mean_dist

            return R_acc, t_acc, s_acc
        def compute_chamfer_dist(vox_pred, vox_gt):

            """
            Measures the average distance (in meters) between occupied voxels.
            """
            # 1. Get centers of occupied voxels
            mask_p = vox_pred.occupied_mask()
            mask_g = vox_gt.occupied_mask()

            if not mask_p.any() or not mask_g.any():
                return torch.tensor(0.0, device=vox_pred.device)

            # Use voxel_centers() helper from your classes
            centers_p = vox_pred.voxel_centers()[mask_p].unsqueeze(0) # (1, N, 3)
            centers_g = vox_gt.voxel_centers()[mask_g].unsqueeze(0) # (1, M, 3)

            # 2. Bidirectional Nearest Neighbors
            # dists are squared Euclidean distances
            dist_p_to_g, _, _ = knn_points(centers_p, centers_g, K=1)
            dist_g_to_p, _, _ = knn_points(centers_g, centers_p, K=1)

            # 3. Mean distance in meters
            chamfer = (torch.sqrt(dist_p_to_g).mean() + torch.sqrt(dist_g_to_p).mean()) / 2.0
            return chamfer

        device = self.device


        self.vox.reset_state()
        self.vox = self.vox.to(self.device)

        # Initialize an empty GT grid structure
        self.prev_vox_real_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device
        )
        self.vox_real_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device
        )

        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device
        )

        self.vox_baseline = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_epochs=2),
            device=self.device
        )

        # Buffers for temporal consistency
        self._prev_keys = None
        self._prev_probs = None

        threshold = 1.0

        # --- METRICS BUFFER ---
        # Initialize all keys so get_avg doesn't crash if a sequence has no dynamic events
        metrics_buffer = {
            "occ_iou": [],
            "occ_precision": [],
            "occ_precision_inter": [],
            "occ_recall": [],

            "gt_occ_iou": [],
            "gt_occ_precision": [],
            "gt_occ_precision_inter": [],
            "gt_occ_recall": [],
            # Baseline Metrics
            "baseline_occ_iou": [],
            "baseline_occ_precision": [],
            "baseline_occ_precision_inter": [],
            "baseline_occ_recall": [],

            "static_occ_iou": [],
            "static_occ_precision": [],
            "static_occ_precision_inter": [],
            "static_occ_recall": [],

            "dyn_iou": [],
            "dyn_recall_appearing": [],
            "dyn_recall_disappearing": [],
            "dyn_ghost_rate": [],

            "gt_dyn_iou": [],
            "gt_dyn_recall_appearing": [],
            "gt_dyn_recall_disappearing": [],
            "gt_dyn_ghost_rate": [],




            "baseline_dyn_iou": [],
            "baseline_dyn_recall_appearing": [],
            "baseline_dyn_recall_disappearing": [],
            "baseline_dyn_ghost_rate": [],

            "static_dyn_iou": [],
            "static_dyn_recall_appearing": [],
            "static_dyn_recall_disappearing": [],
            "static_dyn_ghost_rate": [],


            "chamfer_dist": [],
            "gt_chamfer_dist": [],
            "static_chamfer_dist": [],
            "baseline_chamfer_dist": [],

            "tfs_model": [],
            "tfs_baseline": [],
            "tfs_gt": [],
        }

        T = batch["timesteps"]
        seq_id = batch["seq_id"]

        if self.cfg.real_gt_voxels_file:
            real_gt_root = os.path.join(self.cfg.dataset_root, self.cfg.real_gt_voxels_file)



        gt_root = os.path.join(self.cfg.dataset_root, self.cfg.gt_voxels_file)
        precomputed_root = os.path.join(self.cfg.dataset_root, self.cfg.precomputed_cache_file)
        pose_root = os.path.join(self.cfg.dataset_root, self.cfg.pose_file)

        # Preload GT
        gt_seq = []
        for t in range(T):
            if self.cfg.skip:
                t = t * step
            gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")

            if os.path.exists(gt_path):
                vox_gt_t = load_sparse_voxel_grid(gt_path, device)
            else:
                print(f"Warning: GT missing for {gt_path}")
                vox_gt_t = None
            gt_seq.append(vox_gt_t)

        d = np.load(os.path.join(pose_root, f"{seq_id}_t0000_align.npz"), allow_pickle=True)

        Rmw = d["Rmw"]   # (3,3) float32
        tmw = d["tmw"]   # (3,)  float32
        scale_factor = float(d["scale"])
        Rmw = to_torch(Rmw, device=self.device)
        tmw = to_torch(tmw, device=self.device)
        tmw_scaled = tmw * scale_factor

        mst = False

        bevs = []
        bevs_gt = []
        bevs_baseline = []

        print(f"\n=== Starting Prediction for Seq: {seq_id} (T={T}) ===")

        #step 0 for baseline
        t = 0
        # print(f"Step {t}...")
        imgs = batch["imgs_t"][t]

        self.vox_gt = gt_seq[t]

        p = t * step
        cache_path = os.path.join(precomputed_root, seq_id, f"t{p:04d}.pt")

        predictions = torch.load(cache_path, map_location=self.device)

        R_w2m = np.array([[0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)
        t_w2m = np.zeros(3, dtype=np.float32)

        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)

        # 2. Rotate Points to World Frame
        WPTS_m = rotate_points(predictions["world_points"], R_w2m, t_w2m)
        predictions["world_points"] = rotate_points(WPTS_m, Rmw, tmw)

        # 3. Apply Scaling
        raw_pts = predictions["world_points"]
        predictions["world_points"] = raw_pts * scale_factor

        # 4. Scale Camera Extrinsics
        if isinstance(predictions["extrinsic"], torch.Tensor):
            predictions["extrinsic"][:, :3, 3] *= scale_factor
        elif isinstance(predictions["extrinsic"], list):
            for i in range(len(predictions["extrinsic"])):
                predictions["extrinsic"][i][:3, 3] *= scale_factor

        stride = 1 if t == 0 else self.cfg.stride

        if "world_points_conf" in predictions:
            predictions["world_points_conf"] = predictions["world_points_conf"][..., ::stride, ::stride]
            conf_tensor = predictions["world_points_conf"]
            if isinstance(conf_tensor, list):
                ref = next((x for x in conf_tensor if x is not None), None)
                target_hw = ref.shape[-2:] if ref is not None else (128, 128)
            else:
                target_hw = conf_tensor.shape[-2:]
        else:
            target_hw = (512 // stride, 512 // stride)

        if "world_points" in predictions:
            predictions["world_points"] = predictions["world_points"][..., ::stride, ::stride, :]

        if "images" in predictions:
            img = predictions["images"]
            if img.shape[-3] == 3 and img.shape[-1] != 3:
                 img = img.permute(0, 2, 3, 1) # (S, 3, H, W) -> (S, H, W, 3)
            predictions["images"] = img[..., ::stride, ::stride, :]
            del img

        # --- PROCESS FEATURES ---
        raw_feats_list = predictions["view_feats"]
        projected_feats_map = []

        for f_raw in raw_feats_list:
            f_proj = self.apply_projector_to_map(f_raw, target_hw=target_hw)
            projected_feats_map.append(f_proj)

        predictions["view_feats"] = projected_feats_map
        del projected_feats_map

        with torch.no_grad():
             bev_base, meta_base = self.run_baseline_inference(
                 predictions,
                 Rmw,
                 tmw_scaled,
                 scale_factor,
                 threshold=threshold # Or use config threshold
             )


        # Initialize CUDA events for high-precision timing
        t_start_pre  = torch.cuda.Event(enable_timing=True)
        t_end_pre    = torch.cuda.Event(enable_timing=True)
        t_start_inf  = torch.cuda.Event(enable_timing=True)
        t_end_inf    = torch.cuda.Event(enable_timing=True)
        t_start_base = torch.cuda.Event(enable_timing=True)
        t_end_base   = torch.cuda.Event(enable_timing=True)
        
        prev_probs = None
        latent_history, coord_history, prob_history, gt_history = [], [], [], []

        static_baseline_vox = None

        # --- TFS tracking: previous frame state ---
        prev_model_keys = None
        prev_model_binary = None
        prev_baseline_keys = None
        prev_baseline_binary = None
        prev_gt_keys = None
        prev_gt_binary = None

        for t in range(T):
            # print(f"Step {t}...")


            imgs = batch["imgs_t"][t]

            self.vox_gt = gt_seq[t]
            if self.vox_gt is None:
                continue

            p = t * step
            cache_path = os.path.join(precomputed_root, seq_id, f"t{p:04d}.pt")
            if not os.path.exists(cache_path):
                continue

            predictions = torch.load(cache_path, map_location=self.device)

            if self.cfg.real_gt_voxels_file:
                real_gt_path = os.path.join(real_gt_root, f"{seq_id}_t{p:04d}.pt")
                real_gt_path = real_gt_path.replace(".glb", "")
                if not os.path.exists(real_gt_path):
                    print("skip, real GT not found")
                    print(real_gt_path)
                    continue

                real_gt = torch.load(real_gt_path, map_location=self.device)


 


            
            
            t_start_pre.record()
            
            R_w2m = np.array([[0, 0, -1],
                            [1, 0, 0],
                            [0, -1, 0]], dtype=np.float32)
            t_w2m = np.zeros(3, dtype=np.float32)

            R_w2m = to_torch(R_w2m, device=self.device)
            t_w2m = to_torch(t_w2m, device=self.device)

            # 2. Rotate Points to World Frame
            WPTS_m = rotate_points(predictions["world_points"], R_w2m, t_w2m)
            predictions["world_points"] = rotate_points(WPTS_m, Rmw, tmw)

            # 3. Apply Scaling
            raw_pts = predictions["world_points"]
            predictions["world_points"] = raw_pts * scale_factor

            # 4. Scale Camera Extrinsics
            if isinstance(predictions["extrinsic"], torch.Tensor):
                predictions["extrinsic"][:, :3, 3] *= scale_factor
            elif isinstance(predictions["extrinsic"], list):
                for i in range(len(predictions["extrinsic"])):
                    predictions["extrinsic"][i][:3, 3] *= scale_factor

            stride = 1 if t == 0 else self.cfg.stride

            if "world_points_conf" in predictions:
                predictions["world_points_conf"] = predictions["world_points_conf"][..., ::stride, ::stride]
                conf_tensor = predictions["world_points_conf"]
                if isinstance(conf_tensor, list):
                    ref = next((x for x in conf_tensor if x is not None), None)
                    target_hw = ref.shape[-2:] if ref is not None else (128, 128)
                else:
                    target_hw = conf_tensor.shape[-2:]
            else:
                target_hw = (512 // stride, 512 // stride)

            if "world_points" in predictions:
                predictions["world_points"] = predictions["world_points"][..., ::stride, ::stride, :]

            if "images" in predictions:
                img = predictions["images"]
                if img.shape[-3] == 3 and img.shape[-1] != 3:
                     img = img.permute(0, 2, 3, 1) # (S, 3, H, W) -> (S, H, W, 3)
                predictions["images"] = img[..., ::stride, ::stride, :]
                del img

            # --- PROCESS FEATURES ---
            raw_feats_list = predictions["view_feats"]
            projected_feats_map = []

            for f_raw in raw_feats_list:
                f_proj = self.apply_projector_to_map(f_raw, target_hw=target_hw)
                projected_feats_map.append(f_proj)

            predictions["view_feats"] = projected_feats_map
            del projected_feats_map
            
            
            t_end_pre.record()

            if self.cfg.real_gt_voxels_file:
                with torch.cuda.amp.autocast(enabled=False):
                    gt_pts = real_gt["world_points"].to(self.device)   # (S, H, W, 3)
                    gt_ex = real_gt["extrinsic"].to(self.device)
                    gt_cams = gt_ex[:, :3, 3]

                    pred_pts = predictions["world_points"]  # already strided (S, H', W', 3)

                    if t == 0:
                        # Match stride to predictions
                        _, pH, pW, _ = pred_pts.shape
                        _, gH, gW, _ = gt_pts.shape

                        # Downsample GT to match prediction resolution
                        gt_pts_strided = gt_pts[:, ::max(1, gH//pH), ::max(1, gW//pW), :]
                        # Ensure exact shape match
                        gt_pts_strided = gt_pts_strided[:, :pH, :pW, :]

                        # Corresponding points for Kabsch
                        gt_flat = gt_pts_strided.reshape(-1, 3)
                        pred_flat = pred_pts.reshape(-1, 3)

                        both_valid = torch.isfinite(gt_flat).all(-1) & torch.isfinite(pred_flat).all(-1)

                        gt_corr = gt_flat[both_valid]
                        pred_corr = pred_flat[both_valid]

                        n = min(50000, gt_corr.shape[0])
                        idx = torch.randperm(gt_corr.shape[0], device=self.device)[:n]

                        R_k, t_k, s_k = kabsch_umeyama_sim3(gt_corr[idx], pred_corr[idx])
                        R_k, t_k, s_k = icp_sim3(
                            gt_corr[idx],
                            pred_corr[idx],
                            init_R=R_k,
                            init_t=t_k,
                            init_s=s_k,
                            max_iters=50
                        )

                    # Apply to FULL resolution GT (not strided)
                    aligned_pts = s_k * (gt_pts @ R_k.T) + t_k
                    aligned_cams = s_k * (gt_cams @ R_k.T) + t_k

                    real_gt["world_points"] = aligned_pts
                    new_ex = gt_ex.clone()
                    new_ex[:, :3, 3] = aligned_cams
                    real_gt["extrinsic"] = new_ex

                    if real_gt["images"].dim() == 4 and real_gt["images"].shape[1] == 3:
                        real_gt["images"] = real_gt["images"].permute(0, 2, 3, 1)

                    self.prev_vox_real_gt = copy.deepcopy(self.vox_real_gt)
                    self.vox_real_gt = build_voxel_from_gt_direct(
                        real_gt, voxel_size=self.cfg.voxel_size, device=self.device, scale_factor=scale_factor
                    )
            """
            if self.cfg.real_gt_voxels_file:
                with torch.cuda.amp.autocast(enabled=False):
                    # Apply the SAME transforms as predictions
                    gt_pts = real_gt["world_points"]  # (S, H, W, 3)
                    gt_pts = rotate_points(gt_pts, R_w2m, t_w2m)
                    gt_pts = rotate_points(gt_pts, Rmw, tmw)
                    gt_pts = gt_pts * scale_factor
                    real_gt["world_points"] = gt_pts

                    # Transform camera centers the same way
                    gt_cams = real_gt["extrinsic"][:, :3, 3]  # (S, 3)
                    gt_cams = (gt_cams @ R_w2m.T + t_w2m)
                    gt_cams = (gt_cams @ Rmw.T + tmw)
                    gt_cams = gt_cams * scale_factor

                    new_ex = real_gt["extrinsic"].clone()
                    new_ex[:, :3, :3] = Rmw @ R_w2m @ real_gt["extrinsic"][:, :3, :3]
                    new_ex[:, :3, 3] = gt_cams
                    real_gt["extrinsic"] = new_ex

                    # Images permute if needed
                    if real_gt["images"].dim() == 4 and real_gt["images"].shape[1] == 3:
                        real_gt["images"] = real_gt["images"].permute(0, 2, 3, 1)

                    self.prev_vox_real_gt = copy.deepcopy(self.vox_real_gt)
                    #self.vox_real_gt = build_voxel_from_sim_data(
                    #    real_gt, voxel_size=self.cfg.voxel_size, device=self.device
                    #)
                    self.vox_real_gt = build_voxel_from_gt_direct(real_gt, voxel_size=self.cfg.voxel_size, device=self.device)
            """
            """
            if self.cfg.real_gt_voxels_file:
                with torch.cuda.amp.autocast(enabled=False):
                    if t == 0:
                        gt_cams = real_gt["extrinsic"][:, :3, 3]
                        pred_cams = predictions["extrinsic"][:, :3, 3]
                        R_kabsch, t_kabsch, s_kabsch = kabsch_umeyama_sim3(gt_cams, pred_cams)
                        S_init, H_init, W_init, _ = predictions["world_points"].shape

                gt_pts = real_gt["world_points"]
                aligned_gt_pts = s_kabsch * (gt_pts @ R_kabsch.T) + t_kabsch

                old_ex = real_gt["extrinsic"]
                new_ex = old_ex.clone()

                # Update Rotation: R_new = R_sim3 @ R_old
                # This rotates the camera's orientation to match the new world frame
                new_ex[:, :3, :3] = torch.matmul(R_kabsch, old_ex[:, :3, :3])

                # Update Translation: Camera Centers
                # This is exactly the same transformation as the points
                # (Using the aligned centers we calculated earlier)
                new_ex[:, :3, 3] = s_kabsch * (gt_cams @ R_kabsch.T) + t_kabsch

                # Save it back to the dictionary
                real_gt["extrinsic"] = new_ex

                real_gt["world_points"] = aligned_gt_pts

            """
            """
                pts = real_gt["world_points"].flatten()

                # 2. Calculate how many (x,y,z) triplets we have per frame (S)
                # We use // to make sure it's an integer
                total_triplets = pts.numel() // 3
                triplets_per_frame = total_triplets // S_init

                # 3. Determine H and W (making them as square as possible)
                H = H_init
                W = triplets_per_frame // H_init

                # 1. Calculate the target number of points for the (S, H, W) grid
                target_n = S_init * H * W

                # 2. Flatten current data for sampling (keeping XYZ and RGB triplets together)
                pts_flat = real_gt["world_points"].reshape(-1, 3)
                conf_flat = real_gt["world_points_conf"].reshape(-1)

                # Ensure images are in (S, H, W, 3) format so they align with the points
                # utils.py expects channels at the end during its internal reshape
                if real_gt["images"].dim() == 4 and real_gt["images"].shape[1] == 3:
                    # Convert (S, 3, H, W) -> (S, H, W, 3) before flattening
                    img_flat = real_gt["images"].permute(0, 2, 3, 1).reshape(-1, 3)
                else:
                    img_flat = real_gt["images"].reshape(-1, 3)

                current_n = pts_flat.shape[0]

                # 3. Synchronized Random Sampling
                # We generate indices ONCE and apply them to all three tensors to keep them synced
                indices = torch.randint(0, current_n, (target_n,), device=pts_flat.device)

                # 4. View back to the required 4D shapes
                try:
                    # Update Points
                    real_gt["world_points"] = pts_flat[indices].view(S_init, H, W, 3)
                    
                    # Update Confidence (1 value per point)
                    real_gt["world_points_conf"] = conf_flat[indices].view(S_init, H, W)
                    
                    # Update Images (3 values per point)
                    # Reshaping to (S, H, W, 3) ensures the colors match the points in the voxel grid
                    real_gt["images"] = img_flat[indices].view(S_init, H, W, 3)

                except RuntimeError as e:
                    print(f"Sampling failed: Expected {target_n} points but indices produced an invalid shape.")
                    raise e

            """
            """
                # Images: convert (S, 3, H, W) -> (S, H, W, 3) if needed
                if real_gt["images"].dim() == 4 and real_gt["images"].shape[1] == 3:
                    real_gt["images"] = real_gt["images"].permute(0, 2, 3, 1)


                self.prev_vox_real_gt = copy.deepcopy(self.vox_real_gt)
                self.vox_real_gt = build_voxel_from_sim_data(real_gt, voxel_size=self.cfg.voxel_size, device=self.device)
            """




            
            t_start_inf.record()

            with torch.enable_grad():
                if not mst and t != 0:
                    bev, mst, _, _ = self.inference(1, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)
                else:
                    bev, mst, _, _ = self.inference(t, imgs, mst, Rmw, tmw_scaled, predictions, scale_factor, threshold=threshold)
            
            t_end_inf.record()     
                   
            t_start_base.record()
            
            with torch.no_grad():
                 bev_base, meta_base = self.run_baseline_inference(
                     predictions,
                     Rmw,
                     tmw_scaled,
                     scale_factor,
                     threshold=threshold
                 )
                 
            t_end_base.record()           
            torch.cuda.synchronize()
            
            # Calculate durations in milliseconds
            ms_pre  = t_start_pre.elapsed_time(t_end_pre)
            ms_inf  = t_start_inf.elapsed_time(t_end_inf)
            ms_base = t_start_base.elapsed_time(t_end_base)

            # Total times
            total_model_time = ms_pre + ms_inf
            total_baseline_time = ms_pre + ms_base

            print(f"Step {t} Timing:")
            print(f"  Shared Preprocessing: {ms_pre:.2f} ms")
            print(f"  Model Inference Only: {ms_inf:.2f} ms")
            print(f"  Base Inference Only:  {ms_base:.2f} ms")
            print(f"  >> TOTAL MODEL:       {total_model_time:.2f} ms")
            print(f"  >> TOTAL BASELINE:    {total_baseline_time:.2f} ms")

            if t == 0:
                static_baseline_vox = copy.deepcopy(self.vox)


            if self.cfg.real_gt_voxels_file:
                metrics_buffer["chamfer_dist"].append(compute_chamfer_dist(self.vox, self.vox_real_gt).item())
                metrics_buffer["gt_chamfer_dist"].append(compute_chamfer_dist(self.vox_gt, self.vox_real_gt).item())
                metrics_buffer["baseline_chamfer_dist"].append(compute_chamfer_dist(self.vox_baseline, self.vox_real_gt).item())
                metrics_buffer["static_chamfer_dist"].append(compute_chamfer_dist(static_baseline_vox, self.vox_real_gt).item())
            else:
                metrics_buffer["chamfer_dist"].append(compute_chamfer_dist(self.vox, self.vox_gt).item())
                metrics_buffer["baseline_chamfer_dist"].append(compute_chamfer_dist(self.vox_baseline, self.vox_gt).item())
                metrics_buffer["static_chamfer_dist"].append(compute_chamfer_dist(static_baseline_vox, self.vox_gt).item())


            if self.cfg.real_gt_voxels_file:                   

                    
                # ---------------------------------------------------------
                # METRICS CALCULATION GT
                # ---------------------------------------------------------

                #logit_pred_before = self.vox_gt.vals_st.clamp(-10.0, 10.0)
                #logit_pred_before = torch.sigmoid(logit_pred_before * 10.0) # Sharp GT
                logit_pred_before = self.vox_gt._display_vals().clamp(-10.0, 10.0)





                #NEW
                #logit_pred_before = (logit_pred_before / 0.75) - 1.0

                if not torch.isfinite(logit_pred_before).all():
                    logit_pred_before = torch.nan_to_num(logit_pred_before, nan=0.001)

                # 3. Align GT keys to Prediction keys
                if self.cfg.real_gt_voxels_file:
                    # 1. Prepare Ground Truth
                    logit_gt = self.vox_real_gt.vals_st.clamp(-10.0, 10.0)
                    p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT
                    #p_occ_tgt = (self.vox_real_gt.hit_count > 0).float()
                    #p_occ_tgt = (self.vox_real_gt.vals_st > 0).float()


                    p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                        self.vox_real_gt, p_occ_tgt, self.vox_gt, default=0.0, r_vox=1
                    )


                # 4. Standard IoU Calculation
                pred_intersect = logit_pred_before[valid_mask]
                tgt_intersect  = p_occ_tgt_aligned[valid_mask]
                pred_fp = logit_pred_before[~valid_mask]

                pred_bin_int = (pred_intersect > 0.0) # Logit > 0 is Prob > 0.5
                tgt_bin_int  = (tgt_intersect  > 0.5)

                tp = (pred_bin_int & tgt_bin_int).sum()
                fp_int = (pred_bin_int & ~tgt_bin_int).sum()
                fn = (~pred_bin_int & tgt_bin_int).sum()

                fp_hallucination = (pred_fp > 0.0).sum()
                total_fp = fp_int + fp_hallucination

                occ_iou = tp / (tp + total_fp + fn + 1e-8)
                occ_recall = tp / (tp + fn + 1e-8)
                occ_precision = tp / (tp + total_fp + 1e-8)
                occ_precision_inter = tp / (tp + fp_int + 1e-8)

                metrics_buffer["gt_occ_iou"].append(occ_iou.item())
                metrics_buffer["gt_occ_recall"].append(occ_recall.item())
                metrics_buffer["gt_occ_precision"].append(occ_precision.item())
                metrics_buffer["gt_occ_precision_inter"].append(occ_precision_inter.item())

                # ---------------------------------------------------------
                # DYNAMIC METRICS (GT)
                # ---------------------------------------------------------
                # Compare Current GT (t) vs Previous GT (t-1)
                if t > 0 and gt_seq[t-1] is not None:
                    if self.cfg.real_gt_voxels_file:
                        vox_gt_prev = self.prev_vox_real_gt
                        

                        # Align Prev GT to Current Keys
                        p_occ_prev_aligned, _ = self.align_probs_to_keys_soft(
                            vox_gt_prev,
                            torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                            self.vox_gt,
                            default=0.0, 
                            r_vox=1
                        )

                    # Define Masks
                    gt_curr = (tgt_intersect > 0.5)
                    # Note: We align Previous GT to Current Keys, so we use valid_mask of current keys
                    gt_prev = (p_occ_prev_aligned[valid_mask] > 0.5)

                    mask_appearing    = (~gt_prev & gt_curr)  # Empty -> Occupied
                    mask_disappearing = (gt_prev & ~gt_curr)  # Occupied -> Empty
                    mask_dynamic      = (gt_prev != gt_curr)

                    # A. Appearing Recall (Do we see new objects?)
                    if mask_appearing.sum() > 0:
                        pred_appearing = (pred_intersect[mask_appearing] > 0.0)
                        metrics_buffer["gt_dyn_recall_appearing"].append(pred_appearing.float().mean().item())

                    # B. Disappearing / Ghosting (Do we clear old objects?)
                    if mask_disappearing.sum() > 0:
                        pred_ghosts = (pred_intersect[mask_disappearing] > 0.0)
                        metrics_buffer["gt_dyn_ghost_rate"].append(pred_ghosts.float().mean().item())
                        metrics_buffer["gt_dyn_recall_disappearing"].append((~pred_ghosts).float().mean().item())

                    # C. Dynamic IoU
                    if mask_dynamic.sum() > 0:
                        pred_dyn = (pred_intersect[mask_dynamic] > 0.0)
                        gt_dyn   = gt_curr[mask_dynamic]
                        intersection = (pred_dyn & gt_dyn).sum()
                        union        = (pred_dyn | gt_dyn).sum()
                        metrics_buffer["gt_dyn_iou"].append((intersection / (union + 1e-8)).item())





            # ---------------------------------------------------------
            # METRICS CALCULATION
            # ---------------------------------------------------------

            logit_pred_before = self.vox.decode_occupancy(with_xyz_cond=False)

            #NEW
            #logit_pred_before = (logit_pred_before / 0.75) - 1.0

            if not torch.isfinite(logit_pred_before).all():
                logit_pred_before = torch.nan_to_num(logit_pred_before, nan=0.001)

            # 3. Align GT keys to Prediction keys
            if self.cfg.real_gt_voxels_file:
                print("real gt")
                # 1. Prepare Ground Truth
                logit_gt = self.vox_real_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT
                #p_occ_tgt = (self.vox_real_gt.hit_count > 0).float()
                #p_occ_tgt = (self.vox_real_gt.vals_st > 0).float()


                p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                    self.vox_real_gt, p_occ_tgt, self.vox, default=0.0, r_vox=1
                )
            else:

                # 1. Prepare Ground Truth
                logit_gt = self.vox_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT

                p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                    self.vox_gt, p_occ_tgt, self.vox, default=0.0, r_vox=1
                )

            # 4. Standard IoU Calculation
            pred_intersect = logit_pred_before[valid_mask]
            tgt_intersect  = p_occ_tgt_aligned[valid_mask]
            pred_fp = logit_pred_before[~valid_mask]

            pred_bin_int = (pred_intersect > 0.0) # Logit > 0 is Prob > 0.5
            tgt_bin_int  = (tgt_intersect  > 0.5)

            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()

            fp_hallucination = (pred_fp > 0.0).sum()
            total_fp = fp_int + fp_hallucination

            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            occ_recall = tp / (tp + fn + 1e-8)
            occ_precision = tp / (tp + total_fp + 1e-8)
            occ_precision_inter = tp / (tp + fp_int + 1e-8)

            metrics_buffer["occ_iou"].append(occ_iou.item())
            metrics_buffer["occ_recall"].append(occ_recall.item())
            metrics_buffer["occ_precision"].append(occ_precision.item())
            metrics_buffer["occ_precision_inter"].append(occ_precision_inter.item())
            # ---------------------------------------------------------
            # DYNAMIC METRICS (Model)
            # ---------------------------------------------------------
            # Compare Current GT (t) vs Previous GT (t-1)
            if t > 0 and gt_seq[t-1] is not None:
                if self.cfg.real_gt_voxels_file:
                    vox_gt_prev = self.prev_vox_real_gt

                    # Align Prev GT to Current Keys
                    p_occ_prev_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        self.vox,
                        default=0.0, 
                        r_vox=1
                    )
                else:
                    vox_gt_prev = gt_seq[t-1]

                    # Align Prev GT to Current Keys
                    p_occ_prev_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        self.vox,
                        default=0.0, 
                        r_vox=1
                    )

                # Define Masks
                gt_curr = (tgt_intersect > 0.5)
                # Note: We align Previous GT to Current Keys, so we use valid_mask of current keys
                gt_prev = (p_occ_prev_aligned[valid_mask] > 0.5)

                mask_appearing    = (~gt_prev & gt_curr)  # Empty -> Occupied
                mask_disappearing = (gt_prev & ~gt_curr)  # Occupied -> Empty
                mask_dynamic      = (gt_prev != gt_curr)

                # A. Appearing Recall (Do we see new objects?)
                if mask_appearing.sum() > 0:
                    pred_appearing = (pred_intersect[mask_appearing] > 0.0)
                    metrics_buffer["dyn_recall_appearing"].append(pred_appearing.float().mean().item())

                # B. Disappearing / Ghosting (Do we clear old objects?)
                if mask_disappearing.sum() > 0:
                    pred_ghosts = (pred_intersect[mask_disappearing] > 0.0)
                    metrics_buffer["dyn_ghost_rate"].append(pred_ghosts.float().mean().item())
                    metrics_buffer["dyn_recall_disappearing"].append((~pred_ghosts).float().mean().item())

                # C. Dynamic IoU
                if mask_dynamic.sum() > 0:
                    pred_dyn = (pred_intersect[mask_dynamic] > 0.0)
                    gt_dyn   = gt_curr[mask_dynamic]
                    intersection = (pred_dyn & gt_dyn).sum()
                    union        = (pred_dyn | gt_dyn).sum()
                    metrics_buffer["dyn_iou"].append((intersection / (union + 1e-8)).item())



            # ---------------------------------------------------------
            # METRICS CALCULATION (BASELINE)
            # ---------------------------------------------------------
            
            # 1. Decode Baseline (Log-Odds directly)
            logit_base = self.vox_baseline._display_vals().clamp(-10.0, 10.0)
            


            # 3. Align GT keys to Prediction keys
            if self.cfg.real_gt_voxels_file:
                # 1. Prepare Ground Truth
                logit_gt = self.vox_real_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT
                #p_occ_tgt = (self.vox_real_gt.hit_count > 0).float()
                #p_occ_tgt = (self.vox_real_gt.vals_st > 0).float()


                p_occ_tgt_base_aligned, valid_mask_base = self.align_probs_to_keys_soft(
                    self.vox_real_gt, p_occ_tgt, self.vox_baseline, default=0.0, r_vox=1
                )
            else:

                # 1. Prepare Ground Truth
                logit_gt = self.vox_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT

                p_occ_tgt_base_aligned, valid_mask_base = self.align_probs_to_keys_soft(
                    self.vox_gt, p_occ_tgt, self.vox_baseline, default=0.0, r_vox=1
                )

 
            
            # 3. Intersection Logic
            base_intersect = logit_base[valid_mask_base]
            tgt_intersect_base = p_occ_tgt_base_aligned[valid_mask_base]
            base_fp = logit_base[~valid_mask_base]
            
            # Threshold: > 0.0 log-odds (0.5 prob) or use self.p.occ_thresh
            base_bin_int = (base_intersect > self.vox_baseline.p.occ_thresh)
            tgt_bin_base = (tgt_intersect_base > 0.5)
            
            tp_base = (base_bin_int & tgt_bin_base).sum()
            fp_int_base = (base_bin_int & ~tgt_bin_base).sum()
            fn_base = (~base_bin_int & tgt_bin_base).sum()
            
            fp_hallucination_base = (base_fp > self.vox_baseline.p.occ_thresh).sum()
            total_fp_base = fp_int_base + fp_hallucination_base
            
            base_iou = tp_base / (tp_base + total_fp_base + fn_base + 1e-8)
            base_recall = tp_base / (tp_base + fn_base + 1e-8)
            base_precision = tp_base / (tp_base + total_fp_base + 1e-8)
            base_precision_inter = tp_base / (tp_base + fp_int_base + 1e-8)

            metrics_buffer["baseline_occ_iou"].append(base_iou.item())
            metrics_buffer["baseline_occ_recall"].append(base_recall.item())
            metrics_buffer["baseline_occ_precision"].append(base_precision.item())
            metrics_buffer["baseline_occ_precision_inter"].append(base_precision_inter.item())
           # ---------------------------------------------------------
            # DYNAMIC METRICS (BASELINE)
            # ---------------------------------------------------------
            if t > 0 and gt_seq[t-1] is not None:
                # 1. Align Previous GT to Baseline's current keys
                # (We need to see what the baseline "sees" relative to what changed in GT)
                if self.cfg.real_gt_voxels_file:
                    vox_gt_prev = self.prev_vox_real_gt


                    # Align Prev GT to Current Keys
                    p_occ_prev_base_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        self.vox_baseline,
                        default=0.0, 
                        r_vox=1
                    )
                else:
                    vox_gt_prev = gt_seq[t-1]

                    # Align Prev GT to Current Keys
                    p_occ_prev_base_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        self.vox_baseline,
                        default=0.0, 
                        r_vox=1
                    )


                # 2. Define Dynamic Masks for Baseline spatial context
                # Use the aligned current GT from the baseline intersection logic above
                gt_curr_base = (tgt_intersect_base > 0.5)
                gt_prev_base = (p_occ_prev_base_aligned[valid_mask_base] > 0.5)

                mask_app_base  = (~gt_prev_base & gt_curr_base)
                mask_dis_base  = (gt_prev_base & ~gt_curr_base)
                mask_dyn_base  = (gt_prev_base != gt_curr_base)

                # 3. Baseline Appearing Recall (New objects)
                if mask_app_base.sum() > 0:
                    base_app_pred = (base_intersect[mask_app_base] > self.vox_baseline.p.occ_thresh)
                    metrics_buffer["baseline_dyn_recall_appearing"].append(base_app_pred.float().mean().item())

                # 4. Baseline Ghosting / Disappearing (Clearing old objects)
                if mask_dis_base.sum() > 0:
                    base_ghosts = (base_intersect[mask_dis_base] > self.vox_baseline.p.occ_thresh)
                    metrics_buffer["baseline_dyn_ghost_rate"].append(base_ghosts.float().mean().item())
                    metrics_buffer["baseline_dyn_recall_disappearing"].append((~base_ghosts).float().mean().item())

                # 5. Baseline Dynamic IoU
                if mask_dyn_base.sum() > 0:
                    base_dyn_pred = (base_intersect[mask_dyn_base] > self.vox_baseline.p.occ_thresh)
                    gt_dyn_base   = gt_curr_base[mask_dyn_base]
                    
                    int_dyn_base = (base_dyn_pred & gt_dyn_base).sum()
                    uni_dyn_base = (base_dyn_pred | gt_dyn_base).sum()
                    metrics_buffer["baseline_dyn_iou"].append((int_dyn_base / (uni_dyn_base + 1e-8)).item())






            # ---------------------------------------------------------
            # METRICS CALCULATION (STATIC)


            # 2. Decode Prediction
            if t == 0:
                static_logit_pred_before = self.vox.decode_occupancy(with_xyz_cond=False)

            #NEW
            #logit_pred_before = (logit_pred_before / 0.75) - 1.0

            if not torch.isfinite(logit_pred_before).all():
                static_logit_pred_before = torch.nan_to_num(static_logit_pred_before, nan=0.001)

            # 3. Align GT keys to Prediction keys
            if self.cfg.real_gt_voxels_file:
                # 1. Prepare Ground Truth
                logit_gt = self.vox_real_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT

                #p_occ_tgt = (self.vox_real_gt.hit_count > 0).float()
                #p_occ_tgt = (self.vox_real_gt.vals_st > 0).float()

                static_p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                    self.vox_real_gt, p_occ_tgt, static_baseline_vox, default=0.0, r_vox=1
                )

            else:

                # 1. Prepare Ground Truth
                logit_gt = self.vox_gt.vals_st.clamp(-10.0, 10.0)
                p_occ_tgt = torch.sigmoid(logit_gt * 10.0) # Sharp GT

                static_p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys_soft(
                    self.vox_gt, p_occ_tgt, static_baseline_vox, default=0.0, r_vox=1
                )

 
            
 

            # 4. Standard IoU Calculation
            pred_intersect = static_logit_pred_before[valid_mask]
            tgt_intersect  = static_p_occ_tgt_aligned[valid_mask]
            pred_fp = static_logit_pred_before[~valid_mask]

            pred_bin_int = (pred_intersect > 0.0) # Logit > 0 is Prob > 0.5
            tgt_bin_int  = (tgt_intersect  > 0.5)

            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()

            fp_hallucination = (pred_fp > 0.0).sum()
            total_fp = fp_int + fp_hallucination

            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            occ_recall = tp / (tp + fn + 1e-8)
            occ_precision = tp / (tp + total_fp + 1e-8)
            occ_precision_inter = tp / (tp + fp_int + 1e-8)

            metrics_buffer["static_occ_iou"].append(occ_iou.item())
            metrics_buffer["static_occ_recall"].append(occ_recall.item())
            metrics_buffer["static_occ_precision"].append(occ_precision.item())
            metrics_buffer["static_occ_precision_inter"].append(occ_precision_inter.item())

            # ---------------------------------------------------------
            # DYNAMIC METRICS (Static)
            # ---------------------------------------------------------
            # Compare Current GT (t) vs Previous GT (t-1)
            if t > 0 and gt_seq[t-1] is not None:

                if self.cfg.real_gt_voxels_file:
                    vox_gt_prev = self.prev_vox_real_gt


                    # Align Prev GT to Current Keys
                    p_occ_prev_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        static_baseline_vox,
                        default=0.0,
                        r_vox=1
                    )
                else:
                    vox_gt_prev = gt_seq[t-1]

                    # Align Prev GT to Current Keys
                    p_occ_prev_aligned, _ = self.align_probs_to_keys_soft(
                        vox_gt_prev,
                        torch.sigmoid(vox_gt_prev.vals_st * 10.0),
                        static_baseline_vox,
                        default=0.0,
                        r_vox=1
                    )

                # Define Masks
                gt_curr = (tgt_intersect > 0.5)
                # Note: We align Previous GT to Current Keys, so we use valid_mask of current keys
                gt_prev = (p_occ_prev_aligned[valid_mask] > 0.5)

                mask_appearing    = (~gt_prev & gt_curr)  # Empty -> Occupied
                mask_disappearing = (gt_prev & ~gt_curr)  # Occupied -> Empty
                mask_dynamic      = (gt_prev != gt_curr)

                # A. Appearing Recall (Do we see new objects?)
                if mask_appearing.sum() > 0:
                    pred_appearing = (pred_intersect[mask_appearing] > 0.0)
                    metrics_buffer["static_dyn_recall_appearing"].append(pred_appearing.float().mean().item())

                # B. Disappearing / Ghosting (Do we clear old objects?)
                if mask_disappearing.sum() > 0:
                    pred_ghosts = (pred_intersect[mask_disappearing] > 0.0)
                    metrics_buffer["static_dyn_ghost_rate"].append(pred_ghosts.float().mean().item())
                    metrics_buffer["static_dyn_recall_disappearing"].append((~pred_ghosts).float().mean().item())

                # C. Dynamic IoU
                if mask_dynamic.sum() > 0:
                    pred_dyn = (pred_intersect[mask_dynamic] > 0.0)
                    gt_dyn   = gt_curr[mask_dynamic]
                    intersection = (pred_dyn & gt_dyn).sum()
                    union        = (pred_dyn | gt_dyn).sum()
                    metrics_buffer["static_dyn_iou"].append((intersection / (union + 1e-8)).item())





            # ---------------------------------------------------------
            # END METRICS
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # TEMPORAL FLICKER SCORE (TFS)
            # ---------------------------------------------------------
            # Model TFS
            curr_model_binary = (self.vox.decode_occupancy(with_xyz_cond=False) > 0.0)
            tfs_model = self.compute_tfs(
                prev_model_keys, prev_model_binary,
                self.vox.keys.clone(), curr_model_binary,
            )
            if tfs_model is not None:
                metrics_buffer["tfs_model"].append(tfs_model)
            prev_model_keys = self.vox.keys.clone()
            prev_model_binary = curr_model_binary.detach().clone()

            # Baseline TFS
            curr_baseline_binary = (self.vox_baseline._display_vals().clamp(-10.0, 10.0) > self.vox_baseline.p.occ_thresh)
            tfs_baseline = self.compute_tfs(
                prev_baseline_keys, prev_baseline_binary,
                self.vox_baseline.keys.clone(), curr_baseline_binary,
            )
            if tfs_baseline is not None:
                metrics_buffer["tfs_baseline"].append(tfs_baseline)
            prev_baseline_keys = self.vox_baseline.keys.clone()
            prev_baseline_binary = curr_baseline_binary.detach().clone()

            # GT TFS
            curr_gt_binary = (self.vox_gt._display_vals().clamp(-10.0, 10.0) > 0.0)
            tfs_gt = self.compute_tfs(
                prev_gt_keys, prev_gt_binary,
                self.vox_gt.keys.clone(), curr_gt_binary,
            )
            if tfs_gt is not None:
                metrics_buffer["tfs_gt"].append(tfs_gt)
            prev_gt_keys = self.vox_gt.keys.clone()
            prev_gt_binary = curr_gt_binary.detach().clone()

            # ---------------------------------------------------------

            bev_spec = BevSpec(
                resolution=self.vox_gt.p.voxel_size,
                width_m=float(self.bev_window_m[0]),
                height_m=float(self.bev_window_m[1]),
                origin_xy=self.bev_origin_xy,
                z_band=(self.z_band_bev[0], self.z_band_bev[1]),
            )

            #bev_gt, meta = bev_from_voxels(self.vox_gt, bev_spec, include_free=True)
            
            bev_gt, meta, prev_probs = bev_from_voxels(self.vox_gt, bev_spec, include_free=True, prev_probs=prev_probs, vis_mode="motion")
            bev_base, meta_base, _ = bev_from_voxels(self.vox_baseline, bev_spec, include_free=True, prev_probs=prev_probs, vis_mode="motion")

            bevs.append(bev)
            bevs_gt.append(bev_gt)
            bevs_baseline.append(bev_base)

            self.vox.z_latent = self.vox.z_latent.detach()
            save_dir = f"debug_viz_pred/{batch_idx}"
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/step_{t}.ply"
            #self.export_debug_ply(fname, t)
            self.export_separated_ply(t, save_dir=save_dir)
            #self.export_latent_colored_ply(f"{save_dir}/latent_step_{t}.ply", t)
            #self.export_kmeans_ply(f"{save_dir}/kmeans_{t}.ply", t)


            #self.visualize_latent_final_analysis(save_dir, t, n_samples=5000)
            #self.generate_reliability_diagram(save_dir, t)


            # Store current state of the whole grid
            latent_history.append(self.vox.z_latent.detach())
            coord_history.append(self.vox.voxel_centers().detach())

            logits = self.vox.decode_occupancy(with_xyz_cond=False)
            prob_history.append(torch.sigmoid(logits).detach())

            # Align GT for this specific step
            gt_logits = gt_seq[t].vals_st.clamp(-10.0, 10.0)
            gt_p, _ = self.align_probs_to_keys_soft(gt_seq[t], torch.sigmoid(gt_logits*10), self.vox)
            gt_history.append(gt_p.detach())



            torch.cuda.empty_cache()

        #self.visualize_latent_history(save_dir, seq_id, latent_history, coord_history, prob_history, gt_history)
        # --- FINAL SUMMARY PRINT ---
        print("-" * 60)
        print(f"SEQUENCE REPORT: {seq_id}")
        if self.cfg.real_gt_voxels_file:                   
            print("-" * 60)
            print(f"  GT Metrics:")
            print(f"    IoU:                 {self.get_avg(metrics_buffer, 'gt_occ_iou'):.4f}")
            print(f"    Recall:              {self.get_avg(metrics_buffer, 'gt_occ_recall'):.4f}")
            print(f"    Precision:           {self.get_avg(metrics_buffer, 'gt_occ_precision'):.4f}")
            
        print("-" * 60)
        print(f"  MODEL Metrics:")
        print(f"    IoU:                 {self.get_avg(metrics_buffer, 'occ_iou'):.4f}")
        print(f"    Recall:              {self.get_avg(metrics_buffer, 'occ_recall'):.4f}")
        print(f"    Precision:           {self.get_avg(metrics_buffer, 'occ_precision'):.4f}")
        print("-" * 60)
        print(f"  BASELINE Metrics:")
        print(f"    IoU:                 {self.get_avg(metrics_buffer, 'baseline_occ_iou'):.4f}")
        print(f"    Recall:              {self.get_avg(metrics_buffer, 'baseline_occ_recall'):.4f}")
        print(f"    Precision:           {self.get_avg(metrics_buffer, 'baseline_occ_precision'):.4f}")
        print("-" * 60)
        print(f"  STATIC Metrics:")
        print(f"    IoU:                 {self.get_avg(metrics_buffer, 'static_occ_iou'):.4f}")
        print(f"    Recall:              {self.get_avg(metrics_buffer, 'static_occ_recall'):.4f}")
        print(f"    Precision:           {self.get_avg(metrics_buffer, 'static_occ_precision'):.4f}")
        if self.cfg.real_gt_voxels_file:                   
            print("-" * 60)
            print(f"  Dynamic Metrics GT (Changes Only):")
            print(f"    Dynamic IoU:         {self.get_avg(metrics_buffer, 'gt_dyn_iou'):.4f}")
            print(f"    Appearing Recall:    {self.get_avg(metrics_buffer, 'gt_dyn_recall_appearing'):.4f}  (High = Fast reaction to new objects)")
            print(f"    Disappearing Recall: {self.get_avg(metrics_buffer, 'gt_dyn_recall_disappearing'):.4f}  (High = Good cleanup)")
            print(f"    Ghost Rate:          {self.get_avg(metrics_buffer, 'gt_dyn_ghost_rate'):.4f}  (High = Objects leave trails)")
            print("-" * 60)
        print(f"  Dynamic Metrics Model (Changes Only):")
        print(f"    Dynamic IoU:         {self.get_avg(metrics_buffer, 'dyn_iou'):.4f}")
        print(f"    Appearing Recall:    {self.get_avg(metrics_buffer, 'dyn_recall_appearing'):.4f}  (High = Fast reaction to new objects)")
        print(f"    Disappearing Recall: {self.get_avg(metrics_buffer, 'dyn_recall_disappearing'):.4f}  (High = Good cleanup)")
        print(f"    Ghost Rate:          {self.get_avg(metrics_buffer, 'dyn_ghost_rate'):.4f}  (High = Objects leave trails)")
        print("-" * 60)
        print(f"  Dynamic Metrics BASELINE (Changes Only):")
        print(f"    Dynamic IoU:         {self.get_avg(metrics_buffer, 'baseline_dyn_iou'):.4f}")
        print(f"    Appearing Recall:    {self.get_avg(metrics_buffer, 'baseline_dyn_recall_appearing'):.4f}  (High = Fast reaction to new objects)")
        print(f"    Disappearing Recall: {self.get_avg(metrics_buffer, 'baseline_dyn_recall_disappearing'):.4f}  (High = Good cleanup)")
        print(f"    Ghost Rate:          {self.get_avg(metrics_buffer, 'baseline_dyn_ghost_rate'):.4f}  (High = Objects leave trails)")
        print("-" * 60)
        print(f"  Dynamic Metrics STATIC (Changes Only):")
        print(f"    Dynamic IoU:         {self.get_avg(metrics_buffer, 'static_dyn_iou'):.4f}")
        print(f"    Appearing Recall:    {self.get_avg(metrics_buffer, 'static_dyn_recall_appearing'):.4f}  (High = Fast reaction to new objects)")
        print(f"    Disappearing Recall: {self.get_avg(metrics_buffer, 'static_dyn_recall_disappearing'):.4f}  (High = Good cleanup)")
        print(f"    Ghost Rate:          {self.get_avg(metrics_buffer, 'static_dyn_ghost_rate'):.4f}  (High = Objects leave trails)")
        print("-" * 60)
        if self.cfg.real_gt_voxels_file:                   
            print(f"    GT Chamfer Dist: {self.get_avg(metrics_buffer, 'gt_chamfer_dist'):.4f}  (Lower = Better)")
        print(f"    Chamfer Dist: {self.get_avg(metrics_buffer, 'chamfer_dist'):.4f}  (Lower = Better)")
        print(f"    Baseline Chamfer Dist: {self.get_avg(metrics_buffer, 'baseline_chamfer_dist'):.4f}  (Lower = Better)")
        print(f"    Static Chamfer Dist: {self.get_avg(metrics_buffer, 'static_chamfer_dist'):.4f}  (Lower = Better)")
        print("-" * 60)
        print(f"  Temporal Flicker Score (Lower = More Stable):")
        print(f"    GT TFS:              {self.get_avg(metrics_buffer, 'tfs_gt'):.4f}")
        print(f"    Model TFS:           {self.get_avg(metrics_buffer, 'tfs_model'):.4f}")
        print(f"    Baseline TFS:        {self.get_avg(metrics_buffer, 'tfs_baseline'):.4f}")
        print("-" * 60)
        print("\n")

        return bevs, bevs_gt, bevs_baseline, metrics_buffer


    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        # Check if the checkpoint contains voxel keys
        if "vox.keys" in state_dict:
            saved_keys = state_dict["vox.keys"]
            target_size = saved_keys.shape[0]
            current_size = self.vox.keys.shape[0]

            if target_size != current_size:
                #print(f"[Checkpoint Load] Resizing voxel grid buffers from {current_size} to {target_size}...")

                # List of all sparse buffers in your VoxelGrid
                buffer_names = [
                    "keys", "vals_st", "vals_lt", "vals",
                    "hit_count", "pos_occ_count", "neg_free_count",
                    "last_occ_epoch", "last_free_epoch", "view_bits",
                    "seen_occ_epoch", "seen_view_bits_e", "occ_epoch_count",
                    "view_bits_cum", "lt_promoted_flag"
                ]

                # Resize every buffer to match the checkpoint shape
                for name in buffer_names:
                    full_key = f"vox.{name}"
                    if full_key in state_dict:
                        saved_tensor = state_dict[full_key]
                        current_buffer = getattr(self.vox, name)

                        # Create a new zero-tensor with the shape from checkpoint
                        new_buffer = torch.zeros(
                            saved_tensor.shape,
                            dtype=current_buffer.dtype,
                            device=self.device
                        )
                        setattr(self.vox, name, new_buffer)

                # Resize Latents
                if "vox.z_latent" in state_dict:
                    saved_z = state_dict["vox.z_latent"]
                    self.vox.z_latent = torch.zeros(
                        saved_z.shape,
                        dtype=self.vox.z_latent.dtype,
                        device=self.device
                    )
        
    def get_avg(self, metrics_buffer, name):
        vals = metrics_buffer[name]
        return sum(vals) / len(vals) if len(vals) > 0 else 0.0

    @staticmethod
    def compute_tfs(prev_keys, prev_binary, curr_keys, curr_binary):
        """
        Temporal Flicker Score: fraction of shared voxels that flip occupancy state
        between consecutive frames. Lower = more temporally stable.
        """
        if prev_keys is None or curr_keys is None:
            return None
        # Hash keys to 1D for fast set intersection
        M = 1000003
        prev_hash = prev_keys[:, 0].long() * M * M + prev_keys[:, 1].long() * M + prev_keys[:, 2].long()
        curr_hash = curr_keys[:, 0].long() * M * M + curr_keys[:, 1].long() * M + curr_keys[:, 2].long()

        # Find voxels present in both frames
        common_mask_prev = torch.isin(prev_hash, curr_hash)
        common_mask_curr = torch.isin(curr_hash, prev_hash)

        n_common = common_mask_prev.sum().item()
        if n_common == 0:
            return None

        # Sort by hash so corresponding voxels align
        prev_common_hash, prev_sort = prev_hash[common_mask_prev].sort()
        curr_common_hash, curr_sort = curr_hash[common_mask_curr].sort()

        prev_occ = prev_binary[common_mask_prev][prev_sort]
        curr_occ = curr_binary[common_mask_curr][curr_sort]

        flips = (prev_occ != curr_occ).float().mean()
        return flips.item()

     

    def export_latent_colored_ply(self, filename, step_idx):
        """
        Exports occupancy colored by PCA of z_latent.
        Uses Whitening + Robust Scaling to maximize color distinctness.
        """
        # 1. Decode occupancy
        logit_pred = self.vox.decode_occupancy(with_xyz_cond=False)
        prob_pred = torch.sigmoid(logit_pred)
        mask_pred_occ = prob_pred > 0.5

        # 2. Geometry
        keys_pred = self.vox.keys[mask_pred_occ]
        if keys_pred.numel() == 0:
            return

        xyz = self.vox._unhash_keys(keys_pred).float()
        xyz = self.vox.origin + (xyz + 0.5) * self.vox.p.voxel_size
        pts = xyz.detach().cpu().numpy()

        # 3. Latents
        latents = self.vox.z_latent[mask_pred_occ].detach().cpu().numpy()

        # 4. Polarized Projection
        if latents.shape[0] >= 3:
            # A. Whiten=True forces component variances to be equal.
            # This prevents the first component (Red) from dominating the map.
            pca = PCA(n_components=3, whiten=True)
            rgb_pca = pca.fit_transform(latents)

            # B. Robust Scaling (The "Polarizer")
            # Clip the bottom 5% and top 5% of values.
            # This ignores outliers and stretches the core data to full contrast.
            q_min = np.quantile(rgb_pca, 0.05, axis=0)
            q_max = np.quantile(rgb_pca, 0.95, axis=0)

            rgb_pca = np.clip(rgb_pca, q_min, q_max)

            # Normalize to 0-1
            rgb_norm = (rgb_pca - q_min) / (q_max - q_min + 1e-8)

            colors = (rgb_norm * 255).astype(np.uint8)
        else:
            colors = np.full((pts.shape[0], 3), 128, dtype=np.uint8)

        # 5. Write PLY
        header = f"""ply
        format ascii 1.0
        element vertex {pts.shape[0]}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

        with open(filename, "w") as f:
            f.write(header)
            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

        print(f"Saved polarized latent PLY: {filename}")


    def visualize_latent_history(self, save_folder, seq_id, latent_history, coord_history, prob_history, gt_history, n_samples=8000):
        """
        Plots the cumulative history of all voxels across all timesteps.
        """
        os.makedirs(save_folder, exist_ok=True)

        # 1. Flatten the history lists into single tensors
        all_latents = torch.cat(latent_history, dim=0).float()   # (Total_Obs, D)
        all_coords = torch.cat(coord_history, dim=0).float()     # (Total_Obs, 3)
        all_probs = torch.cat(prob_history, dim=0).float()       # (Total_Obs,)
        all_gt = torch.cat(gt_history, dim=0).float()             # (Total_Obs,)

        # 2. Calculate Entropy for all points
        eps = 1e-6
        all_entropy = -(all_probs * torch.log(all_probs + eps) + (1 - all_probs) * torch.log(1 - all_probs + eps))

        # 3. Subsample (History gets huge, so we sample to keep UMAP fast)
        num_total = all_latents.shape[0]
        idx = np.random.choice(num_total, min(n_samples, num_total), replace=False)

        lat_sub = all_latents[idx].cpu().numpy()
        prob_sub = all_probs[idx].cpu().numpy()
        ent_sub = all_entropy[idx].cpu().numpy()
        coords_sub = all_coords[idx].cpu().numpy()
        gt_sub = all_gt[idx].cpu().numpy()

        # 4. Run UMAP on the entire history
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        embedding = reducer.fit_transform(lat_sub)

        # 5. Plotting (2x3 Grid)
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))

        # Row 1: The "States" of voxels over time
        axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], c=prob_sub, cmap='coolwarm', s=5, alpha=0.4)
        axes[0, 0].set_title("Occupancy Probability (All Steps)")

        # Color by Error (TP, FP, FN, TN)
        pred_bin = prob_sub > 0.5
        gt_bin = gt_sub > 0.5
        err_sub = np.zeros_like(pred_bin, dtype=int)
        err_sub[pred_bin & gt_bin] = 1 # TP
        err_sub[pred_bin & ~gt_bin] = 2 # FP
        err_sub[~pred_bin & gt_bin] = 3 # FN

        colors = ['#d3d3d3', '#2ca02c', '#d62728', '#1f77b4']
        for i in range(4):
            m = err_sub == i
            axes[0, 1].scatter(embedding[m, 0], embedding[m, 1], c=colors[i], s=5, alpha=0.4)
        axes[0, 1].set_title("Classification States (All Steps)")

        axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], c=ent_sub, cmap='magma', s=5, alpha=0.4)
        axes[0, 2].set_title("Entropy / Uncertainty (All Steps)")

        # Row 2: Proving Spatial Invariance across time
        axes[1, 0].scatter(embedding[:, 0], embedding[:, 1], c=coords_sub[:, 0], cmap='RdYlBu_r', s=5, alpha=0.3)
        axes[1, 0].set_title("X-Position")

        axes[1, 1].scatter(embedding[:, 0], embedding[:, 1], c=coords_sub[:, 1], cmap='RdYlBu_r', s=5, alpha=0.3)
        axes[1, 1].set_title("Y-Position")

        axes[1, 2].scatter(embedding[:, 0], embedding[:, 1], c=coords_sub[:, 2], cmap='viridis', s=5, alpha=0.3)
        axes[1, 2].set_title("Z-Position")

        plt.savefig(f"{save_folder}/latent_history_{seq_id}.png", dpi=200)
        plt.close()

    def generate_reliability_diagram(self, save_folder, step_idx, n_bins=10):
        """
        Generates a Reliability Diagram (Calibration Curve) for voxel occupancy.
        Calculates ECE (Expected Calibration Error).
        """
        os.makedirs(save_folder, exist_ok=True)

        # 1. Extract Data and handle BFloat16 cast
        logits = self.vox.decode_occupancy(with_xyz_cond=False)
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()

        if probs.size == 0:
            return

        # 2. Align Ground Truth labels
        gt_logits = self.vox_gt.vals_st.clamp(-10.0, 10.0)
        gt_probs_raw = torch.sigmoid(gt_logits * 10.0) # Sharp GT

        tgt_soft, valid = self.align_probs_to_keys_soft(
            self.vox_gt, gt_probs_raw, self.vox, default=0.0
        )
        labels = (tgt_soft.detach().float().cpu().numpy() > 0.5).astype(int)

        # Only evaluate voxels that have a corresponding GT
        valid_mask = valid.cpu().numpy()
        probs = probs[valid_mask]
        labels = labels[valid_mask]

        if probs.size == 0:
            return

        # 3. Binning logic
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accs = []
        bin_confs = []
        bin_sizes = []
        ece = 0.0

        for lower, upper in zip(bin_lowers, bin_uppers):
            # Calculated per bin
            in_bin = (probs > lower) & (probs <= upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(probs[in_bin])

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_accs.append(accuracy_in_bin)
                bin_confs.append(avg_confidence_in_bin)
                bin_sizes.append(prop_in_bin)
            else:
                bin_accs.append(0)
                bin_confs.append((lower + upper) / 2)
                bin_sizes.append(0)

        # 4. Plotting
        plt.figure(figsize=(8, 8))

        # The "Perfect Calibration" diagonal
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")

        # The actual reliability curve
        plt.bar(bin_lowers, bin_accs, width=1/n_bins, align='edge',
                alpha=0.8, edgecolor='black', color='#1f77b4', label='Model')

        # Formatting
        plt.text(0.05, 0.9, f"ECE: {ece:.4f}", fontsize=14, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8))

        plt.xlabel("Confidence (Predicted Probability)")
        plt.ylabel("Accuracy (GT Frequency)")
        plt.title(f"Reliability Diagram - Step {step_idx}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(save_folder, f"reliability_step_{step_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved Reliability Diagram to: {save_path} (ECE: {ece:.4f})")

    def visualize_latent_final_analysis(self, save_folder, step_idx, n_samples=5000):
        """
        Generates a 2x3 Comprehensive Analysis Figure:
        Row 1: Predicted Occupancy, Classification Errors, Shannon Entropy
        Row 2: X-Influence, Y-Influence, Z-Influence (Elevation)
        """
        os.makedirs(save_folder, exist_ok=True)

        # 1. Extract Data and Cast to Float32 to avoid BFloat16 NumPy errors
        logits = self.vox.decode_occupancy(with_xyz_cond=False)
        probs = torch.sigmoid(logits).detach().float()
        latents = self.vox.z_latent.detach().float()
        centers = self.vox.voxel_centers().detach().float()

        if latents.shape[0] < 10: return

        # 2. Calculate Shannon Entropy
        eps = 1e-6
        entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))

        # 3. Align Ground Truth for Error Plot
        gt_logits = self.vox_gt.vals_st.clamp(-10.0, 10.0)
        gt_probs = torch.sigmoid(gt_logits * 10.0)
        tgt_soft, _ = self.align_probs_to_keys_soft(self.vox_gt, gt_probs, self.vox)

        # 4. Subsample for Consistency across all 6 plots
        idx = np.random.choice(latents.shape[0], min(n_samples, latents.shape[0]), replace=False)

        lat_sub = latents[idx].cpu().numpy()
        prob_sub = probs[idx].cpu().numpy()
        ent_sub = entropy[idx].cpu().numpy()
        x_sub, y_sub, z_sub = centers[idx, 0].cpu().numpy(), centers[idx, 1].cpu().numpy(), centers[idx, 2].cpu().numpy()

        # Determine Error Labels (0:TN, 1:TP, 2:FP, 3:FN)
        pred_bin = prob_sub > 0.5
        gt_bin = tgt_soft[idx].cpu().numpy() > 0.5
        err_sub = np.zeros_like(pred_bin, dtype=int)
        err_sub[pred_bin & gt_bin] = 1
        err_sub[pred_bin & ~gt_bin] = 2
        err_sub[~pred_bin & gt_bin] = 3

        # 5. Run UMAP (Once)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        embedding = reducer.fit_transform(lat_sub)

        # 6. Plotting 2 rows x 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # --- ROW 1: PERFORMANCE & UNCERTAINTY ---
        # Col 1: Predicted Occupancy
        sc00 = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], c=prob_sub, cmap='coolwarm', s=8)
        fig.colorbar(sc00, ax=axes[0, 0], label='Occupancy Probability')
        axes[0, 0].set_title("Predicted Occupancy")

        # Col 2: Classification Errors
        colors = ['#d3d3d3', '#2ca02c', '#d62728', '#1f77b4']
        names = ['True Negative', 'True Positive', 'False Positive', 'False Negative']
        for i in range(4):
            mask = err_sub == i
            axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], c=colors[i], label=names[i], s=10, alpha=0.6)
        axes[0, 1].legend(loc='upper right', markerscale=2)
        axes[0, 1].set_title("Classification Errors")

        # Col 3: Shannon Entropy (Geometric Uncertainty)
        sc02 = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], c=ent_sub, cmap='magma', s=8)
        fig.colorbar(sc02, ax=axes[0, 2], label='H(p)')
        axes[0, 2].set_title("Entropy (Uncertainty)")

        # --- ROW 2: SPATIAL INVARIANCE ---
        # Col 1: X-Coordinate Influence
        sc10 = axes[1, 0].scatter(embedding[:, 0], embedding[:, 1], c=x_sub, cmap='RdYlBu_r', s=8)
        fig.colorbar(sc10, ax=axes[1, 0], label='X (m)')
        axes[1, 0].set_title("X-Position")

        # Col 2: Y-Coordinate Influence
        sc11 = axes[1, 1].scatter(embedding[:, 0], embedding[:, 1], c=y_sub, cmap='RdYlBu_r', s=8)
        fig.colorbar(sc11, ax=axes[1, 1], label='Y (m)')
        axes[1, 1].set_title("Y-Position")

        # Col 3: Z-Coordinate (Elevation) Influence
        sc12 = axes[1, 2].scatter(embedding[:, 0], embedding[:, 1], c=z_sub, cmap='viridis', s=8)
        fig.colorbar(sc12, ax=axes[1, 2], label='Z (m)')
        axes[1, 2].set_title("Z-Position")

        # Final Save
        save_path = os.path.join(save_folder, f"final_latent_analysis_step_{step_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved final multi-pane latent analysis to: {save_path}")

    def export_kmeans_ply(self, filename, step_idx, n_clusters=4):
        """
        Exports occupancy colored by K-Means clustering of latents.
        This demonstrates 'unsupervised semantic segmentation'.
        """
        # 1. Decode occupancy
        logit_pred = self.vox.decode_occupancy(with_xyz_cond=False)
        prob_pred = torch.sigmoid(logit_pred)
        mask_pred_occ = prob_pred > 0.5

        # 2. Geometry
        keys_pred = self.vox.keys[mask_pred_occ]
        if keys_pred.numel() == 0:
            return

        xyz = self.vox._unhash_keys(keys_pred).float()
        xyz = self.vox.origin + (xyz + 0.5) * self.vox.p.voxel_size
        pts = xyz.detach().cpu().numpy()

        # 3. Latents
        latents = self.vox.z_latent[mask_pred_occ].detach().cpu().numpy()

        # 4. K-Means Clustering
        if latents.shape[0] >= n_clusters:
            from sklearn.cluster import KMeans
            # Use a fixed random_state for consistency across timesteps/runs
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(latents)

            # 5. Map Clusters to Distinct Colors (using matplotlib tab10/tab20)
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab10') # Distinct categorical colors

            # Map labels 0..K to colors 0..255
            colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
            for i in range(pts.shape[0]):
                # Get RGBA from cmap, drop alpha, convert to 0-255
                rgba = cmap(labels[i] / max(1, n_clusters - 1)) # Normalize to 0-1
                colors[i] = (np.array(rgba[:3]) * 255).astype(np.uint8)

        else:
            colors = np.full((pts.shape[0], 3), 128, dtype=np.uint8)

        # 6. Write PLY
        header = f"""ply
        format ascii 1.0
        element vertex {pts.shape[0]}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """

        with open(filename, "w") as f:
            f.write(header)
            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

        print(f"Saved K-Means PLY: {filename}")

    def export_separated_ply(self, step_idx, save_dir="debug_viz"):
        """
        Exports GT and Prediction to separate PLY files.
        - GT: Green points (step_X_gt.ply)
        - Pred: Red points (step_X_pred.ply)
        """
        os.makedirs(save_dir, exist_ok=True)

        # -----------------------------
        # 1. Export Ground Truth (Green)
        # -----------------------------
        # GT uses vals_st directly
        mask_gt = self.vox_gt.occupied_mask()

        if mask_gt.any():
            # Get centers of occupied GT voxels
            centers_gt = self.vox_gt.voxel_centers()[mask_gt].detach().cpu().numpy()

            # Create Green colors (0, 255, 0)
            colors_gt = np.zeros_like(centers_gt)
            colors_gt[:, 1] = 255

            self._write_ply(
                os.path.join(save_dir, f"step_{step_idx}_gt.ply"),
                centers_gt,
                colors_gt
            )

        # GT uses vals_st directly
        mask_baseline = self.vox_baseline.occupied_mask()

        if mask_baseline.any():
            # Get centers of occupied GT voxels
            centers_baseline = self.vox_baseline.voxel_centers()[mask_baseline].detach().cpu().numpy()

            # Create Green colors (0, 255, 0)
            colors_baseline = np.zeros_like(centers_baseline)
            colors_baseline[:, 1] = 255

            self._write_ply(
                os.path.join(save_dir, f"step_{step_idx}_baseline.ply"),
                centers_baseline,
                colors_baseline
            )


        if self.cfg.real_gt_voxels_file:
            # GT uses vals_st directly
            mask_real = self.vox_real_gt.occupied_mask()

            if mask_real.any():
                # Get centers of occupied GT voxels
                centers_real = self.vox_real_gt.voxel_centers()[mask_real].detach().cpu().numpy()

                # Create Green colors (0, 255, 0)
                colors_real = np.zeros_like(centers_real)
                colors_real[:, 1] = 255

                self._write_ply(
                    os.path.join(save_dir, f"step_{step_idx}_real_gt.ply"),
                    centers_real,
                    colors_real
                )



        # -----------------------------
        # 2. Export Prediction (Red)
        # -----------------------------
        # Pred needs decoding from latent state
        logits_pred = self.vox.decode_occupancy(with_xyz_cond=False)
        probs_pred = torch.sigmoid(logits_pred)

        # Threshold at 0.5 (or custom threshold)
        mask_pred = probs_pred > 0.5

        if mask_pred.any():
            # Get centers of predicted occupied voxels
            centers_pred = self.vox.voxel_centers()[mask_pred].detach().cpu().numpy()

            # Create Red colors (255, 0, 0)
            colors_pred = np.zeros_like(centers_pred)
            colors_pred[:, 0] = 255

            self._write_ply(
                os.path.join(save_dir, f"step_{step_idx}_pred.ply"),
                centers_pred,
                colors_pred
            )

        print(f"[Viz] Saved separated PLYs to {save_dir}/step_{step_idx}_*.ply")

    def _write_ply(self, filename, points, colors):
        """Helper to write colored PLY file"""
        if len(points) == 0:
            return

        header = f"""ply
        format ascii 1.0
        element vertex {len(points)}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        with open(filename, "w") as f:
            f.write(header)
            for p, c in zip(points, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    def export_debug_ply(self, filename, step_idx):
        """
        Exports the current State vs GT to a color-coded PLY file.
        Green = Correct Wall
        Red   = False Positive (Ghost)
        Blue  = False Negative (Missed Wall)
        """
        # 1. Get Prediction Data
        # Decode occupancy
        logit_pred = self.vox.decode_occupancy(with_xyz_cond=False)
        prob_pred = torch.sigmoid(logit_pred)

        # Threshold (what the model thinks is a wall)
        mask_pred_occ = prob_pred > 0.5
        keys_pred = self.vox.keys[mask_pred_occ]

        # 2. Get GT Data
        # Ensure we are looking at the same coordinate system
        # (Assuming self.vox_gt is already loaded for this timestep)
        logit_gt = self.vox_gt.vals_st
        prob_gt = torch.sigmoid(logit_gt * 10.0) # Sharp GT
        mask_gt_occ = prob_gt > 0.5
        keys_gt = self.vox_gt.keys[mask_gt_occ]

        if keys_pred.numel() == 0 and keys_gt.numel() == 0:
            return

        # 3. Find Intersection (True Positives)
        # We use the unique keys logic
        # Note: keys are int64 hashes

        # Convert to sets for easy set logic (fast enough for <100k voxels)
        # OR use tensor logic if strictly needed, but CPU set is easier for debug
        set_pred = set(keys_pred.detach().cpu().numpy().tolist())
        set_gt   = set(keys_gt.detach().cpu().numpy().tolist())

        tp_keys = list(set_pred & set_gt)
        fp_keys = list(set_pred - set_gt)
        fn_keys = list(set_gt - set_pred)

        # 4. Collect Points and Colors
        all_points = []
        all_colors = []
        
          # Helper to unhash and move to numpy
        def process_keys(k_list, color):
            if not k_list: return
            k_tensor = torch.tensor(k_list, dtype=torch.int64, device=self.device)
            xyz = self.vox._unhash_keys(k_tensor).float()
            # Convert grid coords to world coords
            xyz = self.vox.origin + (xyz + 0.5) * self.vox.p.voxel_size

            pts = xyz.detach().cpu().numpy()
            cols = np.tile(np.array(color), (pts.shape[0], 1))

            all_points.append(pts)
            all_colors.append(cols)

        # GREEN for Match
        process_keys(tp_keys, [0, 255, 0])
        # RED for Ghost
        process_keys(fp_keys, [255, 0, 0])
        # BLUE for Missed
        process_keys(fn_keys, [0, 0, 255])

        if not all_points:
            return

        # 5. Concatenate and Write PLY
        pts_final = np.concatenate(all_points, axis=0)
        col_final = np.concatenate(all_colors, axis=0)

        header = f"""ply
        format ascii 1.0
        element vertex {pts_final.shape[0]}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        with open(filename, "w") as f:
            f.write(header)
            for p, c in zip(pts_final, col_final):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        print(f"Saved debug PLY: {filename}")
    
        
# dataset_auto.py
import os, re, random
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from dust3r.utils.image import load_images as li
import pytorch_lightning as pl

# ---------- utils ----------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _list_dirs(path: str) -> List[str]:
    return sorted(
        [os.path.join(path, d) for d in os.listdir(path)
         if os.path.isdir(os.path.join(path, d))],
        key=lambda p: _natural_key(os.path.basename(p))
    )

def _list_imgs(path: str, exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}) -> List[str]:
    files = [os.path.join(path, f) for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f))
             and os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_natural_key)
    return files



import os
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _has_image(filenames):
    return any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in filenames)


def _sequence_dirs_from_root(root: str) -> List[str]:
    """
    For layouts like:

        root/
          scene1/.../sub_scene_X/
            time0/
              img.*
            time1/
              img.*
          scene2/.../sub_scene_Y/
            time0/
              img.*

    A 'sequence' is defined as the *parent directory* of any `time*` folder
    that actually contains images. There can be many sub-scenes and arbitrary
    nesting above them.
    """
    root = os.path.normpath(root)
    sequence_dirs = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = os.path.normpath(dirpath)
        base = os.path.basename(dirpath)

        # We only care about timestep folders like time0, time1, ...
        if base.startswith("time"):
            if _has_image(filenames):      # or skip this check if you fully trust structure
                parent = os.path.dirname(dirpath)
                sequence_dirs.add(parent)

    if not sequence_dirs:
        raise FileNotFoundError(f"No sequences found under: {root}")

    # Natural sort by basename: scene1_sub1, scene1_sub2, ..., scene10_sub3
    seqs = sorted(sequence_dirs, key=lambda p: _natural_key(os.path.basename(p)))
    return seqs


# ---------- dataset ----------
class HabitatSeqDataset(Dataset):
    """
    Each sample = one sequence with timesteps.
    """
    def __init__(
        self,
        dataset_root: str,
        size: int = 512,
        verbose: bool = False,
        min_images_per_timestep: int = 1,
        sequences: Optional[List[str]] = None,   # pass a subset for train/val if you want
        skip=False,
        seq_list: str = "/cluster/scratch/kochmar/renders/seq_manifest.json",
        step=1

    ):
        self.root = dataset_root
        self.size = size
        self.verbose = verbose
        self.min_images_per_timestep = min_images_per_timestep
        self.skip = skip
        self.seq_list = seq_list
        self.step = step
        
        if sequences is None:
            # seqs = _sequence_dirs_from_root(dataset_root)
            with open(self.seq_list) as f:
                all_entries = json.load(f)

            if self.skip:
                seqs = [e["seq_path"] for e in all_entries if e["has_gt"]]
                all_ids  = [e["seq_id"]  for e in all_entries if e["has_gt"]]
            else:
                seqs = [e["seq_path"] for e in all_entries]
                all_ids  = [e["seq_id"]  for e in all_entries]
                
        else:
            seqs = [p if os.path.isabs(p) else os.path.join(dataset_root, p) for p in sequences]
        for s in seqs:
            if not os.path.isdir(s):
                raise FileNotFoundError(f"Sequence dir missing: {s}")
            
            
        self.seq_paths = seqs

    def __len__(self): return len(self.seq_paths)

    def _list_timesteps(self, seq_dir: str) -> List[str]:
        # timesteps are immediate subfolders; if none, treat the seq_dir itself as one timestep
        all_dirs = _list_dirs(seq_dir)
        valid_dirs = []
        for d in all_dirs:
            base = os.path.basename(d)
            if base.startswith("time") or base.isdigit():
                valid_dirs.append(d)

        # If no subfolders found, maybe the seq_dir itself is the timestep?
        return valid_dirs if valid_dirs else [seq_dir]

    def _load_timestep(self, t_dir: str) -> List[Dict]:
        img_paths = _list_imgs(t_dir)
        if len(img_paths) < self.min_images_per_timestep:
            return []

        try:
            # Attempt to load images using dust3r utils
            return li(img_paths, size=self.size, verbose=False)
        except (PIL.UnidentifiedImageError, OSError, Exception) as e:
            # --- CRASH PROTECTION ---
            # If a file is corrupt (e.g., cam_9.jpg), print a warning and SKIP this timestep.
            print(f"\n[WARN] Corrupted data in {t_dir}")
            print(f"       Skipping this timestep. Error: {e}")
            return []

    def __getitem__(self, idx: int) -> Dict:
        seq_dir = self.seq_paths[idx]
        t_dirs = self._list_timesteps(seq_dir)

        imgs_t: List[List[Dict]] = []
        for t, td in enumerate(t_dirs):
            if t % self.step != 0 and self.skip:
                continue

            
            #if t > 10:
            if t > 120:
                break

            imgs = self._load_timestep(td)

            if imgs:
                imgs_t.append(imgs)

        if not imgs_t:
            raise RuntimeError(f"No images found for sequence: {seq_dir}")

        p = seq_dir.rstrip("/") 
        basis = os.path.basename(os.path.dirname(p)) # "kfPV7w3FaU5.basis" 
        basis = basis.replace(".basis", "") # "kfPV7w3FaU5" 
        final = os.path.basename(p) # "0" 
        seq_id = f"{basis}_{final}"

        return {
            "seq_id": seq_id,
            "seq_path": seq_dir,
            "timesteps": len(imgs_t),
            "imgs_t": imgs_t,   # List[List[dict]]; each inner list is what your inference() expects
        }
        
        

# ---------- datamodule (no seq_list needed) ----------
class HabitatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        batch_size: int = 1,
        num_workers: int = 1,
        size: int = 512,
        verbose: bool = False,
        train_val_split: float = 0.0,  # 0 = all train, else fraction for val (e.g., 0.1)
        seed: int = 42,
        skip=False,
        seq_list: str = "/cluster/scratch/kochmar/renders/seq_manifest.json",
        step=1
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.verbose = verbose
        self.train_val_split = train_val_split
        self.seed = seed

        self.train_set = None
        self.val_set = None
        self.skip = skip
        self.seq_list = seq_list
        self.step = step
    def setup(self, stage: Optional[str] = None):
 
        with open(self.seq_list) as f:
            all_entries = json.load(f)

        if self.skip:
            all_seqs = [e["seq_path"] for e in all_entries if e["has_gt"]]
            all_ids  = [e["seq_id"]  for e in all_entries if e["has_gt"]]
        else:
            all_seqs = [e["seq_path"] for e in all_entries]
            all_ids  = [e["seq_id"]  for e in all_entries]

        #print("got seqs")
        if self.train_val_split > 0.0:
            #random.Random(self.seed).shuffle(all_seqs)
            n_val = max(1, int(len(all_seqs) * self.train_val_split))
            val_seqs = all_seqs[:n_val]
            train_seqs = all_seqs[n_val:]
        else:
            train_seqs, val_seqs = all_seqs, []

        self.train_set = HabitatSeqDataset(
            dataset_root=self.dataset_root,
            size=self.size,
            verbose=self.verbose,
            sequences=train_seqs,
            skip=self.skip,
            seq_list=self.seq_list,
            step=self.step
        )
        
        self.val_set = HabitatSeqDataset(
            dataset_root=self.dataset_root,
            size=self.size,
            verbose=self.verbose,
            sequences=val_seqs,
            skip=self.skip,
            seq_list=self.seq_list,
            step=self.step
        ) if val_seqs else None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,              # keep 1 if your step assumes one sequence
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_one_seq
        )

    def val_dataloader(self):
        if not self.val_set: return None
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_one_seq
        )

    @staticmethod
    def _collate_one_seq(batch_list: List[Dict]) -> Dict:
        assert len(batch_list) == 1, "Set batch_size=1 (one sequence per batch)."
        return batch_list[0]


# --------------------------
# 7) Entrypoint
# --------------------------
def main():
    # Fill your paths here or read from CLI/yaml
    cfg = TrainConfig(
        # dataset_root="/Users/marvin/Documents/Thesis/repo/dataset_generation/habitat/",
        #dataset_root="/home/mpk40/Documents/data/",
        #dataset_root="/cluster/scratch/kochmar/renders/",
        #dataset_root="/cluster/scratch/kochmar/eval_train/",
        dataset_root="/cluster/scratch/kochmar/hm3d_gt_2/",

        real_gt_voxels_file="hm3d_voxels_2",

        #gt_voxels_file="gt_voxels_per_timestep_new",
        #gt_voxels_file="gt_voxels_per_timestep_new_3",
        gt_voxels_file="gt_voxels_per_timestep_new",

        #precomputed_cache_file="precomputed_cache",
        #precomputed_cache_file="precomputed_cache_2",
        precomputed_cache_file="precomputed_cache",

        #pose_file="gt_poses_new",
        #pose_file="gt_poses_new_3",
        pose_file="gt_poses_new",

        seq_file="seq_manifest.json",
        voxel_size=0.2,
        radius_m=1,
        topk=8,
        temp=0.5,
        feature_dim=16,
        #feature_dim=32,

        occ_decoder_hidden=64,
        lr=3e-5,
        #lr=3e-4,
        #lr=3e-3,
        max_epochs=200,
        batch_size=1,
        num_workers=0,
        precision="bf16",
        skip=True,
        weight_decay=0.05,
        #weight_decay=0.00,
        lambda_occ= 1.0,
        lambda_temp = 0.05,      # temporal consistency weight
        #lambda_temp = 0.0,      # temporal consistency weight

        lambda_ent = 5e-3,      # routing entropy reg
        #lambda_ent = 0.0,      # routing entropy reg
        lambda_tv = 5e-3 ,      # (optional) spatial TV on occupancy
        #lambda_tv = 0.0,      # (optional) spatial TV on occupancy
    )

    dm = HabitatDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size=512,
        verbose=False,
        #train_val_split=0.5,  # or whatever you want
        train_val_split=0.2,  # or whatever you want
        skip=True,
        seq_list=os.path.join(cfg.dataset_root, cfg.seq_file),
        step=STEP
        
    )

    sys = VoxelUpdaterSystem(cfg)
    #sys = VoxelUpdaterSystem.load_from_checkpoint(
    #    "/cluster/scratch/kochmar/checkpoints/voxup-epoch=39-val_loss_total=11.3340.ckpt",
    #    strict=False,
    #    # This overrides the saved hparams with your new config
    #    cfg=cfg
    #)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath="/cluster/scratch/kochmar/checkpoints/full_finetune",       # Explicitly set a folder so you can find them
        monitor="val_loss_total",
        save_top_k=5,
        mode="min",
        filename="voxup-{epoch:02d}-{val_loss_total:.4f}" # Match the key logged in validation_step
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")

 
    #print(">>> before Trainer()", flush=True)


    # --- Wandb logger ---
    wandb_logger = WandbLogger(
        project="voxel_dust3r",          # choose a project name
        name="voxup-manualopt",          # optional run name
        config=cfg.__dict__,             # logs all your hyperparams
        save_dir="./wandb_logs",         # where to put local files
    )


    # 2. Load the checkpoint file manually
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full7/voxup-epoch=05-val_loss_total=8.7155.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=09-val_loss_total=7.6451.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu") # Load to CPU first to save GPU mem
    state_dict = checkpoint["state_dict"]

    # 3. Load the state dictionary with strict=False
    #    This tells PyTorch: "If you see 'log_temp' in the file but not in the model, just ignore it."
    keys_to_remove = [
        "vox.keys",
        "vox.vals_st",
        "vox.vals_lt",
        "vox.vals",
        "vox.hit_count",
        "vox.pos_occ_count",
        "vox.neg_free_count",
        "vox.last_occ_epoch",
        "vox.last_free_epoch",
        "vox.view_bits",
        "vox.seen_occ_epoch",
        "vox.seen_view_bits_e",
        "vox.occ_epoch_count",
        "vox.view_bits_cum",
        "vox.lt_promoted_flag",
        "vox.z_latent"  # CRITICAL: Also remove the old latent vectors!
    ]

    # 3. Delete them from the dictionary
    print("Filtering checkpoint: Removing voxel structure, keeping network weights...")
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    keys = sys.load_state_dict(checkpoint["state_dict"], strict=False)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        #gradient_clip_val=1.0,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        callbacks=[ckpt_cb, lr_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        #strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        logger=wandb_logger,
    )
    #print(">>> before trainer.fit()", flush=True)
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full/voxup-epoch=07-val_loss_total=10.9045.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full7/voxup-epoch=05-val_loss_total=8.4459.ckpt"

    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full9/voxup-epoch=01-val_loss_total=7.5519.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full9/voxup-epoch=03-val_loss_total=8.5959.ckpt"
    #ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=03-val_loss_total=7.6833.ckpt"
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full10/voxup-epoch=09-val_loss_total=7.6451.ckpt"



    #trainer.fit(sys, dm, ckpt_path=ckpt_path)

    trainer.fit(sys, dm)
if __name__ == "__main__":
    main()