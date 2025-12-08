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
import traceback
from torch.profiler import profile, ProfilerActivity
import gc
import torch.serialization as serialization
import argparse

import wandb
from pytorch_lightning.loggers import WandbLogger

serialization.add_safe_globals([argparse.Namespace])

import logging
from torch.cuda.amp import autocast


logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)


import numpy as np
from voxel.voxel import TorchSparseVoxelGrid, VoxelParams

def load_sparse_voxel_grid(path, device):
    data = np.load(path)
    origin = data["origin"].astype(np.float32)
    voxel_size = float(data["voxel_size"][0])
    keys = torch.from_numpy(data["keys"]).to(device)
    vals = torch.from_numpy(data["vals"]).to(device, dtype=torch.float32)

    vox_gt = TorchSparseVoxelGrid(
        origin_xyz=origin,
        params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
        device=device,
    )
    vox_gt.keys = keys
    vox_gt.vals_st = vals
    return vox_gt


def _dump_prof(prof, tag="trace"):
    try:
        prof.export_chrome_trace(f"{tag}.json")
        print(f"[profiler] wrote {tag}.json")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
    except Exception as e:
        print("[profiler] export failed:", e)

def _profile_block(tag, fn, *args, **kwargs):
    """
    Runs fn(*args, **kwargs) under a profiler. Always exports a trace,
    even if fn raises, so you get something before the crash.
    """
    prof = profile(
        activities=[ProfilerActivity.CPU],  # add ProfilerActivity.CUDA if you ever use CUDA
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    )
    prof.__enter__()
    try:
        return fn(*args, **kwargs)
    except Exception:
        print(f"[profiler] exception inside '{tag}', exporting trace then re-raising")
        _dump_prof(prof, f"{tag}_CRASH")
        traceback.print_exc()
        raise
    finally:
        prof.__exit__(None, None, None)
        _dump_prof(prof, f"{tag}_OK")

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
    radius_m: float = 0.25
    topk: int = 8
    temp: float = 0.5

    # model
    feature_dim: int = 64         # latent/feature size (match PointNeXt output)
    occ_decoder_hidden: int = 64
    ema_to_st: float = 0.4        # how strongly decoded prob refreshes ST log-odds during training updates

    # optimization
    lr: float = 1e-3
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

# --------------------------
# 4) Dataset (sequence-level)
#   Yields a dict with all frames of one sequence; teacher can be precomputed and cached.
# --------------------------
class HabitatSeqDataset(Dataset):
    def __init__(self, root: str, seq_list_file: str):
        with open(seq_list_file, "r") as f:
            self.seq_paths = [os.path.join(root, line.strip()) for line in f if line.strip()]
        assert len(self.seq_paths) > 0, "Empty sequence list."

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx: int) -> Dict:
        seq_dir = self.seq_paths[idx]
        # TODO: load sequence frames here:
        # - images per camera per timestep OR already-formed partial point clouds per timestep
        # - camera poses / intrinsics if needed
        # - optionally: cached teacher voxel labels per timestep
        # Return a dict:
        # {
        #   "seq_id": str,
        #   "timesteps": int T,
        #   "points_t":   List[Tensor (Nt,3)]  # partial points per t (world)
        #   "rgb_t":      List[Tensor (Nt,3)]  # optional colors per point
        #   "cams_t":     List[Tensor (Nt,3)]  # optional per-point camera centers
        #   "teacher_occ": Optional[List[Dict]] # optional precomputed teacher labels per t
        #   "init_full":  Dict with {"points": Tensor (N0,3), "rgb": (N0,3)} for init
        # }
        raise NotImplementedError

# --------------------------
# 5) LightningModule
# --------------------------
class VoxelUpdaterSystem(pl.LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg
        self.feature_dim = 32
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
        
        self.projector = FeatureProjector(out_dim=self.feature_dim)


        self.automatic_optimization = False   # <<< add this

    def configure_optimizers(self):
        params = list(self.vox.sim_net.parameters()) + \
                 list(self.vox.gru_cell.parameters()) + \
                 list(self.vox.decoder.parameters()) + \
                 list(self.projector.parameters()) + \
                 list(self.vox.gate_mlp.parameters())
        # if your feature extractor is finetuned, extend params with extractor params
        
        # opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # return opt
    
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.max_epochs, eta_min=self.cfg.lr * 0.1
        )

        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

    def inference(self, i, imgs, mst, Rmw=None, tmw=None):


        POINTS = "world_points"
        CONF = "world_points_conf"
        threshold = 1.0     
        z_clip_map = (-0.1, 0.3)   

        R_w2m = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)

        t_w2m = np.zeros(3, dtype=np.float32)

        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)
        
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

            print("Running inference took", length, "seconds!")
            

        else:
        
            start = time.time()

            changed_idx = changed_images(image_tensors, self.keyframes, thresh=0.000005)
            
            print(changed_idx)

            end = time.time()
            length = end - start
            
            if len(changed_idx) < 2:
                    # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                    self.vox.next_epoch()
                    return None, None, None, None
 
            print("Finding changed images took", length, "seconds!")

            changed_idx = [0] + [x for x in changed_idx if x != 0]
            
            index_map = {new: old for new, old in enumerate(changed_idx)}
                    
            idx_t = torch.tensor(changed_idx, device=self.device, dtype=torch.long)
            self.keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
            
            
            print("final changed idx:", changed_idx)

            start = time.time()
   
            print("inference pred")
            mst = True
            predictions = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=self.projector)
                
            end = time.time()
            length = end - start

            print("Running inference took", length, "seconds!")


        # Keep tensors; only extract what we need later.
        # If you truly need NumPy later, convert specific keys then.
        needed = {
            "images","extrinsic", POINTS, CONF, "view_feats"
        }
        for k in list(predictions.keys()):
            if k not in needed:
                del predictions[k]  # drop unneeded heavy stuff early



        start = time.time()
        
        WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
        if Rmw is None or tmw is None:
            Rmw, tmw, info = align_pointcloud_torch_fast(WPTS_m, inlier_dist=self.voxel_size*0.75, ransac_iters=500, point_chunk=5_000_000, cand_chunk=4096)
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)

        """
        if self.vox_gt is not None and self.vox_gt.keys.numel() > 0:
            # 1. Get GT Points
            gt_ijk = self.vox_gt._unhash_keys(self.vox_gt.keys)
            gt_pts = self.vox_gt.origin + (gt_ijk.float() + 0.5) * self.vox_gt.p.voxel_size

            # 2. Flatten Prediction
            pred_flat = WPTS_m.reshape(-1, 3)
            valid = torch.isfinite(pred_flat).all(dim=1)
            pred_valid = pred_flat[valid]

            if pred_valid.shape[0] > 0 and gt_pts.shape[0] > 0:
                # --- A. Centering ---
                pred_c = pred_valid.mean(dim=0)
                gt_c = gt_pts.mean(dim=0)

                pred_centered = pred_valid - pred_c
                gt_centered = gt_pts - gt_c

                # --- B. Scaling (Root Mean Square distance from center) ---
                # How "spread out" are the points?
                dist_pred = torch.norm(pred_centered, dim=1).mean()
                dist_gt = torch.norm(gt_centered, dim=1).mean()

                # Calculate scale factor
                scale = dist_gt / (dist_pred + 1e-8)

                print(f"[Align] Fixing Scale. GT_spread={dist_gt:.2f}, Pred_spread={dist_pred:.2f}, Scale={scale:.4f}")

                # --- C. Apply Transform ---
                # New_Pos = (Old_Pos - Old_Center) * Scale + New_Center

                # Apply to the full (S, H, W, 3) tensor
                # Broadcast center subtraction
                WPTS_m = (WPTS_m - pred_c.view(1,1,1,3)) * scale + gt_c.view(1,1,1,3)

                # Fix Camera Translation too (approximate)
                if tmw is not None:
                     # This is tricky for tmw alone, but sticking to point alignment is key for IoU
                     pass

            else:
                 print("[Align] Warning: Empty clouds, skipping align.")
            """


        predictions[POINTS] = WPTS_m

        end = time.time()
        length = end - start

        print("Aligning frames took", length, "seconds!")
        start = time.time()



        end = time.time()
        length = end - start

        print("Projecting view feats", length, "seconds!")
        
        start = time.time()

        frames_map, conf_map, images_map, features_map, (S,H,W), frame_ids = filter_frames(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            FEAT="view_feats",
            threshold=threshold,
            z_clip_map=z_clip_map,   # or None
        )  
        
        
        print(f"Features require grad: {features_map[0].requires_grad}")
        
        end = time.time()
        length = end - start

        print("Building frames/camera centers took", length, "seconds!")

        start = time.time()

        align_to_voxel = False #(i > 0)
         

        #features_map = [vf_t[i] for i in range(vf.shape[0])]  # one vector per image

        # features_map = [pointnext_inference(preprocess_points(f,i)) for f, i in zip(frames_map, images_map)]
        with autocast(enabled=False):
            vox, bev, meta = build_maps_from_latent_features(
                i,
                frames_map,
                conf_map,
                features_map,
                self.vox,
                voxel_size=self.voxel_size,           # 10 cm
                bev_window_m=(5.0, 5.0), # local 20x20 m
                bev_origin_xy=(-2.0, -2.0),
                z_clip_vox=(-np.inf, np.inf),
                z_band_bev=(0.02, 0.5),
                frame_ids=frame_ids
            )

        self.vox = vox
    
            
        end = time.time()
        length = end - start

        print("Building Voxel and BEV took", length, "seconds!")

        
            
        self.vox.next_epoch()
        
        # after build_frames_and_centers_vectorized(...)
        del predictions  # drops images, view_feats, etc. all at once

        # after build_maps_from_latent_features(...)
        del frames_map, conf_map, images_map, features_map, image_tensors

        return bev, mst, Rmw, tmw
    
    def inference_gt(self, i, imgs):

        POINTS = "world_points"
        CONF = "world_points_conf"
        threshold = 1.0     
        z_clip_map = (-0.1, 0.3)   

        R_w2m = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)

        t_w2m = np.zeros(3, dtype=np.float32)



        R_w2m = to_torch(R_w2m, device=self.device)
        t_w2m = to_torch(t_w2m, device=self.device)
        
       
        
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


        
        start = time.time()

        predictions = get_reconstructed_scene_no_opt(0, ".", imgs, self.model, self.device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0)
        
        # self.keyframes = image_tensors.clone()
        
        end = time.time()
        length = end - start

        print("Running inference took", length, "seconds!")
        


        # Keep tensors; only extract what we need later.
        # If you truly need NumPy later, convert specific keys then.
        needed = {
            "images","extrinsic", POINTS, CONF
        }
        for k in list(predictions.keys()):
            if k not in needed:
                del predictions[k]  # drop unneeded heavy stuff early



        start = time.time()
        with torch.no_grad():

            # WPTS_m = torch.from_numpy(predictions[POINTS]).to(device=self.device)

            WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
            if Rmw is None and tmw is None:
                print("aligning floor")
                Rmw, tmw, info = align_pointcloud_torch_fast(WPTS_m, inlier_dist=self.voxel_size*0.75)

            WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
            predictions[POINTS] = WPTS_m


            camera_R = R_w2m @ Rmw
            camera_t = t_w2m + tmw
            frames_map, cam_centers_map, conf_map, images_map, _, (S,H,W), frame_ids = build_frames_and_centers_vectorized_torch(
                predictions,
                POINTS=POINTS,
                CONF=CONF,
                # IMG="images",
                threshold=threshold,
                Rmw=camera_R, tmw=camera_t,
                z_clip_map=z_clip_map,   # or None
                return_flat=True

            )   
            end = time.time()
            length = end - start

            print("Aligning and building frames/camera centers took", length, "seconds!")

            start = time.time()

            align_to_voxel = False #(i > 0)
        
            with autocast(enabled=False):

                vox, bev, meta = build_maps_from_points_and_centers_torch(
                    frames_map,
                    cam_centers_map,
                    conf_map,
                    self.vox_gt,
                    align_to_voxel=align_to_voxel,
                    voxel_size=self.voxel_size,           # 10 cm
                    bev_window_m=(5.0, 5.0), # local 20x20 m
                    bev_origin_xy=(-2.0, -2.0),
                    z_clip_vox=(-np.inf, np.inf),
                    z_band_bev=(0.02, 0.5),
                    samples_per_voxel=0.7,#1,
                    ray_stride=6,#2,
                    max_free_rays=10000,
                    frame_ids=frame_ids
                 ) 

            self.vox_gt = vox
        
                
            end = time.time()
            length = end - start

            print("Building Voxel and BEV took", length, "seconds!")

            
                
            self.vox_gt.next_epoch()
            
        
          # after build_frames_and_centers_vectorized(...)
        del predictions  # drops images, view_feats, etc. all at once

        # after build_maps_from_latent_features(...)
        del frames_map, cam_centers_map, conf_map, images_map, image_tensors

        return bev, Rmw, tmw
  

    # def _decode_occupancy_now(self) -> torch.Tensor:
    #     """Decode current grid.latents to occupancy ∈ [0,1] aligned to self.grid.keys."""
    #     if self.grid.keys.numel() == 0:
    #         return torch.empty(0, device=self.device)
    #     centers = self.grid.origin + (self.grid._unhash_keys(self.grid.keys).float() + 0.5) * self.grid.p.voxel_size
    #     return self.decoder(self.grid.z_latent.to(self.device), centers.to(self.device))
    
    # def align_probs_to_keys(self, src_keys: torch.Tensor, src_probs: torch.Tensor,
    #                         dst_keys: torch.Tensor, default: float = 0.5) -> torch.Tensor:
    #     """
    #     Map probs from (src_keys, src_probs) onto dst_keys order.
    #     Any dst_key not found in src-> default.
    #     Assumes both key tensors are 1D torch.int64 and (roughly) sorted.
    #     """
    #     # sort dst once to use searchsorted
    #     dst_sorted, inv = torch.sort(dst_keys)          # inv maps sorted -> original order
    #     # find positions where each dst_sorted would appear in src_keys
    #     src_sorted, _ = torch.sort(src_keys)
    #     pos = torch.searchsorted(src_sorted, dst_sorted)

    #     # build a mask for exact matches
    #     # need values at those positions; do a gather safely
    #     pos_clamped = torch.clamp(pos, max=src_sorted.numel()-1)
    #     match_vals = src_sorted[pos_clamped]
    #     is_match = (pos < src_sorted.numel()) & (match_vals == dst_sorted)

    #     # map dst_sorted matches back to src indices:
    #     # get a dict from key->index for src_keys
    #     # (cheap-ish because it’s sparse and done on GPU)
    #     # Build hash map via sorting once:
    #     _, src_inv = torch.sort(src_keys)
    #     src_keys_sorted = src_keys[src_inv]
    #     where_in_src = torch.searchsorted(src_keys_sorted, dst_sorted[is_match])
    #     src_idx_for_match = src_inv[where_in_src]

    #     out_sorted = torch.full((dst_sorted.numel(),), default,
    #                             device=dst_keys.device, dtype=src_probs.dtype)
    #     out_sorted[is_match] = src_probs[src_idx_for_match]

    #     # return in original dst_keys order
    #     return out_sorted[inv]


    # def align_probs_to_keys(self, src_keys, src_probs, dst_keys, default=0.0):
    #     # handle degenerate case
    #     if src_keys.numel() == 0:
    #         return torch.full(
    #             (dst_keys.numel(),), default,
    #             device=dst_keys.device, dtype=src_probs.dtype
    #         )

    #     # 1) sort src once and carry probs with it
    #     src_sorted, src_perm = torch.sort(src_keys)         # src_sorted[i] = src_keys[src_perm[i]]
    #     src_probs_sorted = src_probs[src_perm]

    #     # 2) sort dst (we’ll align in sorted space, then unsort)
    #     dst_sorted, dst_sort_idx = torch.sort(dst_keys)     # dst_sorted[i] = dst_keys[dst_sort_idx[i]]

    #     # 3) For each dst_sorted value, where would it be inserted in src_sorted?
    #     pos = torch.searchsorted(src_sorted, dst_sorted)
    #     # pos[i] = index where dst_sorted[i] would go in src_sorted to keep sorted order

    #     # 4) Check if we actually have an exact match at that position
    #     pos_clamped = torch.clamp(pos, max=src_sorted.numel() - 1)
    #     match_vals = src_sorted[pos_clamped]
    #     is_match = (pos < src_sorted.numel()) & (match_vals == dst_sorted)

    #     # 5) Fill output in *sorted-dst* space
    #     out_sorted = torch.full(
    #         (dst_sorted.numel(),), default,
    #         device=dst_keys.device, dtype=src_probs.dtype
    #     )
    #     # for matches: use the corresponding src_probs_sorted
    #     out_sorted[is_match] = src_probs_sorted[pos_clamped[is_match]]

    #     # 6) Unscramble back to original dst_keys order
    #     orig_to_sorted = torch.argsort(dst_sort_idx)   # original idx -> sorted idx
    #     return out_sorted[orig_to_sorted]


    def align_probs_to_keys2(self, src_keys, src_probs, dst_keys, default=0.5):
        """
        Align (src_keys, src_probs) to dst_keys.

        Returns:
            out_probs: (len(dst_keys),) aligned probs (default for unknowns)
            valid_mask: bool (len(dst_keys),) True where dst_keys[i] exists in src_keys
        """
        # handle degenerate case: no GT keys
        if src_keys.numel() == 0:
            out = torch.full(
                (dst_keys.numel(),),
                default,
                device=dst_keys.device,
                dtype=src_probs.dtype,
            )
            valid = torch.zeros(dst_keys.numel(), dtype=torch.bool, device=dst_keys.device)
            return out, valid

        # 1) sort src and carry probs
        src_sorted, src_perm = torch.sort(src_keys)
        src_probs_sorted = src_probs[src_perm]

        # 2) sort dst
        dst_sorted, dst_sort_idx = torch.sort(dst_keys)

        # 3) positions where dst_sorted would be inserted in src_sorted
        pos = torch.searchsorted(src_sorted, dst_sorted)

        # 4) determine matches in sorted-dst space
        pos_clamped = torch.clamp(pos, max=src_sorted.numel() - 1)
        match_vals = src_sorted[pos_clamped]
        is_match_sorted = (pos < src_sorted.numel()) & (match_vals == dst_sorted)

        # 5) fill output in sorted-dst space
        out_sorted = torch.full(
            (dst_sorted.numel(),),
            default,
            device=dst_keys.device,
            dtype=src_probs.dtype,
        )
        out_sorted[is_match_sorted] = src_probs_sorted[pos_clamped[is_match_sorted]]

        # 6) map back to original dst order
        orig_to_sorted = torch.argsort(dst_sort_idx)
        out = out_sorted[orig_to_sorted]
        valid_mask = is_match_sorted[orig_to_sorted]

        return out, valid_mask


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

        # 2) sort dst
        dst_sorted, dst_sort_idx = torch.sort(dst_keys)

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
    
    # def training_step(self, batch: Dict, batch_idx: int):
    #     """
    #     One batch = one sequence.
    #     Flow:
    #       t=0: init latents from full cloud
    #       t>0: extract point features, learned update, decode occupancy, compute loss vs teacher
    #     """
    #     cfg = self.cfg
    #     device = self.device
    #     opt = self.optimizers()
        
    #     self.vox.reset_state()
    #     self.vox = self.vox.to(self.device)

        
        
    #     self.vox_gt = TorchSparseVoxelGrid(
    #         origin_xyz=np.zeros(3, dtype=np.float32),
    #         params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
    #         device=self.device 
    #     )
        
        
    #     self._prev_keys = None
    #     self._prev_probs = None


    #     # ---- unpack sequence ----
    #     T = batch["timesteps"]
       
       
    #     loss_total = torch.zeros([], device=device)

    #     # all_imgs = batch["imgs_t"]
    #     # numel = all_imgs.numel()
    #     # bytes_per_element = all_imgs.element_size()
    #     # total = numel * bytes_per_element
    #     # print(f"Tensor: shape={tuple(all_imgs.shape)}, dtype={all_imgs.dtype}")
    #     # print(f"  numel={numel:,}")
    #     # print(f"  element size={bytes_per_element} bytes")
    #     # print(f"  total={total/1024**2:.3f} MB ({total:,} bytes)")
        
    #     # ---- iterate timesteps ----
    #     seq_id = batch["seq_id"]

    #     gt_root = os.path.join(self.cfg.dataset_root, "gt_voxels_per_timestep_01")
    #     gt_seq = []
        
    #     # for t in range(T):
    #     #     gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
    #     #     vox_gt_t = load_sparse_voxel_grid(gt_path, device)
    #     #     gt_seq.append(vox_gt_t)
            
    #     if not os.path.exists(os.path.join(gt_root, f"{seq_id}_t0000_gt.npz")):
    #         return loss_total

    #     for t in range(T):
    #         if self.cfg.skip:
    #             t = t*10
    #         gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
    #         if os.path.exists(gt_path):
    #             vox_gt_t = load_sparse_voxel_grid(gt_path, device)
    #         else:
    #             vox_gt_t = None
    #         gt_seq.append(vox_gt_t)
            

    #     mst = False
    #     Rmw = None
    #     tmw = None
    #     for t in range(T):
            
    #         print(f"=============================timestep {t}=============================")
    #         imgs = batch["imgs_t"][t]          # <--- this is your old `imgs`

      
    #         # bev_gt, R, tw = self.inference_gt(t, imgs)
                
                
    #         # seq_id = batch["seq_id"]
    #         # gt_path = os.path.join(
    #         #     self.cfg.dataset_root,
    #         #     "gt_voxels_per_timestep",
    #         #     f"{seq_id}_t{t:04d}_gt.npz"
    #         # )
    #         # self.vox_gt = load_sparse_voxel_grid(gt_path, self.device)
    #         self.vox_gt = gt_seq[t]

    #         if self.vox_gt is None:
    #             print("No GT voxel grid for this timestep, skipping.")
    #             continue
            
    #         if not mst and t != 0:
    #             bev, mst, _, _ = self.inference(1, imgs, mst, Rmw, tmw)
    #         else:
    #             bev, mst, _, _ = self.inference(t, imgs, mst, Rmw,tmw)
        
    #         with autocast(enabled=False):

    #             # p_occ_tgt = self.vox_gt.vals_st
    #             print("gt voxels: ", (self.vox_gt.vals_st))
    #             logit_gt   = self.vox_gt.vals_st.clamp(-8.0, 8.0)   # optional but recommended
    #             p_occ_tgt  = torch.sigmoid(logit_gt)
    #             #p_occ_tgt = torch.sigmoid(self.vox_gt.vals_st)
                
    #             # (D) decode current occupancy
    #             p_occ_pred_before = self.vox.decode_occupancy()
    #             p_occ_tgt, valid_mask = self.align_probs_to_keys(self.vox_gt.keys, p_occ_tgt,
    #                                 self.vox.keys, default=0.0)      



    #             p_occ_pred = p_occ_pred_before[valid_mask]
    #             p_occ_tgt  = p_occ_tgt[valid_mask]
                
                
                
    #             if p_occ_pred.numel() > 0:
    #                 pred_bin = (p_occ_pred > 0.5)
    #                 tgt_bin  = (p_occ_tgt  > 0.5)

    #                 tp = (pred_bin & tgt_bin).sum()
    #                 fp = (pred_bin & ~tgt_bin).sum()
    #                 fn = (~pred_bin & tgt_bin).sum()

    #                 occ_iou = tp / (tp + fp + fn + 1e-8)
    #                 self.log("metric/occ_iou", occ_iou)

    #                 self.log("debug/frac_pos_gt", tgt_bin.float().mean())
    #                 self.log("debug/frac_pos_pred", pred_bin.float().mean())
    #             else:
    #                 self.log("metric/occ_iou", 0.0)
    #                 self.log("debug/frac_pos_gt", 0.0)
    #                 self.log("debug/frac_pos_pred", 0.0)


    #             # Visualize Overlap
    #             intersection = torch.isin(self.vox.keys, self.vox_gt.keys).sum()
    #             union = len(self.vox.keys) + len(self.vox_gt.keys) - intersection
    #             iou = intersection / (union + 1e-8)
    #             print(f"Voxel IoU: {iou:.4f} | Pred Voxels: {len(self.vox.keys)} | GT Voxels: {len(self.vox_gt.keys)}  | Overlap: {intersection}")


    #             # Debug: Compare Centroids
    #             if self.vox.keys.numel() > 0 and self.vox_gt.keys.numel() > 0:
    #                 # Get world centers of predicted voxels
    #                 pred_centers = self.vox.voxel_centers() 
    #                 # Get world centers of GT voxels
    #                 gt_ijk = self.vox_gt._unhash_keys(self.vox_gt.keys).float()
    #                 gt_centers = self.vox_gt.origin + (gt_ijk + 0.5) * self.vox_gt.p.voxel_size
                    
    #                 print(f"Pred Centroid: {pred_centers.mean(0).detach().cpu().numpy()}")
    #                 print(f"GT   Centroid: {gt_centers.mean(0).detach().cpu().numpy()}")
                    
    #                 # Check if they are close
    #                 dist = torch.norm(pred_centers.mean(0) - gt_centers.mean(0))
    #                 print(f"Centroid Distance: {dist.item()} (should be < voxel_size)")

    #             assert len(torch.unique(self.vox.keys)) == len(self.vox.keys)
    #             assert len(torch.unique(self.vox_gt.keys)) == len(self.vox_gt.keys)


         

    #             if p_occ_pred.numel() == 0:
    #                 # Nothing to supervise this step
    #                 loss_occ = torch.tensor(0.0, device=self.device)
    #             else:
    #                 # ---- compute positive class weight (same ratio as before) ----
    #                 pos_mask = (p_occ_tgt > 0.5)
    #                 num_pos  = pos_mask.sum()
    #                 num_neg  = (~pos_mask).sum()

    #                 if num_pos > 0:
    #                     pos_weight = (num_neg.float() / (num_pos.float() + 1e-8)).to(self.device)
    #                 else:
    #                     pos_weight = torch.tensor(1.0, device=self.device)

    #                 weights = torch.ones_like(p_occ_tgt, device=self.device)
    #                 weights[pos_mask] = pos_weight

    #                 loss_occ = F.binary_cross_entropy(
    #                     p_occ_pred.clamp(1e-5, 1-1e-5),
    #                     p_occ_tgt.clamp(1e-5, 1-1e-5),
    #                     weight=weights,
    #                     reduction="mean"
    #                 )

    #             logit_gt = self.vox_gt.vals_st
    #             p_gt = torch.sigmoid(logit_gt)
    #             frac_gt_all = (p_gt > 0.5).float().mean()
    #             print("GT fraction occupied over all gt voxels:", float(frac_gt_all))

    #             # # p_occ_pred, p_occ_tgt already masked with valid_mask
    #             # if p_occ_pred.numel() == 0:
    #             #     loss_occ = torch.tensor(0.0, device=self.device)
    #             # else:
    #             #     tgt_bin = (p_occ_tgt > 0.5)
    #             #     pos_idx = tgt_bin.nonzero(as_tuple=True)[0]
    #             #     neg_idx = (~tgt_bin).nonzero(as_tuple=True)[0]

    #             #     num_pos = pos_idx.numel()
    #             #     num_neg = neg_idx.numel()

    #             #     if num_pos == 0:
    #             #         # nothing occupied in this step → only learn from negatives, small loss
    #             #         loss_occ = F.binary_cross_entropy(
    #             #             p_occ_pred.clamp(1e-5, 1-1e-5),
    #             #             p_occ_tgt.clamp(1e-5, 1-1e-5),
    #             #             reduction="mean",
    #             #         )
    #             #     else:
    #             #         # keep all positives
    #             #         k = 5  # try k in [3, 10]
    #             #         max_neg = min(num_neg, k * num_pos)

    #             #         if max_neg > 0:
    #             #             perm = torch.randperm(num_neg, device=self.device)
    #             #             neg_idx_sample = neg_idx[perm[:max_neg]]
    #             #             idx = torch.cat([pos_idx, neg_idx_sample], dim=0)
    #             #         else:
    #             #             idx = pos_idx

    #             #         p_sub = p_occ_pred[idx]
    #             #         t_sub = p_occ_tgt[idx]

    #             #         loss_occ = F.binary_cross_entropy(
    #             #             p_sub.clamp(1e-5, 1-1e-5),
    #             #             t_sub.clamp(1e-5, 1-1e-5),
    #             #             reduction="mean",
    #             #         )

    #             # Temporal smoothness on logits (optional, encourages stability but not over-smoothing)
    #             # keep a buffer of previous decoded occupancy
                
    #             # if t == 1:
    #             #     self._prev_p_occ = p_occ_pred.detach()
    #             # logit_now  = torch.logit(p_occ_pred.clamp(1e-5, 1-1e-5))
    #             # logit_prev = torch.logit(self._prev_p_occ.clamp(1e-5, 1-1e-5)).to(device)
    #             # loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)
    #             # self._prev_p_occ = p_occ_pred.detach()
                
                
    #             if (t == 0) or (self._prev_keys is None):
    #                 loss_temp = torch.tensor(0.0, device=self.device)
    #             else:
    #                 prev_aligned, valid_mask = self.align_probs_to_keys(self._prev_keys, self._prev_probs,
    #                                                 self.vox.keys, default=0.0)
                    
                    
    #                 logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
    #                 logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                    
    #                 logit_now  = logit_now[valid_mask]
    #                 logit_prev = logit_prev[valid_mask]
                    
                    
    #                 if logit_now.numel() == 0:
    #                     loss_temp = torch.tensor(0.0, device=self.device)
    #                 else:
    #                     loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

    #             # update buffers for next step
    #             self._prev_keys  = self.vox.keys.detach().clone()
    #             self._prev_probs = p_occ_pred_before.detach().clone()


            
    #         # Entropy regularizer on routing (OPTIONAL):
    #         # add a small penalty you compute inside update_with_features_learned (return avg entropy)
    #         # For simplicity, assume you store last entropy in self.grid._last_entropy
    #         loss_ent = torch.tensor(0.0, device=device)
    #         if hasattr(self.vox, "_last_entropy") and self.vox._last_entropy is not None:
    #             loss_ent = self.vox._last_entropy

    #         # TV regularizer on occupancy map (soft smoothness)
    #         # You can build a 3D TV using neighbor shifts (careful—sparse). Simple proxy:
    #         loss_tv = torch.tensor(0.0, device=device)

    #         loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp \
    #                  + cfg.lambda_ent * loss_ent + cfg.lambda_tv * loss_tv



            
    #         self.manual_backward(loss_t)

       
    #         loss_total = loss_total + loss_t.detach()
            
    #         print(loss_t.detach())
    #         print(loss_total)

    #         # # logging
    #         # self.log_dict({
    #         #     "loss/occ": loss_occ,
    #         #     "loss/temp": loss_temp,
    #         #     "loss/ent": loss_ent,
    #         #     "loss/tv": loss_tv,
    #         #     "stats/num_voxels": float(self.vox.keys.numel())
    #         # }, prog_bar=(t == T-1), on_step=True, on_epoch=True, sync_dist=False)
    #         if t == T - 1:
    #             self.log_dict(
    #                 {
    #                     "loss/occ": loss_occ,
    #                     "loss/temp": loss_temp,
    #                     "loss/ent": loss_ent,
    #                     "loss/tv": loss_tv,
    #                     "stats/num_voxels": float(self.vox.keys.numel()),
    #                 },
    #                 prog_bar=True,
    #                 on_step=True,
    #                 on_epoch=True,
    #                 sync_dist=False,
    #             )
            

    #         self.vox.z_latent = self.vox.z_latent.detach()


    #         torch.cuda.empty_cache()
        
    #     # add:
    #     grad_norm = self.compute_grad_norm()
    #     self.log("grad_norm", grad_norm, prog_bar=True, on_step=True, on_epoch=False)

    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)            
    #     # then:
    #     opt.step()            
    #     # --- per-timestep backward + step ---
    #     opt.zero_grad(set_to_none=True)
            


    #     # Add on_epoch=True to smooth the loss curve and save based on the average
    #     self.log("loss/total", loss_total, prog_bar=True, on_epoch=True)        
        
      

    #     return loss_total
    

    
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
        
        N_ACCUM = 4
        
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
            "occ_iou": []
        }

        # ---- iterate timesteps ----
        seq_id = batch["seq_id"]
        gt_root = os.path.join(self.cfg.dataset_root, "gt_voxels_per_timestep_01")
        
        # Preload GT (optional optimization you had)
        gt_seq = []
        if not os.path.exists(os.path.join(gt_root, f"{seq_id}_t0000_gt.npz")):
            return loss_total_seq

        for t in range(T):
            if self.cfg.skip:
                t = t*10 # Adjust indexing if skipping
            
            gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
            if os.path.exists(gt_path):
                vox_gt_t = load_sparse_voxel_grid(gt_path, device)
            else:
                vox_gt_t = None
            gt_seq.append(vox_gt_t)

        mst = False
        Rmw = None
        tmw = None

        for t in range(T):
            imgs = batch["imgs_t"][t]

            self.vox_gt = gt_seq[t]
            if self.vox_gt is None:
                continue
            
            if not mst and t != 0:
                bev, mst, _, _ = self.inference(1, imgs, mst, Rmw, tmw)
            else:
                bev, mst, _, _ = self.inference(t, imgs, mst, Rmw, tmw)
        
            #with autocast(enabled=False):
            # (D) decode current occupancy
            logit_gt   = self.vox_gt.vals_st.clamp(-8.0, 8.0)
            # p_occ_tgt  = torch.sigmoid(logit_gt)
            p_occ_tgt = torch.sigmoid(logit_gt * 10.0)
            p_occ_pred_before = self.vox.decode_occupancy()
            
            # -------------------------------------------------------------
            # DUAL LOSS LOGIC START
            # -------------------------------------------------------------
            
            # 1. Align GT to Prediction Keys
            # valid_mask is TRUE where prediction keys exist in GT
            p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys(
                self.vox_gt.keys, p_occ_tgt, self.vox.keys, default=0.0
            )      

            # ---------------------------------------------------------
            # PART A: Loss on Intersection (Pred & GT)
            # ---------------------------------------------------------
            pred_intersect = p_occ_pred_before[valid_mask]
            tgt_intersect  = p_occ_tgt_aligned[valid_mask]
            
            loss_intersect = torch.tensor(0.0, device=self.device)
            
            if pred_intersect.numel() > 0:
                # Calculate weight for positives just like before
                pos_mask = (tgt_intersect > 0.5)
                num_pos = pos_mask.sum()
                num_neg = (~pos_mask).sum()
                
                if num_pos > 0:
                    pos_weight = (num_neg.float() / (num_pos.float() + 1e-8)).to(self.device)
                else:
                    pos_weight = torch.tensor(1.0, device=self.device)

                weights = torch.ones_like(tgt_intersect, device=self.device)
                weights[pos_mask] = pos_weight

                loss_intersect = F.binary_cross_entropy(
                    pred_intersect.clamp(1e-5, 1-1e-5),
                    tgt_intersect.clamp(1e-5, 1-1e-5),
                    weight=weights,
                    reduction="mean"
                )

            # ---------------------------------------------------------
            # PART B: Loss on False Positives (Pred - GT)
            # ---------------------------------------------------------
            # These are voxels in your prediction that DO NOT exist in GT.
            # Since GT is truth, these must be empty (0.0).
            pred_fp = p_occ_pred_before[~valid_mask]
            loss_fp = torch.tensor(0.0, device=self.device)

            if pred_fp.numel() > 0:
                # Target is all zeros
                tgt_fp = torch.zeros_like(pred_fp)
                
                # Weighting: You might want to weigh this less than intersection
                # but here we start with 1.0 (strict precision).
                loss_fp = F.binary_cross_entropy(
                    pred_fp.clamp(1e-5, 1-1e-5),
                    tgt_fp, 
                    reduction="mean"
                )

            # ---------------------------------------------------------
            # TOTAL OCCUPANCY LOSS & IoU
            # ---------------------------------------------------------
            fp_weight = 0.1
            loss_occ = loss_intersect + (fp_weight * loss_fp)

            # --- Metrics: Global IoU (Including FP Hallucinations) ---
            # Valid/Intersect Part
            pred_bin_int = (pred_intersect > 0.5)
            tgt_bin_int  = (tgt_intersect  > 0.5)
            
            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()
            
            # Hallucination Part (Preds outside GT are all FPs if > 0.5)
            fp_hallucination = (pred_fp > 0.5).sum()
            
            total_fp = fp_int + fp_hallucination
            
            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            metrics_buffer["occ_iou"].append(occ_iou.item())

            # -------------------------------------------------------------
            # DUAL LOSS LOGIC END
            # -------------------------------------------------------------

            # --- Loss: Temporal ---
            if (t == 0) or (self._prev_keys is None):
                loss_temp = torch.tensor(0.0, device=self.device)
            else:
                prev_aligned, valid_mask_temp = self.align_probs_to_keys(
                    self._prev_keys, self._prev_probs, self.vox.keys, default=0.0
                )
                
                logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
                logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                
                logit_now  = logit_now[valid_mask_temp]
                logit_prev = logit_prev[valid_mask_temp]
                
                if logit_now.numel() == 0:
                    loss_temp = torch.tensor(0.0, device=self.device)
                else:
                    loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

            # update buffers for next step
            self._prev_keys  = self.vox.keys.detach().clone()
            self._prev_probs = p_occ_pred_before.detach().clone()

            # --- Loss: Others (Entropy / TV) ---
            loss_ent = torch.tensor(0.0, device=device)
            if hasattr(self.vox, "_last_entropy") and self.vox._last_entropy is not None:
                loss_ent = self.vox._last_entropy
            
            loss_tv = torch.tensor(0.0, device=device)

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
            torch.cuda.empty_cache()
        
        # ---- End of Sequence Loop ----

        # # 1. Clip Gradients
        grad_norm = self.compute_grad_norm()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)            
        
        # # 2. Optimizer Step
        # opt.step()            
        # opt.zero_grad(set_to_none=True)
        
        
        if (batch_idx + 1) % N_ACCUM == 0:
            
            # 2. Step Optimizer
            opt.step()            
            
            # 3. Zero Gradients (Clear buffer for next accumulation cycle)
            opt.zero_grad(set_to_none=True)
            
            # Optional: Log grad_norm only when we actually step
            self.log("grad_norm", grad_norm, prog_bar=True, on_step=True, on_epoch=False)

        # 3. Aggregate Metrics (Mean over sequence)
        def get_avg(name):
            vals = metrics_buffer[name]
            return sum(vals) / len(vals) if len(vals) > 0 else 0.0

        avg_loss_total = loss_total_seq# / max(T, 1)

        # 4. Log Averaged Metrics
        self.log_dict({
            "loss/occ": get_avg("loss_occ"),
            "loss/temp": get_avg("loss_temp"),
            "loss/ent": get_avg("loss_ent"),
            "loss/tv": get_avg("loss_tv"),
            "loss/total_avg": avg_loss_total,
            "metric/occ_iou": get_avg("occ_iou"),
            "stats/num_voxels": float(self.vox.keys.numel()),
            "grad_norm": grad_norm
        }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)

        return loss_total_seq

    # -----------------------------------------------------------
    # ===> PASTE THIS FUNCTION HERE inside VoxelUpdaterSystem <===
    # -----------------------------------------------------------
    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        # Check if the checkpoint contains voxel keys
        if "vox.keys" in state_dict:
            saved_keys = state_dict["vox.keys"]
            target_size = saved_keys.shape[0]
            current_size = self.vox.keys.shape[0]

            if target_size != current_size:
                print(f"[Checkpoint Load] Resizing voxel grid buffers from {current_size} to {target_size}...")

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
    
    
    # def validation_step(self, batch: Dict, batch_idx: int):
    #     device = self.device

    #     cfg = self.cfg

    #     self.vox.reset_state()
    #     self.vox = self.vox.to(self.device)

    #     self.vox_gt = TorchSparseVoxelGrid(
    #         origin_xyz=np.zeros(3, dtype=np.float32),
    #         params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
    #         device=self.device 
    #     )
        
    #     # no optimizer / manual_backward here
    #     T = batch["timesteps"]

    #     val_loss_total = torch.zeros([], device=device)


    #     seq_id = batch["seq_id"]

    #     gt_root = os.path.join(self.cfg.dataset_root, "gt_voxels_per_timestep_01")
    #     gt_seq = []
    #     for t in range(T):
    #         if self.cfg.skip:
    #             t = t*10
    #         gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
    #         vox_gt_t = load_sparse_voxel_grid(gt_path, device)
    #         gt_seq.append(vox_gt_t)
            
    #     mst = False

    #     Rmw = None
    #     tmw = None
    #     for t in range(T):
            
    #         print(f"=============================timestep {t}=============================")

    #         imgs = batch["imgs_t"][t]
        
        
        
    #         # seq_id = batch["seq_id"]
    #         # gt_path = os.path.join(
    #         #     self.cfg.dataset_root,
    #         #     "gt_voxels_per_timestep",
    #         #     f"{seq_id}_t{t:04d}_gt.npz"
    #         # )
    #         # self.vox_gt = load_sparse_voxel_grid(gt_path, self.device)
    #         self.vox_gt = gt_seq[t]

        
    #         with torch.enable_grad():

    #             # bev_gt, R, tw = self.inference_gt(t, imgs)
                
    #             #bev = self.inference(t, imgs)
    #             if not mst and t != 0:
    #                 bev, mst, _, _  = self.inference(1, imgs, mst, Rmw, tmw)
    #             else:
    #                 bev, mst, _, _  = self.inference(t, imgs, mst, Rmw, tmw)
        
        


    #         with autocast(enabled=False):
    #             # p_occ_tgt = self.vox_gt.vals_st
                
    #             p_occ_tgt = torch.sigmoid(self.vox_gt.vals_st)


    #             p_occ_pred_before = self.vox.decode_occupancy()
    #             p_occ_tgt, valid_mask = self.align_probs_to_keys(self.vox_gt.keys, p_occ_tgt,
    #                                 self.vox.keys, default=0.0)      

    #             p_occ_pred = p_occ_pred_before[valid_mask]
    #             p_occ_tgt  = p_occ_tgt[valid_mask]
                
                


    #             # Visualize Overlap
    #             intersection = torch.isin(self.vox.keys, self.vox_gt.keys).sum()
    #             union = len(self.vox.keys) + len(self.vox_gt.keys) - intersection
    #             iou = intersection / (union + 1e-8)
    #             print(f"Voxel IoU: {iou:.4f} | Pred Voxels: {len(self.vox.keys)} | GT Voxels: {len(self.vox_gt.keys)}  | Overlap: {intersection}")

    #                 # Inside training_step, before intersection calculation

    #             # Debug: Compare Centroids
    #             if self.vox.keys.numel() > 0 and self.vox_gt.keys.numel() > 0:
    #                 # Get world centers of predicted voxels
    #                 pred_centers = self.vox.voxel_centers() 
    #                 # Get world centers of GT voxels
    #                 gt_ijk = self.vox_gt._unhash_keys(self.vox_gt.keys).float()
    #                 gt_centers = self.vox_gt.origin + (gt_ijk + 0.5) * self.vox_gt.p.voxel_size
                    
    #                 print(f"Pred Centroid: {pred_centers.mean(0).detach().cpu().numpy()}")
    #                 print(f"GT   Centroid: {gt_centers.mean(0).detach().cpu().numpy()}")
                    
    #                 # Check if they are close
    #                 dist = torch.norm(pred_centers.mean(0) - gt_centers.mean(0))
    #                 print(f"Centroid Distance: {dist.item()} (should be < voxel_size)")

    #             assert len(torch.unique(self.vox.keys)) == len(self.vox.keys)
    #             assert len(torch.unique(self.vox_gt.keys)) == len(self.vox_gt.keys)




    #             if p_occ_pred.numel() == 0:
    #                 # Nothing to supervise this step
    #                 loss_occ = torch.tensor(0.0, device=self.device)
    #             else:

    #                 loss_occ = F.binary_cross_entropy(
    #                     p_occ_pred.clamp(1e-5, 1 - 1e-5),
    #                     p_occ_tgt.clamp(1e-5, 1 - 1e-5),
    #                 )
            
            
    #             if (t == 0) or (self._prev_keys is None):
    #                 loss_temp = torch.tensor(0.0, device=self.device)
    #             else:
    #                 # prev_aligned = self.align_probs_to_keys(self._prev_keys, self._prev_probs,
    #                 #                             self.vox.keys, default=0.0)
    #                 # logit_now  = torch.logit(p_occ_pred.clamp(1e-5, 1-1e-5))
    #                 # logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                    
                    
    #                 prev_aligned, valid_mask = self.align_probs_to_keys(self._prev_keys, self._prev_probs,
    #                                                 self.vox.keys, default=0.0)
                    
                    
    #                 logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
    #                 logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                    
    #                 logit_now  = logit_now[valid_mask]
    #                 logit_prev = logit_prev[valid_mask]
                    
                    
    #                 if logit_now.numel() == 0:
    #                     loss_temp = torch.tensor(0.0, device=self.device)
    #                 else:
    #                     loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

    #             # update buffers for next step
    #             self._prev_keys  = self.vox.keys.detach().clone()
    #             self._prev_probs = p_occ_pred_before.detach().clone()
            
            
    #         loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp \

    #         self.vox.z_latent = self.vox.z_latent.detach()

    #         val_loss_total += loss_t.detach()
            
            
    #         torch.cuda.empty_cache()


    #     self.log("val_loss_total", val_loss_total, prog_bar=True, on_epoch=True)
    #     return val_loss_total

    
    
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
            "occ_iou": []
        }

        seq_id = batch["seq_id"]
        gt_root = os.path.join(self.cfg.dataset_root, "gt_voxels_per_timestep_01")
        
        # Preload GT
        gt_seq = []
        for t in range(T):
            if self.cfg.skip:
                t = t * 10
            gt_path = os.path.join(gt_root, f"{seq_id}_t{t:04d}_gt.npz")
            if os.path.exists(gt_path):
                vox_gt_t = load_sparse_voxel_grid(gt_path, device)
            else:
                vox_gt_t = None
            gt_seq.append(vox_gt_t)
            
        mst = False
        Rmw = None
        tmw = None

        for t in range(T):
            # print(f"== Val Step {t} ==")
            imgs = batch["imgs_t"][t]
            
            self.vox_gt = gt_seq[t]
            if self.vox_gt is None:
                continue

            with torch.enable_grad(): # (Keep grad enabled for inference/update parts if needed by model)
                if not mst and t != 0:
                    bev, mst, _, _  = self.inference(1, imgs, mst, Rmw, tmw)
                else:
                    bev, mst, _, _  = self.inference(t, imgs, mst, Rmw, tmw)

            # Validation Loss Calculation (No Autocast needed strictly, but good for consistency)
            # with autocast(enabled=False):
            # p_occ_tgt = torch.sigmoid(self.vox_gt.vals_st)
            p_occ_tgt = torch.sigmoid(self.vox_gt.vals_st * 10.0)

            p_occ_pred_before = self.vox.decode_occupancy()

            # 1. Align GT to Prediction
            p_occ_tgt_aligned, valid_mask = self.align_probs_to_keys(
                self.vox_gt.keys, p_occ_tgt, self.vox.keys, default=0.0
            )      

            # ---------------------------------------------------------
            # PART A: Intersection Loss
            # ---------------------------------------------------------
            pred_intersect = p_occ_pred_before[valid_mask]
            tgt_intersect  = p_occ_tgt_aligned[valid_mask]
            
            loss_intersect = torch.tensor(0.0, device=self.device)

            if pred_intersect.numel() > 0:
                # Positive Weighting
                pos_mask = (tgt_intersect > 0.5)
                num_pos = pos_mask.sum()
                num_neg = (~pos_mask).sum()
                
                if num_pos > 0:
                    pos_weight = (num_neg.float() / (num_pos.float() + 1e-8))
                else:
                    pos_weight = torch.tensor(1.0, device=self.device)

                weights = torch.ones_like(tgt_intersect)
                weights[pos_mask] = pos_weight

                loss_intersect = F.binary_cross_entropy(
                    pred_intersect.clamp(1e-5, 1-1e-5),
                    tgt_intersect.clamp(1e-5, 1-1e-5),
                    weight=weights
                )

            # ---------------------------------------------------------
            # PART B: False Positive Loss (Hallucinations)
            # ---------------------------------------------------------
            pred_fp = p_occ_pred_before[~valid_mask]
            loss_fp = torch.tensor(0.0, device=self.device)

            if pred_fp.numel() > 0:
                tgt_fp = torch.zeros_like(pred_fp)
                loss_fp = F.binary_cross_entropy(
                    pred_fp.clamp(1e-5, 1-1e-5),
                    tgt_fp
                )

            # Total Val Loss
            fp_weight = 1.0
            loss_occ = loss_intersect + (fp_weight * loss_fp)

            # ---------------------------------------------------------
            # METRICS: Global IoU (Including Hallucinations)
            # ---------------------------------------------------------
            # Intersect part
            pred_bin_int = (pred_intersect > 0.5)
            tgt_bin_int  = (tgt_intersect  > 0.5)
            
            tp = (pred_bin_int & tgt_bin_int).sum()
            fp_int = (pred_bin_int & ~tgt_bin_int).sum()
            fn = (~pred_bin_int & tgt_bin_int).sum()
            
            # Hallucination part
            fp_hallucination = (pred_fp > 0.5).sum()
            
            total_fp = fp_int + fp_hallucination
            occ_iou = tp / (tp + total_fp + fn + 1e-8)
            
            metrics_buffer["occ_iou"].append(occ_iou.item())

            # --- Temporal Loss (Optional for Val, but good to track) ---
            if (t == 0) or (self._prev_keys is None):
                loss_temp = torch.tensor(0.0, device=self.device)
            else:
                prev_aligned, valid_mask_temp = self.align_probs_to_keys(
                    self._prev_keys, self._prev_probs, self.vox.keys, default=0.0
                )
                logit_now  = torch.logit(p_occ_pred_before.clamp(1e-5, 1-1e-5))
                logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                logit_now  = logit_now[valid_mask_temp]
                logit_prev = logit_prev[valid_mask_temp]

                if logit_now.numel() == 0:
                    loss_temp = torch.tensor(0.0, device=self.device)
                else:
                    loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

            self._prev_keys  = self.vox.keys.detach().clone()
            self._prev_probs = p_occ_pred_before.detach().clone()
        
            loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp
            val_loss_total_seq += loss_t.detach()
            
            metrics_buffer["loss_occ"].append(loss_occ.item())
            metrics_buffer["loss_temp"].append(loss_temp.item())
        
            self.vox.z_latent = self.vox.z_latent.detach()
            torch.cuda.empty_cache()

        # Average over sequence
        val_loss_total = val_loss_total_seq #/ max(T, 1)
        avg_iou = sum(metrics_buffer["occ_iou"]) / len(metrics_buffer["occ_iou"]) if metrics_buffer["occ_iou"] else 0.0

        self.log("val_loss_total", val_loss_total, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_metric/occ_iou", avg_iou, prog_bar=True, on_epoch=True, sync_dist=False)
        
        return val_loss_total
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
        seq_list: str = "/cluster/scratch/kochmar/renders/seq_manifest.json"

    ):
        self.root = dataset_root
        self.size = size
        self.verbose = verbose
        self.min_images_per_timestep = min_images_per_timestep
        self.skip = skip
        self.seq_list = seq_list
        
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
        t_dirs = _list_dirs(seq_dir)
        return t_dirs if t_dirs else [seq_dir]

    def _load_timestep(self, t_dir: str) -> List[Dict]:
        img_paths = _list_imgs(t_dir)
        if len(img_paths) < self.min_images_per_timestep:
            return []
        return li(img_paths, size=self.size, verbose=self.verbose)

    def __getitem__(self, idx: int) -> Dict:
        seq_dir = self.seq_paths[idx]
        t_dirs = self._list_timesteps(seq_dir)

        imgs_t: List[List[Dict]] = []
        for t, td in enumerate(t_dirs):
            if t % 10 != 0 and self.skip:
                continue
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
        seq_list: str = "/cluster/scratch/kochmar/renders/seq_manifest.json"
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
    def setup(self, stage: Optional[str] = None):
        print("getting seqs")
        # all_seqs = _sequence_dirs_from_root(self.dataset_root)
        
        
        # gt_root = os.path.join(self.dataset_root, "gt_voxels_per_timestep")
        # if self.skip:
        #     filtered = []
        #     for seq_dir in all_seqs:
        #         # reconstruct seq_id exactly like __getitem__
        #         p = seq_dir.rstrip("/")
        #         basis = os.path.basename(os.path.dirname(p)).replace(".basis", "")
        #         final = os.path.basename(p)
        #         seq_id = f"{basis}_{final}"

        #         # we just check for t=0 GT; adjust if you need stricter checks
        #         gt_path_t0 = os.path.join(gt_root, f"{seq_id}_t0000_gt.npz")
        #         if os.path.exists(gt_path_t0):
        #             filtered.append(seq_dir)
        #     all_seqs = filtered
            
        with open(self.seq_list) as f:
            all_entries = json.load(f)

        if self.skip:
            all_seqs = [e["seq_path"] for e in all_entries if e["has_gt"]]
            all_ids  = [e["seq_id"]  for e in all_entries if e["has_gt"]]
        else:
            all_seqs = [e["seq_path"] for e in all_entries]
            all_ids  = [e["seq_id"]  for e in all_entries]

        print("got seqs")
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
            skip=self.skip
        )
        
        self.val_set = HabitatSeqDataset(
            dataset_root=self.dataset_root,
            size=self.size,
            verbose=self.verbose,
            sequences=val_seqs,
            skip=self.skip

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
        dataset_root="/cluster/scratch/kochmar/renders/",
        voxel_size=0.1,
        radius_m=0.25,
        topk=8,
        temp=0.5,
        feature_dim=64,
        occ_decoder_hidden=64,
        lr=1e-3,
        max_epochs=20,
        batch_size=1,
        num_workers=4,
        precision="bf16",
        skip=True,
    )

    dm = HabitatDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size=512,
        verbose=False,
        train_val_split=0.05,  # or whatever you want
        skip=True
    )

    sys = VoxelUpdaterSystem(cfg)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath="/cluster/scratch/kochmar/checkpoints/",       # Explicitly set a folder so you can find them
        monitor="val_loss_total",
        save_top_k=3,
        mode="min",
        filename="voxup-{epoch:02d}-{val_loss_total:.4f}" # Match the key logged in validation_step
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")

 
    print(">>> before Trainer()", flush=True)


    # --- Wandb logger ---
    wandb_logger = WandbLogger(
        project="voxel_dust3r",          # choose a project name
        name="voxup-manualopt",          # optional run name
        config=cfg.__dict__,             # logs all your hyperparams
        save_dir="./wandb_logs",         # where to put local files
    )

 
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        #gradient_clip_val=1.0,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        callbacks=[ckpt_cb, lr_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger,
    )
    print(">>> before trainer.fit()", flush=True)
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/voxup-epoch=09-val_loss_total=19.5780.ckpt"

    trainer.fit(sys, dm)
if __name__ == "__main__":
    main()
