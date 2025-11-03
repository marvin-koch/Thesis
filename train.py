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

import pow3r.tools.path_to_dust3r
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
        # ---- core components (replace with your actual imports) ----
        self.voxel_size = 0.01
        self.vox = LatentVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device, dtype=torch.float32
        )
        
        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device, dtype=torch.float32
        )
        
        weights_path = "naver/" + "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)

        self.model.eval()
        self.model = self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.keyframes = []
        # convenience buffer for device transfers
        self.register_buffer("_origin", torch.zeros(3), persistent=False)

    def configure_optimizers(self):
        params = list(self.vox.sim_net.parameters()) + \
                 list(self.vox.gru_cell.parameters()) + \
                 list(self.vox.decoder.parameters())
        # if your feature extractor is finetuned, extend params with extractor params
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return opt
    
    def inference(self, i, imgs):


        POINTS = "world_points"
        CONF = "world_points_conf"
        threshold = 1.0     
        z_clip_map = (-0.1, 0.3)   

        R_w2m = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)

        t_w2m = np.zeros(3, dtype=np.float32)

        R_w2m = to_torch(R_w2m, device=device)
        t_w2m = to_torch(t_w2m, device=device)
        
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


        if i < 1:
            
            start = time.time()

            predictions = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0)
            
            # predictions = _profile_block(
            #     f"inference_t{i}_full",
            #     get_reconstructed_scene,
            #     i,                 # itr
            #     ".",               # outdir
            #     imgs,              # imgs
            #     self.model,        # model
            #     self.device,       # device
            #     False,             # silent
            #     512,               # image_size
            #     "",                # filelist
            #     "linear",          # schedule
            #     100,               # niter
            #     1,                 # min_conf_thr
            #     True,              # as_pointcloud
            #     False,             # mask_sky
            #     True,              # clean_depth
            #     False,             # transparent_cams
            #     0.05,              # cam_size
            #     "oneref",          # scenegraph_type
            #     1,                 # winsize
            #     0                  # refid
            #     # changed_gids=None (implicit)
            #     # tau=0.45 (default)
            # )

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
                    return None
 
            print("Finding changed images took", length, "seconds!")

            changed_idx = [0] + [x for x in changed_idx if x != 0]
            
            index_map = {new: old for new, old in enumerate(changed_idx)}
                    
            idx_t = torch.tensor(changed_idx, device=self.device, dtype=torch.long)
            self.keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
            
            
            print("final changed idx:", changed_idx)

            start = time.time()

            # stacked_predictions = []
            # for input_frames in [imgs]:
            
            predictions = get_reconstructed_scene_no_opt(i, ".", imgs, self.model, self.device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx)
            
            
            # predictions = _profile_block(
            #     f"inference_t{i}_changed",
            #     get_reconstructed_scenet,
            #     i,                 # itr
            #     ".",               # outdir
            #     imgs,              # imgs
            #     self.model,        # model
            #     self.device,       # device
            #     False,             # silent
            #     512,               # image_size
            #     "",                # filelist
            #     "linear",          # schedule
            #     100,               # niter
            #     1,                 # min_conf_thr
            #     True,              # as_pointcloud
            #     False,             # mask_sky
            #     True,              # clean_depth
            #     False,             # transparent_cams
            #     0.05,              # cam_size
            #     "oneref",          # scenegraph_type
            #     1,                 # winsize
            #     0,                 # refid
            #     changed_idx        # changed_gids
            #     # tau=0.45 (default)
            # )

            # predictions = run_model(model, vggt_input, attn_mask=adj)s
                
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
        Rmw, tmw, info = align_pointcloud(WPTS_m, inlier_dist=self.voxel_size*0.75)
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
        predictions[POINTS] = WPTS_m


        camera_R = R_w2m @ Rmw
        camera_t = t_w2m + tmw
        frames_map, cam_centers_map, conf_map, images_map, features_map, (S,H,W) = build_frames_and_centers_vectorized(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            IMG="images",
            FEAT="view_feats",
            threshold=threshold,
            Rmw=camera_R, tmw=camera_t,
            z_clip_map=z_clip_map,   # or None
        )   
        end = time.time()
        length = end - start

        print("Aligning and building frames/camera centers took", length, "seconds!")

        start = time.time()

        align_to_voxel = False #(i > 0)
         
        
        print(frames_map[0].shape)
        print(features_map[0].shape)

        #features_map = [vf_t[i] for i in range(vf.shape[0])]  # one vector per image

        # features_map = [pointnext_inference(preprocess_points(f,i)) for f, i in zip(frames_map, images_map)]
        
        vox, bev, meta = build_maps_from_latent_features(
            i,
            frames_map,
            cam_centers_map,
            conf_map,
            features_map,
            self.vox,
            align_to_voxel=align_to_voxel,
            voxel_size=self.voxel_size,           # 10 cm
            bev_window_m=(5.0, 5.0), # local 20x20 m
            bev_origin_xy=(-2.0, -2.0),
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(0.02, 0.5),
        )

        self.vox = vox
    
            
        end = time.time()
        length = end - start

        print("Building Voxel and BEV took", length, "seconds!")

        
            
        self.vox.next_epoch()

        return bev
    
    def inference_gt(self, i, imgs):

        POINTS = "world_points"
        CONF = "world_points_conf"
        threshold = 1.0     
        z_clip_map = (-0.1, 0.3)   

        R_w2m = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=np.float32)

        t_w2m = np.zeros(3, dtype=np.float32)



        R_w2m = to_torch(R_w2m, device=device)
        t_w2m = to_torch(t_w2m, device=device)
        
       
        
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

        predictions = get_reconstructed_scene_no_opt(0, ".", imgs, self.model, self.device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0)
        
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
        
        WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
        Rmw, tmw, info = align_pointcloud(WPTS_m, inlier_dist=self.voxel_size*0.75)
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
        predictions[POINTS] = WPTS_m


        camera_R = R_w2m @ Rmw
        camera_t = t_w2m + tmw
        frames_map, cam_centers_map, conf_map, (S,H,W), frame_ids = build_frames_and_centers_vectorized_torch(
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

        return bev
  

    # def _decode_occupancy_now(self) -> torch.Tensor:
    #     """Decode current grid.latents to occupancy ∈ [0,1] aligned to self.grid.keys."""
    #     if self.grid.keys.numel() == 0:
    #         return torch.empty(0, device=self.device)
    #     centers = self.grid.origin + (self.grid._unhash_keys(self.grid.keys).float() + 0.5) * self.grid.p.voxel_size
    #     return self.decoder(self.grid.z_latent.to(self.device), centers.to(self.device))
    
    def align_probs_to_keys(self, src_keys: torch.Tensor, src_probs: torch.Tensor,
                            dst_keys: torch.Tensor, default: float = 0.5) -> torch.Tensor:
        """
        Map probs from (src_keys, src_probs) onto dst_keys order.
        Any dst_key not found in src-> default.
        Assumes both key tensors are 1D torch.int64 and (roughly) sorted.
        """
        # sort dst once to use searchsorted
        dst_sorted, inv = torch.sort(dst_keys)          # inv maps sorted -> original order
        # find positions where each dst_sorted would appear in src_keys
        src_sorted, _ = torch.sort(src_keys)
        pos = torch.searchsorted(src_sorted, dst_sorted)

        # build a mask for exact matches
        # need values at those positions; do a gather safely
        pos_clamped = torch.clamp(pos, max=src_sorted.numel()-1)
        match_vals = src_sorted[pos_clamped]
        is_match = (pos < src_sorted.numel()) & (match_vals == dst_sorted)

        # map dst_sorted matches back to src indices:
        # get a dict from key->index for src_keys
        # (cheap-ish because it’s sparse and done on GPU)
        # Build hash map via sorting once:
        _, src_inv = torch.sort(src_keys)
        src_keys_sorted = src_keys[src_inv]
        where_in_src = torch.searchsorted(src_keys_sorted, dst_sorted[is_match])
        src_idx_for_match = src_inv[where_in_src]

        out_sorted = torch.full((dst_sorted.numel(),), default,
                                device=dst_keys.device, dtype=src_probs.dtype)
        out_sorted[is_match] = src_probs[src_idx_for_match]

        # return in original dst_keys order
        return out_sorted[inv]

    def training_step(self, batch: Dict, batch_idx: int):
        """
        One batch = one sequence.
        Flow:
          t=0: init latents from full cloud
          t>0: extract point features, learned update, decode occupancy, compute loss vs teacher
        """
        cfg = self.cfg
        device = self.device
        self.vox = LatentVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device, dtype=torch.float32
        )
        self.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=self.voxel_size, promote_hits=2),
            device=self.device, dtype=torch.float32
        )

        # ---- unpack sequence ----
        T = batch["timesteps"]
       
       
        loss_total = torch.zeros([], device=device)

        # ---- iterate timesteps ----
        for t in range(T):
            imgs = batch["imgs_t"][t]          # <--- this is your old `imgs`

            bev_gt = self.inference_gt(t, imgs)
            bev = self.inference(t, imgs)

        

            p_occ_tgt = self.vox_gt.vals_st
            # (D) decode current occupancy
            p_occ_pred = self.vox.decode_occupancy()
            p_occ_tgt = self.align_probs_to_keys(self.vox_gt.keys, p_occ_tgt,
                                self.vox.keys, default=0.5)      


            # (F) losses
            # Occupancy BCE
            loss_occ = F.binary_cross_entropy(
                p_occ_pred.clamp(1e-5, 1-1e-5),
                p_occ_tgt.clamp(1e-5, 1-1e-5)
            )

            # Temporal smoothness on logits (optional, encourages stability but not over-smoothing)
            # keep a buffer of previous decoded occupancy
            
            # if t == 1:
            #     self._prev_p_occ = p_occ_pred.detach()
            # logit_now  = torch.logit(p_occ_pred.clamp(1e-5, 1-1e-5))
            # logit_prev = torch.logit(self._prev_p_occ.clamp(1e-5, 1-1e-5)).to(device)
            # loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)
            # self._prev_p_occ = p_occ_pred.detach()
            
            
            if (t == 0) or (self._prev_keys is None):
                loss_temp = torch.tensor(0.0, device=self.device)
            else:
                prev_aligned = self.align_probs_to_keys(self._prev_keys, self._prev_probs,
                                                self.vox.keys, default=0.5)
                logit_now  = torch.logit(p_occ_pred.clamp(1e-5, 1-1e-5))
                logit_prev = torch.logit(prev_aligned.clamp(1e-5, 1-1e-5))
                loss_temp = F.smooth_l1_loss(logit_now, logit_prev, beta=0.1)

            # update buffers for next step
            self._prev_keys  = self.vox.keys.detach().clone()
            self._prev_probs = p_occ_pred.detach().clone()

            # Entropy regularizer on routing (OPTIONAL):
            # add a small penalty you compute inside update_with_features_learned (return avg entropy)
            # For simplicity, assume you store last entropy in self.grid._last_entropy
            loss_ent = torch.tensor(0.0, device=device)
            if hasattr(self.vox, "_last_entropy") and self.vox._last_entropy is not None:
                loss_ent = self.vox._last_entropy

            # TV regularizer on occupancy map (soft smoothness)
            # You can build a 3D TV using neighbor shifts (careful—sparse). Simple proxy:
            loss_tv = torch.tensor(0.0, device=device)

            loss_t = cfg.lambda_occ * loss_occ + cfg.lambda_temp * loss_temp \
                     + cfg.lambda_ent * loss_ent + cfg.lambda_tv * loss_tv

            loss_total = loss_total + loss_t
            
            print(loss_total)

            # logging
            self.log_dict({
                "loss/occ": loss_occ,
                "loss/temp": loss_temp,
                "loss/ent": loss_ent,
                "loss/tv": loss_tv,
                "stats/num_voxels": float(self.vox.keys.numel())
            }, prog_bar=(t == T-1), on_step=True, on_epoch=True, sync_dist=False)

        self.log("loss/total", loss_total, prog_bar=True)
        
        
        # after build_frames_and_centers_vectorized(...)
        del predictions  # drops images, view_feats, etc. all at once

        # after build_maps_from_latent_features(...)
        del frames_map, cam_centers_map, conf_map, images_map, features_map
        gc.collect()

        return loss_total
    
    
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

def _sequence_dirs_from_root(root: str) -> List[str]:
    """Each immediate subdir of root is a sequence. If root itself contains images,
    treat root as a single sequence as well."""
    seqs = _list_dirs(root)
    # also support 'flat' root as one sequence if it directly has images
    if _list_imgs(root):
        seqs = [root] + seqs
    if not seqs:
        raise FileNotFoundError(f"No sequences found under: {root}")
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
    ):
        self.root = dataset_root
        self.size = size
        self.verbose = verbose
        self.min_images_per_timestep = min_images_per_timestep

        if sequences is None:
            seqs = _sequence_dirs_from_root(dataset_root)
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
        for td in t_dirs:
            imgs = self._load_timestep(td)
            if imgs:
                imgs_t.append(imgs)

        if not imgs_t:
            raise RuntimeError(f"No images found for sequence: {seq_dir}")

        return {
            "seq_id": os.path.basename(seq_dir.rstrip("/")),
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
        num_workers: int = 4,
        size: int = 512,
        verbose: bool = False,
        train_val_split: float = 0.0,  # 0 = all train, else fraction for val (e.g., 0.1)
        seed: int = 42,
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

    def setup(self, stage: Optional[str] = None):
        all_seqs = _sequence_dirs_from_root(self.dataset_root)
        if self.train_val_split > 0.0:
            random.Random(self.seed).shuffle(all_seqs)
            n_val = max(1, int(len(all_seqs) * self.train_val_split))
            val_seqs = all_seqs[:n_val]
            train_seqs = all_seqs[n_val:]
        else:
            train_seqs, val_seqs = all_seqs, []

        self.train_set = HabitatSeqDataset(
            dataset_root=self.dataset_root,
            size=self.size,
            verbose=self.verbose,
            sequences=train_seqs
        )
        self.val_set = HabitatSeqDataset(
            dataset_root=self.dataset_root,
            size=self.size,
            verbose=self.verbose,
            sequences=val_seqs
        ) if val_seqs else None

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,              # keep 1 if your step assumes one sequence
            shuffle=True,
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
        dataset_root="/Users/marvin/Documents/Thesis/repo/dataset_generation/habitat/",
        voxel_size=0.10,
        radius_m=0.25,
        topk=8,
        temp=0.5,
        feature_dim=64,
        occ_decoder_hidden=64,
        lr=1e-3,
        max_epochs=20,
        batch_size=1,
        num_workers=4,
        precision="32",
    )

    dm = HabitatDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        size=512,
        verbose=False,
        train_val_split=0.0,  # or whatever you want
    )

    sys = VoxelUpdaterSystem(cfg)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="loss/total",
        save_top_k=3,
        mode="min",
        filename="voxup-{epoch:02d}-{loss_total:.4f}"
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")

 
 
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        callbacks=[ckpt_cb, lr_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    trainer.fit(sys, dm)
if __name__ == "__main__":
    main()
