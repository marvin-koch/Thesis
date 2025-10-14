"""

Turn VGGT outputs into a 3D occupancy voxel grid (with optional freespace
ray-carving) and a 2D BEV occupancy map at 0.10 m resolution.

Uses Amanatides & Woo (1987) 3D Voxel traversal paper
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from voxel.voxel import *
from dataclasses import dataclass
from voxel.align import *
import glob
import os
import json
from pathlib import Path

from vggt.utils.load_fn import load_and_preprocess_images
from visual_util import predictions_to_glb
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return type(x)(to_numpy(xx) for xx in x)

    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return np.asarray(x)
    
def get_conf_frame(predictions, f, conf_key="confidence"):
    """Return confidence map for frame f as (H,W) float in [0,1]."""
    if conf_key not in predictions:
        for alt in ("depth_confidence", "point_confidence"):
            if alt in predictions:
                conf_key = alt
                break
    if conf_key not in predictions:
        raise KeyError("No confidence map found in predictions "
                       "(tried 'confidence', 'depth_confidence', 'point_confidence').")
    C = to_numpy(predictions[conf_key])
    # normalize shapes → (S,H,W)
    if C.ndim == 4 and C.shape[-1] == 1:   # (S,H,W,1)
        C = C[..., 0]
    elif C.ndim == 4 and C.shape[1] == 1:  # (S,1,H,W)
        C = np.transpose(C, (0,2,3,1))[..., 0]
    elif C.ndim != 3:
        raise ValueError(f"confidence must be (S,H,W)/(S,H,W,1)/(S,1,H,W), got {C.shape}")
    # scale to [0,1] if it looks like uint8 / >1
    if C.dtype == np.uint8 or C.max() > 1.0:
        C = C.astype(np.float32) / 255.0
    return C[f].astype(np.float32)  # (H,W)


def align_pointcloud(points_world_S_HW3,
                     *,
                     max_samples=500_000,
                     ransac_iters=1000,
                     inlier_dist=0.03,
                     min_up_dot=0.8,          # be stricter: within ~45° of +Z
                     floor_bottom_frac=0.30,  # use lowest 40% z as floor-biased sampling pool
                     z_tiebreak_q=0.20,       # break ties by 20th-percentile inlier z
                     seed=0):
    """
    Finds a Z-up frame and aligns the *lowest horizontal plane* to z=0 (the floor).
    Returns:
      R_w2m (3x3), t_w2m (3,), info dict
    """
    rng = np.random.default_rng(seed)

    # 0) Basic OpenCV(cam0)->Z-up (x,y,z)=(x,z,-y)
    # R_basic = np.array([[1,0,0],[0,0,1],[0,-1,0]], np.float32)
    R_basic = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)

    # Flatten + clean
    WPTS = np.asarray(points_world_S_HW3, dtype=np.float32).reshape(-1,3)
    WPTS = WPTS[~np.isnan(WPTS).any(axis=1)]
    if WPTS.size == 0:
        return R_basic, np.zeros(3, np.float32), {"reason":"no_points"}

    # 1) Go to rough Z-up space
    P0 = WPTS @ R_basic.T
    if P0.shape[0] > max_samples:
        P0 = P0[rng.choice(P0.shape[0], max_samples, replace=False)]

    # ----- Floor bias: build a "low-z" pool for RANSAC sampling -----
    z = P0[:, 2]
    if np.isfinite(z).any():
        z_thresh = np.quantile(z, floor_bottom_frac)
        low_mask = z <= z_thresh
        P_low = P0[low_mask]
        # fallback if degenerate
        if P_low.shape[0] < 200:
            P_low = P0
    else:
        P_low = P0

    up = np.array([0,0,1.0], np.float32)

    best = {
        "inliers": -1,
        "n": None, "d": None,
        "z_q": np.inf,   # lower is better
        "idx": None
    }

    # 2) RANSAC plane on floor-biased pool
    N = P_low.shape[0]
    if N < 3:
        return R_basic, np.zeros(3, np.float32), {"reason":"too_few_points_after_filter"}

    for _ in range(int(ransac_iters)):
        i = rng.choice(N, 3, replace=False)
        p0, p1, p2 = P_low[i[0]], P_low[i[1]], P_low[i[2]]

        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        # orient upward
        if n[2] < 0: n = -n

        up_dot = float(n @ up)
        if up_dot < min_up_dot:
            continue  # not horizontal enough

        d = -float(n @ p0)

        # score over ALL points (not only P_low) for robust inlier count
        dist = np.abs(P0 @ n + d)
        inlier_mask = dist < float(inlier_dist)
        inl = int(inlier_mask.sum())
        if inl <= 0:
            continue

        # z-aware tiebreak: lower 20th-percentile inlier z is better
        inlier_z = P0[inlier_mask, 2]
        z_q = float(np.quantile(inlier_z, z_tiebreak_q)) if inlier_z.size else np.inf

        # Choose the model with:
        #   (1) highest inlier count,
        #   (2) if within 2% of best inliers, lowest z_q wins.
        better = False
        if inl > best["inliers"]:
            better = True
        elif best["inliers"] > 0 and inl >= 0.98 * best["inliers"] and z_q < best["z_q"]:
            better = True

        if better:
            best.update({"inliers": inl, "n": n.copy(), "d": d, "z_q": z_q, "idx": np.where(inlier_mask)[0]})

    if best["n"] is None:
        return R_basic, np.zeros(3, np.float32), {"reason":"ransac_failed"}

    # 2b) Optional: refine plane with least-squares on its inliers (more stable normal)
    idx = best["idx"]
    if idx is not None and idx.size >= 50:
        Q = P0[idx]
        # plane via PCA: normal is smallest eigenvector of covariance
        Qc = Q - Q.mean(axis=0, keepdims=True)
        C = (Qc.T @ Qc) / max(1, Q.shape[0]-1)
        w, U = np.linalg.eigh(C)
        n_ref = U[:, 0]
        if n_ref[2] < 0: n_ref = -n_ref
        # ensure near-horizontal
        if float(n_ref @ up) >= min_up_dot:
            d_ref = -float(n_ref @ Q.mean(axis=0))
            best["n"], best["d"] = n_ref, d_ref

    n, d = best["n"], best["d"]
    up_dot = float(n @ up)

    # 3) Rotate so plane normal -> +Z (Rodrigues n->z)
    v = np.cross(n, up)
    s = np.linalg.norm(v)
    c = up_dot
    if s < 1e-8:   # already aligned
        R_corr = np.eye(3, dtype=np.float32)
    else:
        vx = np.array([[0,-v[2],v[1]],
                       [v[2],0,-v[0]],
                       [-v[1],v[0],0]], np.float32)
        R_corr = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1-c)/(s**2))

    # Combined rotation (WORLD->MAP)
    R_w2m = R_corr @ R_basic

    # After rotation, plane is n'·x + d = 0, with n' ~ [0,0,1], so z + d = 0.
    # To move floor to z=0: translate by +d on z.
    t_w2m = np.array([0.0, 0.0, d], dtype=np.float32)

    info = {
        "ransac_inliers": int(best["inliers"]),
        "inlier_dist": float(inlier_dist),
        "floor_bottom_frac": float(floor_bottom_frac),
        "z_tiebreak_q": float(best["z_q"]),
        "angle_to_up_deg": float(np.degrees(np.arccos(np.clip(up_dot, -1, 1)))),
    }
    return R_w2m.astype(np.float32), t_w2m.astype(np.float32), info


def build_frames_and_centers_vectorized(
    predictions: dict,
    *,
    POINTS: str = "world_points_from_depth",
    CONF: str = "world_points_conf",
    EXTR_KEY: str = "extrinsic",
    threshold: float = 50.0,     # percentile in [0,100]
    Rmw: np.ndarray = np.eye(3, dtype=np.float32),   # (optional) world->map rot (not applied to points here)
    tmw: np.ndarray = np.zeros(3, dtype=np.float32), # (optional) world->map trans (not applied to points here)
    z_clip_map: tuple[float,float] | None = None,    # apply on (possibly flipped) points
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int,int,int]]:
    """
    Vectorized prep:
      - computes camera centers for all frames in one shot
      - builds a global confidence threshold (percentile) once
      - builds per-pixel validity mask in batch
      - splits into per-frame arrays without looping over points

    Returns:
      frames_map: [ (N_i,3) float32 ]
      cam_centers_map: [ (3,) float32 ]
      sh: (S,H,W)   # original grid shape of POINTS/CONF
    """ 

    # --- load arrays ---
    P = to_numpy(predictions[POINTS]).astype(np.float32)   # (S,H,W,3)
    C = to_numpy(predictions.get(CONF, np.ones_like(P[...,0], dtype=np.float32))).astype(np.float32)  # (S,H,W)
    
    print(P.shape)
    print(C.shape)
    EXTR = to_numpy(predictions[EXTR_KEY])                 # (S,3,4) or (S,4,4)

    S, H, W = P.shape[:3]
    
    if threshold == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(C, threshold)
        
    # pad extrinsics if needed -> (S,4,4)
    if EXTR.ndim == 3 and EXTR.shape[1:] == (3,4):
        bottom = np.tile(np.array([[0,0,0,1]], EXTR.dtype), (EXTR.shape[0],1,1))
        EXTR = np.concatenate([EXTR, bottom], axis=1)

    # # --- camera centers: Cw = -R^T t (all frames at once) ---
    # Rwc = EXTR[:, :3, :3]                  # (S,3,3)
    # twc = EXTR[:, :3, 3]                   # (S,3)
    # Cw = -np.einsum('sij,sj->si', Rwc.transpose(0,2,1), twc)  # (S,3)
    # Cm = (Rmw @ Cw.T).T + tmw[None, :]      # (S,3)
    # cam_centers_map = [Cm[f].astype(np.float32) for f in range(S)]

    Cw = EXTR[:, :3, 3]  # (S,3)   <-- correct for Twc
    cam_centers_map = [Cw[f].astype(np.float32) for f in range(S)]

    C_flat = C.reshape(-1)
    C_flat = C_flat[np.isfinite(C_flat)]

    # --- validity mask (vectorized) ---
    finite_xyz = np.isfinite(P).all(axis=-1)              # (S,H,W)
    finite_conf = np.isfinite(C)                          # (S,H,W)
    conf_ok = (C >= conf_threshold) & (C > 1e-5)          # (S,H,W)
    valid = finite_xyz & finite_conf & conf_ok            # (S,H,W)

    # --- optional z clip (apply to your current “map-consistent” frame; you’re NOT applying Rmw to points here) ---
    # If you want the clip in Z-up coordinates, apply R_w2m to P (vectorized), compute mask_z on that,
    # but still *return* the original P (to keep your current behavior). Uncomment if needed:
    if z_clip_map is not None:
        z0, z1 = z_clip_map
        # z-clip in current coordinates of P (what you visualize/use)
        z = P[..., 2]                     # (S,H,W)
        valid &= (z >= z0) & (z <= z1)

    # --- gather per-frame points (no per-point loops) ---
    P_flat = P.reshape(S, -1, 3)         # (S, H*W, 3)
    V_flat = valid.reshape(S, -1)        # (S, H*W)
    C_flat = C.reshape(S,-1)
    
    frames_map = []
    conf_map = []

    for f in range(S):
        conf_map.append(C_flat[f][V_flat[f]].astype(np.float32))
        if not np.any(V_flat[f]):
            frames_map.append(np.empty((0,3), dtype=np.float32))
            continue
        frames_map.append(P_flat[f][V_flat[f]].astype(np.float32))

    return frames_map, cam_centers_map, conf_map, (S, H, W)

# =============================
# ----- BEV rasterization -----
# =============================

@dataclass
class BevSpec:
    resolution: float = 0.10  # m/cell
    width_m: float = 20.0     # meters
    height_m: float = 20.0
    origin_xy: Tuple[float, float] = (-10.0, -10.0)  # world coords of cell (0,0)
    z_band: Tuple[float, float] = (0.05, 1.80)


def bev_from_voxels(
    vox,
    spec: BevSpec,
    include_free: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Project occupied (and optionally free) voxels to a 2D BEV occupancy.

    Returns (bev, meta) where bev is int8 with values: -1 unknown, 0 free, 100 occ.
    Works with:
      - dict-based SparseVoxelGrid (has .logodds: Dict[(i,j,k)] -> float, .origin np, .p params)
      - TorchSparseVoxelGrid (has .keys/.vals torch tensors, .origin torch, .p params)
    """
    W = int(round(spec.width_m / spec.resolution))
    H = int(round(spec.height_m / spec.resolution))
    bev = -np.ones((H, W), dtype=np.int8)  # start unknown

    zmin, zmax = spec.z_band

    # --------- Torch-backed voxel grid (fast path) ---------
    is_torch_grid = hasattr(vox, "keys") and hasattr(vox, "vals")
    if is_torch_grid:

        keys: torch.Tensor = vox.keys
        vals: torch.Tensor = vox.vals
        if keys.numel() == 0:
            meta = {
                "resolution": spec.resolution,
                "origin_xy": spec.origin_xy,
                "width": W,
                "height": H,
                "z_band": spec.z_band,
            }
            return bev, meta

        # decode ijk from hashed keys
        off = (1 << 20)
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = ( keys        & ((1 << 21) - 1)) - off
        ijk = torch.stack([i, j, k], dim=-1).to(torch.float32)  # (M,3)

        # centers in world/map coords
        origin = vox.origin.to(torch.float32)
        vs = float(vox.p.voxel_size)
        centers = origin + (ijk + 0.5) * vs                       # (M,3)
        cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]

        # masks
        # occ_mask = vals > vox.p.occ_thresh
        occ_mask = vox.occupied_mask()

        z_mask = (cz >= zmin) & (cz <= zmax)
        # ----- Occupied -----
        occ_idx = occ_mask & z_mask
        
        if occ_idx.any():
            u = torch.floor((cx[occ_idx] - spec.origin_xy[0]) / spec.resolution).to(torch.int64)
            v = torch.floor((cy[occ_idx] - spec.origin_xy[1]) / spec.resolution).to(torch.int64)
            # clamp to image bounds, then write
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.any():
                uu = u[inb].cpu().numpy()
                vv = v[inb].cpu().numpy()
                bev[vv, uu] = 100

        # ----- Free (optional) -----
        if include_free:
            # Treat sufficiently negative log-odds as free (and not occupied)
            free_mask = (~occ_mask) & (vals <= -0.2) & z_mask
            if free_mask.any():
                u = torch.floor((cx[free_mask] - spec.origin_xy[0]) / spec.resolution).to(torch.int64)
                v = torch.floor((cy[free_mask] - spec.origin_xy[1]) / spec.resolution).to(torch.int64)
                inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if inb.any():
                    uu = u[inb].cpu().numpy()
                    vv = v[inb].cpu().numpy()
                    # only set free if not already marked occupied
                    mask = (bev[vv, uu] != 100)
                    bev[vv[mask], uu[mask]] = 0

    # --------- Dict/NumPy voxel grid (compat path) ---------
    else:
        # occupied
        occ_vox = vox.occupied_voxels(zmin=zmin, zmax=zmax)
        for ijk in occ_vox:
            cx, cy, cz = vox.ijk_to_center(ijk)
            u = int(math.floor((cx - spec.origin_xy[0]) / spec.resolution))
            v = int(math.floor((cy - spec.origin_xy[1]) / spec.resolution))
            if 0 <= u < W and 0 <= v < H:
                bev[v, u] = 100

        if include_free:
            # known-free where not occupied
            for ijk, L in vox.logodds.items():
                if L > vox.p.occ_thresh:
                    continue
                if L <= -0.2:
                    cx, cy, _ = vox.ijk_to_center(ijk)
                    u = int(math.floor((cx - spec.origin_xy[0]) / spec.resolution))
                    v = int(math.floor((cy - spec.origin_xy[1]) / spec.resolution))
                    if 0 <= u < W and 0 <= v < H and bev[v, u] != 100:
                        bev[v, u] = 0

    meta = {
        "resolution": spec.resolution,
        "origin_xy": spec.origin_xy,
        "width": W,
        "height": H,
        "z_band": spec.z_band,
    }
    return bev, meta

# -----------------------------
# Optional: simple BEV inflation
# -----------------------------

def inflate_bev(bev: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Binary inflate occupied cells by radius (meters). Pure NumPy (no SciPy).
    Returns a new BEV array (int8) with the same value semantics.
    """
    H, W = bev.shape
    r_cells = int(math.ceil(radius_m / resolution))
    if r_cells <= 0:
        return bev.copy()

    # Precompute disk offsets
    offsets: List[Tuple[int, int]] = []
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx * dx + dy * dy <= r_cells * r_cells:
                offsets.append((dy, dx))

    out = bev.copy()
    occ_ys, occ_xs = np.where(bev == 100)
    for y, x in zip(occ_ys, occ_xs):
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < H and 0 <= xx < W and out[yy, xx] != 100:
                # mark inflated area as free(0) if unknown(-1); keep occupied as 100
                if out[yy, xx] == -1:
                    out[yy, xx] = 0
    return out

def build_maps_from_points_and_centers_torch(
    frames_xyz: list[np.ndarray],          # list of (N_i,3) in MAP frame
    cam_centers: list[np.ndarray],         # list of (3,) in MAP frame
    conf_map: list[np.ndarray],         
    tvox,
    *,
    align_to_voxel=False,
    voxel_size: float = 0.10,
    bev_window_m=(20.0, 20.0),
    bev_origin_xy=(-10.0, -10.0),
    z_clip_vox=(-np.inf, np.inf),
    z_band_bev=(0.05, 1.80),
    max_range_m: float | None = 12.0,
    carve_free: bool = True,
    # new (optional) tuning/throughput controls:
    device: str = "cpu",
    samples_per_voxel: float = 1.1,
    ray_stride: int = 1,
    max_free_rays: int | None = 200_000,
    batch_chunk_points: int | None = None,  # e.g., 2_000_000 to limit RAM
):

    # 1) Concatenate all frames once (no per-frame integrate calls)
    #    Build per-point camera centers by repeating each frame center.
    valid_frames = [(np.asarray(P, np.float32), np.asarray(C, np.float32).reshape(3), np.asarray(CONF, np.float32))
                    for P, C, CONF in zip(frames_xyz, cam_centers, conf_map)
                    if (P is not None and P.size and C is not None)]
    
    print("1")
    if not valid_frames:
        bev_spec = BevSpec(resolution=voxel_size, width_m=float(bev_window_m[0]),
                           height_m=float(bev_window_m[1]), origin_xy=bev_origin_xy, z_band=z_band_bev)
        bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
        return tvox, bev, meta

    print("2")

    lengths = [pf.shape[0] for pf, _, _ in valid_frames]
    P_all = np.concatenate([pf for pf, _, _ in valid_frames], axis=0)              # (M,3)
    C_all = np.concatenate([np.repeat(cf[None, :], n, axis=0)
                            for n, (_, cf, _) in zip(lengths, valid_frames)], axis=0)  # (M,3)
    CONF_all = np.concatenate([c for _, _, c in valid_frames], axis=0)              # (M,3)

    print("3")

    # drop NaNs early (keeps alignment)
    keep = np.isfinite(P_all).all(axis=1) & np.isfinite(C_all).all(axis=1)
    P_all = P_all[keep]; C_all = C_all[keep]; CONF_all = CONF_all[keep]
    
    print("4")

    if align_to_voxel:
        _, pts, cameras = sim3_align_to_voxel(
            pts_world=torch.from_numpy(P_all),
            cams_world=torch.from_numpy(C_all),
            conf_world=torch.from_numpy(CONF_all),
            grid=tvox,
            cfg=AlignCfg(
                iters=5,
                downsample_points=50000,
                pad_vox=48,
                max_block_vox=192,
                samples_per_ray=16,
                lambda_free=0.1,
                lambda_prior_t=1e-4,
                lambda_prior_r=1e-4,
                lambda_prior_s=1e-4,
                step=2e-1,
                use_sim3=True,
                rot_clamp_deg=10.0,
                trans_clamp_m=5*voxel_size,
                scale_clamp=0.1,
                conf_filter=True,
                conf_quantile=0.3,
            )
        )
        
        # _, pts, cameras = sim3_icp_align_to_voxel(
        #     pts_world=torch.from_numpy(P_all),
        #     cams_world=torch.from_numpy(C_all),
        #     grid=tvox,                         # your TorchSparseVoxelGrid
        #     conf_world=torch.from_numpy(CONF_all),
        #     cfg=AlignCfg(
        #         iters=5,
        #         downsample_points=50000,
        #         pad_vox=48,
        #         max_block_vox=192,
        #         samples_per_ray=16,
        #         lambda_free=0.1,
        #         lambda_prior_t=1e-4,
        #         lambda_prior_r=1e-4,
        #         lambda_prior_s=1e-4,
        #         step=2e-1,
        #         use_sim3=True,
        #         rot_clamp_deg=10.0,
        #         trans_clamp_m=5*voxel_size,
        #         scale_clamp=0.1,
        #         conf_filter=True,
        #         conf_quantile=0.3,
        #         icp_max_iters=50,
        #         icp_trim_fraction=0.7,
        #         icp_huber_delta=0.03,         # ~3 cm
        #         icp_dist_thresh=0.05,         # 20 cm cutoff early
        #         icp_nn_subsample=50000,
        #         icp_max_map_points=50000,
        #     )
        # )

        # _, pts, cameras = sim3_align_to_voxel_robust(
        #     pts_world=torch.from_numpy(P_all),
        #     cams_world=torch.from_numpy(C_all),
        #     conf_world=torch.from_numpy(CONF_all),
        #     grid=tvox,
        #     cfg=AlignCfg(
        #         iters=12,
        #         downsample_points=30000,
        #         pad_vox=24,
        #         max_block_vox=192,
        #         samples_per_ray=16,
        #         lambda_free_schedule=[0.001],
        #         lambda_prior_t=1e-4,
        #         lambda_prior_r=1e-4,
        #         lambda_prior_s=1e-4,
        #         step=2e-1,
        #         use_sim3=True,
        #         rot_clamp_deg=10.0,
        #         trans_clamp_m=2*voxel_size,#0.5*voxel_size,
        #         scale_clamp=0.1,
        #         sigmas=[3.0, 2.0, 1.0],
        #         sdf_mode="logit",
        #         free_margin_m=2.0*voxel_size,
        #         free_step_m=0.5*voxel_size,
        #         lambda_norm=0.1,
        #         conf_filter=True,
        #         conf_quantile=0.1,
        #         huber_delta=0.05,
        #         bootstrap_nn_max_m=0.05*voxel_size,
        #         bootstrap_icp_iters=2
        #     )
        # )
    else:
        pts, cameras = torch.from_numpy(P_all).to(device), torch.from_numpy(C_all).to(device),


    print(pts.shape)
    # 2) Integrate once (optionally in chunks to cap memory)
    if batch_chunk_points is None or P_all.shape[0] <= batch_chunk_points:
        tvox.integrate_points_with_cameras(
            pts,
            cameras,
            carve_free=carve_free,
            max_range=max_range_m,
            z_clip=z_clip_vox,
            samples_per_voxel=samples_per_voxel,
            ray_stride=ray_stride,
            max_free_rays=max_free_rays,
        )
        print("INTEGRATED")
    else:
        m = P_all.shape[0]
        for s in range(0, m, batch_chunk_points):
            e = min(s + batch_chunk_points, m)
            tvox.integrate_points_with_cameras(
                pts[s:e],
                cameras[s:e],
                carve_free=carve_free,
                max_range=max_range_m,
                z_clip=z_clip_vox,
                samples_per_voxel=samples_per_voxel,
                ray_stride=ray_stride,
                max_free_rays=max_free_rays,
            )


    # 3) BEV (reuse your existing rasterizer via adapter)
    #vox = TorchVoxelAdapter(tvox)
    bev_spec = BevSpec(
        resolution=voxel_size,
        width_m=float(bev_window_m[0]),
        height_m=float(bev_window_m[1]),
        origin_xy=bev_origin_xy,
        z_band=z_band_bev,
    )
    bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
    return tvox, bev, meta

def build_maps_from_points_and_centers(
    frames_xyz: list[np.ndarray],          # list of (N_i,3) in MAP frame
    cam_centers: list[np.ndarray],         # list of (3,) in MAP frame
    *,
    voxel_size: float = 0.10,
    bev_window_m=(20.0, 20.0),
    bev_origin_xy=(-10.0, -10.0),
    z_clip_vox=(-np.inf, np.inf),
    z_band_bev=(0.05, 1.80),
    max_range_m: float | None = 12.0,
    carve_free: bool = True,
):
    vox = SparseVoxelGrid(origin=np.zeros(3, dtype=np.float32),
                          params=VoxelParams(voxel_size=voxel_size))
    for P, C in zip(frames_xyz, cam_centers):
        if P.size == 0: 
            continue
        P = P[~np.isnan(P).any(axis=1)]
        vox.integrate_frame(P, C, carve_free=carve_free, max_range=max_range_m, z_clip=z_clip_vox)
        

    bev_spec = BevSpec(
        resolution=voxel_size,
        width_m=float(bev_window_m[0]),
        height_m=float(bev_window_m[1]),
        origin_xy=bev_origin_xy,
        z_band=z_band_bev,
    )
    bev, meta = bev_from_voxels(vox, bev_spec, include_free=True)
    return vox, bev, meta




def visualize_vggt_pointcloud(
    predictions,
    *,
    key="world_points",
    conf_key="world_points_conf",
    threshold=50.0,
    max_points=500_000,
    z_band=None,
    frame_stride=1,
):
    """
    Colorized VGGT point cloud with coordinate axes.
    Accepts images in (S,H,W,3) or (S,3,H,W).
    """
    import numpy as np
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required (pip install open3d).") from e

    def to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    if key not in predictions or "images" not in predictions:
        raise KeyError("predictions must contain key and 'images'")

    # World points: (S,H,W,3)
    WPTS = to_numpy(predictions[key])
    if WPTS.ndim != 4 or WPTS.shape[-1] != 3:
        raise ValueError(f"key expected (S,H,W,3), got {WPTS.shape}")
    S, H, W = WPTS.shape[:3]

    # Images: accept (S,H,W,3) OR (S,3,H,W)
    IMGS = to_numpy(predictions["images"])
    if IMGS.ndim == 4 and IMGS.shape[-1] == 3:
        pass  # already (S,H,W,3)
    elif IMGS.ndim == 4 and IMGS.shape[1] == 3:
        IMGS = np.transpose(IMGS, (0, 2, 3, 1))  # (S,3,H,W) -> (S,H,W,3)
    else:
        raise ValueError(f"images must be (S,H,W,3) or (S,3,H,W), got {IMGS.shape}")

    # Ensure dims match world points
    if IMGS.shape[:3] != (S, H, W):
        raise ValueError(f"images shape {IMGS.shape} must match key {(S,H,W,3)}")

    # Flatten selected frames
    frames = np.arange(0, S, max(1, int(frame_stride)))
    P = WPTS[frames].reshape(-1, 3).astype(np.float32)
    C = IMGS[frames].reshape(-1, 3).astype(np.float32)

    # Normalize colors to [0,1]
    if C.max() > 1.0:  # likely uint8
        C = C / 255.0
    C = np.clip(C, 0.0, 1.0)
    
    conf = np.array(predictions[conf_key][frames])
    conf = conf.reshape(-1).astype(np.float32)
    
    if threshold == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, threshold)


    valid = (~np.isnan(P).any(axis=1)) & np.isfinite(P).all(axis=1)
    valid = valid & (conf >= conf_threshold) & (conf > 1e-5)

    P = P[valid]; C = C[valid]

    # Optional z-band filter
    if z_band is not None:
        z0, z1 = float(z_band[0]), float(z_band[1])
        in_band = (P[:, 2] >= z0) & (P[:, 2] <= z1)
        P = P[in_band]; C = C[in_band]

    # Downsample
    n = P.shape[0]
    if n == 0:
        raise RuntimeError("No valid points to visualize (after filtering).")
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        P = P[idx]; C = C[idx]

    print(f"Visualizing {P.shape[0]} points from {len(frames)} frame(s)...")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))

    # Add coordinate frame (X=red, Y=green, Z=blue)
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # scale (meters)
        origin=[0, 0, 0]
    )

    # Visualize together
    o3d.visualization.draw_geometries(
        [pcd, axis_frame],
        window_name="VGGT Colored Point Cloud with Axes"
    )

def load_images(target_dir, device="cpu"): 
    image_names = glob.glob(os.path.join(target_dir, "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    
    return images

def run_model(model, images, dtype=torch.float32):

    print("Running Inference with VGGT")

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)


    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    
    return predictions

def rotate_points(points, R, t):
    WPTS_m = points @ R.T + t[None,None,None,:]            # (S,H,W,3)
    return WPTS_m


@torch.no_grad()
def covisibility_from_world_proximity(
    P_world: torch.Tensor,           # (S, H, W, 3), world-aligned per-frame point clouds
    conf_map: torch.Tensor,          # (S, H, W) or (S, H, W, 1) or (S,H,W,C)
    *,
    stride: int = 4,                 # subsample pixels: use every `stride`th row/col
    max_points: int = 20_000,        # cap points per frame after subsampling (random downsample)
    eps: float = 0.5,               # distance threshold for "same surface"
    sym: str = "mean",               # "mean" | "min" | "max"
    normalize: bool = True,          # return overlap ratio in [0,1] (else raw counts)
    chunk_size: int = 10_000,        # cdist chunking to save memory
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    conf_percentile: float | None = 50.0,  # global percentile over confidences to filter points; None = no percentile
    conf_min: float = 1e-5,          # absolute floor on confidence
):
    """
    Returns:
    W: (S, S) symmetric covisibility matrix (dtype), based on 3D proximity only.
    """
    assert sym in ("mean", "min", "max")
    assert P_world.ndim == 4 and P_world.shape[-1] == 3, f"P_world must be (S,H,W,3), got {tuple(P_world.shape)}"


    # Normalize conf_map to (S,H,W) tensor
    if conf_map.ndim == 4:
        conf_map = conf_map[..., 0]
    elif conf_map.ndim != 3:
        raise ValueError(f"conf_map must be (S,H,W) or (S,H,W,1[+]), got {tuple(conf_map.shape)}")

    P_world = torch.from_numpy(P_world)
    conf_map = torch.from_numpy(conf_map)

    # Keep everything in torch; decide device
    device = device or P_world.device
    P_world = P_world.to(device)
    conf_map = conf_map.to(device)

    S, H, Wimg, _ = P_world.shape

    # Build subsampling grid once
    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, Wimg, stride, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')       # (H/stride, W/stride)
    sub_v = grid_y.reshape(-1)                                    # (M,)
    sub_u = grid_x.reshape(-1)                                    # (M,)

    # Optionally compute a GLOBAL percentile threshold over confidences (only finite points)
    if conf_percentile is not None:
        # gather confidence over all frames at subsampled pixels
        conf_all = conf_map[:, sub_v, sub_u].reshape(-1)          # (S*M,)
        # keep finite confs only (some pipelines use NaNs/Infs for invalid)
        finite = torch.isfinite(conf_all)
        if finite.any():
            # torch.quantile is nan-agnostic only if you filter first
            conf_threshold = torch.quantile(conf_all[finite], conf_percentile / 100.0)
            conf_threshold = torch.maximum(conf_threshold, torch.as_tensor(conf_min, device=device))
        else:
            conf_threshold = torch.as_tensor(conf_min, device=device)
    else:
        conf_threshold = torch.as_tensor(conf_min, device=device)

    # Prepack subsampled, confidence-filtered points per frame
    frames_pts: list[torch.Tensor] = []
    # For deterministic downsampling on CPU, keep generator on CPU but use indices on device
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(S):
        Pi = P_world[i, sub_v, sub_u, :]                          # (M, 3)
        Ci = conf_map[i, sub_v, sub_u]                            # (M,)

        # validity mask: finite 3D & finite conf & conf >= thresholds
        valid3d = torch.isfinite(Pi).all(dim=-1)
        validc  = torch.isfinite(Ci) & (Ci >= conf_threshold) & (Ci > conf_min)
        mask = valid3d & validc

        Pi = Pi[mask]                                             # (m_i, 3)
        if Pi.numel() == 0:
            frames_pts.append(torch.empty((0, 3), device=device, dtype=dtype))
            continue

        # optional per-frame cap
        if Pi.shape[0] > max_points:
            # randperm on CPU generator → bring indices to device
            idx = torch.randperm(Pi.shape[0], generator=rng)[:max_points]
            idx = idx.to(device=device)
            Pi = Pi.index_select(0, idx)

        frames_pts.append(Pi.to(dtype=dtype))

    # Pairwise proximity with chunking
    W = torch.zeros((S, S), dtype=dtype, device=device)

    def nn_consistent(A: torch.Tensor, B: torch.Tensor, eps: float, chunk: int):
        """
        Count of points in A that have at least one NN in B within eps, and total count |A|.
        """
        if A.numel() == 0 or B.numel() == 0:
            return 0, 0
        nA = A.shape[0]
        ok_total = 0
        # Pre-chunk B too if it's very big to reduce peak memory
        # (Here we only chunk A; chunking B as well is easy if needed.)
        for start in range(0, nA, chunk):
            end = min(start + chunk, nA)
            a = A[start:end]                                      # (m,3)
            dmin = torch.cdist(a, B, p=2).min(dim=1).values       # (m,)
            ok_total += (dmin <= eps).sum().item()
        return ok_total, nA

    for i in range(S):
        Ai = frames_pts[i]
        for j in range(i, S):
            if i == j:
                w = 1.0
            else:
                Bj = frames_pts[j]
                ok_ij, n_i = nn_consistent(Ai, Bj, eps=eps, chunk=chunk_size)
                ok_ji, n_j = nn_consistent(Bj, Ai, eps=eps, chunk=chunk_size)

                if normalize:
                    s_ij = ok_ij / max(1, n_i)
                    s_ji = ok_ji / max(1, n_j)
                else:
                    s_ij, s_ji = float(ok_ij), float(ok_ji)

                if sym == "mean":
                    w = 0.5 * (s_ij + s_ji)
                elif sym == "min":
                    w = min(s_ij, s_ji)
                else:  # "max"
                    w = max(s_ij, s_ji)

            W[i, j] = W[j, i] = w

    return W


def covis_to_adj(
    W: torch.Tensor,                 # (S,S) float
    tau: float = 0.5,                # base threshold for "good" edges
    kmin: int = 2,                   # ensure each node has at least kmin neighbors
    window: int | None = None,       # optional |i-j| <= window constraint
    mutual: str = "or",              # 'or' (weakly-undirected) or 'and' (mutual)
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build an adjacency with no isolated nodes:
      - keep edges with W >= tau
      - add top-kmin edges per node as a safety net (k-NN)
      - symmetrize and set diagonal True
    """
    assert mutual in ("or", "and")
    device = device or W.device
    S = W.shape[0]
    W = W.to(device)

    # base thresholded edges
    adj = (W >= tau)

    # optional temporal/spatial band
    if window is not None:
        i = torch.arange(S, device=device)
        band = (i[:, None] - i[None, :]).abs() <= window
        adj = adj & band

    # k-NN backup (exclude self)
    W_knn = W.clone()
    W_knn.fill_diagonal_(-float("inf"))
    if window is not None:
        # forbid outside window in the KNN too
        W_knn = W_knn.masked_fill(~band, -float("inf"))

    if kmin > 0:
        # take top-kmin neighbors per row
        k = min(kmin, max(0, S - 1))
        if k > 0:
            idx = torch.topk(W_knn, k=k, dim=1).indices                        # (S,k)
            knn = torch.zeros((S, S), dtype=torch.bool, device=device)
            knn.scatter_(1, idx, True)                                         # mark top-k per row
            adj = adj | knn                                                     # ensure degree ≥ kmin

    # symmetrize
    adj = (adj | adj.t()) if mutual == "or" else (adj & adj.t())

    # diagonal = True
    adj.fill_diagonal_(True)
    return adj

def compute_tight_components(adj_bool_2d: torch.Tensor,
                            mutual: bool = True,
                            window: int | None = None,
                            jaccard_tau: float | None = None,
                            k_core_k: int | None = None):
    """
    adj_bool_2d: (S,S) bool, True=edge (directed or undirected)
    Returns: list[list[int]] connected components after pruning.
    """

    A = adj_bool_2d.clone()

    S = A.shape[0]
    # 1) symmetrize
    A = A | A.T                # weakly undirected (either direction)
    if mutual:
        A = A & A.T            # require i<->j (mutual)

    # 2) force self-edges
    i = torch.arange(S, device=A.device)
    A[i, i] = True

    # 3) optional: local window
    if window is not None:
        # keep |i-j| <= window
        idx = torch.arange(S, device=A.device)
        band = (idx[:, None] - idx[None, :]).abs() <= window
        A = A & band

    # 4) optional: Jaccard prune (neighborhood similarity)
    if jaccard_tau is not None:
        # avoid diag in sim (but keep diag in A)
        N = A.clone()
        N[i, i] = False
        inter = (N[:, None, :] & N[None, :, :]).sum(dim=-1)                  # (S,S)
        union = (N[:, None, :] | N[None, :, :]).sum(dim=-1).clamp_min_(1)
        jacc = inter.float() / union.float()
        keep = jacc >= jaccard_tau
        A = (A & keep) | torch.diag(torch.ones(S, dtype=torch.bool, device=A.device))

    # 5) optional: k-core prune
    if k_core_k is not None:
        deg = A.sum(dim=1) - 1  # exclude self
        active = torch.ones(S, dtype=torch.bool, device=A.device)
        changed = True
        while changed:
            drop = active & (deg < k_core_k)
            changed = bool(drop.any().item())
            active = active & ~drop
            if changed:
                # zero edges for dropped nodes
                A[drop, :] = False
                A[:, drop] = False
                A[i, i] = True
                deg = A.sum(dim=1) - 1

    # Connected components on the pruned, undirected graph
    S_ = S
    seen = torch.zeros(S_, dtype=torch.bool, device=A.device)
    comps = []
    for s in range(S_):
        if seen[s]:
            continue
        stack = [int(s)]
        seen[s] = True
        comp = [int(s)]
        while stack:
            u = stack.pop()
            nbrs = torch.where(A[u])[0]
            for v in nbrs.tolist():
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
                    comp.append(int(v))
        comps.append(comp)
    return comps


def merge_singletons_to_next(groups, S):
    """
    groups: list[list[int]]  (components over nodes 0..S-1)
    S: total number of frames/nodes

    Rule: if a component has size 1 (singleton {i}), merge i into the group
    that contains (i+1) % S. If *all* components are singletons, merge them all
    into a single group [0,1,...,S-1].
    """
    # keep only non-singletons first
    new_groups = [set(g) for g in groups if len(g) > 1]
    if not new_groups:
        # all singletons -> one big group
        return [list(range(S))]

    # map node -> group index for the current (non-singleton) groups
    node_to_gid = {}
    for gid, g in enumerate(new_groups):
        for v in g:
            node_to_gid[v] = gid

    # collect singletons in sorted order for determinism
    singles = sorted([g[0] for g in groups if len(g) == 1])

    for i in singles:
        # find the next index (wrap) that already sits in a non-singleton group
        t = (i + 1) % S
        hopped = 0
        while t not in node_to_gid and hopped < S:
            t = (t + 1) % S
            hopped += 1
        if hopped >= S:
            # should not happen because we handled "all singletons" above,
            # but just in case, attach to the first group
            target_gid = 0
        else:
            target_gid = node_to_gid[t]

        # merge i into that group
        new_groups[target_gid].add(i)
        node_to_gid[i] = target_gid

    # return as sorted lists (optional: keep original ordering of groups)
    return [sorted(list(g)) for g in new_groups]



import heapq
import math

def dijkstra_to_any(adj, targets, start=0):
    """
    adj: NxN matrix of non-negative weights; 0 or math.inf means no edge.
    targets: target node indices (set/list)
    Returns (target_node, path, dist) or (None, None, inf).
    """
    
    if start in targets:
        return start, [], 0
    
    covis = np.asarray(adj, dtype=float)
    cost = np.full_like(covis, np.inf)
    mask = covis > 0
    # cost[mask] = 1.0 / (covis[mask] + 1e-6)
    
    cost[mask] = -np.log(covis[mask] + 1e-6)

    np.fill_diagonal(cost, np.inf)  # no self-edges
    adj = cost
    
    N = len(adj); targets = set(targets)
    dist = [math.inf]*N
    parent = [-1]*N
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = [False]*N
    
    
    while pq:
        d,u = heapq.heappop(pq)
        print(d,u)
        if visited[u]: continue
        visited[u] = True
        if u in targets:
            # reconstruct path
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return u, path[:-1], d
        for v in range(N):
            w = adj[u][v]
            if w and w < math.inf:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
    return None, None, math.inf

import math, heapq
import numpy as np

def _covis_to_cost(covis, eps=1e-6):
    """Flip covisibility (bigger=better) to cost (smaller=better). 0 -> inf (no edge)."""
    A = np.asarray(covis, dtype=float)
    cost = np.full_like(A, np.inf)
    mask = A > 0
    cost[mask] = 1.0 / (A[mask] + eps)
    np.fill_diagonal(cost, np.inf)
    return cost

def _dijkstra_path(cost, src, dst):
    """Shortest path on dense cost matrix (np.inf = no edge). Returns list of nodes (src..dst) or [] if no path."""
    N = cost.shape[0]
    dist = [math.inf]*N; parent = [-1]*N; vis = [False]*N
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if vis[u]: continue
        vis[u] = True
        if u == dst:
            # reconstruct
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            return list(reversed(path))
        # relax
        row = cost[u]
        # iterate only finite neighbors
        nbrs = np.isfinite(row).nonzero()[0]
        for v in nbrs:
            nd = d + row[v]
            if nd < dist[v]:
                dist[v] = nd; parent[v] = u
                heapq.heappush(pq, (nd, v))
    return []  # unreachable

def _coverage_score(covis, S):
    """
    Coverage = mean over all nodes j of max_{i in S} covis[j,i].
    Returns (score in [0,1], per-node coverage vector).
    """
    A = np.asarray(covis, dtype=float)
    if len(S) == 0:
        return 0.0, np.zeros(A.shape[0], dtype=float)
    cov_to_S = A[:, S]                  # (N, |S|)
    per_node = cov_to_S.max(axis=1)     # (N,)
    score = float(per_node.mean())
    return score, per_node

def grow_set_until_coverage(
    covis,
    thresh=0.80,                 # target mean coverage in [0,1]
    start=None,                  # initial set (list/array); default: argmax row-sum
    max_steps=100,
    verbose=False
):
    """
    Greedy growth:
    - Source each round: node in current set with the LOWEST covis-to-set (sum).
    - Candidate to add: outside node giving largest coverage gain if included.
    - Connectivity: add shortest path (on cost=1/(covis+eps)) from source -> candidate.
    Returns: selected (list), paths_added (list of lists), cov_history (list of floats)
    """
    # prep
    A = np.asarray(covis, dtype=float)
    N = A.shape[0]
    cost = _covis_to_cost(A)

    # init set
    if start is None or len(start) == 0:
        row_sum = A.sum(axis=1)
        seed = int(np.argmax(row_sum))
        S = {seed}
    else:
        S = set(int(x) for x in start)

    cov, per_node = _coverage_score(A, list(S))
    cov_hist = [cov]
    paths = []

    if verbose:
        print(f"[init] |S|={len(S)}  cov={cov:.3f}")

    steps = 0
    while cov < thresh and steps < max_steps:
        steps += 1
        S_list = list(S)

        # ----- choose source: node in S with LOWEST covis-to-set (sum) -----
        # (you could also use mean or max; sum is fine and simple)
        sum_to_S = A[np.ix_(S_list, S_list)].sum(axis=1)  # |S|-vector
        src = S_list[int(np.argmin(sum_to_S))]

        # ----- pick best candidate outside S by marginal coverage gain -----
        outside = [i for i in range(N) if i not in S]
        if not outside:
            break

        best_gain = -1.0
        best_cand = None

        # current per-node coverage to S
        cur_cov = per_node  # shape (N,)

        # evaluate marginal gain if we add each candidate c
        # new per-node coverage would be: max(cur_cov, A[:,c])
        for c in outside:
            new_per = np.maximum(cur_cov, A[:, c])
            new_score = float(new_per.mean())
            gain = new_score - cov
            if gain > best_gain:
                best_gain = gain
                best_cand = c

        if best_cand is None or best_gain <= 0.0:
            if verbose:
                print("[stop] no positive marginal gain; breaking")
            break

        # ----- connect src -> best_cand with shortest path and add nodes -----
        path = _dijkstra_path(cost, src, best_cand)
        if not path:
            # if disconnected, just add the candidate (last resort)
            path = [best_cand]
            if verbose:
                print(f"[warn] no path {src}->{best_cand}; adding candidate alone")

        # add path nodes
        new_nodes = [p for p in path if p not in S]
        S.update(new_nodes)
        paths.append(path)

        # update coverage
        cov, per_node = _coverage_score(A, list(S))
        cov_hist.append(cov)

        if verbose:
            print(f"[iter {steps}] added {new_nodes} via path {path} |S|={len(S)} cov={cov:.3f} (+{best_gain:.3f})")

    return sorted(S), paths, cov_hist



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401


def visualize_bev(bev: np.ndarray, meta: dict, title: str = "BEV occupancy (0.10 m)"):
    """Quick Matplotlib visualization of the BEV occupancy grid.
    -1 = unknown (dark), 0 = free, 100 = occupied
    """
    H, W = bev.shape
    res = float(meta.get("resolution", 0.10))
    ox, oy = meta.get("origin_xy", (0.0, 0.0))
    extent = [ox, ox + W * res, oy, oy + H * res]
    plt.figure()
    plt.imshow(bev, origin="lower", extent=extent)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.colorbar(label="occupancy value")
    plt.tight_layout()
    plt.show()

def visualize_voxels(vox,
                     z_band=(0.05, 1.8),
                     max_dim=96,          # clamp to keep it light
                     title="Occupied voxels (cubes)"):

    # Ensure occupied voxels list is NumPy-compatible
    occ_ijks = vox.occupied_voxels(zmin=float(z_band[0]), zmax=float(z_band[1]))
    occ_ijks = [tuple(int(x) for x in ijk) for ijk in occ_ijks]  # make sure they are tuples of ints

    if not occ_ijks:
        print("No occupied voxels to show.")
        return

    # Find tight bounds in index space
    I = np.array(occ_ijks, dtype=np.int32)
    imin, jmin, kmin = I.min(axis=0)
    imax, jmax, kmax = I.max(axis=0)

    # Make sure origin and voxel size are NumPy scalars
    origin = np.asarray(vox.origin, dtype=np.float32)
    vs = float(vox.p.voxel_size)

    # Clamp the block size (avoid huge dense arrays)
    ni, nj, nk = (imax - imin + 1), (jmax - jmin + 1), (kmax - kmin + 1)
    if max(ni, nj, nk) > max_dim:
        # Center a cropped window
        ci, cj, ck = (imin + imax) // 2, (jmin + jmax) // 2, (kmin + kmax) // 2
        half = max_dim // 2
        imin, imax = ci - half, ci + half
        jmin, jmax = cj - half, cj + half
        kmin, kmax = ck - half, ck + half
        ni, nj, nk = (imax - imin + 1), (jmax - jmin + 1), (kmax - kmin + 1)

    # Fill occupancy grid
    filled = np.zeros((ni, nj, nk), dtype=bool)
    for (i, j, k) in occ_ijks:
        if imin <= i <= imax and jmin <= j <= jmax and kmin <= k <= kmax:
            filled[i - imin, j - jmin, k - kmin] = True

    # Compute voxel corner coordinates in meters
    xe = np.arange(imin, imax + 2) * vs + origin[0]
    ye = np.arange(jmin, jmax + 2) * vs + origin[1]
    ze = np.arange(kmin, kmax + 2) * vs + origin[2]
    X, Y, Z = np.meshgrid(xe, ye, ze, indexing="ij")  # (ni+1, nj+1, nk+1)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(X, Y, Z, filled, edgecolor='k')  # draws cubes
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_points_and_voxels_open3d(
    frames_map, vox, cam_centers_map=None,
    max_points=200_000,
    max_voxels=30_000,
    point_size=2.0,
    voxel_wireframe=False,
    color_by="z",          # "z" | "hash" | "uniform"
    voxel_gap_ratio=0.08,  # fraction of voxel size removed on each edge (0..0.4)
    edge_overlay=True,     # draw thin LineSet edges around voxels
):
    """
    frames_map: list of (N_i,3) MAP-frame points (np.float32)
    cam_centers_map: list of (3,) MAP-frame camera centers
    vox: SparseVoxelGrid (same MAP frame)

    color_by:
        "z"     -> color ramp by world Z (blue->cyan->yellow)
        "hash"  -> pseudo-random stable color per IJK
        "uniform" -> single bluish color
    voxel_gap_ratio: 0 disables gaps, 0.05–0.15 is usually good.
    """
    import numpy as np, open3d as o3d

    def _hash_color(ijk):
        # deterministic pseudo-random color in [0,1]
        i, j, k = np.asarray(ijk, dtype=np.int64)
        h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791)
        r = ((h >>  0) & 255) / 255.0
        g = ((h >>  8) & 255) / 255.0
        b = ((h >> 16) & 255) / 255.0
        # brighten a bit
        return 0.15 + 0.85 * np.array([r, g, b])

    def _z_color(z, zmin, zmax):
        # simple 3-stop ramp: blue -> cyan -> yellow
        if zmax <= zmin:
            t = 0.5
        else:
            t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
        # piecewise: [0,0.5]: blue->cyan, [0.5,1]: cyan->yellow
        if t < 0.5:
            u = t / 0.5
            return np.array([0.0*(1-u) + 0.0*u, 1.0*u, 1.0])
        else:
            u = (t - 0.5) / 0.5
            return np.array([0.0*(1-u) + 1.0*u, 1.0, 1.0*(1-u) + 0.0*u])

    geoms = []

    # 1) point cloud (light gray)
    P = []
    for P_i in frames_map:
        if P_i.size:
            P.append(P_i[~np.isnan(P_i).any(axis=1)])
    if P:
        P = np.concatenate(P, axis=0).reshape(-1, 3)
        if P.shape[0] > max_points:
            idx = np.random.choice(P.shape[0], max_points, replace=False)
            P = P[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        col = np.full((P.shape[0], 3), 0.82, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(col)
        geoms.append(pcd)

    # 2) voxels as shrunken boxes with colors
    occ = vox.occupied_voxels()
    if len(occ) > max_voxels:
        occ = [occ[i] for i in np.random.choice(len(occ), max_voxels, replace=False)]

    if len(occ) > 0:
        vs = float(vox.p.voxel_size)
        shrink = max(0.0, min(0.4, voxel_gap_ratio))   # keep sane
        scale = 1.0 - shrink                           # uniform shrink factor
        big_mesh = o3d.geometry.TriangleMesh()
        edge_lines = []   # for LineSet overlay
        edge_pts = []

        # precompute z range for coloring
        if color_by == "z":
            centers = np.array([vox.ijk_to_center(ijk).astype(np.float64) for ijk in occ])
            zmin, zmax = float(centers[:, 2].min()), float(centers[:, 2].max())

        for ijk in occ:
            c = vox.ijk_to_center(ijk).astype(np.float64)

            box = o3d.geometry.TriangleMesh.create_box(width=vs, height=vs, depth=vs)
            box.translate(c - vs/2.0)
            # shrink around center to create visual gaps between neighbors
            box.scale(scale, center=c)

            # per-voxel color
            if color_by == "hash":
                col = _hash_color(ijk)
            elif color_by == "z":
                col = _z_color(c[2], zmin, zmax)
            else:
                col = np.array([0.2, 0.6, 1.0])  # uniform bluish

            box.paint_uniform_color(col.tolist())
            box.compute_vertex_normals()
            big_mesh += box

            if edge_overlay or voxel_wireframe:
                # collect 12 edges of the box for a LineSet overlay
                # corners of an axis-aligned box are available after translate/scale
                v = np.asarray(box.vertices)
                # index the 12 edges (pairs of vertex indices for open3d's create_box layout)
                edges = [(0,1),(1,3),(2,3),(0,2), (4,5),(5,7),(6,7),(4,6), (0,4),(1,5),(2,6),(3,7)]
                base = len(edge_pts)
                edge_pts.extend(v.tolist())
                edge_lines.extend([(base+a, base+b) for a,b in edges])

        geoms.append(big_mesh)

        if edge_overlay or voxel_wireframe:
            if edge_pts:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.array(edge_pts)),
                    lines=o3d.utility.Vector2iVector(np.array(edge_lines, dtype=np.int32)),
                )
                # darker edges so voxel boundaries pop
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[0.05, 0.05, 0.05]]), (len(edge_lines), 1))
                )
                geoms.append(ls)

    # 3) optional camera centers
    if cam_centers_map:
        for C in cam_centers_map:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=float(vox.p.voxel_size)*0.25)
            sp.translate(np.asarray(C, dtype=np.float64))
            sp.paint_uniform_color([1.0, 0.2, 0.2])
            geoms.append(sp)

    # 4) origin axes
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0]))

    # Use a Visualizer to tweak render options (smoother points, back faces, etc.)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Points + Voxels (MAP frame)", width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) * 0.98  # very light gray helps edges pop

    vis.run()
    vis.destroy_window()

def save_bev_png(bev: np.ndarray, meta: dict, path: str = "bev.png"):
    """Save the BEV to a PNG with axes in meters."""
    H, W = bev.shape
    res = float(meta.get("resolution", 0.10))
    ox, oy = meta.get("origin_xy", (0.0, 0.0))
    extent = [ox, ox + W * res, oy, oy + H * res]
    plt.figure()
    plt.imshow(bev, origin="lower", extent=extent)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("BEV")
    plt.colorbar(label="occupancy value")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")

def save_bev(bev: np.ndarray, meta: dict, png_path="bev.png", npy_path="bev.npy", meta_path="bev_meta.json"):
    # keep your current PNG
    save_bev_png(bev, meta, png_path)
    # save raw values + metadata for evaluation
    np.save(npy_path, bev)
    import json, pathlib
    pathlib.Path(meta_path).write_text(json.dumps(meta, indent=2))


def export_occupied_voxels_as_ply(vox: SparseVoxelGrid, path: str = "voxels.ply", z_band: Tuple[float,float] = (0.0, 3.0)):
    """Write occupied voxel centers to an ASCII PLY (for MeshLab/CloudCompare)."""
    occ = vox.occupied_voxels(zmin=z_band[0], zmax=z_band[1])
    centers = np.array([vox.ijk_to_center(ijk) for ijk in occ], dtype=np.float32)
    N = centers.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in centers:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"Saved {N} voxel centers to {path}")


def _to_tuple3(x):
    """Accept scalar or len-3; return (sx,sy,sz) as floats."""
    if np.isscalar(x):
        x = float(x)
        return (x, x, x)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 1:
        v = float(arr.item())
        return (v, v, v)
    assert arr.size == 3, f"voxel_size must be scalar or len-3, got shape {arr.shape}"
    return (float(arr[0]), float(arr[1]), float(arr[2]))

def export_occupied_voxels(vox, ply_path="voxels.ply", npy_path="voxels_ijk.npy",
                           meta_path="voxels_meta.json", z_band=(0.0, 3.0)):
    """
    Works with TorchSparseVoxelGrid (and similar):
      - PLY of occupied voxel centers (for CloudCompare/MeshLab)
      - NPY of occupied (i,j,k) indices (for exact set ops)
      - JSON meta with origin + voxel size so you can align later
    """
    # --- gather occupied indices ---
    if hasattr(vox, "occupied_voxels"):
        occ_ijk = vox.occupied_voxels(zmin=z_band[0], zmax=z_band[1])  # list[tuple]
        occ_ijk = np.array(occ_ijk, dtype=np.int32)
    elif hasattr(vox, "occupied_ijk_numpy"):
        occ_ijk = np.array(vox.occupied_ijk_numpy(zmin=z_band[0], zmax=z_band[1]), dtype=np.int32)
    else:
        raise AttributeError("voxel grid has no occupied_ijk method")

    # --- compute centers ---
    if occ_ijk.size == 0:
        centers = np.zeros((0,3), dtype=np.float32)
    elif hasattr(vox, "ijk_to_center"):
        centers = np.array([vox.ijk_to_center(tuple(ijk)) for ijk in occ_ijk], dtype=np.float32)
    else:
        # fallback: center = origin + (ijk + 0.5) * voxel_size
        origin = np.asarray(vox.origin.detach().cpu().numpy(), dtype=np.float32).reshape(3)
        sx, sy, sz = _to_tuple3(getattr(vox.p, "voxel_size"))
        centers = origin[None, :] + (occ_ijk.astype(np.float32) + 0.5) * np.array([sx, sy, sz], dtype=np.float32)

    # --- write PLY (ASCII) ---
    N = centers.shape[0]
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in centers:
            f.write(f"{x} {y} {z}\n")

    # --- write NPY of indices ---
    np.save(npy_path, occ_ijk)

    # --- write JSON meta for alignment/eval ---
    origin_xyz = np.asarray(vox.origin.detach().cpu().numpy(), dtype=float).tolist()
    voxel_size_xyz = _to_tuple3(getattr(vox.p, "voxel_size"))
    meta = {
        "origin_xyz": tuple(origin_xyz),
        "voxel_size": tuple(voxel_size_xyz),
        "z_band": tuple(z_band),
        "index_order": "i,j,k",
        "class": type(vox).__name__,
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2))

    print(f"Saved {N} voxel centers → {ply_path}")
    print(f"Saved occupied IJK → {npy_path}")
    print(f"Saved meta → {meta_path}")


