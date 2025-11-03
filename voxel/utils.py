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


def clone(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

def ensure_dir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    
def to_torch(x, device="cuda"):
    """Convert to PyTorch tensor if not already."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.float32)
    else:
        return torch.tensor(x, device=device, dtype=torch.float32)
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.stack([to_numpy(xx) for xx in x])

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

import math
import torch

def align_pointcloud_torch_fast(
    points_world_S_HW3,
    *,
    max_samples=500_000,
    ransac_iters=1000,
    inlier_dist=0.03,
    min_up_dot=0.8,
    floor_bottom_frac=0.30,
    z_tiebreak_q=0.20,
    seed=0,
    shortlist_k=8,           # keep only top-K planes across all batches for tiebreak
    point_chunk=400_000,     # process points in chunks to limit (N x V) memory
    cand_chunk=4096,         # process candidates in chunks when counting inliers
    use_amp=True,            # enable half/bfloat16 matmuls where safe
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    GPU-accelerated Z-up frame alignment with floor detection (optimized).

    Returns:
      R_w2m (3x3) torch.Tensor, t_w2m (3,) torch.Tensor, info dict
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    R_basic = torch.eye(3, dtype=torch.float32, device=device)

    # ---- Load / flatten / filter ----
    if not isinstance(points_world_S_HW3, torch.Tensor):
        WPTS = torch.from_numpy(
            torch.as_tensor(points_world_S_HW3).cpu().numpy()
        )  # defensive; you can just do torch.from_numpy(np.asarray(...,np.float32)) if you prefer
    else:
        WPTS = points_world_S_HW3

    WPTS = WPTS.to(device=device, dtype=torch.float32).view(-1, 3)
    valid = torch.isfinite(WPTS).all(dim=1)
    WPTS = WPTS[valid]
    if WPTS.numel() == 0:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "no_points"}

    # basic rotation (identity)
    P0 = WPTS @ R_basic.T

    # Downsample
    if P0.shape[0] > max_samples:
        idx = torch.randperm(P0.shape[0], device=device, generator=gen)[:max_samples]
        P0 = P0[idx]

    N = P0.shape[0]
    if N < 3:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "too_few_points"}

    # ---- Floor bias pool ----
    z = P0[:, 2]
    finite_z = torch.isfinite(z)
    if finite_z.any():
        z_thresh = torch.quantile(z[finite_z], floor_bottom_frac)
        low_mask = z <= z_thresh
        P_low = P0[low_mask]
        if P_low.shape[0] < 200:
            P_low = P0
    else:
        P_low = P0

    N_low = P_low.shape[0]
    if N_low < 3:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "too_few_points_after_filter"}

    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # ---- Vectorized RANSAC with shortlisting ----
    batch_size = min(ransac_iters, 10000)
    num_batches = (ransac_iters + batch_size - 1) // batch_size

    # Keep global shortlist of best planes by count (and later z-quantile)
    shortlist_counts = torch.empty(0, device=device, dtype=torch.int32)
    shortlist_normals = torch.empty(0, 3, device=device, dtype=torch.float32)
    shortlist_d = torch.empty(0, device=device, dtype=torch.float32)

    # AMP context for fast matmul
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.startswith("cuda")) else torch.cpu.amp.autocast
    amp_kwargs = dict(dtype=autocast_dtype) if (use_amp and device.startswith("cuda")) else {}

    for b in range(num_batches):
        cur = min(batch_size, ransac_iters - b * batch_size)

        # Sample triplets (with replacement). For fewer degenerate triplets,
        # you can sample without replacement per-candidate if N_low is large.
        idxs = torch.randint(0, N_low, (cur, 3), device=device, generator=gen)
        tri = P_low[idxs]                  # (cur, 3, 3)
        p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]

        # Normals from cross product
        v1 = p1 - p0
        v2 = p2 - p0
        normals = torch.cross(v1, v2, dim=1)            # (cur, 3)
        lengths = torch.linalg.vector_norm(normals, dim=1)  # (cur,)
        valid_normals = lengths > 1e-9
        if not valid_normals.any():
            continue

        normals = normals[valid_normals]
        p0_v = p0[valid_normals]
        lengths = lengths[valid_normals]

        normals = normals / lengths.unsqueeze(1)
        # Flip downward normals
        flip = normals[:, 2] < 0
        normals[flip] = -normals[flip]

        # Upward alignment filter
        up_dots = (normals @ up)                        # (V,)
        keep = up_dots >= min_up_dot
        if not keep.any():
            continue

        normals = normals[keep]                         # (V,3)
        p0_v = p0_v[keep]                               # (V,3)
        d_vals = -(normals * p0_v).sum(dim=1)           # (V,)

        V = normals.shape[0]
        if V == 0:
            continue

        # Count inliers for all candidates but do it in chunks to save memory
        counts = torch.zeros(V, device=device, dtype=torch.int32)

        # to avoid big (N x V) allocations, tile either points or candidates
        if N <= point_chunk:
            # Chunk over candidates
            with amp_ctx(**amp_kwargs):
                P0_cast = P0
                for start in range(0, V, cand_chunk):
                    end = min(start + cand_chunk, V)
                    n_blk = normals[start:end]              # (v,3)
                    d_blk = d_vals[start:end]               # (v,)
                    # (N,v) = (N,3) @ (3,v)
                    dist_blk = torch.abs(P0_cast @ n_blk.T + d_blk)
                    counts[start:end] = (dist_blk < inlier_dist).sum(dim=0).to(torch.int32)
                    # free intermediate
                    del dist_blk
        else:
            # Chunk over points
            with amp_ctx(**amp_kwargs):
                for pstart in range(0, N, point_chunk):
                    pend = min(pstart + point_chunk, N)
                    P_blk = P0[pstart:pend]                 # (n,3)
                    # Do all candidates at once for this point block, but possibly in candidate chunks
                    for cstart in range(0, V, cand_chunk):
                        cend = min(cstart + cand_chunk, V)
                        n_blk = normals[cstart:cend]        # (v,3)
                        d_blk = d_vals[cstart:cend]         # (v,)
                        dist_blk = torch.abs(P_blk @ n_blk.T + d_blk)   # (n,v)
                        counts[cstart:cend] += (dist_blk < inlier_dist).sum(dim=0).to(torch.int32)
                        del dist_blk

        # Merge into global shortlist (by count)
        if counts.numel() > 0:
            if counts.numel() <= shortlist_k:
                # just append
                shortlist_counts = torch.cat([shortlist_counts, counts])
                shortlist_normals = torch.cat([shortlist_normals, normals])
                shortlist_d = torch.cat([shortlist_d, d_vals])
            else:
                # local top-k
                c_vals, c_idx = torch.topk(counts, k=min(shortlist_k, counts.numel()), largest=True, sorted=False)
                shortlist_counts = torch.cat([shortlist_counts, c_vals])
                shortlist_normals = torch.cat([shortlist_normals, normals[c_idx]])
                shortlist_d = torch.cat([shortlist_d, d_vals[c_idx]])

            # keep only global top-k
            if shortlist_counts.numel() > shortlist_k:
                g_vals, g_idx = torch.topk(shortlist_counts, k=shortlist_k, largest=True, sorted=False)
                shortlist_counts = g_vals
                shortlist_normals = shortlist_normals[g_idx]
                shortlist_d = shortlist_d[g_idx]

    if shortlist_counts.numel() == 0:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "ransac_failed"}

    # ---- Final tiebreak on small shortlist: recompute precise z-quantile ----
    # For each shortlisted plane, compute inlier mask once and measure z-quantile.
    best_counts = -1
    best_zq = float("inf")
    best_n = None
    best_d = None
    best_mask = None

    # Recompute distances plane-by-plane (shortlist size is tiny, so this is cheap)
    for n_vec, d_val, cnt in zip(shortlist_normals, shortlist_d, shortlist_counts):
        # distances for all points (chunked if needed)
        inliers_total = 0
        z_vals = []

        if N <= point_chunk:
            dist = torch.abs(P0 @ n_vec.unsqueeze(1) + d_val)[:, 0]
            mask = dist < inlier_dist
            inliers_total = int(mask.sum().item())
            if inliers_total > 0:
                z_vals = P0[mask, 2]
        else:
            vals = []
            mcount = 0
            # chunk over points
            for pstart in range(0, N, point_chunk):
                pend = min(pstart + point_chunk, N)
                dist_blk = torch.abs(P0[pstart:pend] @ n_vec + d_val)
                mask_blk = dist_blk < inlier_dist
                mcount += int(mask_blk.sum().item())
                if mask_blk.any():
                    vals.append(P0[pstart:pend, 2][mask_blk])
            inliers_total = mcount
            if vals:
                z_vals = torch.cat(vals, dim=0)

        # Compute z-quantile (only if we have inliers)
        if inliers_total > 0:
            z_q = torch.quantile(z_vals, z_tiebreak_q).item()
        else:
            z_q = float("inf")

        better = (inliers_total > best_counts) or (
            best_counts > 0 and inliers_total >= int(0.98 * best_counts) and z_q < best_zq
        )
        if better:
            best_counts = inliers_total
            best_zq = z_q
            best_n = n_vec.clone()
            best_d = float(d_val.item())
            # Keep a mask for refinement
            if N <= point_chunk:
                best_mask = (torch.abs(P0 @ best_n.unsqueeze(1) + best_d)[:, 0] < inlier_dist)
            else:
                # recompute once fully if needed
                dm = torch.abs(P0 @ best_n.unsqueeze(1) + best_d)[:, 0]
                best_mask = (dm < inlier_dist)

    if best_n is None:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "tiebreak_failed"}

    # ---- Refine plane on inliers with PCA/SVD ----
    if best_mask is not None and int(best_mask.sum().item()) >= 50:
        Q = P0[best_mask]
        Qc = Q - Q.mean(dim=0, keepdim=True)
        # the normal is eigenvector of smallest eigenvalue
        # low-rank pca is faster for large N:
        U, S, Vh = torch.linalg.svd(Qc, full_matrices=False)  # (n,3) -> Vh is (3,3)
        n_ref = Vh[-1, :]  # last row is normal (smallest singular value)
        if n_ref[2] < 0:
            n_ref = -n_ref
        up_dot_ref = float(n_ref @ up)
        if up_dot_ref >= min_up_dot:
            d_ref = -float(n_ref @ Q.mean(dim=0))
            best_n = n_ref
            best_d = d_ref

    n = best_n
    d = best_d
    up_dot = float(n @ up)

    # Rodrigues (rotate n -> z)
    v = torch.cross(n, up)
    s = float(torch.linalg.vector_norm(v))
    c = up_dot

    if s < 1e-8:
        R_corr = torch.eye(3, dtype=torch.float32, device=device)
    else:
        vx = torch.tensor([[ 0.0,   -v[2].item(),  v[1].item()],
                           [ v[2].item(),  0.0,   -v[0].item()],
                           [-v[1].item(),  v[0].item(),  0.0]], dtype=torch.float32, device=device)
        R_corr = torch.eye(3, dtype=torch.float32, device=device) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

    R_w2m = R_corr @ R_basic
    t_w2m = torch.tensor([0.0, 0.0, d], dtype=torch.float32, device=device)

    info = {
        "ransac_inliers": int(best_counts),
        "inlier_dist": float(inlier_dist),
        "floor_bottom_frac": float(floor_bottom_frac),
        "z_tiebreak_q": float(best_zq),
        "angle_to_up_deg": float(torch.rad2deg(torch.arccos(torch.clamp(torch.tensor(up_dot), -1.0, 1.0))).item()),
        "shortlist_k": int(shortlist_k),
    }
    return R_w2m, t_w2m, info

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


def align_pointcloud_torch(
    points_world_S_HW3,
    *,
    max_samples=500_000,
    ransac_iters=1000,
    inlier_dist=0.03,
    min_up_dot=0.8,
    floor_bottom_frac=0.30,
    z_tiebreak_q=0.20,
    seed=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    GPU-accelerated Z-up frame alignment with floor detection.
    
    Returns:
      R_w2m (3x3) torch.Tensor, t_w2m (3,) torch.Tensor, info dict
    """
    torch.manual_seed(seed)
    
    # Basic rotation matrix
    R_basic = torch.eye(3, dtype=torch.float32, device=device)
    
    # Convert and flatten points
    if not isinstance(points_world_S_HW3, torch.Tensor):
        WPTS = torch.from_numpy(np.asarray(points_world_S_HW3, dtype=np.float32))
    else:
        WPTS = points_world_S_HW3.clone()
    
    WPTS = WPTS.to(device=device, dtype=torch.float32).reshape(-1, 3)
    
    # Remove NaNs
    valid_mask = torch.isfinite(WPTS).all(dim=1)
    WPTS = WPTS[valid_mask]
    
    if WPTS.numel() == 0:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "no_points"}
    
    # Apply basic rotation
    P0 = WPTS @ R_basic.T
    
    # Downsample if needed
    if P0.shape[0] > max_samples:
        indices = torch.randperm(P0.shape[0], device=device)[:max_samples]
        P0 = P0[indices]
    
    # ----- Floor bias: build low-z pool -----
    z = P0[:, 2]
    if torch.isfinite(z).any():
        z_thresh = torch.quantile(z[torch.isfinite(z)], floor_bottom_frac)
        low_mask = z <= z_thresh
        P_low = P0[low_mask]
        if P_low.shape[0] < 200:
            P_low = P0
    else:
        P_low = P0
    
    N = P_low.shape[0]
    if N < 3:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "too_few_points_after_filter"}
    
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    
    # ----- VECTORIZED RANSAC -----
    # Sample all triplets at once
    batch_size = min(ransac_iters, 10000)  # Process in batches to limit memory
    num_batches = (ransac_iters + batch_size - 1) // batch_size
    
    best_inliers = -1
    best_n = None
    best_d = None
    best_z_q = float('inf')
    best_idx = None
    
    for batch_idx in range(num_batches):
        current_batch = min(batch_size, ransac_iters - batch_idx * batch_size)
        
        # Sample triplets (current_batch, 3)
        indices = torch.randint(0, N, (current_batch, 3), device=device)
        
        # Get points (current_batch, 3, 3)
        triplets = P_low[indices]
        p0, p1, p2 = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        # Compute normals for all samples at once
        v1 = p1 - p0  # (current_batch, 3)
        v2 = p2 - p0  # (current_batch, 3)
        
        # Cross product: normals (current_batch, 3)
        normals = torch.cross(v1, v2, dim=1)
        
        # Normalize
        lengths = torch.norm(normals, dim=1, keepdim=True)  # (current_batch, 1)
        valid_normals = lengths.squeeze() > 1e-9
        
        if not valid_normals.any():
            continue
        
        normals = normals / (lengths + 1e-9)
        
        # Orient upward (flip if pointing down)
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] = -normals[flip_mask]
        
        # Check up alignment
        up_dots = normals @ up  # (current_batch,)
        horizontal_mask = up_dots >= min_up_dot
        
        # Combine validity checks
        valid_mask = valid_normals & horizontal_mask
        
        if not valid_mask.any():
            continue
        
        # Compute d values for valid planes
        d_values = -(normals * p0).sum(dim=1)  # (current_batch,)
        
        # Compute distances for ALL valid planes at once
        # distances: (current_batch, num_points)
        valid_normals_batch = normals[valid_mask]  # (V, 3)
        valid_d_batch = d_values[valid_mask]  # (V,)
        
        # Distance computation: |n·p + d| for all valid planes
        distances = torch.abs(P0 @ valid_normals_batch.T + valid_d_batch)  # (num_points, V)
        
        # Inlier masks
        inlier_masks = distances < inlier_dist  # (num_points, V)
        inlier_counts = inlier_masks.sum(dim=0)  # (V,)
        
        # Process each valid plane
        valid_indices = torch.where(valid_mask)[0]
        
        for i, (count, local_idx) in enumerate(zip(inlier_counts, valid_indices)):
            count_val = count.item()
            if count_val <= 0:
                continue
            
            # Get inlier z values for tiebreaking
            inlier_mask = inlier_masks[:, i]
            inlier_z = P0[inlier_mask, 2]
            
            if inlier_z.numel() > 0:
                z_q = torch.quantile(inlier_z, z_tiebreak_q).item()
            else:
                z_q = float('inf')
            
            # Check if better
            better = False
            if count_val > best_inliers:
                better = True
            elif best_inliers > 0 and count_val >= 0.98 * best_inliers and z_q < best_z_q:
                better = True
            
            if better:
                best_inliers = count_val
                best_n = normals[local_idx].clone()
                best_d = d_values[local_idx].item()
                best_z_q = z_q
                best_idx = inlier_mask.clone()
    
    if best_n is None:
        return R_basic, torch.zeros(3, dtype=torch.float32, device=device), {"reason": "ransac_failed"}
    
    # ----- Refine plane with least-squares -----
    if best_idx is not None and best_idx.sum() >= 50:
        Q = P0[best_idx]
        
        # PCA on centered points
        Qc = Q - Q.mean(dim=0, keepdim=True)
        C = (Qc.T @ Qc) / max(1, Q.shape[0] - 1)
        
        # Eigendecomposition (smallest eigenvector is normal)
        w, U = torch.linalg.eigh(C)
        n_ref = U[:, 0]  # Smallest eigenvector
        
        # Orient upward
        if n_ref[2] < 0:
            n_ref = -n_ref
        
        # Check if still horizontal
        up_dot_ref = (n_ref @ up).item()
        if up_dot_ref >= min_up_dot:
            d_ref = -(n_ref @ Q.mean(dim=0)).item()
            best_n = n_ref
            best_d = d_ref
    
    n = best_n
    d = best_d
    up_dot = (n @ up).item()
    
    # ----- Rodrigues rotation: n -> z -----
    v = torch.cross(n, up)
    s = torch.norm(v).item()
    c = up_dot
    
    if s < 1e-8:
        R_corr = torch.eye(3, dtype=torch.float32, device=device)
    else:
        # Skew-symmetric matrix
        vx = torch.zeros((3, 3), dtype=torch.float32, device=device)
        vx[0, 1] = -v[2]
        vx[0, 2] = v[1]
        vx[1, 0] = v[2]
        vx[1, 2] = -v[0]
        vx[2, 0] = -v[1]
        vx[2, 1] = v[0]
        
        R_corr = torch.eye(3, dtype=torch.float32, device=device) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    
    # Combined rotation
    R_w2m = R_corr @ R_basic
    
    # Translation to move floor to z=0
    t_w2m = torch.tensor([0.0, 0.0, d], dtype=torch.float32, device=device)
    
    info = {
        "ransac_inliers": best_inliers,
        "inlier_dist": float(inlier_dist),
        "floor_bottom_frac": float(floor_bottom_frac),
        "z_tiebreak_q": float(best_z_q),
        "angle_to_up_deg": float(np.degrees(np.arccos(np.clip(up_dot, -1, 1)))),
    }
    
    return R_w2m, t_w2m, info




def unproject_depth_torch(depth_hw, K_3x3):
    # depth_hw: (H, W), torch.float32, device=*; K_3x3: (3,3)
    depth = depth_hw
    K = K_3x3
    device = depth.device
    H, W = depth.shape

    key = (H, W, device)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs - cx) / fx * depth
    Y = (ys - cy) / fy * depth
    Z = depth
    return torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)



def cam_to_world_torch(P_cam_hw3: torch.Tensor, Twc_4x4: torch.Tensor) -> torch.Tensor:
    """
    Transform camera-space 3D points (H, W, 3) into world coordinates using a 4×4 extrinsic matrix.

    Args:
        P_cam_hw3: (..., H, W, 3) or (H, W, 3) torch tensor of 3D points in camera coordinates.
        Twc_4x4:   (4, 4) or (B, 4, 4) torch tensor(s) representing camera-to-world transform.

    Returns:
        P_world_hw3: torch.Tensor of shape (..., H, W, 3) with transformed points.
    """
    # Ensure torch.float32 and same device
    device = P_cam_hw3.device
    Twc_4x4 = Twc_4x4.to(device=device, dtype=torch.float32)
    P_cam_hw3 = P_cam_hw3.to(dtype=torch.float32)

    H, W, _ = P_cam_hw3.shape
    X = P_cam_hw3.reshape(-1, 3)

    # Add homogeneous coordinate (N, 4)
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype, device=device)
    Xh = torch.cat([X, ones], dim=1)

    # Apply transform (world = Twc * cam)
    Xw = (Xh @ Twc_4x4.T)[:, :3]

    # Reshape back
    return Xw.view(H, W, 3)


def build_frames_and_centers_vectorized(
    predictions: dict,
    *,
    POINTS: str = "world_points_from_depth",
    CONF: str = "world_points_conf",
    IMG: str = "images",
    FEAT: str = None,
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
    
    if FEAT is not None:
        F = to_numpy(predictions[FEAT]).astype(np.float32)   # (S,H,W,3)
        feat_dim = F.shape[-1]

    C = to_numpy(predictions.get(CONF, np.ones_like(P[...,0], dtype=np.float32))).astype(np.float32)  # (S,H,W)
    I = to_numpy(predictions[IMG]).astype(np.float32)
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
    I_flat = I.reshape(S, -1, 3)
    V_flat = valid.reshape(S, -1)        # (S, H*W)
    C_flat = C.reshape(S,-1)
    
    if FEAT is not None:
        F_flat = F.reshape(S, -1, feat_dim)

    frames_map = []
    conf_map = []
    images_map = []
    feat_map = []
    
    for f in range(S):
        conf_map.append(C_flat[f][V_flat[f]].astype(np.float32))
        if not np.any(V_flat[f]):
            frames_map.append(np.empty((0,3), dtype=np.float32))
            continue
        frames_map.append(P_flat[f][V_flat[f]].astype(np.float32))
        images_map.append(I_flat[f][V_flat[f]].astype(np.float32))
        
        if FEAT is not None:
            feat_map.append(F_flat[f][V_flat[f]].astype(np.float32))
        
    return frames_map, cam_centers_map, conf_map, images_map, feat_map if (FEAT is not None) else None , (S, H, W)


import torch
from typing import Optional
def build_frames_and_centers_vectorized_torch(
    predictions: dict,
    *,
    POINTS: str = "world_points_from_depth",
    CONF: str = "world_points_conf",
    EXTR_KEY: str = "extrinsic",
    threshold: float = 50.0,
    Rmw: Optional[torch.Tensor] = None,
    tmw: Optional[torch.Tensor] = None,
    z_clip_map: Optional[tuple[float, float]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_flat: bool = False,  # NEW: skip per-frame splitting
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], tuple[int, int, int]]:
    """
    Heavily optimized version - eliminates the per-frame loop bottleneck.
    
    Set return_flat=True to skip per-frame splitting entirely if you don't need it.
    """
    
    # --- Load tensors (minimize transfers) ---
    P = predictions[POINTS]
    if not isinstance(P, torch.Tensor):
        P = torch.from_numpy(P)
    P = P.to(device=device, dtype=torch.float32)  # (S, H, W, 3)
    
    if CONF in predictions:
        C = predictions[CONF]
        if not isinstance(C, torch.Tensor):
            C = torch.from_numpy(C)
        C = C.to(device=device, dtype=torch.float32)
    else:
        C = torch.ones(P.shape[:-1], dtype=torch.float32, device=device)
    
    EXTR = predictions[EXTR_KEY]
    if not isinstance(EXTR, torch.Tensor):
        EXTR = torch.from_numpy(EXTR)
    EXTR = EXTR.to(device=device, dtype=torch.float32)
    
    S, H, W = P.shape[:3]
    
    # --- Compute confidence threshold (optimized) ---
    if threshold == 0.0:
        conf_threshold = 0.0
    else:
        # Sample for percentile instead of using all values
        C_flat = C.reshape(-1)
        if C_flat.numel() > 1_000_000:
            # Random sample for large tensors
            indices = torch.randperm(C_flat.numel(), device=device)[:1_000_000]
            C_sample = C_flat[indices]
            C_finite = C_sample[torch.isfinite(C_sample)]
        else:
            C_finite = C_flat[torch.isfinite(C_flat)]
        
        if len(C_finite) > 0:
            conf_threshold = torch.quantile(C_finite, threshold / 100.0).item()
        else:
            conf_threshold = 0.0
    
    # --- Pad extrinsics if needed ---
    if EXTR.dim() == 3 and EXTR.shape[1:] == (3, 4):
        bottom = torch.tensor([[0, 0, 0, 1]], dtype=EXTR.dtype, device=device)
        bottom = bottom.unsqueeze(0).expand(S, -1, -1)
        EXTR = torch.cat([EXTR, bottom], dim=1)
    
    # --- Camera centers ---
    Cw = EXTR[:, :3, 3]  # (S, 3)
    
    # --- Validity mask (fully vectorized) ---
    finite_xyz = torch.isfinite(P).all(dim=-1)  # (S, H, W)
    finite_conf = torch.isfinite(C)
    conf_ok = (C >= conf_threshold) & (C > 1e-5)
    valid = finite_xyz & finite_conf & conf_ok
    
    if z_clip_map is not None:
        z0, z1 = z_clip_map
        z = P[..., 2]
        valid &= (z >= z0) & (z <= z1)
    
    # --- OPTIMIZED: Use cumsum for frame boundaries instead of loop ---
    P_flat = P.reshape(S, H * W, 3)  # (S, H*W, 3)
    C_flat = C.reshape(S, H * W)     # (S, H*W)
    V_flat = valid.reshape(S, H * W) # (S, H*W)
    
    if return_flat:
        # Skip per-frame splitting entirely - return concatenated results
        all_valid = V_flat.reshape(-1)
        P_all = P_flat.reshape(-1, 3)[all_valid]
        C_all = C_flat.reshape(-1)[all_valid]
        
        # Build frame indices for later use
        frame_ids = torch.arange(S, device=device)[:, None].expand(S, H * W).reshape(-1)[all_valid]
        
        return [P_all], [Cw], [C_all], (S, H, W), frame_ids
    
    # --- Optimized per-frame extraction using advanced indexing ---
    # Count valid points per frame
    counts = V_flat.sum(dim=1)  # (S,)
    
    # Create frame index tensor for gathering
    max_count = counts.max().item()
    
    if max_count == 0:
        # No valid points
        empty = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_conf = torch.empty(0, dtype=torch.float32, device=device)
        return ([empty] * S, [Cw[i] for i in range(S)], [empty_conf] * S, (S, H, W))
    
    # Batch gather using masked_select + split (faster than loop)
    frames_map = []
    conf_map = []
    cam_centers_map = []
    
    for f in range(S):
        mask = V_flat[f]
        count = counts[f].item()
        
        if count == 0:
            frames_map.append(torch.empty((0, 3), dtype=torch.float32, device=device))
            conf_map.append(torch.empty(0, dtype=torch.float32, device=device))
        else:
            # Use compress/masked_select (faster than boolean indexing for sparse masks)
            frames_map.append(P_flat[f][mask])
            conf_map.append(C_flat[f][mask])
        
        cam_centers_map.append(Cw[f])
    
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

# def build_maps_from_points_and_centers_torch(
#     frames_xyz: list[np.ndarray],          # list of (N_i,3) in MAP frame
#     cam_centers: list[np.ndarray],         # list of (3,) in MAP frame
#     conf_map: list[np.ndarray],         
#     tvox,
#     *,
#     align_to_voxel=False,
#     voxel_size: float = 0.10,
#     bev_window_m=(20.0, 20.0),
#     bev_origin_xy=(-10.0, -10.0),
#     z_clip_vox=(-np.inf, np.inf),
#     z_band_bev=(0.05, 1.80),
#     max_range_m: float | None = 12.0,
#     carve_free: bool = True,
#     # new (optional) tuning/throughput controls:
#     device: str = "cuda",
#     samples_per_voxel: float = 1.1,
#     ray_stride: int = 1,
#     max_free_rays: int | None = 200_000,
#     batch_chunk_points: int | None = None,  # e.g., 2_000_000 to limit RAM
# ):

#     # 1) Concatenate all frames once (no per-frame integrate calls)
#     #    Build per-point camera centers by repeating each frame center.
#     valid_frames = [(np.asarray(P, np.float32), np.asarray(C, np.float32).reshape(3), np.asarray(CONF, np.float32))
#                     for P, C, CONF in zip(frames_xyz, cam_centers, conf_map)
#                     if (P is not None and P.size and C is not None)]
    
#     print("1")
#     if not valid_frames:
#         bev_spec = BevSpec(resolution=voxel_size, width_m=float(bev_window_m[0]),
#                            height_m=float(bev_window_m[1]), origin_xy=bev_origin_xy, z_band=z_band_bev)
#         bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
#         return tvox, bev, meta

#     print("2")

#     lengths = [pf.shape[0] for pf, _, _ in valid_frames]
#     P_all = np.concatenate([pf for pf, _, _ in valid_frames], axis=0)              # (M,3)
#     C_all = np.concatenate([np.repeat(cf[None, :], n, axis=0)
#                             for n, (_, cf, _) in zip(lengths, valid_frames)], axis=0)  # (M,3)
#     CONF_all = np.concatenate([c for _, _, c in valid_frames], axis=0)              # (M,3)

#     print("3")

#     # drop NaNs early (keeps alignment)
#     keep = np.isfinite(P_all).all(axis=1) & np.isfinite(C_all).all(axis=1)
#     P_all = P_all[keep]; C_all = C_all[keep]; CONF_all = CONF_all[keep]
    
#     print("4")

#     if align_to_voxel:
#         _, pts, cameras = sim3_align_to_voxel(
#             pts_world=torch.from_numpy(P_all),
#             cams_world=torch.from_numpy(C_all),
#             conf_world=torch.from_numpy(CONF_all),
#             grid=tvox,
#             cfg=AlignCfg(
#                 iters=5,
#                 downsample_points=50000,
#                 pad_vox=48,
#                 max_block_vox=192,
#                 samples_per_ray=16,
#                 lambda_free=0.1,
#                 lambda_prior_t=1e-4,
#                 lambda_prior_r=1e-4,
#                 lambda_prior_s=1e-4,
#                 step=2e-1,
#                 use_sim3=True,
#                 rot_clamp_deg=10.0,
#                 trans_clamp_m=5*voxel_size,
#                 scale_clamp=0.1,
#                 conf_filter=True,
#                 conf_quantile=0.3,
#             )
#         )
        
#         # _, pts, cameras = sim3_icp_align_to_voxel(
#         #     pts_world=torch.from_numpy(P_all),
#         #     cams_world=torch.from_numpy(C_all),
#         #     grid=tvox,                         # your TorchSparseVoxelGrid
#         #     conf_world=torch.from_numpy(CONF_all),
#         #     cfg=AlignCfg(
#         #         iters=5,
#         #         downsample_points=50000,
#         #         pad_vox=48,
#         #         max_block_vox=192,
#         #         samples_per_ray=16,
#         #         lambda_free=0.1,
#         #         lambda_prior_t=1e-4,
#         #         lambda_prior_r=1e-4,
#         #         lambda_prior_s=1e-4,
#         #         step=2e-1,
#         #         use_sim3=True,
#         #         rot_clamp_deg=10.0,
#         #         trans_clamp_m=5*voxel_size,
#         #         scale_clamp=0.1,
#         #         conf_filter=True,
#         #         conf_quantile=0.3,
#         #         icp_max_iters=50,
#         #         icp_trim_fraction=0.7,
#         #         icp_huber_delta=0.03,         # ~3 cm
#         #         icp_dist_thresh=0.05,         # 20 cm cutoff early
#         #         icp_nn_subsample=50000,
#         #         icp_max_map_points=50000,
#         #     )
#         # )

#         # _, pts, cameras = sim3_align_to_voxel_robust(
#         #     pts_world=torch.from_numpy(P_all),
#         #     cams_world=torch.from_numpy(C_all),
#         #     conf_world=torch.from_numpy(CONF_all),
#         #     grid=tvox,
#         #     cfg=AlignCfg(
#         #         iters=12,
#         #         downsample_points=30000,
#         #         pad_vox=24,
#         #         max_block_vox=192,
#         #         samples_per_ray=16,
#         #         lambda_free_schedule=[0.001],
#         #         lambda_prior_t=1e-4,
#         #         lambda_prior_r=1e-4,
#         #         lambda_prior_s=1e-4,
#         #         step=2e-1,
#         #         use_sim3=True,
#         #         rot_clamp_deg=10.0,
#         #         trans_clamp_m=2*voxel_size,#0.5*voxel_size,
#         #         scale_clamp=0.1,
#         #         sigmas=[3.0, 2.0, 1.0],
#         #         sdf_mode="logit",
#         #         free_margin_m=2.0*voxel_size,
#         #         free_step_m=0.5*voxel_size,
#         #         lambda_norm=0.1,
#         #         conf_filter=True,
#         #         conf_quantile=0.1,
#         #         huber_delta=0.05,
#         #         bootstrap_nn_max_m=0.05*voxel_size,
#         #         bootstrap_icp_iters=2
#         #     )
#         # )
#     else:
#         pts, cameras = torch.from_numpy(P_all).to(device), torch.from_numpy(C_all).to(device),


#     print(pts.shape)
#     # 2) Integrate once (optionally in chunks to cap memory)
#     if batch_chunk_points is None or P_all.shape[0] <= batch_chunk_points:
#         tvox.integrate_points_with_cameras(
#             pts,
#             cameras,
#             carve_free=carve_free,
#             max_range=max_range_m,
#             z_clip=z_clip_vox,
#             samples_per_voxel=samples_per_voxel,
#             ray_stride=ray_stride,
#             max_free_rays=max_free_rays,
#         )
#         print("INTEGRATED")
#     else:
#         m = P_all.shape[0]
#         for s in range(0, m, batch_chunk_points):
#             e = min(s + batch_chunk_points, m)
#             tvox.integrate_points_with_cameras(
#                 pts[s:e],
#                 cameras[s:e],
#                 carve_free=carve_free,
#                 max_range=max_range_m,
#                 z_clip=z_clip_vox,
#                 samples_per_voxel=samples_per_voxel,
#                 ray_stride=ray_stride,
#                 max_free_rays=max_free_rays,
#             )


#     # 3) BEV (reuse your existing rasterizer via adapter)
#     #vox = TorchVoxelAdapter(tvox)
#     bev_spec = BevSpec(
#         resolution=voxel_size,
#         width_m=float(bev_window_m[0]),
#         height_m=float(bev_window_m[1]),
#         origin_xy=bev_origin_xy,
#         z_band=z_band_bev,
#     )
#     bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
#     return tvox, bev, meta


def build_maps_from_points_and_centers_torch(
    frames_xyz: list[torch.Tensor],          # list of (N_i,3) in MAP frame
    cam_centers: list[torch.Tensor],         # list of (3,) in MAP frame
    conf_map: list[torch.Tensor],         
    tvox,
    *,
    align_to_voxel=False,
    voxel_size: float = 0.10,
    bev_window_m=(20.0, 20.0),
    bev_origin_xy=(-10.0, -10.0),
    z_clip_vox=(-float('inf'), float('inf')),
    z_band_bev=(0.05, 1.80),
    max_range_m: float | None = 12.0,
    carve_free: bool = True,
    # new (optional) tuning/throughput controls:
    device: str = "cuda",
    samples_per_voxel: float = 1.1,
    ray_stride: int = 1,
    max_free_rays: int | None = 200_000,
    batch_chunk_points: int | None = None,  # e.g., 2_000_000 to limit RAM
    frame_ids: Optional[torch.Tensor] = None,  # NEW

):
    """
    Optimized version that works directly with PyTorch tensors.
    """
    if frame_ids is not None:
        # All points are already concatenated
        pts = frames_xyz[0]
        # Expand camera centers by frame_ids
        cameras = cam_centers[0][frame_ids]  # Smart indexing
        CONF_all = conf_map[0]
        
        # Skip all the concatenation logic
        keep = torch.isfinite(pts).all(dim=1) & torch.isfinite(cameras).all(dim=1) & torch.isfinite(CONF_all)
        pts = pts[keep]
        cameras = cameras[keep]
        CONF_all = CONF_all[keep]
    else:
        # 1) Filter valid frames and ensure tensors are on the correct device
        valid_frames = []
        for P, C, CONF in zip(frames_xyz, cam_centers, conf_map):
            if P is not None and P.numel() > 0 and C is not None:
                P_dev = P.to(device=device, dtype=torch.float32)
                C_dev = C.to(device=device, dtype=torch.float32).reshape(3)
                CONF_dev = CONF.to(device=device, dtype=torch.float32)
                valid_frames.append((P_dev, C_dev, CONF_dev))
        
        print("1")
        if not valid_frames:
            bev_spec = BevSpec(
                resolution=voxel_size, 
                width_m=float(bev_window_m[0]),
                height_m=float(bev_window_m[1]), 
                origin_xy=bev_origin_xy, 
                z_band=z_band_bev
            )
            bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
            return tvox, bev, meta

        print("2")

        # 2) Concatenate all frames at once
        lengths = [pf.shape[0] for pf, _, _ in valid_frames]
        P_all = torch.cat([pf for pf, _, _ in valid_frames], dim=0)  # (M, 3)
        
        # Repeat camera centers for each point in their respective frames
        C_all = torch.cat([
            cf.unsqueeze(0).expand(n, -1) 
            for n, (_, cf, _) in zip(lengths, valid_frames)
        ], dim=0)  # (M, 3)
        
        CONF_all = torch.cat([c for _, _, c in valid_frames], dim=0)  # (M,)

        print("3")

        # 3) Drop NaNs/Infs early (keeps alignment)
        keep = torch.isfinite(P_all).all(dim=1) & torch.isfinite(C_all).all(dim=1) & torch.isfinite(CONF_all)
        P_all = P_all[keep]
        C_all = C_all[keep]
        CONF_all = CONF_all[keep]
        
        print("4")

        # 4) Optional alignment
        if align_to_voxel:
            _, pts, cameras = sim3_align_to_voxel(
                pts_world=P_all,
                cams_world=C_all,
                conf_world=CONF_all,
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
        else:
            pts = P_all
            cameras = C_all

    print(f"Points shape: {pts.shape}")
    
    # 5) Integrate once (optionally in chunks to cap memory)
    num_points = pts.shape[0]
    if batch_chunk_points is None or num_points <= batch_chunk_points:
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
        for s in range(0, num_points, batch_chunk_points):
            e = min(s + batch_chunk_points, num_points)
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
            print(f"INTEGRATED chunk {s//batch_chunk_points + 1}")

    # 6) Generate BEV
    bev_spec = BevSpec(
        resolution=voxel_size,
        width_m=float(bev_window_m[0]),
        height_m=float(bev_window_m[1]),
        origin_xy=bev_origin_xy,
        z_band=z_band_bev,
    )
    bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
    return tvox, bev, meta



def build_maps_from_latent_features(
    i,
    frames_xyz: list[np.ndarray],          # list of (N_i,3) in MAP frame
    cam_centers: list[np.ndarray],         # list of (3,) in MAP frame
    conf_map: list[np.ndarray],         
    features: list[np.ndarray],
    tvox,
    *,
    align_to_voxel=False,
    voxel_size: float = 0.10,
    bev_window_m=(20.0, 20.0),
    bev_origin_xy=(-10.0, -10.0),
    z_clip_vox=(-np.inf, np.inf),
    z_band_bev=(0.05, 1.80),
 
    # new (optional) tuning/throughput controls:
    device: str = "cpu",

    batch_chunk_points: int | None = None,  # e.g., 2_000_000 to limit RAM
):

    # 1) Concatenate all frames once (no per-frame integrate calls)
    #    Build per-point camera centers by repeating each frame center.
    valid_frames = [(np.asarray(P, np.float32), np.asarray(C, np.float32).reshape(3), np.asarray(CONF, np.float32), np.asarray(F, np.float32))
                    for P, C, CONF, F in zip(frames_xyz, cam_centers, conf_map, features)
                    if (P is not None and P.size and C is not None)]
    
    if not valid_frames:
        bev_spec = BevSpec(resolution=voxel_size, width_m=float(bev_window_m[0]),
                           height_m=float(bev_window_m[1]), origin_xy=bev_origin_xy, z_band=z_band_bev)
        bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
        return tvox, bev, meta


    lengths = [pf.shape[0] for pf, _, _,_ in valid_frames]
    P_all = np.concatenate([pf for pf, _, _,_ in valid_frames], axis=0)              # (M,3)
    C_all = np.concatenate([np.repeat(cf[None, :], n, axis=0)
                            for n, (_, cf, _,_) in zip(lengths, valid_frames)], axis=0)  # (M,3)
    CONF_all = np.concatenate([c for _, _, c,_ in valid_frames], axis=0)              # (M,3)
    F_all = np.concatenate([f for _, _, _,f in valid_frames], axis=0)              # (M,3)


    # drop NaNs early (keeps alignment)
    keep = np.isfinite(P_all).all(axis=1) & np.isfinite(C_all).all(axis=1)
    P_all = P_all[keep]; C_all = C_all[keep]; CONF_all = CONF_all[keep]
    

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
        
    else:
        pts, fts = torch.from_numpy(P_all).to(device), torch.from_numpy(F_all).to(device),


    print(pts.shape)
    # 2) Integrate once (optionally in chunks to cap memory)
    if i == 0:
        # Initialize voxel latents + (optionally) occupancy
        tvox.initialize_latents_from_full_cloud(
            pts_world=pts, f_pts=fts)
    else:
        radius = 0.25
        if batch_chunk_points is None or pts.shape[0] <= batch_chunk_points:
            tvox.update_with_features(
                                pts,  # (N,3)
                                fts,      # (N,D)
                                radius=radius)
            print("INTEGRATED")
        else:
            m = pts.shape[0]
            for s in range(0, m, batch_chunk_points):
                e = min(s + batch_chunk_points, m)
                tvox.update_with_features(
                            pts[s:e],  # (N,3)
                            fts[s:e],      # (N,D)
                            radius=radius)

    
    # Prepare BEV ranges for the to_bev() call
    x0, y0 = float(bev_origin_xy[0]), float(bev_origin_xy[1])
    W, H = float(bev_window_m[0]), float(bev_window_m[1])
    x_range = (x0, x0 + W)
    y_range = (y0, y0 + H)
    res_xy = float(voxel_size)
    z_min, z_max = float(z_band_bev[0]), float(z_band_bev[1])

    bev, meta = tvox.to_bev(
        x_range=x_range,
        y_range=y_range,
        res_xy=res_xy,
        z_min=z_min,
        z_max=z_max,
        agg="max",
        with_xyz_cond=False,
    )
    # bev, meta = bev_from_voxels(tvox, bev_spec, include_free=True)
    return tvox, bev, meta

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
    """
    Apply rotation and translation to points.
    Handles both NumPy and PyTorch inputs automatically.
    
    Args:
        points: array/tensor of shape (..., 3)
        R: rotation matrix (3, 3)
        t: translation vector (3,)
    
    Returns:
        Transformed points in the same format as input
    """
    # Detect input type
    is_numpy = isinstance(points, np.ndarray)
    is_torch_R = isinstance(R, torch.Tensor)
    is_torch_t = isinstance(t, torch.Tensor)
    
    # Convert everything to the same type as points
    if is_numpy:
        # Convert PyTorch to NumPy if needed
        if is_torch_R:
            R = R.cpu().numpy()
        if is_torch_t:
            t = t.cpu().numpy()
        
        # Ensure correct types
        R = np.asarray(R, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        
        # Apply transformation
        result = points @ R.T + t
        
    else:
        # Points are PyTorch tensors
        device = points.device
        
        # Convert NumPy to PyTorch if needed
        if not is_torch_R:
            R = torch.from_numpy(np.asarray(R, dtype=np.float32)).to(device)
        else:
            R = R.to(device)
            
        if not is_torch_t:
            t = torch.from_numpy(np.asarray(t, dtype=np.float32)).to(device)
        else:
            t = t.to(device)
        
        # Apply transformation
        result = points @ R.T + t
    
    return result


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


