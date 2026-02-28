"""
NeuralRecon-style baseline for the MAPP3R pipeline.

This module provides `NeuralReconVoxelGrid`, a drop-in replacement for
`LatentVoxelGrid` that follows the NeuralRecon [Sun et al., CVPR 2021]
architecture:

    Point cloud  →  voxelize (scatter)  →  sparse 3D convolution  →  GRU  →  decode

Key architectural differences from LatentVoxelGrid:
  - LatentVoxelGrid:  ball-query + Gaussian-weighted max-pool  →  fusion MLP  →  GRU
  - NeuralReconVoxelGrid:  scatter voxelize  →  3×3×3 sparse 3D conv  →  GRU

Both share the same pipeline (change detection, selective reconstruction,
cached projection); only the fusion rule differs.

Usage in train.py:
    from neuralrecon_baseline import NeuralReconVoxelGrid
    # Replace:  self.vox = LatentVoxelGrid(...)
    # With:     self.vox = NeuralReconVoxelGrid(...)
    # Everything else (training loop, loss, BEV projection) stays identical.

Reference:
    Sun et al., "NeuralRecon: Real-Time Coherent 3D Reconstruction from
    Monocular Video", CVPR 2021.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reuse existing infrastructure
from voxel.latent_voxel import (
    VoxelParams,
    FeatureProjector,
    LatentToOccupancyDecoder,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Sparse 3D convolution via neighbor gathering
# ---------------------------------------------------------------------------
class Sparse3DConvBlock(nn.Module):
    """
    A single sparse 3×3×3 convolution implemented via hash-table neighbor
    lookups.  No external sparse-conv library required.

    For each active voxel, we gather the latent vectors of its 26 neighbors
    (+ itself = 27 cells), reshape into a (1, C, 3, 3, 3) local volume, and
    apply a standard `nn.Conv3d(kernel_size=3, padding=0)`.

    This faithfully reproduces the spatial reasoning of NeuralRecon's sparse
    3D convolutions while operating on the same hash-based sparse grid used
    by the rest of the MAPP3R pipeline.
    """

    # Pre-computed 3×3×3 offsets (27 neighbors including self)
    _OFFSETS_3x3x3: torch.Tensor | None = None

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=0, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _get_offsets(device: torch.device) -> torch.Tensor:
        """Returns (27, 3) int32 offsets for 3×3×3 neighborhood."""
        if Sparse3DConvBlock._OFFSETS_3x3x3 is None or Sparse3DConvBlock._OFFSETS_3x3x3.device != device:
            rng = torch.arange(-1, 2, device=device, dtype=torch.int32)
            grid = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1)
            Sparse3DConvBlock._OFFSETS_3x3x3 = grid.reshape(-1, 3)  # (27, 3)
        return Sparse3DConvBlock._OFFSETS_3x3x3

    def forward(
        self,
        z: torch.Tensor,          # (M, C_in)  latent vectors for all active voxels
        keys: torch.Tensor,       # (M,)       int64 hash keys (sorted)
        unhash_fn,                 # callable:  keys → (M, 3) int ijk
        hash_fn,                   # callable:  (N, 3) ijk → (N,) int64 keys
    ) -> torch.Tensor:
        """Returns (M, C_out) updated features."""
        M, C = z.shape
        dev = z.device

        if M == 0:
            return torch.zeros(M, self.conv.out_channels, device=dev, dtype=z.dtype)

        offsets = self._get_offsets(dev)  # (27, 3)

        # 1. Compute ijk of all active voxels
        ijk = unhash_fn(keys).to(torch.int32)  # (M, 3)

        # 2. Compute neighbor ijk: (M, 27, 3)
        neighbor_ijk = ijk[:, None, :] + offsets[None, :, :]  # broadcast

        # 3. Hash neighbors and look up in sorted key table
        neighbor_keys = hash_fn(neighbor_ijk.reshape(-1, 3)).view(M, 27)  # (M, 27)
        idx = torch.searchsorted(keys, neighbor_keys)                       # (M, 27)
        idx_safe = idx.clamp(max=M - 1)
        hit = (keys[idx_safe] == neighbor_keys)                             # (M, 27) bool

        # 4. Gather features (zeros for missing neighbors)
        gathered = torch.zeros(M, 27, C, device=dev, dtype=z.dtype)
        # Flatten for index_select then reshape
        flat_idx = idx_safe.reshape(-1)                      # (M*27,)
        flat_feats = z[flat_idx].view(M, 27, C)              # (M, 27, C)
        gathered[hit] = flat_feats[hit]

        # 5. Reshape to (M, C, 3, 3, 3) local volumes
        volumes = gathered.permute(0, 2, 1).view(M, C, 3, 3, 3)  # (M, C, 3, 3, 3)

        # 6. Apply 3D conv (kernel_size=3, no padding → output is 1×1×1)
        out = self.conv(volumes)            # (M, C_out, 1, 1, 1)
        out = out.view(M, -1)              # (M, C_out)
        out = self.act(self.norm(out))      # LayerNorm works on (M, C_out) directly

        return out


class SparseConvStack(nn.Module):
    """
    Two-layer sparse 3D conv stack, mirroring NeuralRecon's per-level
    feature extraction before GRU fusion.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv1 = Sparse3DConvBlock(feature_dim, feature_dim)
        self.conv2 = Sparse3DConvBlock(feature_dim, feature_dim)

    def forward(self, z, keys, unhash_fn, hash_fn):
        h = self.conv1(z, keys, unhash_fn, hash_fn)
        h = self.conv2(h, keys, unhash_fn, hash_fn)
        return h


# ---------------------------------------------------------------------------
# Main baseline module
# ---------------------------------------------------------------------------
class NeuralReconVoxelGrid(nn.Module):
    """
    NeuralRecon-style learned voxel fusion.

    Drop-in replacement for `LatentVoxelGrid`.  Shares the same sparse
    hash-grid infrastructure, initialization, phantom-point generation,
    BEV projection, and decoder.  Only the *update rule* differs:

        LatentVoxelGrid:   ball-query agg  →  fusion MLP  →  GRU
        NeuralReconVoxelGrid:  scatter      →  sparse 3D conv  →  GRU

    This isolates the architectural comparison to the spatial feature
    extraction strategy, as required by the paper's experimental protocol.
    """

    def __init__(
        self,
        origin_xyz,
        params: VoxelParams,
        device="cpu",
        dtype=torch.float32,
        feature_dim: int = 64,
    ):
        super().__init__()

        # ---- sparse grid state (same as LatentVoxelGrid) ----
        self.register_buffer(
            "origin",
            torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3),
            persistent=True,
        )
        self.p = params
        self.device = self.origin.device
        self.dtype = dtype
        self.feature_dim = feature_dim

        # Voxel-aligned state tensors
        self._init_buffers()

        self.epoch: int = 0

        # ---- per-voxel latent memory ----
        self.z_latent = torch.zeros((0, feature_dim), dtype=dtype, device=device)

        # ---- NeuralRecon-style modules ----
        # 1. Scatter + 3D conv for spatial context (replaces ball-query agg)
        self.sparse_conv = SparseConvStack(feature_dim)

        # 2. GRU for temporal fusion (same role as in LatentVoxelGrid / NeuralRecon)
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=feature_dim)

        # 3. Occupancy decoder (identical to LatentVoxelGrid)
        self.decoder = LatentToOccupancyDecoder(feature_dim, cond=None)

        # 4. Free-space token (identical)
        self.free_token = nn.Parameter(torch.randn(1, feature_dim))

        # ---- BEV history ----
        self.prev_probs = None

        # ---- init weights ----
        self.apply(self._init_weights)
        nn.init.constant_(self.decoder.fc3.bias, 0.0)
        nn.init.zeros_(self.decoder.fc3.weight)

    # -----------------------------------------------------------------
    # Buffer management (mirrors LatentVoxelGrid exactly)
    # -----------------------------------------------------------------
    def _init_buffers(self):
        """Register all per-voxel state buffers."""
        def eb(name, shape, dt):
            self.register_buffer(name, torch.empty(shape, dtype=dt), persistent=True)

        eb("keys",            (0,), torch.int64)
        eb("vals_st",         (0,), self.dtype)
        eb("vals_lt",         (0,), self.dtype)
        eb("vals",            (0,), self.dtype)
        eb("hit_count",       (0,), torch.int32)
        eb("pos_occ_count",   (0,), torch.int32)
        eb("neg_free_count",  (0,), torch.int32)
        eb("last_occ_epoch",  (0,), torch.int32)
        eb("last_free_epoch", (0,), torch.int32)
        eb("view_bits",       (0,), torch.int32)
        eb("seen_occ_epoch",  (0,), torch.int32)
        eb("seen_view_bits_e",(0,), torch.int32)
        eb("occ_epoch_count", (0,), torch.int32)
        eb("view_bits_cum",   (0,), torch.int32)
        eb("lt_promoted_flag",(0,), torch.uint8)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRUCell):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    # -----------------------------------------------------------------
    # Hash utilities (identical to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def _world_to_ijk(self, pts: torch.Tensor) -> torch.Tensor:
        return torch.floor((pts - self.origin) / self.p.voxel_size).to(torch.int64)

    @staticmethod
    def _hash_ijk(ijk: torch.Tensor) -> torch.Tensor:
        off = 1 << 20
        i = ijk[..., 0] + off
        j = ijk[..., 1] + off
        k = ijk[..., 2] + off
        return (i << 42) ^ (j << 21) ^ k

    @staticmethod
    def _unhash_keys_static(keys: torch.Tensor) -> torch.Tensor:
        off = 1 << 20
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = (keys & ((1 << 21) - 1)) - off
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    def _unhash_keys(self, keys: torch.Tensor) -> torch.Tensor:
        return self._unhash_keys_static(keys)

    # -----------------------------------------------------------------
    # Sparse grid management (identical to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def _ensure_and_index(self, upd_keys: torch.Tensor) -> torch.Tensor:
        """Merge upd_keys into self.keys; grow all state tensors; return indices."""
        if upd_keys.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)

        all_keys = torch.cat([self.keys, upd_keys])
        uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        idx_old = inv[:n_old]
        idx_upd = inv[n_old:]

        if uk.numel() != n_old:
            def grow(src, fill=0):
                out = torch.zeros(uk.shape[0], dtype=src.dtype, device=src.device)
                if src.numel() > 0:
                    out[idx_old] = src
                if uk.shape[0] > n_old:
                    mask_new = torch.ones(uk.shape[0], dtype=torch.bool, device=src.device)
                    if src.numel() > 0:
                        mask_new[idx_old] = False
                    out[mask_new] = fill
                return out

            self.keys            = uk
            self.vals_st         = grow(self.vals_st, 0)
            self.vals_lt         = grow(self.vals_lt, 0)
            self.vals            = grow(self.vals, 0)
            self.hit_count       = grow(self.hit_count, 0)
            self.pos_occ_count   = grow(self.pos_occ_count, 0)
            self.neg_free_count  = grow(self.neg_free_count, 0)
            self.last_occ_epoch  = grow(self.last_occ_epoch, -2**31 + 1)
            self.last_free_epoch = grow(self.last_free_epoch, -2**31 + 1)
            self.view_bits       = grow(self.view_bits, 0)
            self.seen_occ_epoch  = grow(self.seen_occ_epoch, -2**31 + 1)
            self.seen_view_bits_e = grow(self.seen_view_bits_e, 0)
            self.occ_epoch_count = grow(self.occ_epoch_count, 0)
            self.view_bits_cum   = grow(self.view_bits_cum, 0)
            self.lt_promoted_flag = grow(self.lt_promoted_flag, 0)

            current_dtype = self.z_latent.dtype if self.z_latent.numel() > 0 else self.dtype
            z_new = torch.zeros((uk.shape[0], self.feature_dim), device=self.device, dtype=current_dtype)
            if self.z_latent.numel() > 0:
                z_new[idx_old] = self.z_latent
            self.z_latent = z_new

        return idx_upd

    # -----------------------------------------------------------------
    # Display / query helpers (identical to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def _display_vals(self) -> torch.Tensor:
        return torch.maximum(self.vals_st, self.vals_lt)

    def occupied_mask(self) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        lt_occ = self.vals_lt > self.p.occ_thresh
        st_occ = self.vals_st > (self.p.occ_thresh + self.p.st_margin)
        return lt_occ | st_occ

    def voxel_centers(self) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.zeros(0, 3, device=self.device, dtype=torch.float32)
        ijk = self._unhash_keys(self.keys).to(torch.float32)
        return self.origin + (ijk + 0.5) * self.p.voxel_size

    # -----------------------------------------------------------------
    # Epoch management (simplified — keeps only what the pipeline needs)
    # -----------------------------------------------------------------
    def next_epoch(self):
        self.epoch += 1

    # -----------------------------------------------------------------
    # Reset (identical interface to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def reset_state(self, origin_xyz=None):
        current_device = self.gru_cell.weight_hh.device
        self.device = current_device

        if origin_xyz is not None:
            self.origin = torch.as_tensor(
                origin_xyz, dtype=self.dtype, device=self.device
            ).reshape(3)

        def reset_buf(name, shape, dt):
            self.register_buffer(
                name,
                torch.zeros(shape, dtype=dt, device=current_device),
                persistent=True,
            )

        reset_buf("keys",            (0,), torch.int64)
        reset_buf("vals_st",         (0,), self.dtype)
        reset_buf("vals_lt",         (0,), self.dtype)
        reset_buf("vals",            (0,), self.dtype)
        reset_buf("hit_count",       (0,), torch.int32)
        reset_buf("pos_occ_count",   (0,), torch.int32)
        reset_buf("neg_free_count",  (0,), torch.int32)
        reset_buf("last_occ_epoch",  (0,), torch.int32)
        reset_buf("last_free_epoch", (0,), torch.int32)
        reset_buf("view_bits",       (0,), torch.int32)
        reset_buf("seen_occ_epoch",  (0,), torch.int32)
        reset_buf("seen_view_bits_e",(0,), torch.int32)
        reset_buf("occ_epoch_count", (0,), torch.int32)
        reset_buf("view_bits_cum",   (0,), torch.int32)
        reset_buf("lt_promoted_flag",(0,), torch.uint8)

        self.z_latent = torch.zeros(
            (0, self.feature_dim), dtype=self.dtype, device=current_device
        )
        self.epoch = 0
        self.prev_probs = None

    # -----------------------------------------------------------------
    # Phantom points (identical to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def generate_phantom_points(self, origins, terminations, n_samples=16):
        N = origins.shape[0]
        max_range = 0.85
        steps = torch.linspace(0, max_range, n_samples, device=self.device).unsqueeze(0)
        step_size = max_range / (n_samples - 1)
        noise = (torch.rand(N, n_samples, device=self.device) - 0.5) * step_size
        ratios = (steps + noise).clamp(0.0, max_range)
        vec = (terminations - origins).unsqueeze(1)
        phantom_pts = origins.unsqueeze(1) + vec * ratios.unsqueeze(-1)
        phantom_pts = phantom_pts.reshape(-1, 3)
        phantom_feats = self.free_token.expand(phantom_pts.shape[0], -1)
        return phantom_pts, phantom_feats

    # =================================================================
    # CORE: Initialization from full cloud (t=0)
    # =================================================================
    def initialize_latents_from_full_cloud(
        self,
        pts_world: torch.Tensor,
        f_pts: torch.Tensor,
        cam_centers: torch.Tensor,
        *,
        init_lt: bool = True,
        lt_level: float | None = None,
        carve_free: bool = True,
        z_clip: Tuple[float, float] | None = (-float("inf"), float("inf")),
        stride: int = 4,
        samples_per_voxel: float = 2.0,
        max_rays: int = 10000,
        max_range: float = 12.0,
    ):
        """
        Seed z_latent from the full initial point cloud.
        Logic: scatter features into voxels → sparse 3D conv → GRU step.
        Free-space voxels are identified via ray marching (same as LatentVoxelGrid).
        """
        dev, dt = self.device, self.dtype

        if pts_world.numel() == 0:
            return

        pts_world = pts_world.to(dev, dt)
        f_pts = f_pts.to(dev, dt)

        # --- sanitize ---
        if not torch.isfinite(f_pts).all():
            f_pts = torch.nan_to_num(f_pts, nan=0.0, posinf=0.0, neginf=0.0)
        finite = torch.isfinite(pts_world).all(dim=1) & torch.isfinite(cam_centers).all(dim=1)
        pts_world, cam_centers, f_pts = pts_world[finite], cam_centers[finite], f_pts[finite]
        if pts_world.numel() == 0:
            return

        if z_clip is not None:
            z0, z1 = z_clip
            keep = (pts_world[:, 2] >= z0) & (pts_world[:, 2] <= z1)
            pts_world, cam_centers, f_pts = pts_world[keep], cam_centers[keep], f_pts[keep]

        if max_range is not None and cam_centers is not None:
            cam_centers = cam_centers.to(dev, dt)
            d_all = torch.norm(pts_world - cam_centers, dim=1)
            keep = d_all <= max_range
            pts_world, f_pts, cam_centers = pts_world[keep], f_pts[keep], cam_centers[keep]

        # --- identify surface voxels ---
        ijk = self._world_to_ijk(pts_world)
        keys_surf = self._hash_ijk(ijk)

        # --- identify free-space voxels via ray marching ---
        keys_free = torch.zeros(0, dtype=torch.int64, device=dev)
        if carve_free and cam_centers is not None:
            cam_centers = cam_centers.to(dev, dt)
            P_sub, C_sub = pts_world[::stride], cam_centers[::stride]
            if max_rays is not None and P_sub.shape[0] > max_rays:
                perm = torch.randperm(P_sub.shape[0], device=dev)[:max_rays]
                P_sub, C_sub = P_sub[perm], C_sub[perm]

            if P_sub.numel() > 0:
                V = P_sub - C_sub
                seg_len = torch.linalg.norm(V, dim=1)
                steps = torch.clamp(
                    (seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32),
                    min=1,
                )
                max_steps = int(steps.max().item())
                base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
                t = base[None, :] / steps.to(dt)[:, None]
                mask = t < 1.0
                samples = C_sub.unsqueeze(1) + t.unsqueeze(-1) * V.unsqueeze(1)
                samples = samples[mask]
                if samples.numel() > 0:
                    ijk_free = self._world_to_ijk(samples)
                    keys_free_all = self._hash_ijk(ijk_free)
                    unique_walls = torch.unique(keys_surf)
                    keep = ~torch.isin(keys_free_all, unique_walls)
                    keys_free = torch.unique(keys_free_all[keep])

        # --- allocate all voxels ---
        all_keys = torch.cat([keys_free, keys_surf])
        self._ensure_and_index(all_keys)

        # A. Free-space init
        idx_free = torch.searchsorted(self.keys, keys_free)
        if idx_free.numel() > 0:
            self.z_latent.index_copy_(0, idx_free, self.free_token.expand(idx_free.shape[0], -1))

        # B. Scatter surface features via max-pool
        idx_surf_pts = torch.searchsorted(self.keys, keys_surf)
        unique_surf_idx = torch.unique(idx_surf_pts)

        u_all = torch.full((self.keys.shape[0], self.feature_dim), -1e9, device=dev, dtype=dt)
        idx_expanded = idx_surf_pts.unsqueeze(-1).expand(-1, self.feature_dim)
        u_all.scatter_reduce_(0, idx_expanded, f_pts, reduce="amax", include_self=True)

        u_surf = u_all[unique_surf_idx]
        u_surf = torch.where(u_surf <= -1e8, torch.zeros_like(u_surf), u_surf)

        # C. Apply sparse 3D conv for spatial context (NeuralRecon-style)
        # We temporarily write scattered features, run conv, then GRU
        self.z_latent[unique_surf_idx] = u_surf

        conv_out = self.sparse_conv(
            self.z_latent, self.keys, self._unhash_keys, self._hash_ijk
        )

        # D. GRU update on surface voxels
        z_prev = torch.zeros_like(self.z_latent[unique_surf_idx])  # first observation
        z_new = self.gru_cell(conv_out[unique_surf_idx], z_prev)
        if not torch.isfinite(z_new).all():
            z_new = torch.nan_to_num(z_new, nan=0.0, posinf=0.0, neginf=0.0)
        self.z_latent.index_copy_(0, unique_surf_idx, z_new.to(self.z_latent.dtype))

        # E. Initialize log-odds from decoder
        if init_lt:
            if lt_level is not None:
                p_occ = torch.full((self.keys.shape[0],), float(lt_level), device=dev, dtype=dt)
                self.vals_lt[:] = p_occ
                self.vals_st[:] = p_occ
            elif self.decoder is not None:
                logit = self.decoder(self.z_latent, None)
                self.vals_lt[:] = logit
                self.vals_st[:] = logit
            self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)
            self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)
            self.vals = self._display_vals()

    # =================================================================
    # CORE: Per-timestep update (t > 0)  — THE KEY DIFFERENCE
    # =================================================================
    def update_with_features(
        self,
        pts_world: torch.Tensor,   # (N, 3)
        f_pts: torch.Tensor,       # (N, D)
        radius: float = 0.25,      # unused (kept for API compat)
        **kwargs,
    ):
        """
        NeuralRecon-style update:
            1. Scatter-voxelize new points (max-pool per voxel)
            2. Sparse 3D convolution for spatial context
            3. GRU update on touched voxels

        This replaces LatentVoxelGrid's ball-query + Gaussian-weighted
        max-pool + fusion MLP pipeline.
        """
        if pts_world.numel() == 0 or self.keys.numel() == 0:
            return

        dev, dt = self.device, self.z_latent.dtype
        M = self.keys.shape[0]

        # --- 1. Scatter: assign points to voxels via max-pool ---
        ijk_pts = self._world_to_ijk(pts_world)
        keys_pts = self._hash_ijk(ijk_pts)
        unique_src_keys, inv = torch.unique(keys_pts, return_inverse=True)

        # Max-pool features per voxel
        f_src = torch.full(
            (unique_src_keys.size(0), self.feature_dim), -1e9, device=dev, dtype=torch.float32
        )
        f_src = torch.scatter_reduce(
            f_src, 0,
            inv.unsqueeze(-1).expand(-1, self.feature_dim),
            f_pts.float(), reduce="amax", include_self=True,
        )
        f_src = torch.where(f_src <= -1e8, torch.zeros_like(f_src), f_src)

        # Map source voxel keys to grid indices
        idx_in_grid = torch.searchsorted(self.keys, unique_src_keys)
        idx_in_grid = idx_in_grid.clamp(max=M - 1)
        valid = self.keys[idx_in_grid] == unique_src_keys
        idx_in_grid = idx_in_grid[valid]
        f_src = f_src[valid]

        if idx_in_grid.numel() == 0:
            return

        # Write scattered features into a temporary buffer
        obs_buf = torch.zeros_like(self.z_latent)
        obs_buf[idx_in_grid] = f_src.to(dt)

        # --- 2. Sparse 3D convolution (NeuralRecon's spatial reasoning) ---
        conv_out = self.sparse_conv(obs_buf, self.keys, self._unhash_keys, self._hash_ijk)

        # --- 3. GRU update on touched voxels ---
        idx_upd = idx_in_grid
        z_sel = self.z_latent[idx_upd]
        z_new = self.gru_cell(conv_out[idx_upd], z_sel)
        self.z_latent.index_copy_(0, idx_upd, torch.nan_to_num(z_new).to(dt))

    # -----------------------------------------------------------------
    # Decode occupancy (identical interface to LatentVoxelGrid)
    # -----------------------------------------------------------------
    def decode_occupancy(self, with_xyz_cond: bool = False) -> torch.Tensor:
        if self.z_latent.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=torch.float32)
        if with_xyz_cond:
            centers = self.voxel_centers()
            return self.decoder(self.z_latent, centers)
        return self.decoder(self.z_latent, None)

    # -----------------------------------------------------------------
    # BEV projection (identical to LatentVoxelGrid)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def to_bev(
        self,
        x_range,
        y_range,
        res_xy: float,
        z_min: float,
        z_max: float,
        agg: str = "max",
        with_xyz_cond: bool = False,
        occ_thresh: float = 0.5,
        vis_mode: str = "occupancy",
    ):
        x0, x1 = x_range
        y0, y1 = y_range
        Hx = int((x1 - x0) / res_xy)
        Hy = int((y1 - y0) / res_xy)

        prev_probs = self.prev_probs
        bev_probs = torch.full((Hy, Hx), -1.0, device=self.device, dtype=torch.float32)
        meta = {"x0": x0, "y0": y0, "res": res_xy, "width": Hx, "height": Hy}

        centers = self.voxel_centers()
        if centers.numel() == 0:
            if vis_mode == "motion":
                return np.zeros((Hy, Hx, 3), dtype=np.uint8), meta
            return np.full((Hy, Hx), -1, dtype=np.int8), meta

        z = centers[:, 2]
        z_mask = (z >= z_min) & (z <= z_max)

        if z_mask.any():
            centers_z = centers[z_mask]
            z_lat = self.z_latent[z_mask]
            probs = torch.sigmoid(self.decoder(z_lat, None)).to(bev_probs.dtype)

            gx = ((centers_z[:, 0] - x0) / res_xy).floor().to(torch.long)
            gy = ((centers_z[:, 1] - y0) / res_xy).floor().to(torch.long)
            keep = (gx >= 0) & (gx < Hx) & (gy >= 0) & (gy < Hy)
            gx, gy, probs = gx[keep], gy[keep], probs[keep]

            if probs.numel() > 0:
                idx = gy * Hx + gx
                bev_probs.view(-1).scatter_reduce_(0, idx, probs, reduce="amax", include_self=True)

        raw_probs = bev_probs.clone()
        raw_probs[raw_probs < 0] = 0.0
        self.prev_probs = raw_probs

        bev_np = bev_probs.cpu().numpy()

        if vis_mode == "motion":
            out_img = np.zeros((Hy, Hx, 3), dtype=np.uint8)
            curr_occ = bev_np > occ_thresh
            known = bev_np > -0.5
            if prev_probs is not None and prev_probs.shape == bev_probs.shape:
                prev_np = prev_probs.cpu().numpy()
                static_mask = curr_occ
                out_img[static_mask] = [200, 200, 200]
            else:
                out_img[curr_occ] = [200, 200, 200]
            return out_img, meta
        else:
            out_map = np.full_like(bev_np, -1, dtype=np.int8)
            known_mask = bev_np > -0.5
            out_map[(bev_np <= occ_thresh) & known_mask] = 0
            out_map[(bev_np > occ_thresh) & known_mask] = 100
            return out_map, meta