from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np


# ----------------- Params -----------------
@dataclass
class VoxelParams:
    # Geometry
    voxel_size: float = 0.10  # [m], 10 cm

    # Log-odds (base steps)
    occ_inc: float = +0.5
    free_inc: float = -0.5
    l_min: float = -2.0
    l_max: float = +3.5
    occ_thresh: float = 0.0

    # Numerics
    endpoint_eps: float = 1e-4

    # ---------- L1/L2 cache & hysteresis ----------
    st_margin: float = 0.20
    promote_hits: int = 3
    lt_occ_scale: float = 0.5
    lt_free_scale: float = 0.5

    # LT carving (slow free) conditions (used only if lt_allow_free=True)
    lt_free_k_neg: int = 4
    lt_free_recent_occ_epochs: int = 2

    # ST decay
    st_decay_gamma: float = 0.0  # 0.0 -> ST clears every epoch

    # ---------- NEW: Epoch-based LT promotion ----------
    promote_epochs: int = 5
    lt_min_view_sectors: int = 1
    lt_promotion_mode: str = "once"   # "once" or "accumulate"
    lt_promote_value: float = 0.8

    # ---------- NEW: LT carving policy ----------
    lt_allow_free: bool = True

    # ---------- NEW: LT demotion (epoch-based) ----------
    lt_demote_enable: bool = True
    lt_demote_k_neg: int = 10
    lt_demote_min_no_occ_epochs: int = 3
    lt_demote_step: float = 0.5
    lt_demote_floor: float = 0.0
    lt_reset_promotion_on_demote: bool = True


# ----------------- Similarity (point->voxel) -----------------
class FeatureVoxelSimilarity(nn.Module):
    """
    Small MLP to score compatibility of a point feature f_i and a voxel latent z_j,
    conditioned on relative offset delta_xyz = p_i - center(v_j).
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, f_pts_flat: torch.Tensor, z_vox_flat: torch.Tensor, delta_xyz_flat: torch.Tensor):
        """
        Flat/batched mode for efficiency.
        Args:
            f_pts_flat:   (R, D)
            z_vox_flat:   (R, D)
            delta_xyz_flat: (R, 3)
        Returns:
            sim_flat: (R,) unnormalized scores
        """
        x = torch.cat([f_pts_flat, z_vox_flat, delta_xyz_flat], dim=-1)  # (R, 2D+3)
        return self.net(x).squeeze(-1)  # (R,)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------- (A) latent -> occupancy decoder -----------------
class LatentToOccupancyDecoder(nn.Module):
    """
    Decode a voxel latent z_j (optionally conditioned on the voxel center)
    into an occupancy probability in [0,1].

    Two modes:
      - cond=None: p = sigma(MLP(z))
      - cond='xyz': p = sigma(MLP([z, pos_enc(center_xyz)]))
    """
    def __init__(self, latent_dim: int, hidden: int = 96, cond: str | None = None,
                 xyz_pe_bands: int = 4):
        super().__init__()
        self.cond = cond
        in_dim = latent_dim

        if cond == 'xyz':
            # simple Fourier positional encoding of centers (x,y,z)
            self.xyz_pe_bands = xyz_pe_bands
            pe_dim = 3 * 2 * xyz_pe_bands
            in_dim = latent_dim + pe_dim
        elif cond is None:
            pass
        else:
            raise ValueError("cond must be None or 'xyz'")

        # small but expressive head (LN + two residual blocks)
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def _fourier_pe(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: (..., 3) in meters. Returns (..., 3*2*B).
        """
        B = self.xyz_pe_bands
        freqs = xyz.new_tensor([2.0**k * math.pi for k in range(B)])  # (B,)
        # (..., 3, B)
        ang = xyz[..., None, :] * freqs[None, :, None]  # broadcast to (..., B, 3)
        ang = ang.movedim(-3, -1)  # (..., 3, B)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (..., 3, 2B)
        return pe.reshape(*xyz.shape[:-1], -1)

    def forward(self, z: torch.Tensor, centers_xyz: torch.Tensor | None = None) -> torch.Tensor:
        """
        z: (M, Dz)
        centers_xyz: (M,3) if cond='xyz', else None
        returns: (M,) occupancy probs
        """
        if self.cond == 'xyz':
            assert centers_xyz is not None, "centers_xyz required when cond='xyz'"
            pe = self._fourier_pe(centers_xyz)
            x = torch.cat([z, pe], dim=-1)
        else:
            x = z

        x = self.ln(x)
        h = F.relu(self.fc1(x))
        h = h + F.relu(self.fc2(h))  # tiny residual
        logit = self.fc3(h).squeeze(-1)
        return torch.sigmoid(logit)



class FeatureProjector(nn.Module):
    """
    Projects high-dimensional point features (e.g., 768-D from a vision backbone)
    into a smaller latent space (e.g., 64-D) suitable for voxel updates.

    Optionally includes normalization or a small MLP head.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 64, hidden_dim: int | None = None,
                 use_layernorm: bool = True, activation: str = "relu"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_layernorm = use_layernorm

        if hidden_dim is None:
            # simple linear projection
            self.net = nn.Linear(in_dim, out_dim)
        else:
            # small MLP head for richer mapping
            act = nn.ReLU() if activation == "relu" else nn.GELU()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, out_dim)
            )

        if use_layernorm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim)
        returns: (N, out_dim)
        """
        x = self.net(x)
        x = self.norm(x)
        return x


# ----------------- Grid -----------------
class LatentVoxelGrid(nn.Module):
    """
    Sparse log-odds voxel map + learned latent memory per voxel.

    - Classic ST/LT log-odds integration (ray carving) preserved.
    - Learned update: top-K, low-temperature routing from point features to voxels,
      a per-voxel gate, and GRU fusion to update `z_latent`.
    """
    def __init__(self, origin_xyz, params: VoxelParams,
                 device="cpu", dtype=torch.float32, feature_dim: int = 64):
        super().__init__()

        # ---- core state ----
        self.origin = torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3)
        self.p = params
        self.device = self.origin.device
        self.dtype = dtype

        def eb(name, shape, dtype_, persistent=True):
            self.register_buffer(name, torch.empty(shape, dtype=dtype_), persistent=persistent)

        eb("keys",           (0,),     torch.int64)
        eb("vals_st",        (0,),     dtype)
        eb("vals_lt",        (0,),     dtype)
        eb("vals",           (0,),     dtype)
        eb("hit_count",      (0,),     torch.int32)
        eb("pos_occ_count",  (0,),     torch.int16)
        eb("neg_free_count", (0,),     torch.int16)
        eb("last_occ_epoch", (0,),     torch.int32)
        eb("last_free_epoch",(0,),     torch.int32)
        eb("view_bits",      (0,),     torch.int16)
        eb("seen_occ_epoch", (0,),     torch.int32)
        eb("seen_view_bits_e",(0,),    torch.int16)
        eb("occ_epoch_count",(0,),     torch.int16)
        eb("view_bits_cum",  (0,),     torch.int16)
        eb("lt_promoted_flag",(0,),    torch.uint8)

        self.epoch: int = 0

        # ---- learned latent memory ----
        self.feature_dim = feature_dim
        # self.input_dim = 768
        self.z_latent = torch.empty((0, feature_dim), dtype=self.dtype, device=self.device)
        # self.z_proj = nn.Linear(self.input_dim, self.feature_dim)
        self.z_proj = nn.Identity()
        # update modules
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=feature_dim)
        self.sim_net  = FeatureVoxelSimilarity(feature_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        ).to(self.device)

        # routing controls
        self.routing_tau: float = 0.3   # temperature for softmax
        self.routing_topk: int = 8      # voxels per point (after radius prefilter)
        
        self.decoder = LatentToOccupancyDecoder(feature_dim)

    # ---------- utilities ----------
    def _world_to_ijk(self, pts: torch.Tensor) -> torch.Tensor:
        rel = (pts - self.origin) / self.p.voxel_size
        return torch.floor(rel).to(torch.int64)

    @staticmethod
    def _hash_ijk(ijk: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ijk[..., 0] + off
        j = ijk[..., 1] + off
        k = ijk[..., 2] + off
        return (i << 42) ^ (j << 21) ^ k

    @staticmethod
    def _unhash_keys_static(keys: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = ( keys        & ((1 << 21) - 1)) - off
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    def _unhash_keys(self, keys: torch.Tensor) -> torch.Tensor:
        return self._unhash_keys_static(keys)

    def _ensure_and_index(self, upd_keys: torch.Tensor) -> torch.Tensor:
        """Merge upd_keys into self.keys; realign all state; return positions of upd_keys in the new key set."""
        if upd_keys.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        all_keys = torch.cat([self.keys, upd_keys], dim=0)
        uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        idx_old = inv[:n_old]
        idx_upd = inv[n_old:]

        if uk.numel() != n_old:
            def grow_like(src: torch.Tensor, fill=0):
                out = torch.empty(uk.shape[0], dtype=src.dtype, device=src.device)
                if src.numel() > 0:
                    out[idx_old] = src
                if uk.shape[0] > n_old:
                    mask_new = torch.ones(uk.shape[0], dtype=torch.bool, device=src.device)
                    if src.numel() > 0:
                        mask_new[idx_old] = False
                    out[mask_new] = fill
                return out

            self.keys            = uk
            self.vals_st         = grow_like(self.vals_st,       0)
            self.vals_lt         = grow_like(self.vals_lt,       0)
            self.vals            = grow_like(self.vals,          0)
            self.hit_count       = grow_like(self.hit_count,     0)
            self.pos_occ_count   = grow_like(self.pos_occ_count, 0)
            self.neg_free_count  = grow_like(self.neg_free_count,0)
            self.last_occ_epoch  = grow_like(self.last_occ_epoch,-2**31+1)
            self.last_free_epoch = grow_like(self.last_free_epoch,-2**31+1)
            self.view_bits       = grow_like(self.view_bits,     0)
            self.seen_occ_epoch  = grow_like(self.seen_occ_epoch,   -2**31+1)
            self.seen_view_bits_e= grow_like(self.seen_view_bits_e, 0)
            self.occ_epoch_count = grow_like(self.occ_epoch_count,  0)
            self.view_bits_cum   = grow_like(self.view_bits_cum,    0)
            self.lt_promoted_flag= grow_like(self.lt_promoted_flag, 0)

            # grow latents (zeros for new voxels)
            z_new = torch.zeros((uk.shape[0], self.feature_dim), device=self.device, dtype=self.dtype)
            if self.z_latent.numel() > 0:
                z_new[idx_old] = self.z_latent
            self.z_latent = z_new

        return idx_upd

    # ---------- display ----------
    def _display_vals(self) -> torch.Tensor:
        return torch.maximum(self.vals_st, self.vals_lt)

    # ---------- epoch advance & promotion finalization ----------
    def next_epoch(self):
        if self.keys.numel() > 0:
            now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
            seen_now = (self.seen_occ_epoch == now)

            if seen_now.any():
                self.occ_epoch_count[seen_now] = torch.clamp(
                    self.occ_epoch_count[seen_now] + 1,
                    max=torch.iinfo(self.occ_epoch_count.dtype).max
                )
                self.view_bits_cum[seen_now] = self.view_bits_cum[seen_now] | self.seen_view_bits_e[seen_now]

                epochs_ok = self.occ_epoch_count[seen_now] >= int(self.p.promote_epochs)
                if int(self.p.lt_min_view_sectors) > 1:
                    vb = self.view_bits_cum[seen_now].to(torch.int16)
                    pop = ((vb & 1 > 0).to(torch.int16) +
                           ((vb >> 1) & 1 > 0).to(torch.int16) +
                           ((vb >> 2) & 1 > 0).to(torch.int16) +
                           ((vb >> 3) & 1 > 0).to(torch.int16) +
                           ((vb >> 4) & 1 > 0).to(torch.int16) +
                           ((vb >> 5) & 1 > 0).to(torch.int16) +
                           ((vb >> 6) & 1 > 0).to(torch.int16) +
                           ((vb >> 7) & 1 > 0).to(torch.int16))
                    mv_ok = pop >= int(self.p.lt_min_view_sectors)
                else:
                    mv_ok = torch.ones_like(self.occ_epoch_count[seen_now], dtype=torch.bool, device=self.device)

                prom_local = epochs_ok & mv_ok
                if prom_local.any():
                    idx_all = torch.arange(self.keys.numel(), device=self.device, dtype=torch.int64)
                    idx_prom = idx_all[seen_now][prom_local]

                    if self.p.lt_promotion_mode == "once":
                        tgt = torch.full_like(self.vals_lt[idx_prom], float(self.p.lt_promote_value))
                        self.vals_lt[idx_prom] = torch.maximum(self.vals_lt[idx_prom], tgt)
                        self.lt_promoted_flag[idx_prom] = 1
                    else:
                        step = float(self.p.lt_occ_scale * self.p.occ_inc)
                        self.vals_lt.index_add_(0, idx_prom,
                                                torch.full_like(idx_prom, step, dtype=self.vals_lt.dtype))
                    self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

            if self.seen_view_bits_e.numel() > 0:
                self.seen_view_bits_e.zero_()

        # LT demotion
        if self.p.lt_demote_enable and self.keys.numel() > 0:
            enough_neg = self.neg_free_count >= int(self.p.lt_demote_k_neg)
            no_recent_occ = (self.epoch - self.last_occ_epoch) >= int(self.p.lt_demote_min_no_occ_epochs)
            demote_mask = enough_neg & no_recent_occ
            if demote_mask.any():
                idx_demote = torch.nonzero(demote_mask, as_tuple=False).squeeze(-1)
                step = float(self.p.lt_demote_step)
                self.vals_lt.index_add_(0, idx_demote,
                                        torch.full_like(idx_demote, -step, dtype=self.vals_lt.dtype))
                self.vals_lt.clamp_(min=max(self.p.l_min, float(self.p.lt_demote_floor)),
                                    max=self.p.l_max)
                if self.p.lt_reset_promotion_on_demote:
                    not_occ_anymore = self.vals_lt[idx_demote] <= self.p.occ_thresh
                    if not_occ_anymore.any() and self.lt_promoted_flag.numel() > 0:
                        self.lt_promoted_flag[idx_demote[not_occ_anymore]] = 0

        self.epoch += 1

        if self.vals_st.numel() > 0 and self.p.st_decay_gamma < 0.9999:
            self.vals_st.mul_(self.p.st_decay_gamma).clamp_(min=self.p.l_min, max=self.p.l_max)

    # ---------- queries ----------
    def occupied_mask(self) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        lt_occ = self.vals_lt > self.p.occ_thresh
        st_occ = self.vals_st > (self.p.occ_thresh + self.p.st_margin)
        return lt_occ | st_occ

    def occupied_indices(self):
        if self.keys.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return self.keys[self.occupied_mask()]

    # ---------- guarded updates (per-ray) ----------
    def _update_occupied_guarded_(self, idx: torch.Tensor, cam_center_world: Optional[torch.Tensor]):
        if idx.numel() == 0: return
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.occ_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        self.hit_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.hit_count.dtype))
        self.neg_free_count.index_fill_(0, idx, torch.tensor(0, dtype=self.neg_free_count.dtype, device=self.device))
        self.pos_occ_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.pos_occ_count.dtype))
        self.last_occ_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_occ_epoch.dtype, device=self.device))

        if cam_center_world is not None:
            ijk = self._unhash_keys(self.keys[idx])
            centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
            v = centers - torch.as_tensor(cam_center_world, dtype=self.origin.dtype, device=self.device).reshape(1,3)
            yaw = torch.atan2(v[:,1], v[:,0])
            sector = torch.floor((yaw + math.pi) / (2*math.pi) * 8.0) % 8.0
            bits = (1 << sector.to(torch.int16)).to(self.view_bits.dtype)
            self.view_bits[idx] = self.view_bits[idx] | bits
            bits_e = (1 << sector.to(torch.int16)).to(self.seen_view_bits_e.dtype)
            self.seen_view_bits_e[idx] = self.seen_view_bits_e[idx] | bits_e

        self.seen_occ_epoch[idx] = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)

    def _update_free_guarded_(self, idx: torch.Tensor):
        if idx.numel() == 0: return
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.free_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)
        self.pos_occ_count.index_fill_(0, idx, torch.tensor(0, dtype=self.pos_occ_count.dtype, device=self.device))
        self.neg_free_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.neg_free_count.dtype))
        self.last_free_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_free_epoch.dtype, device=self.device))

        if not self.p.lt_allow_free:
            return
        enough_neg = self.neg_free_count[idx] >= self.p.lt_free_k_neg
        not_recent_occ = (self.epoch - self.last_occ_epoch[idx]) > self.p.lt_free_recent_occ_epochs
        carve_mask = enough_neg & not_recent_occ
        if carve_mask.any():
            idx_carve = idx[carve_mask]
            self.vals_lt.index_add_(0, idx_carve,
                torch.full_like(idx_carve, self.p.lt_free_scale * self.p.free_inc, dtype=self.vals_lt.dtype))
            self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

    # ---------- classic integration (vectorized) ----------
    @torch.no_grad()
    def integrate_frame_torch(
        self,
        points_world: torch.Tensor,     # (N,3)
        cam_center_world: torch.Tensor, # (3,)
        *,
        carve_free: bool = True,
        max_range: float | None = 12.0,
        z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
        ray_stride: int = 3,
        max_free_rays: int | None = 40000,
        samples_per_voxel: float = 1.05,
    ):
        if points_world.numel() == 0:
            return
        dev, dt = self.device, self.dtype
        pts = points_world.to(dev, dt)
        cam = cam_center_world.to(dev, dt).reshape(1,3)

        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if pts.numel() == 0: return

        if max_range is not None:
            d = torch.linalg.norm(pts - cam, dim=1)
            pts = pts[d <= max_range]
        if z_clip is not None:
            z0, z1 = z_clip
            m = (pts[:,2] >= z0) & (pts[:,2] <= z1)
            pts = pts[m]
        if pts.numel() == 0: return

        # occupied endpoints
        ijk_occ = self._world_to_ijk(pts)
        keys_occ = torch.unique(self._hash_ijk(ijk_occ))
        idx_occ = self._ensure_and_index(keys_occ)
        self._update_occupied_guarded_(idx_occ, cam_center_world)

        if not carve_free:
            self.vals = self._display_vals(); return

        # free carving
        pts_free = pts[::ray_stride] if ray_stride > 1 else pts
        if (max_free_rays is not None) and (pts_free.shape[0] > max_free_rays):
            ridx = torch.randperm(pts_free.shape[0], device=dev)[:max_free_rays]
            pts_free = pts_free[ridx]
        if pts_free.numel() == 0:
            self.vals = self._display_vals(); return

        vec = pts_free - cam
        seg_len = torch.linalg.norm(vec, dim=1)
        steps_per_ray = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
        max_steps = int(steps_per_ray.max().item())

        base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
        t = base[None, :] / steps_per_ray.to(dt)[:, None]
        t = torch.minimum(t,
            torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt),
                            torch.tensor(0.0, device=dev, dtype=dt)))
        mask = (t < 1.0)
        samples = cam + t.unsqueeze(-1) * vec.unsqueeze(1)
        samples = samples[mask]

        ijk_free = self._world_to_ijk(samples)
        keys_free = self._hash_ijk(ijk_free)
        if keys_occ.numel() > 0 and keys_free.numel() > 0:
            keep = ~torch.isin(keys_free, keys_occ)
            keys_free = keys_free[keep]
        if keys_free.numel() > 0:
            keys_free = torch.unique(keys_free)
            idx_free = self._ensure_and_index(keys_free)
            self._update_free_guarded_(idx_free)

        self.vals = self._display_vals()

    # # ---------- learned latent update (point features -> voxel latents) ----------
    # def update_with_features(self,
    #                          pts_world: torch.Tensor,  # (N,3)
    #                          f_pts: torch.Tensor,      # (N,D)
    #                          radius: float = 0.25):
    #     """
    #     Route point features to nearby voxels, then gated-GRU fuse into z_latent.
    #     Uses radius prefilter + per-point top-K + temperatured softmax.
    #     """
    #     if pts_world.numel() == 0 or self.keys.numel() == 0:
    #         return

    #     dev = self.device
    #     N = pts_world.shape[0]
    #     M = self.keys.shape[0]
    #     D = self.feature_dim

    #     # voxel centers
    #     ijk_all   = self._unhash_keys(self.keys).to(torch.float32)  # (M,3)
    #     voxel_xyz = self.origin + (ijk_all + 0.5) * self.p.voxel_size  # (M,3)

    #     # candidate voxels per point: radius ∪ top-K nearest
    #     with torch.no_grad():
    #         dists = torch.cdist(pts_world.to(dev), voxel_xyz)       # (N,M)
    #         mask  = dists <= radius                                 # (N,M)

    #         K = min(self.routing_topk, M)
    #         # nearest K even if outside radius
    #         neg_d = -dists
    #         topk_vals, topk_idx = torch.topk(neg_d, k=K, dim=1)     # (N,K)
    #         cand_mask = mask.clone()
    #         cand_mask.scatter_(1, topk_idx, True)

    #         i_idx, j_idx = cand_mask.nonzero(as_tuple=True)         # (R,), (R,)

    #     if i_idx.numel() == 0:
    #         return

    #     # gather flat tensors for selected pairs
    #     p_sel = pts_world[i_idx]                  # (R,3)
    #     f_sel = f_pts[i_idx]                      # (R,D)
    #     v_sel = voxel_xyz[j_idx]                  # (R,3)
    #     z_sel = self.z_latent[j_idx]              # (R,D)
    #     delta = p_sel - v_sel                     # (R,3)

    #     # similarity (flat)
    #     sim_flat = self.sim_net(f_sel, z_sel, delta)   # (R,)

    #     # per-point softmax with temperature: w = softmax_j( sim/τ )
    #     # 1) segment-wise max for stability
    #     #    (requires PyTorch 2.0+ scatter_reduce; else implement a custom segment-max)
    #     max_per_i = torch.full((N,), -1e9, device=sim_flat.device, dtype=sim_flat.dtype)
    #     max_per_i = max_per_i.scatter_reduce(0, i_idx, sim_flat, reduce="amax", include_self=True)
    #     sim_shift = sim_flat - max_per_i[i_idx]
    #     w_unnorm  = torch.exp(sim_shift / max(self.routing_tau, 1e-6))
    #     sum_per_i = torch.zeros(N, device=sim_flat.device, dtype=sim_flat.dtype).scatter_add(0, i_idx, w_unnorm)
    #     weights   = w_unnorm / (sum_per_i[i_idx] + 1e-8)           # (R,)

    #     # fuse into voxel inputs via scatter-add
    #     upd = torch.zeros(M, D, device=dev, dtype=f_sel.dtype)
    #     upd.index_add_(0, j_idx, (weights.unsqueeze(-1) * f_sel))  # (M,D)

    #     # only update voxels that received mass
    #     got_mass = torch.zeros(M, dtype=torch.bool, device=dev)
    #     got_mass[j_idx] = True
    #     idx_upd = torch.nonzero(got_mass, as_tuple=False).squeeze(-1)  # (M_upd,)
    #     if idx_upd.numel() == 0:
    #         return

    #     u = upd[idx_upd]                  # (M_upd,D)
    #     z = self.z_latent[idx_upd]        # (M_upd,D)

    #     # gate (scalar per voxel) to suppress noisy updates
    #     gamma = torch.sigmoid(self.gate_mlp(torch.cat([u, z], dim=-1)))  # (M_upd,1)
    #     x_in = gamma * u                                                 # broadcast to (M_upd,D)

    #     # GRU fuse
    #     z_new = self.gru_cell(x_in, z)                                   # (M_upd,D)
    #     self.z_latent.index_copy_(0, idx_upd, z_new)

    def update_with_features(self,
                            pts_world: torch.Tensor,  # (N,3)
                            f_pts: torch.Tensor,      # (N,D)
                            radius: float = 0.25,
                            *,
                            chunkN: int = 80_000,     # tune
                            chunkT: int = 512,        # CHUNK OFFSETS!
                            neighbor_pad: int = 0,
                            r_vox_cap: int = 6,       # safety cap on neighbor radius in voxels
                            use_amp: bool = True):
        if pts_world.numel() == 0 or self.keys.numel() == 0:
            return

        dev = self.device
        N   = pts_world.shape[0]
        M   = self.keys.shape[0]
        D   = self.feature_dim
        K   = min(self.routing_topk, M)
        vox = float(self.p.voxel_size)

        # neighbor offsets
        r_vox = int(math.ceil(radius / max(vox, 1e-8))) + int(neighbor_pad)
        r_vox = min(r_vox, int(r_vox_cap))  # prevent T from blowing up
        rng   = torch.arange(-r_vox, r_vox + 1, device=dev, dtype=torch.int32)
        ox, oy, oz = torch.meshgrid(rng, rng, rng, indexing="ij")
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=-1)  # (T,3)
        T = int(offsets.size(0))

        keys_sorted = self.keys  # should already be sorted

        all_i_parts, all_j_parts = [], []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # running global topK storage
            topk_vals = torch.full((N, K), -float("inf"), device=dev) if K > 0 else None
            topk_idx  = torch.zeros((N, K), dtype=torch.long, device=dev)          if K > 0 else None

            for i0 in range(0, N, chunkN):
                i1 = min(i0 + chunkN, N)
                n  = i1 - i0
                pts = pts_world[i0:i1].to(dev, torch.float32)             # (n,3)
                ijk0 = self._world_to_ijk(pts).to(torch.int32)            # (n,3)

                # running block topK for this point chunk
                if K > 0:
                    blk_vals = torch.full((n, K), -float("inf"), device=dev)
                    blk_idx  = torch.zeros((n, K), dtype=torch.long, device=dev)

                for t0 in range(0, T, chunkT):
                    t1 = min(t0 + chunkT, T)
                    off = offsets[t0:t1]                                  # (t,3)
                    # (n,t,3) — small because t <= chunkT
                    cand_ijk  = ijk0[:, None, :] + off[None, :, :]        # (n,t,3)
                    cand_keys = self._hash_ijk(cand_ijk.reshape(-1, 3)).reshape(n, -1)  # (n,t)

                    # searchsorted to map keys -> voxel indices
                    pos = torch.searchsorted(keys_sorted, cand_keys.reshape(-1)).reshape(n, -1)
                    valid = (pos < M) & (keys_sorted[pos] == cand_keys)   # (n,t)

                    # centers for dist calc (compute just for this chunk)
                    centers = self.origin.to(torch.float32) + (cand_ijk.to(torch.float32) + 0.5) * vox  # (n,t,3)
                    d2 = ((pts[:, None, :] - centers) ** 2).sum(dim=-1)   # (n,t)

                    # radius hits
                    if radius > 0:
                        rad_hits = valid & (d2 <= (radius * radius))
                        if rad_hits.any():
                            hi, ht = torch.nonzero(rad_hits, as_tuple=True)
                            all_i_parts.append(hi + i0)
                            all_j_parts.append(pos[hi, ht])

                    # per-point topK among neighbors (masked invalids)
                    if K > 0:
                        d2m = d2.masked_fill(~valid, float("inf"))
                        k_eff = min(K, d2m.size(1))
                        vals, idx_local = torch.topk(-d2m, k=k_eff, dim=1)         # (n,k_eff)
                        cand_idx = pos.gather(1, idx_local)                         # (n,k_eff)
                        # merge with running block topK
                        mv = torch.cat([blk_vals, vals], dim=1)                     # (n, K+k_eff)
                        mi = torch.cat([blk_idx,  cand_idx], dim=1)                 # (n, K+k_eff)
                        blk_vals, sel = torch.topk(mv, k=K, dim=1)
                        blk_idx = mi.gather(1, sel)

                    # free per-offset chunk
                    del cand_ijk, cand_keys, pos, valid, centers, d2

                # merge block topK into global
                if K > 0:
                    gv = torch.cat([topk_vals[i0:i1], blk_vals], dim=1)             # (n, 2K)
                    gi = torch.cat([topk_idx[i0:i1],  blk_idx ], dim=1)
                    new_vals, sel = torch.topk(gv, k=K, dim=1)
                    new_idx = gi.gather(1, sel)
                    topk_vals[i0:i1] = new_vals
                    topk_idx[i0:i1]  = new_idx
                    del blk_vals, blk_idx, gv, gi, new_vals, sel

                del pts, ijk0

            # union(radius hits, topK)
            if all_i_parts:
                i_hits = torch.cat(all_i_parts, dim=0)
                j_hits = torch.cat(all_j_parts, dim=0)
            else:
                i_hits = torch.empty(0, dtype=torch.long, device=dev)
                j_hits = torch.empty(0, dtype=torch.long, device=dev)

            if K > 0:
                i_topk = torch.arange(N, device=dev).unsqueeze(1).expand(-1, K).reshape(-1)
                j_topk = topk_idx.reshape(-1)
                i_idx = torch.cat([i_hits, i_topk], dim=0)
                j_idx = torch.cat([j_hits, j_topk], dim=0)
            else:
                i_idx, j_idx = i_hits, j_hits

            # dedup
            if i_idx.numel() == 0:
                return
            pairs = torch.stack([i_idx, j_idx], dim=1)
            pairs = torch.unique(pairs, dim=0)
            i_idx, j_idx = pairs[:, 0], pairs[:, 1]

        # ---- unchanged fuse path below ----
        vox_xyz = self.origin + (self._unhash_keys(self.keys).to(torch.float32) + 0.5) * vox
        p_sel = pts_world[i_idx].to(dev)
        f_sel = f_pts[i_idx].to(dev)
        f_sel = self.z_proj(f_sel)               # (R,64)  <-- project to latent dim
        v_sel = vox_xyz[j_idx]
        z_sel = self.z_latent[j_idx]
        delta = p_sel - v_sel

        sim_flat = self.sim_net(f_sel, z_sel, delta)

        Nfull = int(N)
        tau = max(float(self.routing_tau), 1e-6)
        max_per_i = torch.full((Nfull,), -1e9, device=sim_flat.device, dtype=sim_flat.dtype)
        max_per_i = max_per_i.scatter_reduce(0, i_idx, sim_flat, reduce="amax", include_self=True)
        sim_shift = sim_flat - max_per_i[i_idx]
        w_unnorm  = torch.exp(sim_shift / tau)
        sum_per_i = torch.zeros(Nfull, device=sim_flat.device, dtype=sim_flat.dtype).scatter_add(0, i_idx, w_unnorm)
        weights   = w_unnorm / (sum_per_i[i_idx] + 1e-8)

        M = self.keys.shape[0]
        D = self.feature_dim
        upd = torch.zeros(M, D, device=dev, dtype=f_sel.dtype)
        upd.index_add_(0, j_idx, weights.unsqueeze(-1) * f_sel)

        got_mass = torch.zeros(M, dtype=torch.bool, device=dev); got_mass[j_idx] = True
        idx_upd = got_mass.nonzero(as_tuple=False).squeeze(-1)
        if idx_upd.numel() == 0:
            return

        u = upd[idx_upd]
        z = self.z_latent[idx_upd]
        gamma = torch.sigmoid(self.gate_mlp(torch.cat([u, z], dim=-1)))
        x_in  = gamma * u
        z_new = self.gru_cell(x_in, z)
        self.z_latent.index_copy_(0, idx_upd, z_new)


    # ---------- exports ----------
    def to_numpy(self):
        return self.keys.detach().cpu().numpy(), self._display_vals().detach().cpu().numpy()

    def occupied_ijk_numpy(self, zmin=None, zmax=None):
        if self.keys.numel() == 0:
            return np.empty((0,3), dtype=np.int32)
        ijk = self._unhash_keys(self.keys)
        occ_mask = self.occupied_mask()
        if zmin is not None or zmax is not None:
            centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
            cz = centers[:,2]
            if zmin is not None: occ_mask &= (cz >= zmin)
            if zmax is not None: occ_mask &= (cz <= zmax)
        ijk_sel = ijk[occ_mask]
        if ijk_sel.numel() == 0: return np.empty((0,3), dtype=np.int32)
        arr = ijk_sel.cpu().numpy()
        return [(int(a), int(b), int(c)) for a,b,c in arr]

    def occupied_voxels(self, zmin: float | None = None, zmax: float | None = None):
        return self.occupied_ijk_numpy(zmin=zmin, zmax=zmax)

    def ijk_to_center(self, ijk) -> np.ndarray:
        ijk_t = torch.as_tensor(ijk, device=self.device, dtype=torch.float32)
        centers = self.origin + (ijk_t + 0.5) * self.p.voxel_size
        return centers.detach().cpu().numpy()


    # --- voxel centers helper ---
    def voxel_centers(self) -> torch.Tensor:
        """
        Returns (M,3) world centers for current keys (float32, device=self.device).
        """
        if self.keys.numel() == 0:
            return torch.empty(0, 3, device=self.device, dtype=torch.float32)
        ijk = self._unhash_keys(self.keys).to(torch.float32)
        return self.origin + (ijk + 0.5) * self.p.voxel_size

    # --- decode all current voxels with a provided decoder ---
    #@torch.no_grad()
    def decode_occupancy(self,with_xyz_cond: bool = False) -> torch.Tensor:
        """
        decoder: LatentToOccupancyDecoder (or compatible)
        with_xyz_cond: pass centers to decoder if it expects xyz conditioning
        returns: (M,) probabilities aligned with self.keys
        """
        if self.z_latent.numel() == 0:
            print("empty")
            return torch.empty(0, device=self.device, dtype=torch.float32)
        if with_xyz_cond:
            centers = self.voxel_centers()
            return self.decoder(self.z_latent, centers)
        else:
            return self.decoder(self.z_latent, None)

    # --- rasterize to 2D BEV occupancy ---
    @torch.no_grad()
    def to_bev(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        res_xy: float,
        z_min: float,
        z_max: float,
        agg: str = "max",
        with_xyz_cond: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Produce a BEV occupancy grid by aggregating decoded voxel probs in a Z band.

        Returns:
          bev: (Hy, Hx) float in [0,1]
          meta: dict with origin/resolution for plotting
        """
        centers = self.voxel_centers()
        if centers.numel() == 0:
            Hx = int((x_range[1]-x_range[0]) / res_xy)
            Hy = int((y_range[1]-y_range[0]) / res_xy)
            return torch.zeros(Hy, Hx), {"x0": x_range[0], "y0": y_range[0], "res": res_xy}

        # slice by height
        z = centers[:, 2]
        z_mask = (z >= z_min) & (z <= z_max)
        if not z_mask.any():
            Hx = int((x_range[1]-x_range[0]) / res_xy)
            Hy = int((y_range[1]-y_range[0]) / res_xy)
            return torch.zeros(Hy, Hx, device=self.device), {"x0": x_range[0], "y0": y_range[0], "res": res_xy}

        centers = centers[z_mask]
        z_lat = self.z_latent[z_mask]
        probs = self.decoder(z_lat, centers if with_xyz_cond else None)  # (Mz,)

        # index into BEV grid
        x0, x1 = x_range
        y0, y1 = y_range
        Hx = int((x1 - x0) / res_xy)
        Hy = int((y1 - y0) / res_xy)

        gx = ((centers[:, 0] - x0) / res_xy).floor().to(torch.long)
        gy = ((centers[:, 1] - y0) / res_xy).floor().to(torch.long)
        keep = (gx >= 0) & (gx < Hx) & (gy >= 0) & (gy < Hy)
        gx, gy, probs = gx[keep], gy[keep], probs[keep]

        bev = torch.zeros(Hy, Hx, device=self.device, dtype=probs.dtype)
        if probs.numel() == 0:
            return bev, {"x0": x0, "y0": y0, "res": res_xy}

        if agg == "max":
            # scatter max (PyTorch 2.0: scatter_reduce)
            idx = gy * Hx + gx
            flat = bev.view(-1)
            flat.scatter_reduce_(0, idx, probs, reduce="amax", include_self=True)
        elif agg == "mean":
            idx = gy * Hx + gx
            flat_sum = bev.view(-1)
            flat_cnt = torch.zeros_like(flat_sum)
            flat_sum.scatter_add_(0, idx, probs)
            flat_cnt.scatter_add_(0, idx, torch.ones_like(probs))
            flat = torch.where(flat_cnt > 0, flat_sum / flat_cnt.clamp_min(1), flat_sum*0)
            bev = flat.view(Hy, Hx)
        else:
            raise ValueError("agg must be 'max' or 'mean'")

        return bev, {"x0": x0, "y0": y0, "res": res_xy}



    def _ensure_feature_storage_(self):
        """internal: make sure z_latent exists & is aligned with keys."""
        if getattr(self, "z_latent", None) is None or self.z_latent.shape[0] != self.keys.shape[0]:
            D = getattr(self, "feature_dim", 64)
            z = torch.zeros((self.keys.shape[0], D), device=self.device, dtype=self.dtype)
            if getattr(self, "z_latent", None) is not None and self.z_latent.numel() > 0:
                # try to preserve old in case only size changed (rare)
                z[:min(z.shape[0], self.z_latent.shape[0])] = self.z_latent[:min(z.shape[0], self.z_latent.shape[0])]
            self.z_latent = z

    #@torch.no_grad()  # remove this decorator during training so gradients flow into sim_net & GRU
    def update_with_features_learned(
        self,
        pts_world: torch.Tensor,    # (N,3)
        f_pts: torch.Tensor,        # (N,D) features (e.g., PointNeXt)
        *,
        radius_m: float = 0.25,
        temp: float = 0.5,
        topk: int = 8,
        ema_to_st: float = 0.4,     # how strongly decoded prob refreshes ST log-odds (0 disables)
        write_display: bool = True,
    ):
        """
        Learned fusion: route points to nearby voxels with a learned similarity, then GRU-fuse.

        - radius_m: geometric candidate radius for voxels around each point
        - temp: softmax temperature (lower = sharper routing)
        - topk: keep only top-k candidate voxels per point (sparse/fast)
        - decoder: optional LatentToOccupancyDecoder to refresh vals_st after latent update
        - ema_to_st: if >0, write decoded occupancy to ST via logit-EMA
        """
        dev, dt = self.device, self.dtype
        if pts_world.numel() == 0:
            return

        # 0) make sure latent storage matches current key set
        self._ensure_feature_storage_()

        # 1) candidate voxel keys for each point (small cube around point's voxel)
        vs = float(self.p.voxel_size)
        r_vox = max(1, int(math.ceil(radius_m / vs)))
        # voxel index of each point
        ijk_p = self._world_to_ijk(pts_world.to(dev, dt))     # (N,3)

        # precompute the integer offsets in the (2r+1)^3 cube
        ofs = torch.stack(torch.meshgrid(
            torch.arange(-r_vox, r_vox+1, device=dev),
            torch.arange(-r_vox, r_vox+1, device=dev),
            torch.arange(-r_vox, r_vox+1, device=dev),
            indexing='ij'), dim=-1).reshape(-1,3)             # (K,3)
        # points → candidate ijk → candidate keys
        ijk_cand = (ijk_p[:, None, :] + ofs[None, :, :]).to(torch.int64)   # (N,K,3)
        keys_cand = self._hash_ijk(ijk_cand)                               # (N,K)

        # 2) ensure all candidate voxels exist in the grid (grows keys/state/latents)
        idx_cand = self._ensure_and_index(keys_cand.reshape(-1)).reshape(keys_cand.shape)  # (N,K)
        self._ensure_feature_storage_()  # grew => reattach z_latent

        # centers for those candidates (for geometric conditioning)
        ijk_cand_float = self._unhash_keys(self.keys[idx_cand.reshape(-1)]).to(torch.float32).reshape_as(ijk_cand)
        centers_cand = self.origin + (ijk_cand_float + 0.5) * vs          # (N,K,3)
        delta = pts_world[:, None, :].to(dev, dt) - centers_cand          # (N,K,3)

        # gather candidate voxel latents
        z_cand = self.z_latent[idx_cand]                                  # (N,K,D)

        # 3) learned similarity per (point, voxel)
        # sim_net expects (N,K) point features and voxel latents + delta xyz
        # flatten to feed efficiently
        N, K = keys_cand.shape
        D = f_pts.shape[-1]
        f_rep = f_pts.to(dev, dt)[:, None, :].expand(N, K, D)             # (N,K,D)
        sim = self.sim_net(                                               # (N,K)
            f_rep.reshape(-1, D),                                         # (N*K, D) point feats
            z_cand.reshape(-1, D),                                        # (N*K, D) voxel latents
            delta.reshape(-1, 3)                                          # (N*K, 3)
        ).reshape(N, K)

        # 4) geometric mask (exact radius) + top-k sparsification
        # keep only voxels whose center is within radius_m from the point
        keep_geom = (delta.square().sum(-1).sqrt() <= radius_m)           # (N,K)
        sim[~keep_geom] = -1e4

        if topk is not None and topk < K:
            topv, topi = torch.topk(sim, k=topk, dim=1)                   # (N,topk)
            mask = torch.full_like(sim, fill_value=-1e4)
            mask.scatter_(1, topi, topv)
            sim = mask

        # 5) softmax routing (temperature)
        weights = torch.softmax(sim / max(temp, 1e-6), dim=1)             # (N,K)

        # 6) fuse per-voxel input = Σ_i α_ij f_i  via scatter-add on flattened voxel indices
        flat_idx = idx_cand.reshape(-1)                                    # (N*K,)
        flat_w   = weights.reshape(-1, 1)                                  # (N*K,1)
        flat_f   = f_rep.reshape(-1, D)                                    # (N*K,D)

        fused = torch.zeros_like(self.z_latent)                            # (M,D) M=#voxels
        norm  = torch.zeros((self.keys.shape[0], 1), device=dev, dtype=dt)

        # fused: (M, D), norm: (M, 1)
        fused.index_add_(0, flat_idx, flat_w * flat_f)   # (N*K, D) added into rows flat_idx
        norm.index_add_(0,  flat_idx, flat_w)            # (N*K, 1)

        # avoid div-by-zero
        fused = torch.where(norm > 0, fused / norm.clamp_min(1e-6), torch.zeros_like(fused))

        # 7) GRU fusion (voxel-wise)
        # GRUCell input=(M,D), hidden=(M,D) — run only where norm>0
        touched = (norm.squeeze(-1) > 0).nonzero(as_tuple=False).squeeze(-1)  # (T,)
        if touched.numel() > 0:
            z_old = self.z_latent[touched]
            z_new = self.gru_cell(fused[touched], z_old)
            self.z_latent[touched] = z_new

        # 8) (optional) refresh ST occupancy from decoder (small EMA in log-odds)
        if self.decoder is not None and ema_to_st > 0 and touched.numel() > 0:
            centers = self.origin + (self._unhash_keys(self.keys[touched]).to(torch.float32) + 0.5) * vs
            with torch.enable_grad():  # allow training of decoder if needed
                p_occ = self.decoder(self.z_latent[touched], centers)           # (T,)
            # convert prob → logit (log-odds) and EMA into ST
            logit = torch.logit(p_occ.clamp(1e-5, 1 - 1e-5))
            # initialize vals_st if needed
            if self.vals_st.numel() == 0:
                self.vals_st = torch.zeros(self.keys.shape[0], device=dev, dtype=dt)
            self.vals_st[touched] = (1 - ema_to_st) * self.vals_st[touched] + ema_to_st * logit
            self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        if write_display:
            # refresh compatibility display buffer
            if self.vals.numel() == 0:
                self.vals = torch.zeros_like(self.vals_st)
            self.vals = self._display_vals()
            
    @torch.no_grad()
    def initialize_latents_from_full_cloud(
        self,
        pts_world: torch.Tensor,   # (N,3) full scene points (aligned)
        f_pts: torch.Tensor,       # (N,D) point features (e.g., PointNeXt)
        *,
        pool: str = "mean",        # "mean" or "mean+max"
        init_lt: bool = True,      # initialize LT/ST from decoded probs
        lt_level: float | None = None,  # if set, override decoded probs with constant occupancy level in LT
        z_whiten: bool = False     # optional: per-voxel whitening of pooled features
    ):
        """
        Seed z_latent for all touched voxels using the *full* point cloud.
        Optionally compute initial occupancy via the decoder and write ST/LT log-odds.

        Typical call right after your first DUSt3R run:
            grid.initialize_latents_from_full_cloud(P_world, point_features, decoder=dec)
        """
        dev, dt = self.device, self.dtype

        if pts_world.numel() == 0:
            return

        pts_world = pts_world.to(dev, dt)
        f_pts = f_pts.to(dev, dt)
        assert pts_world.shape[0] == f_pts.shape[0]

        print(self.keys.get_device())
        # 1) insert all voxels touched by points
        ijk = self._world_to_ijk(pts_world)                       # (N,3)
        keys = self._hash_ijk(ijk)                                # (N,)

        idx = self._ensure_and_index(keys)                        # (N,) positions in self.keys
        self._ensure_feature_storage_()

        M = self.keys.shape[0]
        Dp = f_pts.shape[-1]
        Dl = self.feature_dim

        # 2) voxel pooling of point features
        feat_sum = torch.zeros((M, Dp), device=dev, dtype=dt)
        feat_cnt = torch.zeros((M, 1),  device=dev, dtype=dt)

        # sum / count
        # torch.scatter_add_(feat_sum, 0, idx[:, None].expand(-1, Dp), f_pts)
        # torch.scatter_add_(feat_cnt, 0, idx[:, None],                 torch.ones_like(idx, dtype=dt).unsqueeze(-1))


        feat_sum.index_add_(0, idx, f_pts)  # (M, Dp) += (N, Dp) at rows idx
        feat_cnt.index_add_(0, idx, torch.ones((idx.shape[0], 1), device=dev, dtype=dt))

        # mean
        z_mean = torch.where(feat_cnt > 0, feat_sum / feat_cnt.clamp_min(1e-6), torch.zeros_like(feat_sum))

        if pool == "mean+max":
            # compute voxel-wise max features too (approx via segment-max)
            # implement a simple max by bucketizing: initialize -inf and scatter_max
            z_max = torch.full((M, Dp), -1e9, device=dev, dtype=dt)
            # emulate scatter_max: compare and write
            # (PyTorch has scatter_reduce_ with "amax" in 2.x; if available, use that for speed.)
            # Fallback loop over feature dims (kept tiny Dp): robust & simple
            for d in range(Dp):
                flat = torch.full((M,), -1e9, device=dev, dtype=dt)
                torch.scatter_reduce_(flat, 0, idx, f_pts[:, d], reduce="amax", include_self=True)
                z_max[:, d] = flat
            z_pool = torch.cat([z_mean, z_max], dim=-1)  # (M, 2*Dp)
            # optional projection to latent size
            if isinstance(self.z_proj, nn.Identity):
                # if latent_dim == 2*Dp you can change latent_dim beforehand,
                # else define self.z_proj = nn.Linear(2*Dp, Dl)
                pass
        else:
            z_pool = z_mean  # (M, Dp)

        # 3) optional per-voxel whitening to normalize scale (helps early training)
        if z_whiten:
            mu = z_pool.mean(dim=-1, keepdim=True)
            sd = z_pool.std(dim=-1, keepdim=True).clamp_min(1e-4)
            z_pool = (z_pool - mu) / sd

        # 4) write into z_latent (through projection if needed)
        if isinstance(self.z_proj, nn.Identity):
            # if dims match, direct; else assert
            if z_pool.shape[1] != Dl:
                raise ValueError(f"latent_dim {Dl} != pooled feature dim {z_pool.shape[1]} "
                                f"(set self.z_proj to Linear({z_pool.shape[1]}->{Dl}))")
            self.z_latent = z_pool
        else:
            self.z_latent = self.z_proj(z_pool)  # (M, Dl)

        # 5) optional: initialize occupancy (LT/ST) using decoder or constant level
        if init_lt:
            if lt_level is not None:
                # write constant occupancy (in prob space) where voxel was touched
                p_occ = torch.full((M,), float(lt_level), device=dev, dtype=dt)
                touched = (feat_cnt.squeeze(-1) > 0)
                logit = torch.logit(p_occ.clamp(1e-5, 1-1e-5))
                self.vals_lt[touched] = logit[touched]
                self.vals_st[touched] = logit[touched]
            elif self.decoder is not None:
                centers = self.voxel_centers()
                p_occ = self.decoder(self.z_latent, centers if getattr(self.decoder, "cond", None) == "xyz" else None)
                logit = torch.logit(p_occ.clamp(1e-5, 1-1e-5))
                # only commit for voxels that actually saw points
                touched = (feat_cnt.squeeze(-1) > 0)
                self.vals_lt[touched] = logit[touched]
                self.vals_st[touched] = logit[touched]

            # clamp & refresh display buffer
            self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)
            self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)
            self.vals = self._display_vals()

        # 6) housekeeping: mark as seen this epoch (so promotion logic can kick in later if you use it)
        if pts_world.shape[0] > 0:
            # all touched indices this pass:
            touched_idx = torch.nonzero(feat_cnt.squeeze(-1) > 0, as_tuple=False).squeeze(-1)
            if touched_idx.numel() > 0:
                now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
                if self.seen_occ_epoch.shape[0] != self.keys.shape[0]:
                    # grow aux arrays just in case
                    self._ensure_and_index(torch.empty(0, dtype=torch.int64, device=self.device))
                self.seen_occ_epoch[touched_idx] = now
