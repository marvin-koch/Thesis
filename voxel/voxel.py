from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
from typing import Tuple
from dataclasses import dataclass

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
    promote_hits: int = 3          # (legacy) no longer used for LT; kept for API compat
    lt_occ_scale: float = 0.5
    lt_free_scale: float = 0.5

    # LT carving (slow free) conditions (used only if lt_allow_free=True)
    lt_free_k_neg: int = 4
    lt_free_recent_occ_epochs: int = 2

    # ST decay
    st_decay_gamma: float = 0.0       # 0.0 -> ST clears every epoch

    # ---------- NEW: Epoch-based LT promotion ----------
    promote_epochs: int = 50         # promote to LT if seen occupied in >= N distinct epochs
    lt_min_view_sectors: int = 1      # require >= M distinct yaw sectors across epochs (1 disables)
    lt_promotion_mode: str = "once"   # "once" -> set to target; "accumulate" -> add each qualifying epoch
    lt_promote_value: float = 0.8     # target LT level for "once" mode

    # ---------- NEW: LT carving policy ----------
    lt_allow_free: bool = True        # if False, never carve LT once promoted


    # ---------- NEW: LT demotion (epoch-based) ----------
    lt_demote_enable: bool = True       # turn LT demotion on/off
    lt_demote_k_neg: int = 50            # need ≥ this many consecutive free hits
    lt_demote_min_no_occ_epochs: int = 3# and ≥ this many epochs since last occupancy
    lt_demote_step: float = 0.5         # subtract from LT per qualifying epoch
    lt_demote_floor: float = 0.0        # clamp lower bound for LT during demotion
    lt_reset_promotion_on_demote: bool = True  # allow re-promotion after big demotion
    

# ----------------- Grid -----------------
class TorchSparseVoxelGrid:
    """
    Sparse log-odds voxel map with:
      - ST (short-term) fast updates (±occ_inc/free_inc) + per-epoch decay/reset,
      - LT (long-term) updated ONLY at epoch boundaries if voxel was seen occupied
        in >= promote_epochs distinct epochs (and optionally multi-view),
      - optional guarded LT carving (or disable entirely via lt_allow_free=False).

    Keys: int64 hashes of (i,j,k). State tensors are aligned to `self.keys`.
    """
    def __init__(self, origin_xyz, params: VoxelParams, device=None, dtype=torch.float32):
        self.origin = torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3)
        self.p = params
        self.device = self.origin.device
        self.dtype = dtype

        # master keyset
        self.keys = torch.empty(0, dtype=torch.int64, device=self.device)  # (N,)

        # L1/L2 log-odds (aligned to keys)
        self.vals_st = torch.empty(0, dtype=dtype, device=self.device)     # short-term (L1)
        self.vals_lt = torch.empty(0, dtype=dtype, device=self.device)     # long-term  (L2)

        # compatibility "display" (computed via _display_vals)
        self.vals = torch.empty(0, dtype=dtype, device=self.device)

        # per-voxel counters/guards (aligned to keys)
        self.hit_count       = torch.empty(0, dtype=torch.int32, device=self.device)  # total + hits (legacy)
        self.pos_occ_count   = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive + hits (legacy)
        self.neg_free_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive frees
        self.last_occ_epoch  = torch.empty(0, dtype=torch.int32, device=self.device)
        self.last_free_epoch = torch.empty(0, dtype=torch.int32, device=self.device)
        self.view_bits       = torch.empty(0, dtype=torch.int16, device=self.device)  # 8-dir bitmask (cumulative this session)

        # ---------- NEW: epoch-based promotion state ----------
        self.seen_occ_epoch   = torch.empty(0, dtype=torch.int32, device=self.device)  # last epoch index when seen occupied
        self.seen_view_bits_e = torch.empty(0, dtype=torch.int16, device=self.device)  # per-epoch view bits
        self.occ_epoch_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # number of distinct epochs seen occupied
        self.view_bits_cum    = torch.empty(0, dtype=torch.int16, device=self.device)  # OR of per-epoch view bits
        self.lt_promoted_flag = torch.empty(0, dtype=torch.uint8, device=self.device)  # 0/1: already promoted (for "once")

        self.epoch: int = 0  # advance once per integration round

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

    def _unhash_keys(self, keys: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = ( keys        & ((1 << 21) - 1)) - off
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    def _ensure_and_index(self, upd_keys: torch.Tensor) -> torch.Tensor:
        """Merge upd_keys into self.keys; realign all state; return positions of upd_keys."""
        if upd_keys.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        all_keys = torch.cat([self.keys, upd_keys], dim=0)
        uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        idx_old = inv[:n_old]
        idx_upd = inv[n_old:]

        if uk.numel() != n_old:
            def grow_like(src, fill=0):
                out = torch.empty(uk.shape[0], dtype=src.dtype, device=src.device)
                if src.numel() > 0:
                    out[idx_old] = src
                if uk.shape[0] > n_old:
                    mask_new = torch.ones(uk.shape[0], dtype=torch.bool, device=src.device)
                    if src.numel() > 0:
                        mask_new[idx_old] = False
                    out[mask_new] = fill
                return out

            self.keys = uk
            # grow state
            self.vals_st       = grow_like(self.vals_st,       0)
            self.vals_lt       = grow_like(self.vals_lt,       0)
            self.vals          = grow_like(self.vals,          0)
            self.hit_count     = grow_like(self.hit_count,     0)
            self.pos_occ_count = grow_like(self.pos_occ_count, 0)
            self.neg_free_count= grow_like(self.neg_free_count,0)
            self.last_occ_epoch= grow_like(self.last_occ_epoch,-2**31+1)
            self.last_free_epoch=grow_like(self.last_free_epoch,-2**31+1)
            self.view_bits     = grow_like(self.view_bits,     0)

            # NEW epoch-based state
            self.seen_occ_epoch   = grow_like(self.seen_occ_epoch,   -2**31+1)
            self.seen_view_bits_e = grow_like(self.seen_view_bits_e, 0)
            self.occ_epoch_count  = grow_like(self.occ_epoch_count,  0)
            self.view_bits_cum    = grow_like(self.view_bits_cum,    0)
            self.lt_promoted_flag = grow_like(self.lt_promoted_flag, 0)

        return idx_upd

    # ---------- display ----------
    def _display_vals(self) -> torch.Tensor:
        # Simple and effective: show whichever memory is more confident of occupancy.
        return torch.maximum(self.vals_st, self.vals_lt)

    # ---------- epoch advance & promotion finalization ----------
    def next_epoch(self):
        """
        Finalize current epoch (epoch-based promotion), then advance epoch and handle ST decay/reset.
        """
        # 1) finalize BEFORE incrementing epoch
        if self.keys.numel() > 0:
            now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
            seen_now = (self.seen_occ_epoch == now)  # voxels that had any occupied endpoint this epoch

            if seen_now.any():
                # count this epoch once per voxel
                self.occ_epoch_count[seen_now] = torch.clamp(
                    self.occ_epoch_count[seen_now] + 1,
                    max=torch.iinfo(self.occ_epoch_count.dtype).max
                )
                # OR per-epoch view bits into cumulative mask
                self.view_bits_cum[seen_now] = self.view_bits_cum[seen_now] | self.seen_view_bits_e[seen_now]

                # Promotion gates
                epochs_ok = self.occ_epoch_count[seen_now] >= int(self.p.promote_epochs)
                if int(self.p.lt_min_view_sectors) > 1:
                    # popcount of 8-bit mask (branchless)
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
                        # set LT to at least lt_promote_value exactly once
                        tgt = torch.full_like(self.vals_lt[idx_prom], float(self.p.lt_promote_value))
                        self.vals_lt[idx_prom] = torch.maximum(self.vals_lt[idx_prom], tgt)
                        self.lt_promoted_flag[idx_prom] = 1
                    else:
                        # accumulate small LT step per qualifying epoch
                        step = float(self.p.lt_occ_scale * self.p.occ_inc)
                        self.vals_lt.index_add_(0, idx_prom,
                            torch.full_like(idx_prom, step, dtype=self.vals_lt.dtype))
                    self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

            # clear per-epoch view bits for the NEW epoch
            if self.seen_view_bits_e.numel() > 0:
                self.seen_view_bits_e.zero_()

        # 2) increment epoch index
        self.epoch += 1

        # 3) ST decay/reset
        if self.vals_st.numel() > 0 and self.p.st_decay_gamma < 0.9999:
            # gamma=0 -> reset to 0
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
    def _update_occupied_guarded_(self, idx: torch.Tensor, cam_center_world: torch.Tensor | None):
        if idx.numel() == 0: return

        # ST (fast)
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.occ_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        # counters (legacy + useful timestamps)
        self.hit_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.hit_count.dtype))
        self.neg_free_count.index_fill_(0, idx, torch.tensor(0, dtype=self.neg_free_count.dtype, device=self.device))
        self.pos_occ_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.pos_occ_count.dtype))
        self.last_occ_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_occ_epoch.dtype, device=self.device))

        # view sector bit (8-way yaw)
        if cam_center_world is not None:
            ijk = self._unhash_keys(self.keys[idx])
            centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
            v = centers - torch.as_tensor(cam_center_world, dtype=self.origin.dtype, device=self.device).reshape(1,3)
            yaw = torch.atan2(v[:,1], v[:,0])
            sector = torch.floor((yaw + math.pi) / (2*math.pi) * 8.0) % 8.0
            bits = (1 << sector.to(torch.int16)).to(self.view_bits.dtype)
            self.view_bits[idx] = self.view_bits[idx] | bits

            # --- NEW: per-epoch view bits (for promotion gating) ---
            bits_e = (1 << sector.to(torch.int16)).to(self.seen_view_bits_e.dtype)
            self.seen_view_bits_e[idx] = self.seen_view_bits_e[idx] | bits_e

        # --- NEW: mark seen this epoch for epoch-based promotion ---
        self.seen_occ_epoch[idx] = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)

        # NOTE: old "promote by pos_occ_count" removed; promotion happens in next_epoch()

    def _update_free_guarded_(self, idx: torch.Tensor):
        if idx.numel() == 0: return

        # ST (fast)
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.free_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        # counters
        self.pos_occ_count.index_fill_(0, idx, torch.tensor(0, dtype=self.pos_occ_count.dtype, device=self.device))
        self.neg_free_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.neg_free_count.dtype))
        self.last_free_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_free_epoch.dtype, device=self.device))

        # LT carving (optional)
        if not self.p.lt_allow_free:
            return  # never carve LT

        # Guarded LT carving (slow)
        enough_neg = self.neg_free_count[idx] >= self.p.lt_free_k_neg
        not_recent_occ = (self.epoch - self.last_occ_epoch[idx]) > self.p.lt_free_recent_occ_epochs
        carve_mask = enough_neg & not_recent_occ
        if carve_mask.any():
            idx_carve = idx[carve_mask]
            self.vals_lt.index_add_(0, idx_carve,
                torch.full_like(idx_carve, self.p.lt_free_scale * self.p.free_inc, dtype=self.vals_lt.dtype))
            self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

    # ---------- integration (vectorized) ----------
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

        # Occupied endpoints
        ijk_occ = self._world_to_ijk(pts)
        keys_occ = torch.unique(self._hash_ijk(ijk_occ))
        idx_occ = self._ensure_and_index(keys_occ)
        self._update_occupied_guarded_(idx_occ, cam_center_world)

        if not carve_free:
            self.vals = self._display_vals(); return

        # Free carving by sampled points along rays
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
        # exclude endpoint voxels
        if keys_occ.numel() > 0 and keys_free.numel() > 0:
            keep = ~torch.isin(keys_free, keys_occ)
            keys_free = keys_free[keep]
        if keys_free.numel() > 0:
            keys_free = torch.unique(keys_free)
            idx_free = self._ensure_and_index(keys_free)
            self._update_free_guarded_(idx_free)

        self.vals = self._display_vals()

    @torch.no_grad()
    def integrate_points_with_cameras(
        self,
        points_world: torch.Tensor,     # (M,3)
        cams_world: torch.Tensor,       # (M,3) per-point camera center
        *,
        carve_free: bool = True,
        max_range: float | None = 12.0,
        z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
        samples_per_voxel: float = 1.1,
        ray_stride: int = 1,
        max_free_rays: int | None = 200_000,
    ):
        if points_world.numel() == 0:
            return
        dev, dt = self.device, self.dtype
        P = points_world.to(dev, dt)
        C = cams_world.to(dev, dt)

        finite = torch.isfinite(P).all(dim=1) & torch.isfinite(C).all(dim=1)
        P, C = P[finite], C[finite]
        if P.numel() == 0: return

        if max_range is not None:
            d = torch.linalg.norm(P - C, dim=1)
            keep = d <= max_range
            P, C = P[keep], C[keep]
        if z_clip is not None and P.numel() > 0:
            z0, z1 = z_clip
            keep = (P[:,2] >= z0) & (P[:,2] <= z1)
            P, C = P[keep], C[keep]
        if P.numel() == 0: return

        # Occupied (unique endpoints)
        ijk_occ = self._world_to_ijk(P)
        keys_occ = torch.unique(self._hash_ijk(ijk_occ))
        idx_occ = self._ensure_and_index(keys_occ)
        self._update_occupied_guarded_(idx_occ, cam_center_world=None)

        if not carve_free:
            self.vals = self._display_vals(); return

        # Free carving
        if ray_stride > 1:
            P = P[::ray_stride]; C = C[::ray_stride]
        if (max_free_rays is not None) and (P.shape[0] > max_free_rays):
            ridx = torch.randperm(P.shape[0], device=dev)[:max_free_rays]
            P, C = P[ridx], C[ridx]
        if P.numel() == 0:
            self.vals = self._display_vals(); return

        V = P - C
        seg_len = torch.linalg.norm(V, dim=1)
        steps = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
        max_steps = int(steps.max().item())

        base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
        t = base[None, :] / steps.to(dt)[:, None]
        t = torch.minimum(t,
            torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt),
                            torch.tensor(0.0, device=dev, dtype=dt)))
        mask = (t < 1.0)

        samples = C.unsqueeze(1) + t.unsqueeze(-1) * V.unsqueeze(1)
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
        return [ (int(a), int(b), int(c)) for a,b,c in arr ]

    def occupied_voxels(self, zmin: float | None = None, zmax: float | None = None):
        return self.occupied_ijk_numpy(zmin=zmin, zmax=zmax)

    def ijk_to_center(self, ijk) -> np.ndarray:
        ijk_t = torch.as_tensor(ijk, device=self.device, dtype=torch.float32)
        centers = self.origin + (ijk_t + 0.5) * self.p.voxel_size
        return centers.detach().cpu().numpy()




# ----------------- Grid -----------------
class TorchSparseVoxelGrid:
    """
    Sparse log-odds voxel map with:
      - ST (short-term) fast updates (±occ_inc/free_inc) + per-epoch decay/reset,
      - LT (long-term) updated ONLY at epoch boundaries if voxel was seen occupied
        in >= promote_epochs distinct epochs (and optionally multi-view),
      - optional guarded LT carving (or disable entirely via lt_allow_free=False).

    Keys: int64 hashes of (i,j,k). State tensors are aligned to `self.keys`.
    """
    def __init__(self, origin_xyz, params: VoxelParams, device=None, dtype=torch.float32):
        self.origin = torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3)
        self.p = params
        self.device = self.origin.device
        self.dtype = dtype

        # master keyset
        self.keys = torch.empty(0, dtype=torch.int64, device=self.device)  # (N,)

        # L1/L2 log-odds (aligned to keys)
        self.vals_st = torch.empty(0, dtype=dtype, device=self.device)     # short-term (L1)
        self.vals_lt = torch.empty(0, dtype=dtype, device=self.device)     # long-term  (L2)

        # compatibility "display" (computed via _display_vals)
        self.vals = torch.empty(0, dtype=dtype, device=self.device)

        # per-voxel counters/guards (aligned to keys)
        self.hit_count       = torch.empty(0, dtype=torch.int32, device=self.device)  # total + hits (legacy)
        self.pos_occ_count   = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive + hits (legacy)
        self.neg_free_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive frees
        self.last_occ_epoch  = torch.empty(0, dtype=torch.int32, device=self.device)
        self.last_free_epoch = torch.empty(0, dtype=torch.int32, device=self.device)
        self.view_bits       = torch.empty(0, dtype=torch.int16, device=self.device)  # 8-dir bitmask (cumulative this session)

        # ---------- NEW: epoch-based promotion state ----------
        self.seen_occ_epoch   = torch.empty(0, dtype=torch.int32, device=self.device)  # last epoch index when seen occupied
        self.seen_view_bits_e = torch.empty(0, dtype=torch.int16, device=self.device)  # per-epoch view bits
        self.occ_epoch_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # number of distinct epochs seen occupied
        self.view_bits_cum    = torch.empty(0, dtype=torch.int16, device=self.device)  # OR of per-epoch view bits
        self.lt_promoted_flag = torch.empty(0, dtype=torch.uint8, device=self.device)  # 0/1: already promoted (for "once")

        self.epoch: int = 0  # advance once per integration round

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

    def _unhash_keys(self, keys: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = ( keys        & ((1 << 21) - 1)) - off
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    def _ensure_and_index(self, upd_keys: torch.Tensor) -> torch.Tensor:
        """Merge upd_keys into self.keys; realign all state; return positions of upd_keys."""
        if upd_keys.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        all_keys = torch.cat([self.keys, upd_keys], dim=0)
        uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        idx_old = inv[:n_old]
        idx_upd = inv[n_old:]

        if uk.numel() != n_old:
            def grow_like(src, fill=0):
                out = torch.empty(uk.shape[0], dtype=src.dtype, device=src.device)
                if src.numel() > 0:
                    out[idx_old] = src
                if uk.shape[0] > n_old:
                    mask_new = torch.ones(uk.shape[0], dtype=torch.bool, device=src.device)
                    if src.numel() > 0:
                        mask_new[idx_old] = False
                    out[mask_new] = fill
                return out

            self.keys = uk
            # grow state
            self.vals_st       = grow_like(self.vals_st,       0)
            self.vals_lt       = grow_like(self.vals_lt,       0)
            self.vals          = grow_like(self.vals,          0)
            self.hit_count     = grow_like(self.hit_count,     0)
            self.pos_occ_count = grow_like(self.pos_occ_count, 0)
            self.neg_free_count= grow_like(self.neg_free_count,0)
            self.last_occ_epoch= grow_like(self.last_occ_epoch,-2**31+1)
            self.last_free_epoch=grow_like(self.last_free_epoch,-2**31+1)
            self.view_bits     = grow_like(self.view_bits,     0)

            # NEW epoch-based state
            self.seen_occ_epoch   = grow_like(self.seen_occ_epoch,   -2**31+1)
            self.seen_view_bits_e = grow_like(self.seen_view_bits_e, 0)
            self.occ_epoch_count  = grow_like(self.occ_epoch_count,  0)
            self.view_bits_cum    = grow_like(self.view_bits_cum,    0)
            self.lt_promoted_flag = grow_like(self.lt_promoted_flag, 0)

        return idx_upd

    # ---------- display ----------
    def _display_vals(self) -> torch.Tensor:
        # Simple and effective: show whichever memory is more confident of occupancy.
        return torch.maximum(self.vals_st, self.vals_lt)

    # ---------- epoch advance & promotion finalization ----------
    def next_epoch(self):
        """
        Finalize current epoch (epoch-based promotion), then advance epoch and handle ST decay/reset.
        """
        # 1) finalize BEFORE incrementing epoch
        if self.keys.numel() > 0:
            now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
            seen_now = (self.seen_occ_epoch == now)  # voxels that had any occupied endpoint this epoch

            if seen_now.any():
                # count this epoch once per voxel
                self.occ_epoch_count[seen_now] = torch.clamp(
                    self.occ_epoch_count[seen_now] + 1,
                    max=torch.iinfo(self.occ_epoch_count.dtype).max
                )
                # OR per-epoch view bits into cumulative mask
                self.view_bits_cum[seen_now] = self.view_bits_cum[seen_now] | self.seen_view_bits_e[seen_now]

                # Promotion gates
                epochs_ok = self.occ_epoch_count[seen_now] >= int(self.p.promote_epochs)
                if int(self.p.lt_min_view_sectors) > 1:
                    # popcount of 8-bit mask (branchless)
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
                        # set LT to at least lt_promote_value exactly once
                        tgt = torch.full_like(self.vals_lt[idx_prom], float(self.p.lt_promote_value))
                        self.vals_lt[idx_prom] = torch.maximum(self.vals_lt[idx_prom], tgt)
                        self.lt_promoted_flag[idx_prom] = 1
                    else:
                        # accumulate small LT step per qualifying epoch
                        step = float(self.p.lt_occ_scale * self.p.occ_inc)
                        self.vals_lt.index_add_(0, idx_prom,
                            torch.full_like(idx_prom, step, dtype=self.vals_lt.dtype))
                    self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

            # clear per-epoch view bits for the NEW epoch
            if self.seen_view_bits_e.numel() > 0:
                self.seen_view_bits_e.zero_()


        # -------- NEW: LT demotion (epoch-based) --------
        if self.p.lt_demote_enable and self.keys.numel() > 0:
            # voxels with enough consecutive frees
            enough_neg = self.neg_free_count >= int(self.p.lt_demote_k_neg)
            # ...and not seen occupied recently
            no_recent_occ = (self.epoch - self.last_occ_epoch) >= int(self.p.lt_demote_min_no_occ_epochs)

            demote_mask = enough_neg & no_recent_occ
            if demote_mask.any():
                idx_demote = torch.nonzero(demote_mask, as_tuple=False).squeeze(-1)

                # Subtract a step from LT (epoch-based; gentle, independent of per-ray carving)
                step = float(self.p.lt_demote_step)
                self.vals_lt.index_add_(0, idx_demote,
                    torch.full_like(idx_demote, -step, dtype=self.vals_lt.dtype))
                # Clamp to a soft floor (don’t go too negative unless you want to)
                self.vals_lt.clamp_(min=max(self.p.l_min, float(self.p.lt_demote_floor)),
                                    max=self.p.l_max)

                # Optional: if LT fell below occupancy, “forget” promotion so voxel can re-earn LT later
                if self.p.lt_reset_promotion_on_demote:
                    not_occ_anymore = self.vals_lt[idx_demote] <= self.p.occ_thresh
                    if not_occ_anymore.any() and self.lt_promoted_flag.numel() > 0:
                        idx_reset = idx_demote[not_occ_anymore]
                        # allow re-promotion
                        self.lt_promoted_flag[idx_reset] = 0
                        # you may also want to soften historical evidence:
                        # keep occ_epoch_count/view_bits_cum, or decay them:
                        # self.occ_epoch_count[idx_reset] = torch.clamp(
                        #     self.occ_epoch_count[idx_reset] - 1, min=0)
                        # self.view_bits_cum[idx_reset] = 0
                        # (choose policy you prefer)

        # 2) increment epoch index
        self.epoch += 1

        # 3) ST decay/reset
        if self.vals_st.numel() > 0 and self.p.st_decay_gamma < 0.9999:
            # gamma=0 -> reset to 0
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
    def _update_occupied_guarded_(self, idx: torch.Tensor, cam_center_world: torch.Tensor | None):
        if idx.numel() == 0: return

        # ST (fast)
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.occ_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        # counters (legacy + useful timestamps)
        self.hit_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.hit_count.dtype))
        self.neg_free_count.index_fill_(0, idx, torch.tensor(0, dtype=self.neg_free_count.dtype, device=self.device))
        self.pos_occ_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.pos_occ_count.dtype))
        self.last_occ_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_occ_epoch.dtype, device=self.device))

        # view sector bit (8-way yaw)
        if cam_center_world is not None:
            ijk = self._unhash_keys(self.keys[idx])
            centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
            v = centers - torch.as_tensor(cam_center_world, dtype=self.origin.dtype, device=self.device).reshape(1,3)
            yaw = torch.atan2(v[:,1], v[:,0])
            sector = torch.floor((yaw + math.pi) / (2*math.pi) * 8.0) % 8.0
            bits = (1 << sector.to(torch.int16)).to(self.view_bits.dtype)
            self.view_bits[idx] = self.view_bits[idx] | bits

            # --- NEW: per-epoch view bits (for promotion gating) ---
            bits_e = (1 << sector.to(torch.int16)).to(self.seen_view_bits_e.dtype)
            self.seen_view_bits_e[idx] = self.seen_view_bits_e[idx] | bits_e

        # --- NEW: mark seen this epoch for epoch-based promotion ---
        self.seen_occ_epoch[idx] = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)

        # NOTE: old "promote by pos_occ_count" removed; promotion happens in next_epoch()

    def _update_free_guarded_(self, idx: torch.Tensor):
        if idx.numel() == 0: return

        # ST (fast)
        self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.free_inc, dtype=self.vals_st.dtype))
        self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

        # counters
        self.pos_occ_count.index_fill_(0, idx, torch.tensor(0, dtype=self.pos_occ_count.dtype, device=self.device))
        self.neg_free_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.neg_free_count.dtype))
        self.last_free_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_free_epoch.dtype, device=self.device))

        # LT carving (optional)
        if not self.p.lt_allow_free:
            return  # never carve LT

        # Guarded LT carving (slow)
        enough_neg = self.neg_free_count[idx] >= self.p.lt_free_k_neg
        not_recent_occ = (self.epoch - self.last_occ_epoch[idx]) > self.p.lt_free_recent_occ_epochs
        carve_mask = enough_neg & not_recent_occ
        if carve_mask.any():
            idx_carve = idx[carve_mask]
            self.vals_lt.index_add_(0, idx_carve,
                torch.full_like(idx_carve, self.p.lt_free_scale * self.p.free_inc, dtype=self.vals_lt.dtype))
            self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

    # ---------- integration (vectorized) ----------
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

        # Occupied endpoints
        ijk_occ = self._world_to_ijk(pts)
        keys_occ = torch.unique(self._hash_ijk(ijk_occ))
        idx_occ = self._ensure_and_index(keys_occ)
        self._update_occupied_guarded_(idx_occ, cam_center_world)

        if not carve_free:
            self.vals = self._display_vals(); return

        # Free carving by sampled points along rays
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
        # exclude endpoint voxels
        if keys_occ.numel() > 0 and keys_free.numel() > 0:
            keep = ~torch.isin(keys_free, keys_occ)
            keys_free = keys_free[keep]
        if keys_free.numel() > 0:
            keys_free = torch.unique(keys_free)
            idx_free = self._ensure_and_index(keys_free)
            self._update_free_guarded_(idx_free)

        self.vals = self._display_vals()

    @torch.no_grad()
    def integrate_points_with_cameras(
        self,
        points_world: torch.Tensor,     # (M,3)
        cams_world: torch.Tensor,       # (M,3) per-point camera center
        *,
        carve_free: bool = True,
        max_range: float | None = 12.0,
        z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
        samples_per_voxel: float = 1.1,
        ray_stride: int = 1,
        max_free_rays: int | None = 200_000,
    ):
        if points_world.numel() == 0:
            return
        dev, dt = self.device, self.dtype
        P = points_world.to(dev, dt)
        C = cams_world.to(dev, dt)

        finite = torch.isfinite(P).all(dim=1) & torch.isfinite(C).all(dim=1)
        P, C = P[finite], C[finite]
        if P.numel() == 0: return

        if max_range is not None:
            d = torch.linalg.norm(P - C, dim=1)
            keep = d <= max_range
            P, C = P[keep], C[keep]
        if z_clip is not None and P.numel() > 0:
            z0, z1 = z_clip
            keep = (P[:,2] >= z0) & (P[:,2] <= z1)
            P, C = P[keep], C[keep]
        if P.numel() == 0: return

        # Occupied (unique endpoints)
        ijk_occ = self._world_to_ijk(P)
        keys_occ = torch.unique(self._hash_ijk(ijk_occ))
        idx_occ = self._ensure_and_index(keys_occ)
        self._update_occupied_guarded_(idx_occ, cam_center_world=None)

        if not carve_free:
            self.vals = self._display_vals(); return

        # Free carving
        if ray_stride > 1:
            P = P[::ray_stride]; C = C[::ray_stride]
        if (max_free_rays is not None) and (P.shape[0] > max_free_rays):
            ridx = torch.randperm(P.shape[0], device=dev)[:max_free_rays]
            P, C = P[ridx], C[ridx]
        if P.numel() == 0:
            self.vals = self._display_vals(); return

        V = P - C
        seg_len = torch.linalg.norm(V, dim=1)
        steps = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
        max_steps = int(steps.max().item())

        base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
        t = base[None, :] / steps.to(dt)[:, None]
        t = torch.minimum(t,
            torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt),
                            torch.tensor(0.0, device=dev, dtype=dt)))
        mask = (t < 1.0)

        samples = C.unsqueeze(1) + t.unsqueeze(-1) * V.unsqueeze(1)
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
        return [ (int(a), int(b), int(c)) for a,b,c in arr ]

    def occupied_voxels(self, zmin: float | None = None, zmax: float | None = None):
        return self.occupied_ijk_numpy(zmin=zmin, zmax=zmax)

    def ijk_to_center(self, ijk) -> np.ndarray:
        ijk_t = torch.as_tensor(ijk, device=self.device, dtype=torch.float32)
        centers = self.origin + (ijk_t + 0.5) * self.p.voxel_size
        return centers.detach().cpu().numpy()




# # ----------------- Grid -----------------
# class TorchSparseVoxelGrid:
#     """
#     Sparse log-odds voxel map with:
#       - ST (short-term) fast updates (±occ_inc/free_inc) + per-epoch decay/reset,
#       - LT (long-term) updated ONLY at epoch boundaries if voxel was seen occupied
#         in >= promote_epochs distinct epochs (and optionally multi-view),
#       - optional guarded LT carving (or disable entirely via lt_allow_free=False).

#     Keys: int64 hashes of (i,j,k). State tensors are aligned to `self.keys`.
#     """
#     def __init__(self, origin_xyz, params: VoxelParams, device=None, dtype=torch.float32):
#         self.origin = torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3)
#         self.p = params
#         self.device = self.origin.device
#         self.dtype = dtype

#         # master keyset
#         self.keys = torch.empty(0, dtype=torch.int64, device=self.device)  # (N,)

#         # L1/L2 log-odds (aligned to keys)
#         self.vals_st = torch.empty(0, dtype=dtype, device=self.device)     # short-term (L1)
#         self.vals_lt = torch.empty(0, dtype=dtype, device=self.device)     # long-term  (L2)

#         # compatibility "display" (computed via _display_vals)
#         self.vals = torch.empty(0, dtype=dtype, device=self.device)

#         # per-voxel counters/guards (aligned to keys)
#         self.hit_count       = torch.empty(0, dtype=torch.int32, device=self.device)  # total + hits (legacy)
#         self.pos_occ_count   = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive + hits (legacy)
#         self.neg_free_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # consecutive frees
#         self.last_occ_epoch  = torch.empty(0, dtype=torch.int32, device=self.device)
#         self.last_free_epoch = torch.empty(0, dtype=torch.int32, device=self.device)
#         self.view_bits       = torch.empty(0, dtype=torch.int16, device=self.device)  # 8-dir bitmask (cumulative this session)

#         # ---------- NEW: epoch-based promotion state ----------
#         self.seen_occ_epoch   = torch.empty(0, dtype=torch.int32, device=self.device)  # last epoch index when seen occupied
#         self.seen_view_bits_e = torch.empty(0, dtype=torch.int16, device=self.device)  # per-epoch view bits
#         self.occ_epoch_count  = torch.empty(0, dtype=torch.int16, device=self.device)  # number of distinct epochs seen occupied
#         self.view_bits_cum    = torch.empty(0, dtype=torch.int16, device=self.device)  # OR of per-epoch view bits
#         self.lt_promoted_flag = torch.empty(0, dtype=torch.uint8, device=self.device)  # 0/1: already promoted (for "once")

#         self.epoch: int = 0  # advance once per integration round
        
#         self._bootstrap_mode = False   # True while ingesting the full map (epoch 0 seeding)
#         self._bootstrap_done = False   # True after LT is seeded and frozen

#     def begin_bootstrap(self):
#         """
#         Enable 'epoch 0' long-term seeding mode.
#         Call this before you feed the *full* reconstruction frames.
#         """
#         self._bootstrap_mode = True
#         self._bootstrap_done = False
#         # (Optional) make ST persistent during seeding if you want to accumulate many frames
#         # self.p.st_decay_gamma = 1.0

#     @torch.no_grad()
#     def end_bootstrap(self):
#         """
#         Promote current occupied voxels to LT once, clear ST, and mark LT as frozen.
#         Call this after you have fed the full reconstruction.
#         """
#         if self.keys.numel() > 0:
#             occ_mask = self.occupied_mask()  # use current display occupancy (LT|ST)
#             if occ_mask.any():
#                 idx_all = torch.arange(self.keys.numel(), device=self.device, dtype=torch.int64)
#                 idx_prom = idx_all[occ_mask]

#                 # Promote to at least lt_promote_value
#                 tgt = torch.full_like(self.vals_lt[idx_prom], float(self.p.lt_promote_value))
#                 self.vals_lt[idx_prom] = torch.maximum(self.vals_lt[idx_prom], tgt)
#                 self.lt_promoted_flag[idx_prom] = 1
#                 self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

#             # Clear ST so display starts from clean LT
#             if self.vals_st.numel() > 0:
#                 self.vals_st.zero_()

#         # Mark bootstrap complete
#         self._bootstrap_mode = False
#         self._bootstrap_done = True

#     def lock_long_term(self):
#         """
#         Permanently disallow any future LT changes (no promotion, no carving).
#         Call this after end_bootstrap() if you want LT strictly frozen.
#         """
#         self._bootstrap_done = True
#         self.p.lt_allow_free = False   # never carve LT again

#     # ---------- utilities ----------
#     def _world_to_ijk(self, pts: torch.Tensor) -> torch.Tensor:
#         rel = (pts - self.origin) / self.p.voxel_size
#         return torch.floor(rel).to(torch.int64)

#     @staticmethod
#     def _hash_ijk(ijk: torch.Tensor) -> torch.Tensor:
#         off = (1 << 20)
#         i = ijk[..., 0] + off
#         j = ijk[..., 1] + off
#         k = ijk[..., 2] + off
#         return (i << 42) ^ (j << 21) ^ k

#     def _unhash_keys(self, keys: torch.Tensor) -> torch.Tensor:
#         off = (1 << 20)
#         i = ((keys >> 42) & ((1 << 21) - 1)) - off
#         j = ((keys >> 21) & ((1 << 21) - 1)) - off
#         k = ( keys        & ((1 << 21) - 1)) - off
#         return torch.stack([i, j, k], dim=-1).to(torch.int32)

#     def _ensure_and_index(self, upd_keys: torch.Tensor) -> torch.Tensor:
#         """Merge upd_keys into self.keys; realign all state; return positions of upd_keys."""
#         if upd_keys.numel() == 0:
#             return torch.empty(0, dtype=torch.int64, device=self.device)

#         all_keys = torch.cat([self.keys, upd_keys], dim=0)
#         uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
#         n_old = self.keys.numel()
#         idx_old = inv[:n_old]
#         idx_upd = inv[n_old:]

#         if uk.numel() != n_old:
#             def grow_like(src, fill=0):
#                 out = torch.empty(uk.shape[0], dtype=src.dtype, device=src.device)
#                 if src.numel() > 0:
#                     out[idx_old] = src
#                 if uk.shape[0] > n_old:
#                     mask_new = torch.ones(uk.shape[0], dtype=torch.bool, device=src.device)
#                     if src.numel() > 0:
#                         mask_new[idx_old] = False
#                     out[mask_new] = fill
#                 return out

#             self.keys = uk
#             # grow state
#             self.vals_st       = grow_like(self.vals_st,       0)
#             self.vals_lt       = grow_like(self.vals_lt,       0)
#             self.vals          = grow_like(self.vals,          0)
#             self.hit_count     = grow_like(self.hit_count,     0)
#             self.pos_occ_count = grow_like(self.pos_occ_count, 0)
#             self.neg_free_count= grow_like(self.neg_free_count,0)
#             self.last_occ_epoch= grow_like(self.last_occ_epoch,-2**31+1)
#             self.last_free_epoch=grow_like(self.last_free_epoch,-2**31+1)
#             self.view_bits     = grow_like(self.view_bits,     0)

#             # NEW epoch-based state
#             self.seen_occ_epoch   = grow_like(self.seen_occ_epoch,   -2**31+1)
#             self.seen_view_bits_e = grow_like(self.seen_view_bits_e, 0)
#             self.occ_epoch_count  = grow_like(self.occ_epoch_count,  0)
#             self.view_bits_cum    = grow_like(self.view_bits_cum,    0)
#             self.lt_promoted_flag = grow_like(self.lt_promoted_flag, 0)

#         return idx_upd

#     # ---------- display ----------
#     def _display_vals(self) -> torch.Tensor:
#         # Simple and effective: show whichever memory is more confident of occupancy.
#         return torch.maximum(self.vals_st, self.vals_lt)

#     # ---------- epoch advance & promotion finalization ----------
#     def next_epoch(self):
#         """
#         Finalize current epoch (epoch-based promotion) *only if* bootstrap not done,
#         then advance epoch and handle ST decay/reset.
#         """
#         # 1) finalize BEFORE incrementing epoch — only if LT is not frozen
#         if (self.keys.numel() > 0) and (not self._bootstrap_done):
#             now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
#             seen_now = (self.seen_occ_epoch == now)  # voxels that had any occupied endpoint this epoch

#             if seen_now.any():
#                 # increment per-voxel distinct-epoch counter
#                 self.occ_epoch_count[seen_now] = torch.clamp(
#                     self.occ_epoch_count[seen_now] + 1,
#                     max=torch.iinfo(self.occ_epoch_count.dtype).max
#                 )
#                 # OR per-epoch view bits into cumulative
#                 self.view_bits_cum[seen_now] = self.view_bits_cum[seen_now] | self.seen_view_bits_e[seen_now]

#                 # Promotion gates
#                 epochs_ok = self.occ_epoch_count[seen_now] >= int(self.p.promote_epochs)
#                 if int(self.p.lt_min_view_sectors) > 1:
#                     vb = self.view_bits_cum[seen_now].to(torch.int16)
#                     pop = ((vb & 1 > 0).to(torch.int16) +
#                         ((vb >> 1) & 1 > 0).to(torch.int16) +
#                         ((vb >> 2) & 1 > 0).to(torch.int16) +
#                         ((vb >> 3) & 1 > 0).to(torch.int16) +
#                         ((vb >> 4) & 1 > 0).to(torch.int16) +
#                         ((vb >> 5) & 1 > 0).to(torch.int16) +
#                         ((vb >> 6) & 1 > 0).to(torch.int16) +
#                         ((vb >> 7) & 1 > 0).to(torch.int16))
#                     mv_ok = pop >= int(self.p.lt_min_view_sectors)
#                 else:
#                     mv_ok = torch.ones_like(self.occ_epoch_count[seen_now], dtype=torch.bool, device=self.device)

#                 prom_local = epochs_ok & mv_ok
#                 if prom_local.any():
#                     idx_all = torch.arange(self.keys.numel(), device=self.device, dtype=torch.int64)
#                     idx_prom = idx_all[seen_now][prom_local]

#                     if self.p.lt_promotion_mode == "once":
#                         tgt = torch.full_like(self.vals_lt[idx_prom], float(self.p.lt_promote_value))
#                         self.vals_lt[idx_prom] = torch.maximum(self.vals_lt[idx_prom], tgt)
#                         self.lt_promoted_flag[idx_prom] = 1
#                     else:
#                         step = float(self.p.lt_occ_scale * self.p.occ_inc)
#                         self.vals_lt.index_add_(0, idx_prom,
#                             torch.full_like(idx_prom, step, dtype=self.vals_lt.dtype))
#                     self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

#             # clear per-epoch view bits for the NEW epoch
#             if self.seen_view_bits_e.numel() > 0:
#                 self.seen_view_bits_e.zero_()

#         # 2) increment epoch index
#         self.epoch += 1

#         # 3) ST decay/reset (unchanged)
#         if self.vals_st.numel() > 0 and self.p.st_decay_gamma < 0.9999:
#             self.vals_st.mul_(self.p.st_decay_gamma).clamp_(min=self.p.l_min, max=self.p.l_max)


#     # ---------- queries ----------
#     def occupied_mask(self) -> torch.Tensor:
#         if self.keys.numel() == 0:
#             return torch.empty(0, dtype=torch.bool, device=self.device)
#         lt_occ = self.vals_lt > self.p.occ_thresh
#         st_occ = self.vals_st > (self.p.occ_thresh + self.p.st_margin)
#         return lt_occ | st_occ
    
#     def occupied_indices(self):
#         if self.keys.numel() == 0:
#             return torch.empty(0, dtype=torch.int64, device=self.device)
#         return self.keys[self.occupied_mask()]

#     # ---------- guarded updates (per-ray) ----------
#     def _update_occupied_guarded_(self, idx: torch.Tensor, cam_center_world: torch.Tensor | None):
#         if idx.numel() == 0: return

#         # ST (fast)
#         self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.occ_inc, dtype=self.vals_st.dtype))
#         self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

#         # counters (legacy + useful timestamps)
#         self.hit_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.hit_count.dtype))
#         self.neg_free_count.index_fill_(0, idx, torch.tensor(0, dtype=self.neg_free_count.dtype, device=self.device))
#         self.pos_occ_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.pos_occ_count.dtype))
#         self.last_occ_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_occ_epoch.dtype, device=self.device))

#         # view sector bit (8-way yaw)
#         if cam_center_world is not None:
#             ijk = self._unhash_keys(self.keys[idx])
#             centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
#             v = centers - torch.as_tensor(cam_center_world, dtype=self.origin.dtype, device=self.device).reshape(1,3)
#             yaw = torch.atan2(v[:,1], v[:,0])
#             sector = torch.floor((yaw + math.pi) / (2*math.pi) * 8.0) % 8.0
#             bits = (1 << sector.to(torch.int16)).to(self.view_bits.dtype)
#             self.view_bits[idx] = self.view_bits[idx] | bits

#             # --- NEW: per-epoch view bits (for promotion gating) ---
#             bits_e = (1 << sector.to(torch.int16)).to(self.seen_view_bits_e.dtype)
#             self.seen_view_bits_e[idx] = self.seen_view_bits_e[idx] | bits_e

#         # --- NEW: mark seen this epoch for epoch-based promotion ---
#         self.seen_occ_epoch[idx] = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)

#         # NOTE: old "promote by pos_occ_count" removed; promotion happens in next_epoch()

#     def _update_free_guarded_(self, idx: torch.Tensor):
#         if idx.numel() == 0: return

#         # ST (fast)
#         self.vals_st.index_add_(0, idx, torch.full_like(idx, self.p.free_inc, dtype=self.vals_st.dtype))
#         self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)

#         # counters
#         self.pos_occ_count.index_fill_(0, idx, torch.tensor(0, dtype=self.pos_occ_count.dtype, device=self.device))
#         self.neg_free_count.index_add_(0, idx, torch.ones_like(idx, dtype=self.neg_free_count.dtype))
#         self.last_free_epoch.index_fill_(0, idx, torch.tensor(self.epoch, dtype=self.last_free_epoch.dtype, device=self.device))

#         # LT carving (optional)
#         if not self.p.lt_allow_free:
#             return  # never carve LT

#         # Guarded LT carving (slow)
#         enough_neg = self.neg_free_count[idx] >= self.p.lt_free_k_neg
#         not_recent_occ = (self.epoch - self.last_occ_epoch[idx]) > self.p.lt_free_recent_occ_epochs
#         carve_mask = enough_neg & not_recent_occ
#         if carve_mask.any():
#             idx_carve = idx[carve_mask]
#             self.vals_lt.index_add_(0, idx_carve,
#                 torch.full_like(idx_carve, self.p.lt_free_scale * self.p.free_inc, dtype=self.vals_lt.dtype))
#             self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)

#     # ---------- integration (vectorized) ----------
#     @torch.no_grad()
#     def integrate_frame_torch(
#         self,
#         points_world: torch.Tensor,     # (N,3)
#         cam_center_world: torch.Tensor, # (3,)
#         *,
#         carve_free: bool = True,
#         max_range: float | None = 12.0,
#         z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
#         ray_stride: int = 3,
#         max_free_rays: int | None = 40000,
#         samples_per_voxel: float = 1.05,
#     ):
#         if points_world.numel() == 0:
#             return
#         dev, dt = self.device, self.dtype
#         pts = points_world.to(dev, dt)
#         cam = cam_center_world.to(dev, dt).reshape(1,3)

#         finite = torch.isfinite(pts).all(dim=1)
#         pts = pts[finite]
#         if pts.numel() == 0: return

#         if max_range is not None:
#             d = torch.linalg.norm(pts - cam, dim=1)
#             pts = pts[d <= max_range]
#         if z_clip is not None:
#             z0, z1 = z_clip
#             m = (pts[:,2] >= z0) & (pts[:,2] <= z1)
#             pts = pts[m]
#         if pts.numel() == 0: return

#         # Occupied endpoints
#         ijk_occ = self._world_to_ijk(pts)
#         keys_occ = torch.unique(self._hash_ijk(ijk_occ))
#         idx_occ = self._ensure_and_index(keys_occ)
#         self._update_occupied_guarded_(idx_occ, cam_center_world)

#         if not carve_free:
#             self.vals = self._display_vals(); return

#         # Free carving by sampled points along rays
#         pts_free = pts[::ray_stride] if ray_stride > 1 else pts
#         if (max_free_rays is not None) and (pts_free.shape[0] > max_free_rays):
#             ridx = torch.randperm(pts_free.shape[0], device=dev)[:max_free_rays]
#             pts_free = pts_free[ridx]
#         if pts_free.numel() == 0:
#             self.vals = self._display_vals(); return

#         vec = pts_free - cam
#         seg_len = torch.linalg.norm(vec, dim=1)
#         steps_per_ray = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
#         max_steps = int(steps_per_ray.max().item())

#         base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
#         t = base[None, :] / steps_per_ray.to(dt)[:, None]
#         t = torch.minimum(t,
#             torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt),
#                             torch.tensor(0.0, device=dev, dtype=dt)))
#         mask = (t < 1.0)
#         samples = cam + t.unsqueeze(-1) * vec.unsqueeze(1)
#         samples = samples[mask]

#         ijk_free = self._world_to_ijk(samples)
#         keys_free = self._hash_ijk(ijk_free)
#         # exclude endpoint voxels
#         if keys_occ.numel() > 0 and keys_free.numel() > 0:
#             keep = ~torch.isin(keys_free, keys_occ)
#             keys_free = keys_free[keep]
#         if keys_free.numel() > 0:
#             keys_free = torch.unique(keys_free)
#             idx_free = self._ensure_and_index(keys_free)
#             self._update_free_guarded_(idx_free)

#         self.vals = self._display_vals()

#     @torch.no_grad()
#     def integrate_points_with_cameras(
#         self,
#         points_world: torch.Tensor,     # (M,3)
#         cams_world: torch.Tensor,       # (M,3) per-point camera center
#         *,
#         carve_free: bool = True,
#         max_range: float | None = 12.0,
#         z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
#         samples_per_voxel: float = 1.1,
#         ray_stride: int = 1,
#         max_free_rays: int | None = 200_000,
#     ):
#         if points_world.numel() == 0:
#             return
#         dev, dt = self.device, self.dtype
#         P = points_world.to(dev, dt)
#         C = cams_world.to(dev, dt)

#         finite = torch.isfinite(P).all(dim=1) & torch.isfinite(C).all(dim=1)
#         P, C = P[finite], C[finite]
#         if P.numel() == 0: return

#         if max_range is not None:
#             d = torch.linalg.norm(P - C, dim=1)
#             keep = d <= max_range
#             P, C = P[keep], C[keep]
#         if z_clip is not None and P.numel() > 0:
#             z0, z1 = z_clip
#             keep = (P[:,2] >= z0) & (P[:,2] <= z1)
#             P, C = P[keep], C[keep]
#         if P.numel() == 0: return
        

#         # Occupied (unique endpoints)
#         ijk_occ = self._world_to_ijk(P)
#         keys_occ = torch.unique(self._hash_ijk(ijk_occ))
#         idx_occ = self._ensure_and_index(keys_occ)
#         self._update_occupied_guarded_(idx_occ, cam_center_world=None)


#         if not carve_free:
#             self.vals = self._display_vals(); return

#         # Free carving
#         if ray_stride > 1:
#             P = P[::ray_stride]; C = C[::ray_stride]
#         if (max_free_rays is not None) and (P.shape[0] > max_free_rays):
#             ridx = torch.randperm(P.shape[0], device=dev)[:max_free_rays]
#             P, C = P[ridx], C[ridx]
#         if P.numel() == 0:
#             self.vals = self._display_vals(); return


#         V = P - C
#         seg_len = torch.linalg.norm(V, dim=1)
#         steps = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
#         max_steps = int(steps.max().item())

#         base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
#         t = base[None, :] / steps.to(dt)[:, None]
#         t = torch.minimum(t,
#             torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt),
#                             torch.tensor(0.0, device=dev, dtype=dt)))
#         mask = (t < 1.0)

#         samples = C.unsqueeze(1) + t.unsqueeze(-1) * V.unsqueeze(1)
#         samples = samples[mask]
#         ijk_free = self._world_to_ijk(samples)
#         keys_free = self._hash_ijk(ijk_free)


#         # if keys_occ.numel() > 0 and keys_free.numel() > 0:
#         #     keep = ~torch.isin(keys_free, keys_occ)
#         #     keys_free = keys_free[keep]
#         # if keys_free.numel() > 0:
#         #     keys_free = torch.unique(keys_free)
#         #     idx_free = self._ensure_and_index(keys_free)
#         #     self._update_free_guarded_(idx_free)
            
            
#         # --- remove endpoint voxels from free set, efficiently ---
#         print(keys_free.numel())
#         if keys_free.numel() > 0:
#             keys_free = torch.unique(keys_free)  # de-dup early

#         # if keys_free.numel() > 200_000:
#         #     print("too many")
#         #     ridx = torch.randperm(keys_free.numel(), device=keys_free.device)[:200_000]
#         #     keys_free = keys_free[ridx]

#         if keys_occ.numel() > 0 and keys_free.numel() > 0:
#             keys_occ = torch.unique(keys_occ)

#             # Fast set difference without torch.isin
#             all_keys = torch.cat([keys_free, keys_occ])
#             u, inv, counts = torch.unique(all_keys, sorted=True,
#                                         return_inverse=True, return_counts=True)

#             mask_free = counts[inv[:keys_free.numel()]] == 1
#             keys_free = keys_free[mask_free]
            


#         if keys_free.numel() > 0:
#             # keys_free already unique; proceed

#             idx_free = self._ensure_and_index(keys_free)

#             self._update_free_guarded_(idx_free)
            


        
#         del keys_free, ijk_free, keys_occ

#         self.vals = self._display_vals()

#     # ---------- exports ----------
#     def to_numpy(self):
#         return self.keys.detach().cpu().numpy(), self._display_vals().detach().cpu().numpy()

#     def occupied_ijk_numpy(self, zmin=None, zmax=None):
#         if self.keys.numel() == 0:
#             return np.empty((0,3), dtype=np.int32)
#         ijk = self._unhash_keys(self.keys)
#         occ_mask = self.occupied_mask()
#         if zmin is not None or zmax is not None:
#             centers = self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.p.voxel_size
#             cz = centers[:,2]
#             if zmin is not None: occ_mask &= (cz >= zmin)
#             if zmax is not None: occ_mask &= (cz <= zmax)
#         ijk_sel = ijk[occ_mask]
#         if ijk_sel.numel() == 0: return np.empty((0,3), dtype=np.int32)
#         arr = ijk_sel.cpu().numpy()
#         return [ (int(a), int(b), int(c)) for a,b,c in arr ]

#     def occupied_voxels(self, zmin: float | None = None, zmax: float | None = None):
#         return self.occupied_ijk_numpy(zmin=zmin, zmax=zmax)

#     def ijk_to_center(self, ijk) -> np.ndarray:
#         ijk_t = torch.as_tensor(ijk, device=self.device, dtype=torch.float32)
#         centers = self.origin + (ijk_t + 0.5) * self.p.voxel_size
#         return centers.detach().cpu().numpy()
