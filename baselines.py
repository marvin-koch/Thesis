"""
baselines.py — Additional fusion baselines for ECCV 2026 evaluation.

All baselines share the same pipeline (change detection → selective
reconstruction → cached projection).  Only the fusion rule differs.

Each class exposes the same interface used by align_probs_to_keys_soft
and the metrics loop in predict_step:
    .keys               : int64 sorted hash keys
    .p.occ_thresh       : occupancy threshold (float)
    .p.voxel_size       : voxel size (float)
    ._display_vals()    : (M,) logit-like values for occupancy
    ._unhash_keys(keys) : (M,3) int32
    .voxel_centers()    : (M,3) float
    .occupied_mask()    : (M,) bool

Baselines:
    1. TSDFFusion         — Curless & Levoy (1996) running weighted average
    2. EMAFusion          — Exponential moving average of occupancy evidence
    3. LastFrameFusion    — No temporal memory; current frame replaces map
    4. ConfWeightedFusion — DUSt3R confidence-weighted hit counting
    5. SimpleLogOdds      — Plain Bayesian log-odds (no ST/LT guarding)
"""
from __future__ import annotations

import torch
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# Param stub — metric code reads .p.occ_thresh / .p.voxel_size
# ─────────────────────────────────────────────────────────────────
@dataclass
class _BaselineParams:
    occ_thresh: float = 0.0
    voxel_size: float = 0.2


# ─────────────────────────────────────────────────────────────────
# Base class: sparse-hash infrastructure (matches TorchSparseVoxelGrid)
# ─────────────────────────────────────────────────────────────────
class _SparseBaselineGrid:
    """Minimal sparse voxel hash shared by every baseline."""

    def __init__(self, voxel_size: float = 0.2, device="cuda",
                 occ_thresh: float = 0.0):
        self.device = torch.device(device)
        self.dtype  = torch.float32
        self.p      = _BaselineParams(occ_thresh=occ_thresh,
                                       voxel_size=voxel_size)
        self.origin = torch.zeros(3, dtype=self.dtype, device=self.device)

        self.keys    = torch.empty(0, dtype=torch.int64, device=self.device)
        self.vals_st = torch.empty(0, dtype=self.dtype,  device=self.device)

    # ── hash / unhash (identical to TorchSparseVoxelGrid) ────────
    def _world_to_ijk(self, pts: torch.Tensor) -> torch.Tensor:
        return torch.floor((pts - self.origin) / self.p.voxel_size).to(torch.int64)

    @staticmethod
    def _hash_ijk(ijk: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ijk[..., 0] + off
        j = ijk[..., 1] + off
        k = ijk[..., 2] + off
        return (i << 42) ^ (j << 21) ^ k

    @staticmethod
    def _unhash_keys(keys: torch.Tensor) -> torch.Tensor:
        off = (1 << 20)
        i = ((keys >> 42) & ((1 << 21) - 1)) - off
        j = ((keys >> 21) & ((1 << 21) - 1)) - off
        k = ( keys        & ((1 << 21) - 1)) - off
        return torch.stack([i, j, k], dim=-1).to(torch.int32)

    # ── grow sparse state ────────────────────────────────────────
    def _grow_keys(self, new_keys: torch.Tensor):
        if new_keys.numel() == 0:
            return
        merged = torch.cat([self.keys, new_keys])
        uk, inv = torch.unique(merged, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        if uk.numel() == n_old:
            return
        idx_old = inv[:n_old]

        def _grow_buf(src, fill=0.0):
            out = torch.full((uk.shape[0],), fill, dtype=src.dtype,
                             device=src.device)
            if src.numel() > 0:
                out[idx_old] = src
            return out

        old_vals = self.vals_st
        self.keys    = uk
        self.vals_st = _grow_buf(old_vals, 0.0)
        self._grow_extra(uk, idx_old)

    def _grow_extra(self, uk, idx_old):
        pass

    # ── queries ──────────────────────────────────────────────────
    def voxel_centers(self) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.empty((0, 3), device=self.device, dtype=self.dtype)
        ijk = self._unhash_keys(self.keys).to(self.dtype)
        return self.origin + (ijk + 0.5) * self.p.voxel_size

    def _display_vals(self) -> torch.Tensor:
        return self.vals_st

    def occupied_mask(self) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device)
        return self._display_vals() > self.p.occ_thresh

    def next_epoch(self):
        pass

    # ── shared helpers ───────────────────────────────────────────
    def _sanitise(self, pts, cams, conf=None, max_range=20.0):
        dev = self.device
        pts  = pts.to(dev, torch.float32)
        cams = cams.to(dev, torch.float32)
        ok = torch.isfinite(pts).all(1) & torch.isfinite(cams).all(1)
        if conf is not None:
            conf = conf.to(dev, torch.float32)
            ok = ok & torch.isfinite(conf)
        pts, cams = pts[ok], cams[ok]
        if conf is not None:
            conf = conf[ok]
        if max_range is not None and pts.numel() > 0:
            d = (pts - cams).norm(dim=1)
            keep = d <= max_range
            pts, cams = pts[keep], cams[keep]
            if conf is not None:
                conf = conf[keep]
        return (pts, cams, conf) if conf is not None else (pts, cams)

    def _march_free_keys(self, pts, cams, unique_surf_keys,
                         ray_stride=4, max_free_rays=10_000,
                         samples_per_voxel=0.7):
        """Unique hashed keys of free-space voxels (excluding surface)."""
        dev = self.device
        vs  = self.p.voxel_size

        P_sub = pts[::ray_stride]
        C_sub = cams[::ray_stride]
        if max_free_rays and P_sub.shape[0] > max_free_rays:
            perm = torch.randperm(P_sub.shape[0], device=dev)[:max_free_rays]
            P_sub, C_sub = P_sub[perm], C_sub[perm]

        if P_sub.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=dev)

        V = P_sub - C_sub
        seg_len = V.norm(dim=1)
        steps = (seg_len / vs * samples_per_voxel).ceil().clamp(min=1).int()
        max_steps = int(steps.max().item())

        base = torch.arange(max_steps, device=dev, dtype=torch.float32) + 0.5
        t = base[None, :] / steps.float()[:, None]
        mask = t < 1.0
        samples = C_sub[:, None, :] + t[:, :, None] * V[:, None, :]
        samples = samples[mask]

        if samples.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=dev)

        keys_free_all = self._hash_ijk(self._world_to_ijk(samples))
        keep = ~torch.isin(keys_free_all, unique_surf_keys)
        return torch.unique(keys_free_all[keep])


# ═════════════════════════════════════════════════════════════════
# 1. TSDF Fusion  (Curless & Levoy 1996 / KinectFusion 2011)
# ═════════════════════════════════════════════════════════════════
class TSDFFusion(_SparseBaselineGrid):
    """
    Running weighted-average TSDF along each camera ray within a
    truncation band around the observed surface.  Occupancy ~ TSDF < 0.
    """

    def __init__(self, voxel_size=0.2, device="cuda",
                 trunc_ratio: float = 3.0, occ_thresh: float = 0.0):
        super().__init__(voxel_size=voxel_size, device=device,
                         occ_thresh=occ_thresh)
        self.trunc_dist = trunc_ratio * voxel_size
        self._tsdf    = torch.empty(0, dtype=self.dtype, device=self.device)
        self._weights = torch.empty(0, dtype=self.dtype, device=self.device)

    def _grow_extra(self, uk, idx_old):
        def g(src):
            out = torch.zeros(uk.shape[0], dtype=src.dtype, device=src.device)
            if src.numel() > 0: out[idx_old] = src
            return out
        self._tsdf    = g(self._tsdf)
        self._weights = g(self._weights)

    @torch.no_grad()
    def integrate(self, pts, cams, conf=None, max_range=20.0):
        out = self._sanitise(pts, cams, conf, max_range)
        if conf is not None:
            pts, cams, conf = out
        else:
            pts, cams = out
            conf = torch.ones(pts.shape[0], device=self.device)
        if pts.numel() == 0:
            return

        dev   = self.device
        trunc = self.trunc_dist
        vs    = self.p.voxel_size

        ray     = pts - cams
        ray_len = ray.norm(dim=1, keepdim=True).clamp(min=1e-8)
        ray_n   = ray / ray_len

        n_band  = max(2, int(2 * trunc / vs) + 1)
        offsets = torch.linspace(-trunc, trunc, n_band, device=dev)

        sample_pts = pts[:, None, :] + ray_n[:, None, :] * offsets[None, :, None]
        sdf_vals   = -offsets[None, :].expand(pts.shape[0], -1)
        tsdf_vals  = (sdf_vals / trunc).clamp(-1.0, 1.0)
        w = conf[:, None] * (1.0 - (offsets.abs() / trunc).clamp(0, 1))[None, :]

        flat_pts  = sample_pts.reshape(-1, 3)
        flat_tsdf = tsdf_vals.reshape(-1)
        flat_w    = w.reshape(-1)

        keys = self._hash_ijk(self._world_to_ijk(flat_pts))
        self._grow_keys(torch.unique(keys))

        idx = torch.searchsorted(self.keys, keys).clamp(max=self.keys.shape[0] - 1)
        valid = (self.keys[idx] == keys)
        idx_v, tsdf_v, w_v = idx[valid], flat_tsdf[valid], flat_w[valid]

        old_w = self._weights[idx_v]
        old_t = self._tsdf[idx_v]
        new_w = old_w + w_v
        self._tsdf[idx_v]    = (old_w * old_t + w_v * tsdf_v) / new_w.clamp(min=1e-8)
        self._weights[idx_v] = new_w

        observed = self._weights > 0.5
        self.vals_st = -self._tsdf * observed.float()


# ═════════════════════════════════════════════════════════════════
# 2. Exponential Moving Average (EMA) Occupancy
# ═════════════════════════════════════════════════════════════════
class EMAFusion(_SparseBaselineGrid):
    """
    p_v <- (1-alpha) p_v + alpha * evidence,  evidence in {0, 1}.
    """

    def __init__(self, voxel_size=0.2, device="cuda",
                 alpha: float = 0.3, occ_thresh: float = 0.0):
        super().__init__(voxel_size=voxel_size, device=device,
                         occ_thresh=occ_thresh)
        self.alpha = alpha
        self._prob = torch.empty(0, dtype=self.dtype, device=self.device)
        self._seen = torch.empty(0, dtype=torch.bool, device=self.device)

    def _grow_extra(self, uk, idx_old):
        def gf(src, fill=0.0):
            out = torch.full((uk.shape[0],), fill, dtype=torch.float32, device=src.device)
            if src.numel() > 0: out[idx_old] = src
            return out
        def gb(src, fill=False):
            out = torch.full((uk.shape[0],), fill, dtype=torch.bool, device=src.device)
            if src.numel() > 0: out[idx_old] = src
            return out
        self._prob = gf(self._prob, 0.0)
        self._seen = gb(self._seen, False)

    @torch.no_grad()
    def integrate(self, pts, cams, conf=None, max_range=20.0,
                  carve_free=True, ray_stride=4, max_free_rays=10_000):
        pts, cams = self._sanitise(pts, cams, max_range=max_range)
        if pts.numel() == 0:
            return

        dev, a = self.device, self.alpha
        keys_surf = self._hash_ijk(self._world_to_ijk(pts))
        u_surf    = torch.unique(keys_surf)

        keys_free = self._march_free_keys(pts, cams, u_surf, ray_stride, max_free_rays) \
                    if carve_free else torch.zeros(0, dtype=torch.int64, device=dev)

        self._grow_keys(torch.cat([u_surf, keys_free]))

        # surface (evidence = 1)
        idx_s = torch.searchsorted(self.keys, u_surf).clamp(max=self.keys.shape[0]-1)
        hit_s = self.keys[idx_s] == u_surf;  idx_s = idx_s[hit_s]
        new_s = ~self._seen[idx_s]
        self._prob[idx_s[new_s]] = 1.0
        self._seen[idx_s] = True
        old_s = ~new_s
        if old_s.any():
            self._prob[idx_s[old_s]] = (1 - a) * self._prob[idx_s[old_s]] + a

        # free (evidence = 0)
        if keys_free.numel() > 0:
            idx_f = torch.searchsorted(self.keys, keys_free).clamp(max=self.keys.shape[0]-1)
            hit_f = self.keys[idx_f] == keys_free;  idx_f = idx_f[hit_f]
            new_f = ~self._seen[idx_f]
            self._prob[idx_f[new_f]] = 0.0
            self._seen[idx_f] = True
            old_f = ~new_f
            if old_f.any():
                self._prob[idx_f[old_f]] *= (1 - a)

        p = self._prob.clamp(1e-6, 1 - 1e-6)
        self.vals_st = torch.log(p / (1 - p))


# ═════════════════════════════════════════════════════════════════
# 3. Last-Frame Snapshot  (no temporal memory)
# ═════════════════════════════════════════════════════════════════
class LastFrameFusion(_SparseBaselineGrid):
    """Completely replaces the map each timestep."""

    @torch.no_grad()
    def integrate(self, pts, cams, conf=None, max_range=20.0,
                  carve_free=True, ray_stride=4, max_free_rays=10_000):
        pts, cams = self._sanitise(pts, cams, max_range=max_range)
        if pts.numel() == 0:
            return

        dev = self.device
        keys_surf = self._hash_ijk(self._world_to_ijk(pts))
        u_surf    = torch.unique(keys_surf)

        keys_free = self._march_free_keys(pts, cams, u_surf, ray_stride, max_free_rays) \
                    if carve_free else torch.zeros(0, dtype=torch.int64, device=dev)

        all_keys = torch.unique(torch.cat([u_surf, keys_free]), sorted=True)
        self.keys    = all_keys
        self.vals_st = torch.zeros(all_keys.shape[0], dtype=self.dtype, device=dev)

        idx_s = torch.searchsorted(self.keys, u_surf)
        self.vals_st[idx_s] = 1.0
        if keys_free.numel() > 0:
            idx_f = torch.searchsorted(self.keys, keys_free)
            self.vals_st[idx_f] = -1.0


# ═════════════════════════════════════════════════════════════════
# 4. Confidence-Weighted Hit Counting
# ═════════════════════════════════════════════════════════════════
class ConfWeightedFusion(_SparseBaselineGrid):
    """
    score[v] += mean_conf_in_voxel  (surface)
    score[v] -= free_penalty        (free-space)
    """

    def __init__(self, voxel_size=0.2, device="cuda",
                 free_penalty: float = 0.15, occ_thresh: float = 0.0):
        super().__init__(voxel_size=voxel_size, device=device, occ_thresh=occ_thresh)
        self.free_penalty = free_penalty
        self._score = torch.empty(0, dtype=self.dtype, device=self.device)

    def _grow_extra(self, uk, idx_old):
        out = torch.zeros(uk.shape[0], dtype=self._score.dtype, device=self._score.device)
        if self._score.numel() > 0: out[idx_old] = self._score
        self._score = out

    @torch.no_grad()
    def integrate(self, pts, cams, conf=None, max_range=20.0,
                  carve_free=True, ray_stride=4, max_free_rays=10_000):
        if conf is not None:
            pts, cams, conf = self._sanitise(pts, cams, conf, max_range)
        else:
            pts, cams = self._sanitise(pts, cams, max_range=max_range)
            conf = torch.ones(pts.shape[0], device=self.device)
        if pts.numel() == 0:
            return

        dev = self.device
        keys_surf = self._hash_ijk(self._world_to_ijk(pts))
        u_surf    = torch.unique(keys_surf)

        keys_free = self._march_free_keys(pts, cams, u_surf, ray_stride, max_free_rays) \
                    if carve_free else torch.zeros(0, dtype=torch.int64, device=dev)

        self._grow_keys(torch.cat([u_surf, keys_free]))

        idx_pts  = torch.searchsorted(self.keys, keys_surf).clamp(max=self.keys.shape[0]-1)
        conf_sum = torch.zeros(self.keys.shape[0], device=dev)
        cnt      = torch.zeros(self.keys.shape[0], device=dev)
        conf_sum.scatter_add_(0, idx_pts, conf)
        cnt.scatter_add_(0, idx_pts, torch.ones_like(conf))
        has = cnt > 0
        self._score[has] += conf_sum[has] / cnt[has]

        if keys_free.numel() > 0:
            idx_f = torch.searchsorted(self.keys, keys_free).clamp(max=self.keys.shape[0]-1)
            hit_f = self.keys[idx_f] == keys_free
            self._score[idx_f[hit_f]] -= self.free_penalty

        self.vals_st = self._score.clone()


# ═════════════════════════════════════════════════════════════════
# 5. Simple Log-Odds  (plain Bayesian, no ST/LT, no guarding)
# ═════════════════════════════════════════════════════════════════
class SimpleLogOdds(_SparseBaselineGrid):
    """
    Standard Bayesian log-odds occupancy [Elfes 1989, Thrun 2005].
    NO heuristic hardening: no ST/LT split, no epoch-based promotion,
    no guarded carving, no demotion.

    Isolates the value of the ST/LT engineering in your Sec 3.4.
    Expected: brittle under misalignment — aggressive clearing of
    valid walls and ghost trails from moving objects.
    """

    def __init__(self, voxel_size=0.2, device="cuda",
                 occ_inc: float = 0.5, free_inc: float = -0.3,
                 l_min: float = -3.0, l_max: float = 3.5,
                 occ_thresh: float = 0.0):
        super().__init__(voxel_size=voxel_size, device=device, occ_thresh=occ_thresh)
        self.occ_inc  = occ_inc
        self.free_inc = free_inc
        self.l_min    = l_min
        self.l_max    = l_max

    @torch.no_grad()
    def integrate(self, pts, cams, conf=None, max_range=20.0,
                  carve_free=True, ray_stride=4, max_free_rays=10_000):
        pts, cams = self._sanitise(pts, cams, max_range=max_range)
        if pts.numel() == 0:
            return

        dev = self.device
        keys_surf = self._hash_ijk(self._world_to_ijk(pts))
        u_surf    = torch.unique(keys_surf)

        keys_free = self._march_free_keys(pts, cams, u_surf, ray_stride, max_free_rays) \
                    if carve_free else torch.zeros(0, dtype=torch.int64, device=dev)

        self._grow_keys(torch.cat([u_surf, keys_free]))

        # occupied: +occ_inc per unique hit
        idx_s = torch.searchsorted(self.keys, u_surf).clamp(max=self.keys.shape[0]-1)
        hit_s = self.keys[idx_s] == u_surf
        self.vals_st[idx_s[hit_s]] += self.occ_inc

        # free: +free_inc (negative) per unique hit
        if keys_free.numel() > 0:
            idx_f = torch.searchsorted(self.keys, keys_free).clamp(max=self.keys.shape[0]-1)
            hit_f = self.keys[idx_f] == keys_free
            self.vals_st[idx_f[hit_f]] += self.free_inc

        self.vals_st.clamp_(self.l_min, self.l_max)


# ═════════════════════════════════════════════════════════════════
# Metric helper: compute IoU / dynamic / TFS for all extra baselines
# ═════════════════════════════════════════════════════════════════

def compute_extra_baseline_metrics(
    system,             # VoxelUpdaterSystem (for align_probs_to_keys_soft)
    extra_baselines,    # dict {name: baseline_obj}
    metrics_buffer,     # the global metrics_buffer dict
    vox_gt,             # GT voxel grid for timestep t
    gt_seq,             # list of all GT grids
    t,                  # current timestep index
    prev_extra_keys,    # dict {name: tensor|None} — mutated
    prev_extra_binary,  # dict {name: tensor|None} — mutated
    compute_tfs_fn,     # VoxelUpdaterSystem.compute_tfs
    use_real_gt=False,
    vox_real_gt=None,
    prev_vox_real_gt=None,
):
    """
    Compute IoU, dynamic metrics, and TFS for every extra baseline.
    Mirrors the existing baseline metrics block exactly.
    """
    if vox_gt is None:
        return

    ref_vox = vox_real_gt if (use_real_gt and vox_real_gt is not None) else vox_gt

    logit_gt  = ref_vox.vals_st.clamp(-10, 10)
    p_occ_tgt = torch.sigmoid(logit_gt * 10.0)

    for bname, bobj in extra_baselines.items():
        if bobj.keys.numel() == 0:
            continue

        # align GT → baseline keys
        tgt_aligned, valid = system.align_probs_to_keys_soft(
            ref_vox, p_occ_tgt, bobj, default=0.0, r_vox=1)

        # Reverse: align baseline pred to GT keys to find uncovered GT voxels
        logit_b  = bobj._display_vals().clamp(-10, 10)
        b_probs_for_rev = torch.sigmoid(logit_b)
        _, gt_has_b_coverage = system.align_probs_to_keys_soft(
            bobj, b_probs_for_rev, ref_vox, default=0.0, r_vox=1)
        fn_uncovered = ((p_occ_tgt > 0.5) & ~gt_has_b_coverage).sum()

        pred_int = logit_b[valid]
        tgt_int  = tgt_aligned[valid]
        pred_fp  = logit_b[~valid]

        pred_bin = pred_int > bobj.p.occ_thresh
        tgt_bin  = tgt_int > 0.5

        tp     = (pred_bin &  tgt_bin).sum()
        fp_int = (pred_bin & ~tgt_bin).sum()
        fn     = (~pred_bin & tgt_bin).sum() + fn_uncovered
        fp_h   = (pred_fp > bobj.p.occ_thresh).sum()
        total_fp = fp_int + fp_h

        metrics_buffer[f"{bname}_occ_iou"].append(
            (tp / (tp + total_fp + fn + 1e-8)).item())
        metrics_buffer[f"{bname}_occ_recall"].append(
            (tp / (tp + fn + 1e-8)).item())
        metrics_buffer[f"{bname}_occ_precision"].append(
            (tp / (tp + total_fp + 1e-8)).item())

        # dynamic metrics
        if t > 0:
            prev_ref = (prev_vox_real_gt
                        if (use_real_gt and prev_vox_real_gt is not None)
                        else (gt_seq[t-1] if t-1 < len(gt_seq) else None))
            if prev_ref is not None and prev_ref.keys.numel() > 0:
                p_prev_al, vp = system.align_probs_to_keys_soft(
                    prev_ref,
                    torch.sigmoid(prev_ref.vals_st.clamp(-10, 10) * 10),
                    bobj, default=0.0, r_vox=1)
                both = valid & vp
                bi   = both[valid]
                if bi.any():
                    gc = tgt_int[bi] > 0.5
                    gp = p_prev_al[both] > 0.5
                    pb = pred_int[bi]

                    m_app = ~gp & gc
                    m_dis = gp & ~gc
                    m_dyn = gp != gc

                    if m_app.sum() > 0:
                        metrics_buffer[f"{bname}_dyn_recall_appearing"].append(
                            (pb[m_app] > bobj.p.occ_thresh).float().mean().item())
                    if m_dis.sum() > 0:
                        ghosts = pb[m_dis] > bobj.p.occ_thresh
                        metrics_buffer[f"{bname}_dyn_ghost_rate"].append(
                            ghosts.float().mean().item())
                        metrics_buffer[f"{bname}_dyn_recall_disappearing"].append(
                            (~ghosts).float().mean().item())
                    if m_dyn.sum() > 0:
                        pd = pb[m_dyn] > bobj.p.occ_thresh
                        gd = gc[m_dyn]
                        metrics_buffer[f"{bname}_dyn_iou"].append(
                            ((pd & gd).sum() / ((pd | gd).sum() + 1e-8)).item())

        # TFS
        curr_keys   = bobj.keys.clone()
        curr_binary = (bobj._display_vals() > bobj.p.occ_thresh).clone()

        if prev_extra_keys[bname] is not None:
            tfs = compute_tfs_fn(
                prev_extra_keys[bname], prev_extra_binary[bname],
                curr_keys, curr_binary)
            if tfs is not None:
                metrics_buffer[f"{bname}_tfs"].append(tfs)

        prev_extra_keys[bname]   = curr_keys
        prev_extra_binary[bname] = curr_binary.detach().clone()