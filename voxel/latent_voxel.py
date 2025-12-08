from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import time

from pytorch3d.ops import knn_points, ball_query
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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



class FeatureVoxelSimilarity(nn.Module):
    """
    sim(f,z,Δ) = temp * ( <Wf f, Wz z> + wΔ^T Δ )
    - Two-tower projections into a shared space (dot product)
    - Tiny linear head on 3D offset Δ
    - Optional cosine normalization for stability
    """
    def __init__(self, feat_dim: int, proj_dim: int = 16, use_cosine: bool = True):
        super().__init__()
        # use bias=False so projections are pure linear embeddings
        self.f_proj = nn.Linear(feat_dim, proj_dim, bias=False)
        self.z_proj = nn.Linear(feat_dim, proj_dim, bias=False)
        self.delta_lin = nn.Linear(3, 1, bias=True)
        self.use_cosine = use_cosine
        # learned temperature to calibrate scale
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self,
                f_pts_flat: torch.Tensor,   # (R, D)
                z_vox_flat: torch.Tensor,   # (R, D)
                delta_xyz_flat: torch.Tensor # (R, 3)
               ) -> torch.Tensor:
        # ensure contiguous for matmul throughput
        f = f_pts_flat.contiguous()
        z = z_vox_flat.contiguous()
        d = delta_xyz_flat.contiguous()

        # project each side (R,D) -> (R,P)
        Fp = self.f_proj(f)
        Zp = self.z_proj(z)

        if self.use_cosine:
            Fp = Fp / (Fp.norm(dim=-1, keepdim=True) + 1e-6)
            Zp = Zp / (Zp.norm(dim=-1, keepdim=True) + 1e-6)

        # dot product core (R,)
        core = (Fp * Zp).sum(dim=-1)

        # tiny delta term (R,)
        dterm = self.delta_lin(d).squeeze(-1)

        # scale
        temp = self.log_temp.exp()
        return temp * (core + dterm)
    
    
    



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

        nn.init.constant_(self.fc3.bias, -2.0) # Sigmoid(-2.0) ~= 0.12
        
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


class GatedEMAUpdate(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=True),   # scalar α per voxel
        )
    def forward(self, u, z):
        alpha = torch.sigmoid(self.gate(torch.cat([u, z], dim=-1)))  # (U,1)
        return z + alpha * (u - z)


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
        #self.origin = torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3)
        self.register_buffer(
            "origin",
            torch.as_tensor(origin_xyz, dtype=dtype, device=device).reshape(3),
            persistent=True
        )
        self.p = params
        self.device = self.origin.device
        self.dtype = dtype



        self.eb("keys",           (0,),     torch.int64)
        self.eb("vals_st",        (0,),     dtype)
        self.eb("vals_lt",        (0,),     dtype)
        self.eb("vals",           (0,),     dtype)
        self.eb("hit_count",      (0,),     torch.int32)
        self.eb("pos_occ_count",  (0,),     torch.int32)
        self.eb("neg_free_count", (0,),     torch.int32)
        self.eb("last_occ_epoch", (0,),     torch.int32)
        self.eb("last_free_epoch",(0,),     torch.int32)
        self.eb("view_bits",      (0,),     torch.int32)
        self.eb("seen_occ_epoch", (0,),     torch.int32)
        self.eb("seen_view_bits_e",(0,),    torch.int32)
        self.eb("occ_epoch_count",(0,),     torch.int32)
        self.eb("view_bits_cum",  (0,),     torch.int32)
        self.eb("lt_promoted_flag",(0,),    torch.uint8)

        self.epoch: int = 0

        # ---- learned latent memory ----
        self.feature_dim = feature_dim
        gate_hidden_dim = int(feature_dim/2)

        # self.input_dim = 768
        self.z_latent = torch.empty((0, feature_dim), dtype=self.dtype, device=self.device)
        # self.z_proj = nn.Linear(self.input_dim, self.feature_dim)
        self.z_proj = nn.Identity()
        # update modules
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=feature_dim)
        
        
        self.sim_net  = FeatureVoxelSimilarity(feature_dim)
        
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(2*feature_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 1)
        ).to(self.device)

        # routing controls
        self.routing_tau: float = 0.3   # temperature for softmax
        self.routing_topk: int = 8      # voxels per point (after radius prefilter)
        
        self.decoder = LatentToOccupancyDecoder(feature_dim)
        
        self.apply(self.kaiming_init)

        
    def eb(self, name, shape, dtype_, persistent=True):
        self.register_buffer(name, torch.empty(shape, dtype=dtype_), persistent=persistent)
        
    def reset_state(self, origin_xyz: Optional[np.ndarray] = None) -> None:
        """
        Reset all *map state* (keys, log-odds, counts, latent codes, epoch)
        while keeping the learnable networks (sim_net, gru_cell, gate_mlp, ...)
        and hyperparameters intact.

        Call this at the start of a new sequence/scene instead of creating
        a brand new LatentVoxelGrid, so the optimizer still sees the same params.
        """
        
        
        
        current_device = self.gate_mlp[0].weight.device
        
        self.device = current_device
        
        # 2. Update self.device so other methods (like _ensure_and_index) use the correct device
        
        # If you want to change the map origin for a new scene
        if origin_xyz is not None:
            # keep dtype/device aligned with the module
            self.origin = torch.as_tensor(
                origin_xyz, dtype=self.dtype, device=self.device
            ).reshape(3)
        #else:
        #    self.origin.to(self.device)


        # self.eb("keys",           (0,),     torch.int64)
        # self.eb("vals_st",        (0,),     self.dtype)
        # self.eb("vals_lt",        (0,),     self.dtype)
        # self.eb("vals",           (0,),     self.dtype)
        # self.eb("hit_count",      (0,),     torch.int32)
        # self.eb("pos_occ_count",  (0,),     torch.int32)
        # self.eb("neg_free_count", (0,),     torch.int32)
        # self.eb("last_occ_epoch", (0,),     torch.int32)
        # self.eb("last_free_epoch",(0,),     torch.int32)
        # self.eb("view_bits",      (0,),     torch.int32)
        # self.eb("seen_occ_epoch", (0,),     torch.int32)
        # self.eb("seen_view_bits_e",(0,),    torch.int32)
        # self.eb("occ_epoch_count",(0,),     torch.int32)
        # self.eb("view_bits_cum",  (0,),     torch.int32)
        # self.eb("lt_promoted_flag",(0,),    torch.uint8)


        def reset_buf(name, shape, dtype):
                    self.register_buffer(
                        name, 
                        torch.empty(shape, dtype=dtype, device=current_device), 
                        persistent=True
                    )

        reset_buf("keys",           (0,),     torch.int64)
        reset_buf("vals_st",        (0,),     self.dtype)
        reset_buf("vals_lt",        (0,),     self.dtype)
        reset_buf("vals",           (0,),     self.dtype)
        reset_buf("hit_count",      (0,),     torch.int32)
        reset_buf("pos_occ_count",  (0,),     torch.int32)
        reset_buf("neg_free_count", (0,),     torch.int32)
        reset_buf("last_occ_epoch", (0,),     torch.int32)
        reset_buf("last_free_epoch",(0,),     torch.int32)
        reset_buf("view_bits",      (0,),     torch.int32)
        reset_buf("seen_occ_epoch", (0,),     torch.int32)
        reset_buf("seen_view_bits_e",(0,),    torch.int32)
        reset_buf("occ_epoch_count",(0,),     torch.int32)
        reset_buf("view_bits_cum",  (0,),     torch.int32)
        reset_buf("lt_promoted_flag",(0,),    torch.uint8)
        


        # ---- latent memory per voxel ----
        self.z_latent = torch.empty((0, self.feature_dim), dtype=self.dtype, device=current_device)


        # reset logical time
        self.epoch = 0
        
    def kaiming_init(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, torch.nn.GRUCell):
            for name, param in module.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)  # safer for GRU
                elif "bias" in name:
                    torch.nn.init.zeros_(param) 
                    
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
                    vb = self.view_bits_cum[seen_now].to(torch.int32)
                    pop = ((vb & 1 > 0).to(torch.int32) +
                           ((vb >> 1) & 1 > 0).to(torch.int32) +
                           ((vb >> 2) & 1 > 0).to(torch.int32) +
                           ((vb >> 3) & 1 > 0).to(torch.int32) +
                           ((vb >> 4) & 1 > 0).to(torch.int32) +
                           ((vb >> 5) & 1 > 0).to(torch.int32) +
                           ((vb >> 6) & 1 > 0).to(torch.int32) +
                           ((vb >> 7) & 1 > 0).to(torch.int32))
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


    def _invalidate_faiss(self):
        self._faiss = None
        self._faiss_res = None
        self._faiss_is_gpu = False
        self._faiss_M = None

    def _ensure_faiss(self):
        """
        Build (or rebuild) a FAISS index over voxel centers self.vox_xyz (M,3).
        Uses GPU if available; falls back to CPU otherwise.
        """
        M = int(self.keys.shape[0])
        if getattr(self, "_faiss", None) is not None and self._faiss_M == M:
            return  # already good

        # voxel centers: (M,3) float32, contiguous, **CPU** (FAISS wants host unless using torch utils)
        vox = float(self.p.voxel_size)
        vox_xyz = (self.origin +
                (self._unhash_keys(self.keys) + 0.5) * vox)  # (M,3) on device
        self.vox_xyz = vox_xyz  # keep cached on device for downstream math
        x_cpu = vox_xyz.detach().to("cpu").contiguous().numpy()

        index_cpu = faiss.IndexFlatL2(3)  # exact L2 (squared)
        self._faiss_res = None
        self._faiss_is_gpu = False

        # Try GPU if available
        try:
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                # optional: limit temp memory, e.g., res.setTempMemory(256 * 1024 * 1024)
                index_gpu = faiss.index_cpu_to_gpu(res, torch.cuda.current_device(), index_cpu)
                index_gpu.add(x_cpu)
                self._faiss = index_gpu
                self._faiss_res = res
                self._faiss_is_gpu = True
            else:
                index_cpu.add(x_cpu)
                self._faiss = index_cpu
        except Exception:
            # Fallback to CPU if GPU faiss not built
            index_cpu.add(x_cpu)
            self._faiss = index_cpu
            self._faiss_is_gpu = False

        self._faiss_M = M




    def update_with_features(self,
                            pts_world: torch.Tensor,  # (N,3)
                            f_pts: torch.Tensor,      # (N,D)
                            radius: float = 0.25,
                            *,
                            chunkN: int = 100_000,     # tune
                            chunkT: int = 512,        # CHUNK OFFSETS!
                            neighbor_pad: int = 0,
                            r_vox_cap: int = 6,       # safety cap on neighbor radius in voxels
                            use_amp: bool = True):
        if pts_world.numel() == 0 or self.keys.numel() == 0:
            return
        
      

        dev = self.device
        N   = int(pts_world.shape[0])
        M   = int(self.keys.shape[0])
        K   = min(int(self.routing_topk), M)
        D   = self.feature_dim
        vox = float(self.p.voxel_size)

        # --- voxel centers once, on device ---
        vox_xyz = (self.origin +
                (self._unhash_keys(self.keys) + 0.5) * vox).to(dev).contiguous()

        # --- query in chunks, radius-only neighbors ---
        torch.cuda.synchronize()

        start = time.time()
        CH = 512_000
        if radius > 0:
            r_vox = int(math.ceil(radius / max(vox, 1e-8))) + int(neighbor_pad)
            r_vox = min(r_vox, int(r_vox_cap))
            K_ball_bound = (2 * r_vox + 1) ** 3
            K_ball_cap   = 32
            K_ball = min(M, K_ball_bound, K_ball_cap)
        else:
            K_ball = 0

        all_i_parts, all_j_parts = [], []
        
        got_mass = torch.zeros(M, dtype=torch.bool, device=dev)

        for i0 in range(0, N, CH):
            i1 = min(i0 + CH, N)
            pts_chunk = pts_world[i0:i1].to(dev, torch.float32).contiguous()
            pts_b = pts_chunk.unsqueeze(0)
            vox_b = vox_xyz.unsqueeze(0)

            if K_ball > 0:
                res = ball_query(pts_b, vox_b, K=K_ball, radius=float(radius), return_nn=False)
                idx_ball = res.idx.squeeze(0)                  # (n, K_ball)
                valid = idx_ball >= 0
                
                if valid.any():
                    ii_local, kk = torch.where(valid)
                    jj = idx_ball[ii_local, kk]
                    all_i_parts.append(ii_local.to(torch.long) + i0)   # <-- add i0 (globalize)
                    all_j_parts.append(jj.to(torch.long))
                    jj_u = torch.unique_consecutive(torch.sort(jj).values)  # unique within chunk
                    got_mass[jj_u] = True

        # --- union + dedup ---
        if not all_i_parts:
            return
        i_idx = torch.cat(all_i_parts, dim=0)
        j_idx = torch.cat(all_j_parts, dim=0)
        h = (i_idx.to(torch.int64) << 32) | j_idx.to(torch.int64)
        h_u = torch.unique(h)
        i_idx = (h_u >> 32).to(torch.long)
        j_idx = (h_u & ((1 << 32) - 1)).to(torch.long)
        idx_upd = got_mass.nonzero(as_tuple=False).squeeze(-1)  # touched voxels (U,)
        


        torch.cuda.synchronize()

        print("Pairs took", time.time() - start, "seconds!")

        # # --- quick sanity (optional) ---
        # counts_per_point = torch.bincount(i_idx, minlength=N)
        # print("Mean neighbors:", float(counts_per_point.float().mean()),
        #       "Max:", int(counts_per_point.max()), "Min:", int(counts_per_point.min()))

        # ---------- SIM / WEIGHTS / ACCUM ----------
        
        
     
        start = time.time()

        # 1) project features ONCE (don’t re-project per pair)
        # f_proj_all = self.z_proj(f_pts.to(dev))    # (N, Z_lat)
        f_proj_all = (f_pts.to(dev))    # (N, Z_lat)

        # 2) gather per pair
        p_sel = pts_world[i_idx].to(dev)
        v_sel = vox_xyz[j_idx]
        z_sel = self.z_latent[j_idx]
        f_sel = f_proj_all[i_idx]                  # use precomputed projection
        delta = p_sel - v_sel

       
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):

            # 3) sim
            sim_flat = self.sim_net(f_sel, z_sel, delta)
            # sim_flat = fast_sim_mlp_pairs(self.sim_net, i_idx, j_idx, f_proj_all, self.z_latent, delta)
        torch.cuda.synchronize()

        print("Sim flat took", time.time() - start, "seconds!")
        start = time.time()

        # 4) stable softmax per point
        # Nfull = int(N)
        # tau = max(float(self.routing_tau), 1e-6)
        # max_per_i = torch.full((Nfull,), -1e9, device=sim_flat.device, dtype=sim_flat.dtype)
        # max_per_i = max_per_i.scatter_reduce(0, i_idx, sim_flat, reduce="amax", include_self=True)
        # sim_shift = sim_flat - max_per_i[i_idx]
        # w_unnorm  = torch.exp(sim_shift / tau)
        # sum_per_i = torch.zeros(Nfull, device=sim_flat.device, dtype=sim_flat.dtype).scatter_add(0, i_idx, w_unnorm)
        # weights   = w_unnorm / (sum_per_i[i_idx] + 1e-8)
        
        
        # 4) stable softmax per point
        Nfull = int(N)
        tau = max(float(self.routing_tau), 1e-6)

        # Force Float32 for precision in exponentials
        sim_flat_f32 = sim_flat.float() 
        
        max_per_i = torch.full((Nfull,), -1e9, device=dev, dtype=torch.float32)
        max_per_i = max_per_i.scatter_reduce(0, i_idx, sim_flat_f32, reduce="amax", include_self=True)
        
        sim_shift = sim_flat_f32 - max_per_i[i_idx]
        w_unnorm  = torch.exp(sim_shift / tau) # Safer in float32

        sum_per_i = torch.zeros(Nfull, device=dev, dtype=torch.float32).scatter_add(0, i_idx, w_unnorm)
        weights   = w_unnorm / (sum_per_i[i_idx] + 1e-8)
        
        torch.cuda.synchronize()



        print("Softmax took", time.time() - start, "seconds!")

        
        start = time.time()

        # # Get unique indices and their inverse mapping
        # idx_upd, inverse_indices = torch.unique(j_idx.long(), return_inverse=True)

        # if idx_upd.numel() == 0:
        #     return

        # print(f"unique operation took {time.time() - start:.4f} seconds")

        # # Accumulate directly into sparse buffer (only U voxels, not M)
        # u_sparse = torch.zeros(idx_upd.numel(), self.feature_dim, device=dev, dtype=torch.float32)
        # contrib32 = weights.to(torch.float32).unsqueeze(-1) * f_sel.to(torch.float32)

        # u_sparse.index_add_(0, inverse_indices, contrib32)

        # # GRU on touched voxels
        # u_sel = u_sparse.to(self.z_latent.dtype)
        # z_sel = self.z_latent[idx_upd]
        
        
        
        # compact map j -> [0..U-1] without unique/bincount
        comp = torch.full((M,), -1, device=dev, dtype=torch.long)
        comp[idx_upd] = torch.arange(idx_upd.numel(), device=dev, dtype=torch.long)
        inv = comp[j_idx]  # (R,) compact row ids

        # reduce pairs -> (U, D) without atomics to M×D
        # contrib32 = weights.to(torch.float32).unsqueeze(-1) * f_sel.to(torch.float32)
        # u = torch.zeros(idx_upd.numel(), D, device=dev, dtype=torch.float32)
        # u.index_add_(0, inv, contrib32)


        u = torch.zeros(idx_upd.numel(), D, device=dev, dtype=torch.float32)
        chunk_size = 20_000 
        num_pairs = weights.shape[0]

        for chunk_start in range(0, num_pairs, chunk_size):
            end = min(chunk_start + chunk_size, num_pairs)

            # Slice the inputs
            w_chunk = weights[chunk_start:end].unsqueeze(-1)
            f_chunk = f_sel[chunk_start:end]
            inv_chunk = inv[chunk_start:end]

            contrib_chunk = w_chunk * f_chunk

            # Accumulate into the main buffer
            u.index_add_(0, inv_chunk, contrib_chunk.to(torch.float32))
            
            # Explicitly free memory (optional but safe)
            del contrib_chunk, w_chunk, f_chunk
            

            
        # GRU only on touched voxels
        u_sel = u.to(self.z_latent.dtype)
        z_sel = self.z_latent[idx_upd]
        
        torch.cuda.synchronize()

        print("accum took", time.time() - start, "seconds!")
        start = time.time()


        # z = self.z_latent #[idx_upd]
        # gamma = torch.sigmoid(self.gate_mlp(torch.cat([u.to(z.dtype), z], dim=-1)))

        gamma = torch.sigmoid(self.gate_mlp(torch.cat([u_sel, z_sel], dim=-1)))
        x_in  = u_sel * gamma
        # # 6) GRU update
        # u = upd[idx_upd].to(self.z_latent.dtype)
        # z = self.z_latent[idx_upd]
        # gamma = torch.sigmoid(self.gate_mlp(torch.cat([u, z], dim=-1)))
        # x_in  = gamma * u
        torch.cuda.synchronize()

        print("gate took", time.time() - start, "seconds!")
        start = time.time()
        # inp = x_in.unsqueeze(1)    # (U, 1, D)
        # h0  = z_sel.unsqueeze(0)    # (1, U, D)
        
      

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z_new = self.gru_cell(x_in, z_sel)
            
            
            # out, h = self.gru_cell(inp, h0)
            # z_new = out[:, 0, :]     # (U, D)

            # z_new = self.ema_upd(x_in, z_sel)
        z_new = z_new.to(self.z_latent.dtype)


        self.z_latent.index_copy_(0, idx_upd, z_new)
            
            # self.z_latent = self.gru_cell(x_in, self.z_latent)
        torch.cuda.synchronize()

        print("Gru took", time.time() - start, "seconds!")

            
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

            
    #@torch.no_grad()
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


   
