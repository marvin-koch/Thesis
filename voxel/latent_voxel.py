from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

class LatentToOccupancyDecoder(nn.Module):
    """
    Decode a voxel latent z_j (optionally conditioned on the voxel center)
    into an occupancy probability in [0,1].

    Two modes:
      - cond=None: p = sigma(MLP(z))
      - cond='xyz': p = sigma(MLP([z, pos_enc(center_xyz)]))
    """
    def __init__(self, latent_dim: int, hidden: int = 96, cond: str | None = None,
                 xyz_pe_bands: int = 6):
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
        
        # self.ln = nn.LayerNorm(in_dim)
        self.ln = nn.Identity()
        
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

        #nn.init.constant_(self.fc3.bias, -3.0)      
        nn.init.constant_(self.fc3.bias, 0.0)      
        nn.init.zeros_(self.fc3.weight)
          
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
            norm_xyz = centers_xyz / 5.0  # Scale down roughly
            pe = self._fourier_pe(norm_xyz)
            x = torch.cat([z, pe], dim=-1)
        else:
            x = z

        x = self.ln(x)
        h = F.relu(self.fc1(x))
        h = h + F.relu(self.fc2(h))  # tiny residual
        logit = self.fc3(h).squeeze(-1)
        #logit = logit.clamp(-10.0, 10.0)
        # return torch.sigmoid(logit)
        return logit


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

        # self.input_dim = 768
        self.z_latent = torch.zeros((0, feature_dim), dtype=self.dtype, device=self.device)
        # self.z_proj = nn.Linear(self.input_dim, self.feature_dim)
        self.z_proj = nn.Identity()
        # update modules

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=feature_dim)



        """
        bias_ih = self.gru_cell.bias_ih
        bias_hh = self.gru_cell.bias_hh

        # 2. Force the Update Gate (middle chunk) to be positive (+1.0)
        # This sets the default behavior to "Update" (Forget history), preventing freezing.
        # Set Update gate bias (indices dim to 2*dim) to +1.0
        with torch.no_grad():
            # 1. Initialize all biases to 0 first
            self.gru_cell.bias_ih.zero_()
            self.gru_cell.bias_hh.zero_()
            bias_ih[feature_dim : 2*feature_dim].fill_(-2.0)
            bias_hh[feature_dim : 2*feature_dim].fill_(-2.0)
        
        """

               
        # routing controls
        self.routing_tau: float = 0.5   # temperature for softmax
        self.routing_topk: int = 16     # voxels per point (after radius prefilter)
        
        
        self.decoder = LatentToOccupancyDecoder(feature_dim, cond=None)
        
        self.free_token = nn.Parameter(torch.randn(1, feature_dim))
        #self.input_norm = nn.LayerNorm(feature_dim)

        # 1. Initialize WHOLE network (Good for fc1, fc2 hidden layers)
        # This sets all biases to 0.0, including fc3
        self.apply(self.kaiming_init)

        # 2. Overwrite ONLY the final layer (Fixes the output prior)
        # Bias -3.0 -> ~5% probability
        #nn.init.constant_(self.decoder.fc3.bias, -3.0) 
        nn.init.constant_(self.decoder.fc3.bias, 0.0) 
        # Weight 0.0 -> Prevents random noise from overriding the bias
        nn.init.zeros_(self.decoder.fc3.weight)



        


        
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
        
        
        
        #current_device = self.gru_cell[0].weight.device
        current_device = self.gru_cell.weight_hh.device
        
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
                        torch.zeros(shape, dtype=dtype, device=current_device), 
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
        self.z_latent = torch.zeros((0, self.feature_dim), dtype=self.dtype, device=current_device)


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
        #rel = (pts.float() - self.origin.float()) / float(self.p.voxel_size)
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
            return torch.zeros(0, dtype=torch.int64, device=self.device)

        all_keys = torch.cat([self.keys, upd_keys], dim=0)
        uk, inv = torch.unique(all_keys, sorted=True, return_inverse=True)
        n_old = self.keys.numel()
        idx_old = inv[:n_old]
        idx_upd = inv[n_old:]

        if uk.numel() != n_old:
            def grow_like(src: torch.Tensor, fill=0):
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
            current_dtype = self.z_latent.dtype if self.z_latent.numel() > 0 else self.dtype
            #z_new = torch.zeros((uk.shape[0], self.feature_dim), device=self.device, dtype=self.dtypew)
            z_new = torch.zeros((uk.shape[0], self.feature_dim), device=self.device, dtype=current_dtype)
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
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        lt_occ = self.vals_lt > self.p.occ_thresh
        st_occ = self.vals_st > (self.p.occ_thresh + self.p.st_margin)
        return lt_occ | st_occ

    def occupied_indices(self):
        if self.keys.numel() == 0:
            return torch.zeros(0, dtype=torch.int64, device=self.device)
        return self.keys[self.occupied_mask()]


    def generate_phantom_points_none(self, origins, terminations, n_samples=3):
        return torch.empty((0, 3), device=self.device), torch.empty((0, self.feature_dim), device=self.device)

    def generate_phantom_points(self, origins, terminations, n_samples=16):
        # n_samples=16 is Safe for VRAM, but "Sparse" spatially.

        N = origins.shape[0]
        max_range = 0.85

        # 1. Create base steps (0.0, 0.05, 0.10, ... 0.85)
        # Shape: (1, n_samples)
        steps = torch.linspace(0, max_range, n_samples, device=self.device).unsqueeze(0)

        # 2. ADD JITTER (The Magic Fix)
        # Calculate the gap size between samples
        step_size = max_range / (n_samples - 1)

        # Add random noise: +/- half a step
        # This ensures the sample could be ANYWHERE along the segment


        """
        #REMOVE AFTER
        seed = 42
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        # Pass the generator to the random function
        rand_val = torch.rand(N, n_samples, device=self.device, generator=gen)
        noise = (rand_val - 0.5) * step_size
        """

        noise = (torch.rand(N, n_samples, device=self.device) - 0.5) * step_size

        # Apply noise to the steps
        ratios = steps + noise
        ratios = ratios.clamp(0.0, max_range) # Keep within ray bounds

        # 3. Generate Points
        vec = (terminations - origins).unsqueeze(1)
        phantom_pts = origins.unsqueeze(1) + vec * ratios.unsqueeze(-1)
        phantom_pts = phantom_pts.reshape(-1, 3)

        # 4. Features (LOUD "Air" Token)
        phantom_feats = self.free_token.expand(phantom_pts.shape[0], -1)

        return phantom_pts, phantom_feats

    def generate_phantom_points_old(self, origins, terminations, n_samples=3):
            """
            origins: (N, 3) Camera centers corresponding to each point
            terminations: (N, 3) The wall points found by depth
            n_samples: How many phantom points to generate per ray
            """
            N = origins.shape[0]
            
            # 1. Create random ratios between 0.0 (camera) and 0.90 (near wall)
            # We stop at 0.90 to avoid putting a phantom point inside the wall

            ratios = torch.rand(N, n_samples, device=self.device) * 0.90
            #ratios = torch.rand(N, n_samples, device=self.device) * 0.85
           
            # 2. Interpolate: P_phantom = Origin + t * (Wall - Origin)
            # (N, 1, 3)
            vec = (terminations - origins).unsqueeze(1) 
            
            # (N, n_samples, 3)
            phantom_pts = origins.unsqueeze(1) + vec * ratios.unsqueeze(-1)
            
            # Flatten to (N*n_samples, 3)
            phantom_pts = phantom_pts.view(-1, 3)
            
            # 3. Create features for them
            # Expand the learned free_token to match the number of points
            phantom_feats = self.free_token.expand(phantom_pts.shape[0], -1)
            
            return phantom_pts, phantom_feats
 
    def update_with_features(self,
                            pts_world: torch.Tensor,  # (N,3)
                            f_pts: torch.Tensor,      # (N,D)
                            radius: float = 0.25,     # <--- Increase this! (e.g., 0.20 -> 0.40)
                            *,
                            chunkN: int = 100_000,
                            chunkT: int = 512,
                            neighbor_pad: int = 0,
                            r_vox_cap: int = 10,
                            use_amp: bool = True):

        if pts_world.numel() == 0 or self.keys.numel() == 0:
            return

        if not torch.isfinite(f_pts).all():
            f_pts = torch.nan_to_num(f_pts, nan=0.0, posinf=0.0, neginf=0.0)

        """
        # 1. Convert new points to voxel keys
        ijk_new = self._world_to_ijk(pts_world)
        keys_new = self._hash_ijk(ijk_new)

        # 2. Filter unique keys to save memory
        keys_new = torch.unique(keys_new)

        # 3. Expand the grid
        # This adds 0-initialized latent vectors for the new locations
        # and updates self.keys so the subsequent query can find them.
        self._ensure_and_index(keys_new)
        """




        dev = self.device
        N   = int(pts_world.shape[0])
        M   = int(self.keys.shape[0])
        K   = min(int(self.routing_topk), M) # <--- Increase this in config too (e.g., 16 or 32)
        D   = self.feature_dim
        vox = float(self.p.voxel_size)

        # --- voxel centers once, on device ---
        vox_xyz = (self.origin + (self._unhash_keys(self.keys) + 0.5) * vox).to(dev).contiguous()

        # --- query in chunks ---
        if radius > 0:
            r_vox = int(math.ceil(radius / max(vox, 1e-8))) + int(neighbor_pad)
            r_vox = min(r_vox, int(r_vox_cap))
            K_ball_bound = (2 * r_vox + 1) ** 3
            #K_ball_cap   = 32  # <--- Increased Cap to handle larger radius
            K_ball_cap   = 64  # <--- Increased Cap to handle larger radius
            K_ball = min(M, K_ball_bound, K_ball_cap)
        else:
            K_ball = 0

        all_i_parts, all_j_parts = [], []
        got_mass = torch.zeros(M, dtype=torch.bool, device=dev)
        CH = 512_000



        #WEIGHTED
        sigma = self.p.voxel_size * 3.0

        for i0 in range(0, N, CH):
            i1 = min(i0 + CH, N)
            pts_chunk = pts_world[i0:i1].to(dev, torch.float32).contiguous()
            pts_b = pts_chunk.unsqueeze(0)
            vox_b = vox_xyz.unsqueeze(0)

            if K_ball > 0:
                # res.dists contains Squared L2 distances
                res = ball_query(pts_b, vox_b, K=K_ball, radius=float(radius), return_nn=False)
                idx_ball = res.idx.squeeze(0)
                dists_sq = res.dists.squeeze(0) # (Chunk, K)

                valid = idx_ball >= 0

                if valid.any():
                    ii_local, kk = torch.where(valid)
                    jj = idx_ball[ii_local, kk]

                    all_i_parts.append(ii_local.to(torch.long) + i0)
                    all_j_parts.append(jj.to(torch.long))

                    # --- NEW: Compute Weights ---
                    # Retrieve valid squared distances
                    d2 = dists_sq[ii_local, kk]

                    # Gaussian Weighting: w = exp( -dist^2 / (2*sigma^2) )
                    weights = torch.exp(-d2 / (2 * (sigma ** 2)))

                    # Store weights for this batch (we will concat them later)
                    # We can hack this by appending to a temporary list or
                    # re-calculating during aggregation.
                    # Efficient approach: Append to a new list 'all_w_parts'
                    if not hasattr(self, '_temp_w_parts'): self._temp_w_parts = []
                    self._temp_w_parts.append(weights)

                    jj_u = torch.unique(jj)
                    got_mass[jj_u] = True

        if not all_i_parts:
            if hasattr(self, '_temp_w_parts'): del self._temp_w_parts
            return

        i_idx = torch.cat(all_i_parts, dim=0)
        j_idx = torch.cat(all_j_parts, dim=0)
        w_weights = torch.cat(self._temp_w_parts, dim=0).unsqueeze(-1) # (Pairs, 1)
        del self._temp_w_parts # Cleanup

        """
        for i0 in range(0, N, CH):
            i1 = min(i0 + CH, N)
            pts_chunk = pts_world[i0:i1].to(dev, torch.float32).contiguous()
            pts_b = pts_chunk.unsqueeze(0)
            vox_b = vox_xyz.unsqueeze(0)

            if K_ball > 0:
                res = ball_query(pts_b, vox_b, K=K_ball, radius=float(radius), return_nn=False)
                idx_ball = res.idx.squeeze(0)
                valid = idx_ball >= 0

                if valid.any():
                    ii_local, kk = torch.where(valid)
                    jj = idx_ball[ii_local, kk]
                    all_i_parts.append(ii_local.to(torch.long) + i0)
                    all_j_parts.append(jj.to(torch.long))
                    # Mark touched voxels
                    jj_u = torch.unique(jj)
                    got_mass[jj_u] = True

        if not all_i_parts:
            return

        i_idx = torch.cat(all_i_parts, dim=0)
        j_idx = torch.cat(all_j_parts, dim=0)
        """

        # Identify which unique voxels are being updated
        idx_upd = got_mass.nonzero(as_tuple=False).squeeze(-1)
        if idx_upd.numel() == 0: return

        # Map global voxel indices to compact [0..U-1]
        comp = torch.full((M,), -1, device=dev, dtype=torch.long)
        comp[idx_upd] = torch.arange(idx_upd.numel(), device=dev, dtype=torch.long)
        inv = comp[j_idx]  # (Pair_Count,) -> Maps every pair to a row in 'u'

        # -----------------------------------------------------------------------
        # NEW AGGREGATION: MAX POOLING (Robust to signal dilution)
        # -----------------------------------------------------------------------

        # 1. Prepare features
        f_proj_all = f_pts.to(dev)  # Assuming already projected
        f_sel = f_proj_all[i_idx]   # (Pairs, D)

        #NEW MULTIPLE WEIGHTS
        f_sel = f_sel * w_weights

        # 2. Initialize accumulator with -Infinity
        # We process in chunks to avoid OOM if Pairs is huge

        #MAX POOLING
        u = torch.full((idx_upd.numel(), D), -1e9, device=dev, dtype=torch.float32)
        #u = torch.zeros((idx_upd.numel(), D), device=dev, dtype=torch.float32)

        chunk_size = 50_000
        num_pairs = f_sel.shape[0]

        for chunk_start in range(0, num_pairs, chunk_size):
            end = min(chunk_start + chunk_size, num_pairs)

            f_chunk = f_sel[chunk_start:end]
            inv_chunk = inv[chunk_start:end]

            # Expand indices for scatter_reduce: (N, D)
            inv_expanded = inv_chunk.unsqueeze(-1).expand(-1, D)

            # Max Pool: Takes the strongest feature among all points touching the voxel
            u = torch.scatter_reduce(
                u, 
                0, 
                inv_expanded, 
                f_chunk.float(), 
                reduce="amax", 
                #reduce="mean", 
                include_self=True
                #include_self=False
            )

        # Safety: Replace -inf with 0 (for any voxels that somehow got no updates, though unlikely)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u[u <= -1e8] = 0.0 # Clean up untouched initialization


        # 1. Calculate Counts (Density)
        # 'inv' maps every point-voxel pair to a unique voxel index.
        # We just count occurrences of each index.
        ones = torch.ones_like(inv, dtype=torch.float32)
        counts = torch.zeros((idx_upd.numel(),), device=dev)
        counts.scatter_add_(0, inv, ones)

        # 2. Log-Encode the Count
        # We use log1p because counts can vary wildly (1 vs 1000).
        # Log brings them into a neural-friendly range (0.69 vs 6.9).
        density_feature = torch.log1p(counts).unsqueeze(-1) # Shape: (M, 1)

        # 3. Concatenate and Update
        # u_sel is your max-pooled feature (M, D)
        u_sel = u.to(self.z_latent.dtype)
        z_sel = self.z_latent[idx_upd]

        # Concatenate: [Features | Count]
        gru_input = torch.cat([u_sel, density_feature], dim=-1)

        refined_input = self.fusion_mlp(gru_input) # (M, D)
        # Pass expanded input to GRU
        z_new = self.gru_cell(refined_input, z_sel)

        """
        # -----------------------------------------------------------------------
        # GRU Update
        # -----------------------------------------------------------------------
        u_sel = u.to(self.z_latent.dtype)
        z_sel = self.z_latent[idx_upd]

        x_in = u_sel

        z_new = self.gru_cell(x_in, z_sel)


        """

        if not torch.isfinite(z_new).all():
             z_new = torch.nan_to_num(z_new, nan=0.0, posinf=0.0, neginf=0.0)

        self.z_latent.index_copy_(0, idx_upd, z_new.to(self.z_latent.dtype))

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
            return torch.zeros(0, 3, device=self.device, dtype=torch.float32)
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
        t = 1.0
        if self.z_latent.numel() == 0:
            #print("empty")
            return torch.zeros(0, device=self.device, dtype=torch.float32)
        if with_xyz_cond:
            centers = self.voxel_centers()
            return self.decoder(self.z_latent, centers) * t
        else:
            return self.decoder(self.z_latent, None) * t

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
        occ_thresh: float = 0.5,
    ) -> tuple[np.ndarray, dict]:

        # --- 1. Setup Grid ---
        x0, x1 = x_range
        y0, y1 = y_range
        Hx = int((x1 - x0) / res_xy)
        Hy = int((y1 - y0) / res_xy)

        # Initialize with -1.0 (Unknown)
        bev = torch.full((Hy, Hx), -1.0, device=self.device, dtype=torch.float32)
        meta = {"x0": x0, "y0": y0, "res": res_xy, "width": Hx, "height": Hy}

        # --- 2. Get Centers ---
        centers = self.voxel_centers()
        if centers.numel() == 0:
            print("DEBUG: No voxel centers found (grid empty).")
            return np.full((Hy, Hx), -1, dtype=np.int8), meta

        # --- 3. Filter Z-Band ---
        z = centers[:, 2]
        z_mask = (z >= z_min) & (z <= z_max)
        if not z_mask.any():
            return np.full((Hy, Hx), -1, dtype=np.int8), meta

        centers = centers[z_mask]
        z_lat = self.z_latent[z_mask]

        # --- 4. Decode ---
        probs = self.decoder(z_lat, centers if with_xyz_cond else None)
        probs = torch.sigmoid(probs)
        probs = probs.to(dtype=bev.dtype)   # match float32 destination

        # --- 5. Project to 2D ---
        gx = ((centers[:, 0] - x0) / res_xy).floor().to(torch.long)
        gy = ((centers[:, 1] - y0) / res_xy).floor().to(torch.long)


        keep = (gx >= 0) & (gx < Hx) & (gy >= 0) & (gy < Hy)
        gx, gy, probs = gx[keep], gy[keep], probs[keep]

        if probs.numel() == 0:
            return np.full((Hy, Hx), -1, dtype=np.int8), meta


        # --- 6. Rasterize ---
        idx = gy * Hx + gx
        flat = bev.view(-1)
        flat.scatter_reduce_(0, idx, probs, reduce="amax", include_self=True)

        # --- 7. Convert to Int ---
        bev_np = bev.cpu().numpy()

        # Check raw values in the grid before thresholding

        out_map = np.full_like(bev_np, -1, dtype=np.int8)

        known_mask = bev_np > -0.5
        occupied_mask = (bev_np > occ_thresh) & known_mask
        free_mask = (bev_np <= occ_thresh) & known_mask

        out_map[free_mask] = 0
        out_map[occupied_mask] = 100


        return out_map, meta

    def _ensure_feature_storage_(self):
        """internal: make sure z_latent exists & is aligned with keys."""
        if getattr(self, "z_latent", None) is None or self.z_latent.shape[0] != self.keys.shape[0]:
            D = getattr(self, "feature_dim", 64)
            z = torch.zeros((self.keys.shape[0], D), device=self.device, dtype=self.dtype)
            if getattr(self, "z_latent", None) is not None and self.z_latent.numel() > 0:
                # try to preserve old in case only size changed (rare)
                z[:min(z.shape[0], self.z_latent.shape[0])] = self.z_latent[:min(z.shape[0], self.z_latent.shape[0])]
            self.z_latent = z

    def initialize_latents_from_full_cloud(
                self,
                pts_world: torch.Tensor,   # (N,3) full scene points (aligned)
                f_pts: torch.Tensor,       # (N,D) point features
                cam_centers: torch.Tensor, # (N,3) Camera center for each point
                *,
                init_lt: bool = True,      # initialize LT/ST from decoded probs
                lt_level: float | None = None,
                carve_free: bool = True,   
                z_clip: Tuple[float,float] | None = (-float('inf'), float('inf')),
                stride: int = 4,         
                samples_per_voxel: float = 2.0, # Match GT density (usually 2.0)
                max_rays: int = 10000,          # Match GT cap (usually 10k or 20k)
                max_range: float = 12.0         # Match GT clip (usually 12.0)
            ):
                """
                Seed z_latent using the full point cloud.
                Logic aligns with update_with_features: Max Pool -> GRU Update.
                """
                dev, dt = self.device, self.dtype

                if pts_world.numel() == 0:
                    return

                pts_world = pts_world.to(dev, dt)
                f_pts = f_pts.to(dev, dt)

                # 1. Input Sanitization
                if not torch.isfinite(f_pts).all():
                    f_pts = torch.nan_to_num(f_pts, nan=0.0, posinf=0.0, neginf=0.0)

                finite = torch.isfinite(pts_world).all(dim=1) & torch.isfinite(cam_centers).all(dim=1)
                pts_world, cam_centers, f_pts = pts_world[finite], cam_centers[finite], f_pts[finite]
                if pts_world.numel() == 0: return

                if z_clip is not None:
                    z0, z1 = z_clip
                    keep = (pts_world[:,2] >= z0) & (pts_world[:,2] <= z1)
                    pts_world, cam_centers, f_pts = pts_world[keep], cam_centers[keep], f_pts[keep]

                if max_range is not None and cam_centers is not None:
                    cam_centers = cam_centers.to(dev, dt)
                    d_all = torch.norm(pts_world - cam_centers, dim=1)
                    keep = d_all <= max_range
                    pts_world = pts_world[keep]
                    f_pts = f_pts[keep]
                    cam_centers = cam_centers[keep]
                
                # --- 2. Identify Surface Voxels (Walls) ---
                ijk = self._world_to_ijk(pts_world)
                keys_surf = self._hash_ijk(ijk) # (N,)

                # --- 3. Identify Free Space Voxels (Air) ---
                keys_free = torch.zeros(0, dtype=torch.int64, device=dev)
                
                if carve_free and cam_centers is not None:
                    cam_centers = cam_centers.to(dev, dt)
                    
                    # Subsample for Ray Marching
                    P_sub = pts_world[::stride]
                    C_sub = cam_centers[::stride]
                    
                    # Cap rays to match GT generation limits
                    if max_rays is not None and P_sub.shape[0] > max_rays:

                        
                        """
                        #REMOVE AFTER
                        seed = 42
                        gen = torch.Generator(device=self.device)

                        gen.manual_seed(seed)
                        # Generate permutation using the specific seed
                        perm = torch.randperm(P_sub.shape[0], device=self.device, generator=gen)[:max_rays]
                        """


                        perm = torch.randperm(P_sub.shape[0], device=dev)[:max_rays]
                        P_sub = P_sub[perm]
                        C_sub = C_sub[perm]

                    if P_sub.numel() > 0:
                        V = P_sub - C_sub
                        seg_len = torch.linalg.norm(V, dim=1)
                        
                        # Steps based on density
                        steps = torch.clamp((seg_len / self.p.voxel_size * samples_per_voxel).ceil().to(torch.int32), min=1)
                        max_steps = int(steps.max().item())

                        base = torch.arange(max_steps, device=dev, dtype=dt) + 0.5
                        t = base[None, :] / steps.to(dt)[:, None]
                        
                        # Strictly before the wall
                        t = torch.minimum(t, torch.nextafter(torch.tensor(1.0, device=dev, dtype=dt), torch.tensor(0.0, device=dev, dtype=dt)))
                        mask = (t < 1.0)
                        
                        samples = C_sub.unsqueeze(1) + t.unsqueeze(-1) * V.unsqueeze(1)
                        samples = samples[mask]
                        
                        if samples.numel() > 0:
                            ijk_free = self._world_to_ijk(samples)
                            keys_free_all = self._hash_ijk(ijk_free)
                            
                            # Set Difference (Air - Wall)
                            unique_walls = torch.unique(keys_surf)
                            keep = ~torch.isin(keys_free_all, unique_walls)
                            keys_free = torch.unique(keys_free_all[keep])

                # --- 4. Allocate Everything (Walls + Air) ---
                all_keys = torch.cat([keys_free, keys_surf]) 
                self._ensure_and_index(all_keys)
                
                # A. Initialize Free Space with Token
                idx_free = torch.searchsorted(self.keys, keys_free)
                if idx_free.numel() > 0:
                    token_expanded = self.free_token.expand(idx_free.shape[0], -1)
                    self.z_latent.index_copy_(0, idx_free, token_expanded)

                # --- 5. Feature Pooling (Surface Voxels) - MAX POOLING ---
                # Map N points to M unique voxel indices
                idx_surf_pts = torch.searchsorted(self.keys, keys_surf)
                
                # Identify which voxels are actually walls
                unique_surf_idx = torch.unique(idx_surf_pts)
                
                # Initialize accumulator with -Infinity for Max Pooling
                u_all = torch.full((self.keys.shape[0], self.feature_dim), -1e9, device=dev, dtype=dt)
                #u_all = torch.zeros((self.keys.shape[0], self.feature_dim), device=dev, dtype=dt)


                # FIX: Removed the chunking loop to avoid In-place Autograd error.
                # Perform scatter_reduce in one shot.
                # Expand indices: (N, D)
                idx_expanded = idx_surf_pts.unsqueeze(-1).expand(-1, self.feature_dim)

                # Max Pool
                u_all.scatter_reduce_(0, idx_expanded, f_pts, reduce="amax", include_self=True)
                #u_all.scatter_reduce_(0, idx_expanded, f_pts, reduce="mean", include_self=False)

                # --- 6. GRU Update (Surface Voxels) ---
                # Extract inputs for surface voxels
                u_surf = u_all[unique_surf_idx]


                # Cleanup -inf from initialization (should generally be covered, but for safety)
                u_surf = torch.where(u_surf <= -1e8, torch.tensor(0.0, device=dev, dtype=dt), u_surf)

                # Get current state (will be 0.0 for new voxels, or existing if re-initializing)
                z_prev = self.z_latent[unique_surf_idx]

                # Apply GRU: z_new = GRU(input=u, hidden=z_prev)
                z_new = self.gru_cell(u_surf, z_prev)
                
                # Sanitize GRU output
                if not torch.isfinite(z_new).all():
                    z_new = torch.nan_to_num(z_new, nan=0.0, posinf=0.0, neginf=0.0)

                # Write Back
                self.z_latent.index_copy_(0, unique_surf_idx, z_new.to(self.z_latent.dtype))

                # --- 7. Initialize Occupancy Logits ---
                if init_lt:
                    if lt_level is not None:
                        p_occ = torch.full((self.keys.shape[0],), float(lt_level), device=dev, dtype=dt)
                        # Force surface voxels to be occupied if relying on simple counts
                        # (Here we just trust the lt_level passed in, or the decoder below)
                        self.vals_lt[:] = p_occ
                        self.vals_st[:] = p_occ
                    elif self.decoder is not None:
                        # Decode everything (including air)
                        centers = self.voxel_centers()
                        p_occ = self.decoder(self.z_latent, centers if getattr(self.decoder, "cond", None) == "xyz" else None)
                        logit = p_occ
                        self.vals_lt[:] = logit
                        self.vals_st[:] = logit

                    self.vals_lt.clamp_(min=self.p.l_min, max=self.p.l_max)
                    self.vals_st.clamp_(min=self.p.l_min, max=self.p.l_max)
                    self.vals = self._display_vals()

                # 8. Update timestamps for surface voxels
                if pts_world.shape[0] > 0:
                    now = torch.tensor(self.epoch, dtype=self.seen_occ_epoch.dtype, device=self.device)
                    if self.seen_occ_epoch.shape[0] != self.keys.shape[0]:
                        self._ensure_and_index(torch.zeros(0, dtype=torch.int64, device=self.device))
                    self.seen_occ_epoch[unique_surf_idx] = now

