import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# @dataclass
# class AlignCfg:
#     pad_vox: int = 12                 # padding (voxels) around the points' AABB
#     max_block_vox: int = 128          # cap on dense block size per axis
#     iters: int = 4                    # optimization steps
#     samples_per_ray: int = 4          # free-space samples per ray
#     lambda_free: float = 0.5
#     lambda_prior_t: float = 1e-3      # prior on translation
#     lambda_prior_r: float = 1e-3      # prior on rotation (axis-angle)
#     lambda_prior_s: float = 1e-3      # prior on log-scale
#     step: float = 1e-1                # Adam step size
#     use_sim3: bool = True             # True = Sim(3); False = SE(3)
#     downsample_points: int = 10000    # subsample for speed
#     rot_clamp_deg: float = 1.5        # max accepted update magnitude
#     trans_clamp_m: float = 0.03
#     scale_clamp: float = 0.01         # |log s| <= 1% allowed
#     conf_filter: bool = True
#     conf_quantile: float = 0.10        # drop bottom 10% conf points

# ---------- helpers: transforms ----------

def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) -> 3x3 rotation matrix."""
    theta = torch.linalg.norm(omega)
    if theta < 1e-8:
        return torch.eye(3, device=omega.device, dtype=omega.dtype)
    axis = omega / theta
    x, y, z = axis
    K = torch.tensor([[0, -z, y],[z, 0, -x],[-y, x, 0]], device=omega.device, dtype=omega.dtype)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    R = I + torch.sin(theta)*K + (1 - torch.cos(theta))*(K@K)
    return R

def sim3_from_params(omega, t, log_s=None):
    """Return 4x4 Sim(3) (if log_s provided) or SE(3) otherwise."""
    R = so3_exp_map(omega)
    if log_s is None:
        S = 1.0
    else:
        S = torch.exp(log_s)
    T = torch.eye(4, device=omega.device, dtype=omega.dtype)
    T[:3,:3] = S * R
    T[:3, 3] = t
    return T

def apply_sim3(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 Sim(3) to (...,3) points."""
    return (pts @ T[:3,:3].T) + T[:3,3]

# ---------- build dense occupancy block from sparse grid ----------

def build_dense_block_from_sparse(grid, aabb_min, aabb_max, max_block_vox=128):
    """
    Make a dense [1,1,D,H,W] tensor of log-odds cropped from the sparse hash.
    Returns: occ_logodds, block_origin_xyz, voxel_size
    """
    vs = grid.p.voxel_size
    # compute integer ijk bounds
    ijk_min = torch.floor((aabb_min - grid.origin) / vs).to(torch.int64)
    ijk_max = torch.floor((aabb_max - grid.origin) / vs).to(torch.int64)

    # clamp block size
    size_ijk = (ijk_max - ijk_min + 1).clamp(min=1)
    size_ijk = torch.minimum(size_ijk, torch.tensor(max_block_vox, device=size_ijk.device).to(size_ijk.dtype))
    # recompute max with clamped size
    ijk_max = ijk_min + size_ijk - 1

    # dense tensor
    D, H, W = int(size_ijk[2]), int(size_ijk[1]), int(size_ijk[0])  # z,y,x
    occ = torch.zeros((1,1,D,H,W), device=grid.device, dtype=grid.dtype)

    if grid.keys.numel() > 0:
        ijk_all = grid._unhash_keys(grid.keys)  # (N,3) int32
        ijk_all = ijk_all.to(torch.int64)
        # mask voxels within block
        m0 = (ijk_all[:,0] >= ijk_min[0]) & (ijk_all[:,0] <= ijk_max[0])
        m1 = (ijk_all[:,1] >= ijk_min[1]) & (ijk_all[:,1] <= ijk_max[1])
        m2 = (ijk_all[:,2] >= ijk_min[2]) & (ijk_all[:,2] <= ijk_max[2])
        m = m0 & m1 & m2
        if m.any():
            sel_ijk = ijk_all[m]
            sel_vals = grid.vals[m]
            # local indices
            li = (sel_ijk[:,0] - ijk_min[0]).to(torch.int64)
            lj = (sel_ijk[:,1] - ijk_min[1]).to(torch.int64)
            lk = (sel_ijk[:,2] - ijk_min[2]).to(torch.int64)
            occ[0,0, lk, lj, li] = sel_vals

    block_origin = grid.origin + ijk_min.to(grid.dtype) * vs  # world xyz of voxel corner (i,j,k)
    return occ, block_origin, vs, (D,H,W)

def smooth3d(occ, k=3):
    """Simple 3D average blur; occ is [1,1,D,H,W]."""
    pad = k//2
    kernel = torch.ones((1,1,k,k,k), device=occ.device, dtype=occ.dtype) / (k**3)
    return F.conv3d(F.pad(occ, (pad,pad,pad,pad,pad,pad), mode='replicate'), kernel)



# ---------- sampling ----------

def world_to_block_coords(pts_world, block_origin, voxel_size, DHW):
    """Map world pts to normalized grid coords for F.grid_sample (z,y,x in [-1,1])."""
    D,H,W = DHW
    rel = (pts_world - block_origin) / voxel_size  # in voxel units (x,y,z)
    # center-of-voxel to index space; our dense tensor index order is (z,y,x)
    # x = rel[...,0]; y = rel[...,1]; z = rel[...,2]
    
    x = rel[...,0] - 0.5
    y = rel[...,1] - 0.5
    z = rel[...,2] - 0.5
    
    # normalize to [-1,1]
    gx = 2*(x / (W-1)) - 1
    gy = 2*(y / (H-1)) - 1
    gz = 2*(z / (D-1)) - 1
    grid = torch.stack([gx, gy, gz], dim=-1)  # (...,3)
    return grid

def sample_occ_prob(occ_block, grid_norm):
    """Trilinear sample prob S in [0,1]. occ_block: [1,1,D,H,W], grid_norm: [N,1,1,1,3] or [N,3]."""
    if grid_norm.dim() == 2:
        grid = grid_norm.view(1,1,1,-1,3)  # sample N points in one call
        S = F.grid_sample(occ_block, grid, mode='bilinear', align_corners=True)  # [1,1,1,1,N]
        p = torch.sigmoid(S.view(-1))  # (N,)
    else:
        S = F.grid_sample(occ_block, grid_norm, mode='bilinear', align_corners=True)
        p = torch.sigmoid(S)
        
    eps = 1e-4
    return torch.clamp(p, eps, 1 - eps)

# # ---------- main: Sim(3) alignment ----------

# def sim3_align_to_voxel(
#     pts_world: torch.Tensor,              # (S,H,W,3) or (N,3)
#     cams_world: torch.Tensor,             # same shape as pts_world (...,3)
#     grid,
#     conf_world,
#     cfg: AlignCfg = AlignCfg()
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns (T_4x4, pts_aligned). Optimizes a tiny Sim(3) (or SE(3)) so endpoints land on occupied
#     and the rays before endpoints traverse free space.
#     """
#     dev = grid.device
#     dt  = grid.dtype

#     # flatten inputs
#     P = pts_world.reshape(-1, 3).to(dev, dt)
#     C = cams_world.reshape(-1, 3).to(dev, dt)

#     # filter finite
#     finite = torch.isfinite(P).all(dim=1) & torch.isfinite(C).all(dim=1)
#     P, C = P[finite], C[finite]
#     if P.numel() == 0:
#         T = torch.eye(4, device=dev, dtype=dt)
#         return T, pts_world

#     # confidences
#     if conf_world is not None:
#         W = conf_world.reshape(-1).to(dev, dt)[finite]
#         if getattr(cfg, "conf_filter", True):
#             thr = torch.quantile(W, getattr(cfg, "conf_quantile", 0.10))
#             keep = W >= thr
#             P, C, W = P[keep], C[keep], W[keep]
#     else:
#         W = torch.ones(P.shape[0], device=dev, dtype=dt)
        
#     # optional subsample for speed
#     if cfg.downsample_points is not None and P.shape[0] > cfg.downsample_points:
#         idx = torch.randperm(P.shape[0], device=dev)[:cfg.downsample_points]
#         P_sub, C_sub, W_sub = P[idx], C[idx], W[idx]
#     else:
#         P_sub, C_sub, W_sub = P, C, W

#     # build dense block around AABB of P_sub with padding
#     aabb_min = torch.min(P_sub, dim=0).values - cfg.pad_vox * grid.p.voxel_size
#     aabb_max = torch.max(P_sub, dim=0).values + cfg.pad_vox * grid.p.voxel_size
#     occ_L, block_origin, vs, DHW = build_dense_block_from_sparse(grid, aabb_min, aabb_max, cfg.max_block_vox)
#     # smooth a bit to get a softer field
#     occ_L_s = smooth3d(occ_L, k=3)

#     # parameters: axis-angle rot (3), trans (3), log-scale (1 optional)
#     omega = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
#     t     = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
#     log_s = torch.zeros(1, device=dev, dtype=dt, requires_grad=True) if cfg.use_sim3 else None

#     opt = torch.optim.Adam([omega, t] + ([log_s] if cfg.use_sim3 else []), lr=cfg.step)

#     # precompute free-space sample ts
#     K = cfg.samples_per_ray
#     ts = torch.linspace(0.05, 0.95, steps=K, device=dev, dtype=dt)  # avoid endpoints

#     for it in range(cfg.iters):
#         opt.zero_grad()
#         # current transform
#         T = sim3_from_params(omega, t, log_s if cfg.use_sim3 else None)

#         # transform endpoints
#         X = apply_sim3(T, P_sub)  # (M,3)
#         Ct = apply_sim3(T, C_sub)           # transformed camera centers (NEW)

#         # ---- hit term: endpoints should be occupied ----
#         g_end = world_to_block_coords(X, block_origin, vs, DHW)  # (M,3)
#         S_end = sample_occ_prob(occ_L_s, g_end)                  # (M,)
#         # robust clamp to avoid log(0)
#         eps = 1e-4
        
#         w_hit = W_sub
#         L_hit = -(w_hit * torch.log(S_end)).sum() / (w_hit.sum() + 1e-9)

#         # L_hit = -torch.mean(torch.log(torch.clamp(S_end, min=eps)))  # maximize occupancy at endpoints

#         # ---- free term: along ray BEFORE endpoint should be free ----
#         # sample along rays
#         V  = X - Ct
#         Y = Ct.unsqueeze(1) + ts.view(1,-1,1) * V.unsqueeze(1)  # (M,K,3)
#         g_free = world_to_block_coords(Y, block_origin, vs, DHW).view(-1,3)
#         S_free = sample_occ_prob(occ_L_s, g_free).view(-1)         # (M*K,)
        
        
#         # per-ray weights: nearer samples a bit higher (shape: M*K)
#         w_ray = torch.linspace(1.0, 0.6, steps=K, device=dev, dtype=dt)  # (K,)
#         w_ray = w_ray.view(1, K).expand(W_sub.shape[0], K).reshape(-1)   # (M*K,)

        
#         w_pt = W_sub.view(-1,1).expand(-1, K).reshape(-1)
#         w_free = w_pt * w_ray

#         # -∑ w log(1 - p) / ∑ w
#         L_free = -(w_free * torch.log(1.0 - S_free)).sum() / (w_free.sum() + 1e-9)
        
#         # L_free = -torch.mean(torch.log(torch.clamp(1.0 - S_free, min=eps)))  # maximize free space

#         # ---- priors (keep update tiny) ----
#         L_prior = cfg.lambda_prior_t * (t*t).sum() + cfg.lambda_prior_r * (omega*omega).sum()
#         if cfg.use_sim3:
#             L_prior = L_prior + cfg.lambda_prior_s * (log_s*log_s).sum()

#         loss = L_hit + cfg.lambda_free * L_free + L_prior
#         loss.backward()
        
#         print("Loss: ", loss)
#         print(L_hit)
#         print(cfg.lambda_free * L_free)
#         print(L_prior)
#         print(cfg.lambda_prior_s * (log_s*log_s).sum())
        
#         opt.step()

#         # clamp step (safety)
#         # convert omega to magnitude
#         with torch.no_grad():
#             # clamp rotation magnitude
#             th = torch.linalg.norm(omega)
#             th_max = (cfg.rot_clamp_deg / 180.0) * torch.pi
#             if th > th_max:
#                 omega *= (th_max / (th + 1e-8))
#             # clamp translation
#             t_norm = torch.linalg.norm(t)
#             if t_norm > cfg.trans_clamp_m:
#                 t *= (cfg.trans_clamp_m / (t_norm + 1e-8))
#             # clamp scale
#             if cfg.use_sim3:
#                 log_s.clamp_(min=-cfg.scale_clamp, max=cfg.scale_clamp)
   

#     # final transform & apply to full (unsubsampled) points
#     T_final = sim3_from_params(omega.detach(), t.detach(), log_s.detach() if cfg.use_sim3 else None)
#     P_all_aligned = apply_sim3(T_final, pts_world.reshape(-1,3).to(dev, dt)).view_as(pts_world)
#     C_all_aligned  = apply_sim3(T_final, cams_world.reshape(-1,3)).view_as(cams_world)   # NEW

#     return T_final, P_all_aligned, C_all_aligned







@dataclass
class AlignCfg:
    pad_vox: int = 12
    max_block_vox: int = 128
    iters: int = 4
    samples_per_ray: int = 4
    lambda_free: float = 0.5
    lambda_prior_t: float = 1e-3
    lambda_prior_r: float = 1e-3
    lambda_prior_s: float = 1e-3
    step: float = 1e-1
    use_sim3: bool = True          # legacy
    use_sl4: bool = True          # << NEW: enable projective Sim3+p
    lambda_prior_p: float = 1e-1   # << NEW: prior on projective row p
    p_clamp: float = 0.1          # << NEW: |p|∞ clamp (keeps denom stable)
    downsample_points: int = 10000
    rot_clamp_deg: float = 1.5
    trans_clamp_m: float = 0.03
    scale_clamp: float = 0.01
    conf_filter: bool = True
    conf_quantile: float = 0.10


    # ---- ICP options (NEW) ----
    icp_max_iters: int = 30
    icp_trim_fraction: float | None = 0.7   # keep best fraction; None to disable
    icp_huber_delta: float | None = 0.05    # meters; None to disable
    icp_dist_thresh: float | None = 0.20    # meters; None to disable
    icp_nn_subsample: int | None = 50000    # subsample target cloud for NN
    icp_max_map_points: int | None = 200000 # cap target points in block



def sim3p_from_params(omega, t, log_s, p):
    """
    Build H = [[sR, t],
               [p^T, 1]]
    where s = exp(log_s). (If you want SE(3)+p, set log_s=0.)
    """
    R = so3_exp_map(omega)
    s = torch.exp(log_s) if log_s is not None else torch.tensor(1.0, device=omega.device, dtype=omega.dtype)
    H = torch.eye(4, device=omega.device, dtype=omega.dtype)
    H[:3,:3] = s * R
    H[:3, 3] = t
    H[3, :3] = p.reshape(3)
    H[3, 3]  = 1.0
    return H

def apply_sim3p(H: torch.Tensor, pts: torch.Tensor, denom_eps: float = 1e-3) -> torch.Tensor:
    """
    Apply projective transform to (...,3) points: x' = (A x + t) / (p^T x + 1).
    Safe homogeneous divide with clamping on the denominator.
    """
    A = H[:3,:3]; t = H[:3,3]; p = H[3,:3]
    x = pts
    num = (x @ A.T) + t          # (...,3)
    den = (x @ p) + 1.0          # (...,)

    # keep gradients but avoid blow-ups near zero
    den_safe = torch.sign(den).detach() * torch.clamp(den.abs(), min=denom_eps)
    y = num / den_safe.unsqueeze(-1)
    return y



def sim3_align_to_voxel(
    pts_world: torch.Tensor,              # (S,H,W,3) or (N,3)
    cams_world: torch.Tensor,             # same shape as pts_world (...,3)
    grid,
    conf_world,
    cfg: AlignCfg = AlignCfg()
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (T_4x4, pts_aligned). Optimizes a tiny Sim(3) (or SE(3)) so endpoints land on occupied
    and the rays before endpoints traverse free space.
    """
    dev = grid.device
    dt  = grid.dtype

    # flatten inputs
    P = pts_world.reshape(-1, 3).to(dev, dt)
    C = cams_world.reshape(-1, 3).to(dev, dt)

    # filter finite
    finite = torch.isfinite(P).all(dim=1) & torch.isfinite(C).all(dim=1)
    P, C = P[finite], C[finite]
    if P.numel() == 0:
        T = torch.eye(4, device=dev, dtype=dt)
        return T, pts_world

    # confidences
    if conf_world is not None:
        W = conf_world.reshape(-1).to(dev, dt)[finite]
        if getattr(cfg, "conf_filter", True):
            thr = torch.quantile(W, getattr(cfg, "conf_quantile", 0.10))
            keep = W >= thr
            P, C, W = P[keep], C[keep], W[keep]
    else:
        W = torch.ones(P.shape[0], device=dev, dtype=dt)
        
    # optional subsample for speed
    if cfg.downsample_points is not None and P.shape[0] > cfg.downsample_points:
        idx = torch.randperm(P.shape[0], device=dev)[:cfg.downsample_points]
        P_sub, C_sub, W_sub = P[idx], C[idx], W[idx]
    else:
        P_sub, C_sub, W_sub = P, C, W

    # build dense block around AABB of P_sub with padding
    aabb_min = torch.min(P_sub, dim=0).values - cfg.pad_vox * grid.p.voxel_size
    aabb_max = torch.max(P_sub, dim=0).values + cfg.pad_vox * grid.p.voxel_size
    occ_L, block_origin, vs, DHW = build_dense_block_from_sparse(grid, aabb_min, aabb_max, cfg.max_block_vox)
    # smooth a bit to get a softer field
    occ_L_s = smooth3d(occ_L, k=3)


    # parameters (now include p)
    omega = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
    t     = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
    log_s = torch.zeros(1, device=dev, dtype=dt, requires_grad=True) if cfg.use_sim3 or cfg.use_sl4 else None
    p     = torch.zeros(3, device=dev, dtype=dt, requires_grad=True) if cfg.use_sl4 else None

    opt = torch.optim.Adam([omega, t] + ([log_s] if (cfg.use_sim3 or cfg.use_sl4) else []) + ([p] if cfg.use_sl4 else []),
                        lr=cfg.step)

    # precompute free-space sample ts
    K = cfg.samples_per_ray
    ts = torch.linspace(0.05, 0.95, steps=K, device=dev, dtype=dt)  # avoid endpoints

    for it in range(cfg.iters):
        opt.zero_grad()
        # current transform


        if cfg.use_sl4:
            H = sim3p_from_params(omega, t, log_s, p)
            X  = apply_sim3p(H, P_sub)   # endpoints
            Ct = apply_sim3p(H, C_sub)   # camera centers
        else:
            T = sim3_from_params(omega, t, log_s if cfg.use_sim3 else None)
            X  = apply_sim3(T, P_sub)
            Ct = apply_sim3(T, C_sub)

  
        # ---- hit term: endpoints should be occupied ----
        g_end = world_to_block_coords(X, block_origin, vs, DHW)  # (M,3)
        S_end = sample_occ_prob(occ_L_s, g_end)                  # (M,)
        # robust clamp to avoid log(0)
        eps = 1e-4
        
        w_hit = W_sub
        L_hit = -(w_hit * torch.log(S_end)).sum() / (w_hit.sum() + 1e-9)

        # L_hit = -torch.mean(torch.log(torch.clamp(S_end, min=eps)))  # maximize occupancy at endpoints

        # ---- free term: along ray BEFORE endpoint should be free ----
        # sample along rays
        V  = X - Ct
        Y = Ct.unsqueeze(1) + ts.view(1,-1,1) * V.unsqueeze(1)  # (M,K,3)
        g_free = world_to_block_coords(Y, block_origin, vs, DHW).view(-1,3)
        S_free = sample_occ_prob(occ_L_s, g_free).view(-1)         # (M*K,)
        
        
        # per-ray weights: nearer samples a bit higher (shape: M*K)
        w_ray = torch.linspace(1.0, 0.6, steps=K, device=dev, dtype=dt)  # (K,)
        w_ray = w_ray.view(1, K).expand(W_sub.shape[0], K).reshape(-1)   # (M*K,)

        
        w_pt = W_sub.view(-1,1).expand(-1, K).reshape(-1)
        w_free = w_pt * w_ray

        # -∑ w log(1 - p) / ∑ w
        L_free = -(w_free * torch.log(1.0 - S_free)).sum() / (w_free.sum() + 1e-9)
        
        # L_free = -torch.mean(torch.log(torch.clamp(1.0 - S_free, min=eps)))  # maximize free space

        # clamp step (safety)
        # convert omega to magnitude
       # Priors
        L_prior = cfg.lambda_prior_t * (t*t).sum() + cfg.lambda_prior_r * (omega*omega).sum()
        if (cfg.use_sim3 or cfg.use_sl4):
            L_prior = L_prior + cfg.lambda_prior_s * (log_s*log_s).sum()
        if cfg.use_sl4:
            L_prior = L_prior + cfg.lambda_prior_p * (p*p).sum()

        loss = L_hit + cfg.lambda_free * L_free + L_prior
        
        loss.backward()
        
        print("Loss: ", loss)
        print(L_hit)
        print(cfg.lambda_free * L_free)
        print(L_prior)
        print(cfg.lambda_prior_s * (log_s*log_s).sum())
        
        opt.step()
        
        # After opt.step(): safety clamps
        with torch.no_grad():
            th = torch.linalg.norm(omega)
            th_max = (cfg.rot_clamp_deg / 180.0) * torch.pi
            if th > th_max: omega *= (th_max / (th + 1e-8))
            t_norm = torch.linalg.norm(t)
            if t_norm > cfg.trans_clamp_m: t *= (cfg.trans_clamp_m / (t_norm + 1e-8))
            if (cfg.use_sim3 or cfg.use_sl4):
                log_s.clamp_(min=-cfg.scale_clamp, max=cfg.scale_clamp)
            if cfg.use_sl4:
                p.clamp_(min=-cfg.p_clamp, max=cfg.p_clamp)  # keeps denominator away from zero

    # final transform & apply to full (unsubsampled) points
    
    if cfg.use_sl4:
        H_final = sim3p_from_params(omega.detach(), t.detach(), log_s.detach(), p.detach())
        P_all_aligned = apply_sim3p(H_final, pts_world.reshape(-1,3).to(dev, dt)).view_as(pts_world)
        C_all_aligned = apply_sim3p(H_final, cams_world.reshape(-1,3).to(dev, dt)).view_as(cams_world)
        return H_final, P_all_aligned, C_all_aligned
    else:
        T_final = sim3_from_params(omega.detach(), t.detach(), log_s.detach() if cfg.use_sim3 else None)
        P_all_aligned = apply_sim3(T_final, pts_world.reshape(-1,3).to(dev, dt)).view_as(pts_world)
        C_all_aligned = apply_sim3(T_final, cams_world.reshape(-1,3)).view_as(cams_world)
        return T_final, P_all_aligned, C_all_aligned




   
def extract_map_points_in_block(grid, ijk_min, ijk_max, max_points=None):
    """
    Return occupied voxel centers within [ijk_min, ijk_max] (inclusive) as (N,3) float tensor.
    """
    dev, dt = grid.device, grid.dtype
    if grid.keys.numel() == 0:
        return torch.empty(0, 3, device=dev, dtype=dt)

    ijk_all = grid._unhash_keys(grid.keys).to(torch.int64)     # (N,3)
    occ_mask = grid.occupied_mask()                            # (N,) bool on device

    m0 = (ijk_all[:,0] >= ijk_min[0]) & (ijk_all[:,0] <= ijk_max[0])
    m1 = (ijk_all[:,1] >= ijk_min[1]) & (ijk_all[:,1] <= ijk_max[1])
    m2 = (ijk_all[:,2] >= ijk_min[2]) & (ijk_all[:,2] <= ijk_max[2])
    m  = m0 & m1 & m2 & occ_mask

    if not m.any():
        return torch.empty(0, 3, device=dev, dtype=dt)

    ijk = ijk_all[m].to(grid.origin.dtype)
    centers = grid.origin + (ijk + 0.5) * grid.p.voxel_size     # (M,3)

    if (max_points is not None) and (centers.shape[0] > max_points):
        idx = torch.randperm(centers.shape[0], device=dev)[:max_points]
        centers = centers[idx]
    return centers



def _sim3_from_Rts(R, t, s, device, dtype):
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3,:3] = s * R
    T[:3, 3] = t
    return T

def _solve_sim3_procrustes(P, Q, w=None):
    """
    P,Q: (N,3) correspondences; w: (N,) weights or None. Returns 4x4 T aligning P->Q.
    """
    device, dtype = P.device, P.dtype
    N = P.shape[0]
    if w is None:
        w = torch.ones(N, device=device, dtype=dtype)
    w = torch.clamp(w, min=0)
    w_sum = torch.clamp(w.sum(), min=1e-12)
    wn = w / w_sum

    muP = (wn[:,None] * P).sum(0)
    muQ = (wn[:,None] * Q).sum(0)
    Pc, Qc = P - muP, Q - muQ

    C = Pc.mul(wn[:,None]).T @ Qc                   # 3x3
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    R = Vh.T @ U.T
    if torch.linalg.det(R) < 0:
        Vh[-1,:] *= -1
        R = Vh.T @ U.T
    denom = (wn[:,None] * (Pc*Pc)).sum()
    s = S.sum() / torch.clamp(denom, min=1e-12)
    t = muQ - s * (R @ muP)
    return _sim3_from_Rts(R, t, s, device, dtype)


def sim3_icp_align_to_voxel(
    pts_world: torch.Tensor,      # (S,H,W,3) or (N,3) predicted endpoints (source)
    cams_world: torch.Tensor,     # same shape; only used to return transformed cams
    grid,                          # TorchSparseVoxelGrid (target map)
    conf_world,                    # confidences per point (broadcasted/flattened)
    cfg: AlignCfg = AlignCfg()
):
    """
    Closed-form Sim(3) ICP aligning point set P (from pts_world) to map points extracted from 'grid'.
    Returns (T_4x4, pts_aligned, cams_aligned) to match your existing API.
    """
    dev, dt = grid.device, grid.dtype

    # --- flatten + filter finite ---
    P = pts_world.reshape(-1, 3).to(dev, dt)
    C = cams_world.reshape(-1, 3).to(dev, dt)
    finite = torch.isfinite(P).all(1) & torch.isfinite(C).all(1)
    P, C = P[finite], C[finite]
    if P.numel() == 0:
        T = torch.eye(4, device=dev, dtype=dt)
        return T, pts_world, cams_world


    # --- confidences as weights (optional) ---
    if conf_world is not None:
        W = conf_world.reshape(-1).to(dev, dt)[finite]
        if getattr(cfg, "conf_filter", True):
            thr = torch.quantile(W, getattr(cfg, "conf_quantile", 0.10))
            keep = W >= thr
            P, C, W = P[keep], C[keep], W[keep]
    else:
        W = torch.ones(P.shape[0], device=dev, dtype=dt)


    # --- optional subsample source for speed (weights kept) ---
    if cfg.downsample_points is not None and P.shape[0] > cfg.downsample_points:
        idx = torch.randperm(P.shape[0], device=dev)[:cfg.downsample_points]
        P_src, W_src = P[idx], W[idx]
    else:
        P_src, W_src = P, W

    # --- build same AABB you use for dense block; extract target map points inside it ---
    pad = cfg.pad_vox * grid.p.voxel_size
    aabb_min = torch.min(P_src, dim=0).values - pad
    aabb_max = torch.max(P_src, dim=0).values + pad
    ijk_min = torch.floor((aabb_min - grid.origin) / grid.p.voxel_size).to(torch.int64)
    ijk_max = torch.floor((aabb_max - grid.origin) / grid.p.voxel_size).to(torch.int64)


    Q_tgt = extract_map_points_in_block(grid, ijk_min, ijk_max, max_points=cfg.icp_max_map_points)
    if Q_tgt.shape[0] < 3:
        # no target points in crop -> identity
        T = torch.eye(4, device=dev, dtype=dt)
        return T, pts_world, cams_world

    # --- subsample target for NN if requested ---
    if cfg.icp_nn_subsample is not None and Q_tgt.shape[0] > cfg.icp_nn_subsample:
        j = torch.randperm(Q_tgt.shape[0], device=dev)[:cfg.icp_nn_subsample]
        Q_nn = Q_tgt[j]
    else:
        Q_nn = Q_tgt


    # --- ICP loop ---
    T = torch.eye(4, device=dev, dtype=dt)
    rmse_prev = None
    
    # --- before ICP loop ---
    
    import faiss

    d = 3
    index = faiss.IndexFlatL2(d)

    Q_np = Q_nn.detach().cpu().numpy().astype("float32")
    index.add(Q_np)
    

    
    for itr in range(cfg.icp_max_iters):
        print(itr)
        P_t = (P_src @ T[:3,:3].T) + T[:3,3]             # transform source

        # nearest neighbors
        # NOTE: For very large clouds, replace with KD-tree/FAISS
        
        
        # d2 = torch.cdist(P_t, Q_nn) ** 2                 # (Ns, Nt)
        # nn_idx = torch.argmin(d2, dim=1)
        # Q_corr = Q_nn[nn_idx]
        # resid = P_t - Q_corr
        # r_norm = torch.linalg.norm(resid, dim=1)
        

        P_np = P_t.detach().cpu().numpy().astype("float32")
        D, I = index.search(P_np, 1)                         # D: (Ns,1) squared L2

        nn_idx = torch.from_numpy(I[:, 0]).to(P_t.device, torch.long)
        d2     = torch.from_numpy(D[:, 0]).to(P_t.device)    # squared distance

        Q_corr = Q_nn[nn_idx]                                 # (Ns,3)
        # If you need residual vectors (e.g., Huber on vector norm):
        resid  = P_t - Q_corr
        # But for scalar residual norm, use d2 directly:
        r_norm = torch.sqrt(torch.clamp(d2, min=0.0))

        # inlier mask: distance threshold + trimming
        mask = torch.isfinite(r_norm)
        if cfg.icp_dist_thresh is not None:
            mask &= (r_norm <= cfg.icp_dist_thresh)
        if cfg.icp_trim_fraction is not None and 0 < cfg.icp_trim_fraction < 1.0:
            k = max(3, int(cfg.icp_trim_fraction * mask.sum().item()))
            if k < mask.sum().item():
                vals, idxs = torch.topk(r_norm[mask], k=k, largest=False)
                new_mask = torch.zeros_like(mask)
                new_mask[mask.nonzero(as_tuple=True)[0][idxs]] = True
                mask = new_mask

        if mask.sum() < 3:
            break

        # robust per-point weights
        if cfg.icp_huber_delta is not None:
            d = r_norm.detach()
            w = torch.ones_like(d)
            over = d > cfg.icp_huber_delta
            w[over] = cfg.icp_huber_delta / (d[over] + 1e-9)
            w = w[mask]
        else:
            w = torch.ones(mask.sum(), device=dev, dtype=dt)

        # combine with source confidences
        w = w * (W_src[mask] / (W_src[mask].mean() + 1e-9))

        # closed-form Sim(3) on correspondences (P_t -> Q_corr)
        T_step = _solve_sim3_procrustes(P_t[mask], Q_corr[mask], w)
        T = T_step @ T

        rmse = torch.sqrt((r_norm[mask] ** 2).mean())
        if (rmse_prev is not None) and (abs(rmse_prev - rmse) < 1e-6):
            break
        rmse_prev = rmse

    # --- apply to full (unsubsampled) points/cameras to match your API ---
    P_all_aligned = (pts_world.reshape(-1,3).to(dev, dt) @ T[:3,:3].T) + T[:3,3]
    P_all_aligned = P_all_aligned.view_as(pts_world)
    C_all_aligned = (cams_world.reshape(-1,3).to(dev, dt) @ T[:3,:3].T) + T[:3,3]
    C_all_aligned = C_all_aligned.view_as(cams_world)
    
    return T, P_all_aligned, C_all_aligned




# from dataclasses import dataclass
# from typing import Tuple, Optional, Sequence
# import torch
# import torch.nn.functional as F

# # ---------- New / extended config ----------

# @dataclass
# class AlignCfg:
#     # general
#     use_sim3: bool = True
#     iters: int = 120
#     step: float = 1e-2
#     downsample_points: Optional[int] = 150_000  # None = no downsample
#     pad_vox: int = 6
#     max_block_vox: int = 256  # per side cap

#     # multi-resolution schedule (coarse -> fine)
#     sigmas: Optional[Sequence[float]] = (3.0, 2.0, 1.0)   # vox smoothing kernel radii (approx)
#     lambda_free_schedule: Optional[Sequence[float]] = (0.25, 0.5, 1.0)

#     # “SDF-like” loss
#     sdf_mode: str = "logit"  # "logit" (robust fallback) or "edt" (if you wire a true EDT)
#     sdf_eps: float = 1e-3
#     free_margin_m: float = 2.0 * 0.05  # require free up to this margin before hit (~2 vox)
#     free_step_m: float = 0.05          # ray marching step (≈ voxel size)
#     min_ray_m: float = 2e-2
#     samples_per_ray: int = 48          # only used if fixed-K sampling is enabled

#     # weights / priors
#     lambda_prior_t: float = 1e-4
#     lambda_prior_r: float = 1e-4
#     lambda_prior_s: float = 1e-4
#     lambda_norm: float = 0.05          # normal alignment strength

#     # IRLS / confidence
#     conf_filter: bool = True
#     conf_quantile: float = 0.10        # drop bottom 10% conf points
#     huber_delta: float = 0.05          # for residual reweight

#     # clamps
#     rot_clamp_deg: float = 15.0
#     trans_clamp_m: float = 0.30
#     scale_clamp: float = 0.10          # ~ ±10% scale

#     # advanced
#     use_lm: bool = False               # keep Adam default; LM hook left for later
    
#     lambda_scale_from_raycast: float = 0.5  # strong anchor; tune 1e-2..1
#     freeze_scale_on_coarse: bool = True
    
#     allow_sim3_last_stage: bool = True
#     tiny_scale_clamp: float = 0.01
    
#     bootstrap_icp_iters: int = 10
#     bootstrap_huber: float = 0.05
#     bootstrap_nn_max_m: float = 0.15
#     allow_sim3_last_stage: bool = False      # keep refinement SE(3)-only at first



# # ---------- tiny utils ----------

# def per_ray_mean(x: torch.Tensor, K: int) -> torch.Tensor:
#     """x is (M*K,), returns (M,) mean over K."""
#     return x.view(-1, K).mean(dim=1)

# def huber_weight(r: torch.Tensor, delta: float) -> torch.Tensor:
#     a = r.abs()
#     return torch.where(a <= delta, torch.ones_like(a), delta / (a + 1e-8))

# def sample_trilinear(vol: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
#     """
#     vol: (1,1,D,H,W) tensor
#     g:   (N,3) in voxel grid coords (z,y,x) with origin at (0,0,0)
#     returns (N,)
#     """
#     D, H, W = vol.shape[-3:]
#     # normalize to [-1,1] for grid_sample with (x,y,z) order
#     x = (g[:,2] / max(W-1,1)) * 2 - 1
#     y = (g[:,1] / max(H-1,1)) * 2 - 1
#     z = (g[:,0] / max(D-1,1)) * 2 - 1
#     grid = torch.stack((x,y,z), dim=-1).view(1,1,-1,1,3)
#     # grid_sample expects (N,C,D,H,W); align corners True to avoid drift
#     val = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=True)
#     return val.view(-1)

# def sample_trilinear_grad(vol: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
#     """
#     Central differences in voxel units on vol to estimate ∇ at g (in grid space z,y,x).
#     Returns normals in WORLD xyz (so we swap order and sign appropriately later).
#     """
#     # small step: one voxel in grid coords
#     dz = torch.tensor([1.0,0.0,0.0], device=g.device, dtype=g.dtype)
#     dy = torch.tensor([0.0,1.0,0.0], device=g.device, dtype=g.dtype)
#     dx = torch.tensor([0.0,0.0,1.0], device=g.device, dtype=g.dtype)
#     pz1 = sample_trilinear(vol, g+dz)
#     pz0 = sample_trilinear(vol, g-dz)
#     py1 = sample_trilinear(vol, g+dy)
#     py0 = sample_trilinear(vol, g-dy)
#     px1 = sample_trilinear(vol, g+dx)
#     px0 = sample_trilinear(vol, g-dx)
#     # gradient in (z,y,x)
#     gz = (pz1 - pz0)*0.5
#     gy = (py1 - py0)*0.5
#     gx = (px1 - px0)*0.5
#     # convert to WORLD xyz direction: (gx, gy, gz)
#     grad = torch.stack((gx, gy, gz), dim=-1)
#     grad = F.normalize(grad, dim=-1, eps=1e-8)
#     return grad

# def build_signed_distance_from_logit_occ(occ_prob: torch.Tensor, vs_m: float) -> torch.Tensor:
#     """
#     Turn a soft occupancy prob (0..1) into a smooth, signed distance proxy.
#     phi ≈ -α * logit(p), scaled to meters; α sets slope around surface.
#     """
#     eps = 1e-4
#     p = torch.clamp(occ_prob, eps, 1.0-eps)
#     logit = torch.log(p) - torch.log(1-p)
#     # scale so that |phi| ~ 1 voxel around p=0.5; derivative dphi/dx ≈ vs at boundary
#     alpha = vs_m / 4.0
#     phi = -alpha * logit
#     return phi


# import torch
# import torch.nn.functional as F

# # ---- Huber weights (like VGGT-Long IRLS) ----
# def _huber_w(residuals: torch.Tensor, delta: float) -> torch.Tensor:
#     a = residuals.abs()
#     return torch.where(a <= delta, torch.ones_like(a), delta / (a + 1e-8))

# # ---- Extract surface sample (world coords) from your sparse grid in the AABB of P ----
# def extract_voxel_surface_points(grid, aabb_min, aabb_max, max_block_vox=192, occ_thr=0.5, stride=1):
#     """
#     Returns (S_world: (Ns,3)).
#     Surface = occupied voxels with at least one 6-neighbor not occupied.
#     """
#     occ_L, block_origin, vs_m, DHW = build_dense_block_from_sparse(grid, aabb_min, aabb_max, max_block_vox)
#     occ = (occ_L > occ_thr).squeeze(0).squeeze(0)  # (D,H,W) bool/float

#     # 6-neighborhood differences to find boundary voxels
#     D, H, W = occ.shape
#     pad = F.pad(occ.float(), (0,0,0,0,0,0), value=0.0)

#     def shift(vol, dz=0, dy=0, dx=0):
#         z0, z1 = max(0,-dz), D - max(0,dz)
#         y0, y1 = max(0,-dy), H - max(0,dy)
#         x0, x1 = max(0,-dx), W - max(0,dx)
#         out = torch.zeros_like(vol)
#         out[z0:z1, y0:y1, x0:x1] = vol[z0+dz:z1+dz, y0+dy:y1+dy, x0+dx:x1+dx]
#         return out

#     nbh = (shift(occ,1,0,0) + shift(occ,-1,0,0) +
#            shift(occ,0,1,0) + shift(occ,0,-1,0) +
#            shift(occ,0,0,1) + shift(occ,0,0,-1))

#     surface = (occ > 0.5) & (nbh < 6)   # boundary-ish
#     if stride > 1:
#         surface[::stride, ::stride, ::stride] = surface[::stride, ::stride, ::stride]  # cheap decimation

#     idx = surface.nonzero(as_tuple=False)  # (Ns,3) in (z,y,x)
#     if idx.numel() == 0:
#         return torch.empty(0,3, device=occ.device, dtype=occ.dtype)

#     zyxs = idx.to(torch.float32)
#     # voxel center in world: origin + (idx + 0.5) * vs (pay attention to axis order)
#     Z = block_origin[2] + (zyxs[:,0] + 0.5) * vs_m
#     Y = block_origin[1] + (zyxs[:,1] + 0.5) * vs_m
#     X = block_origin[0] + (zyxs[:,2] + 0.5) * vs_m
#     S_world = torch.stack([X, Y, Z], dim=-1).to(grid.device, grid.dtype)  # (Ns,3), xyz
#     return S_world

# # ---- Nearest neighbors from P (source) to S (map surface) with reject threshold ----
# def match_nn(P: torch.Tensor, S: torch.Tensor, max_dist: float, batch: int = 200_000):
#     """
#     Returns indices into P and S for accepted pairs, plus distances.
#     """
#     if P.numel() == 0 or S.numel() == 0:
#         return (torch.empty(0, dtype=torch.long, device=P.device),
#                 torch.empty(0, dtype=torch.long, device=P.device),
#                 torch.empty(0, device=P.device, dtype=P.dtype))
#     sel_p = []
#     sel_s = []
#     sel_d = []
#     for start in range(0, P.shape[0], batch):
#         end = min(start+batch, P.shape[0])
#         d = torch.cdist(P[start:end], S)  # (b, Ns)
#         m, j = d.min(dim=1)               # NN in S
#         keep = m <= max_dist
#         if keep.any():
#             i_keep = torch.nonzero(keep, as_tuple=False).squeeze(1) + start
#             sel_p.append(i_keep)
#             sel_s.append(j[keep])
#             sel_d.append(m[keep])
#     if not sel_p:
#         return (torch.empty(0, dtype=torch.long, device=P.device),
#                 torch.empty(0, dtype=torch.long, device=P.device),
#                 torch.empty(0, device=P.device, dtype=P.dtype))
#     I = torch.cat(sel_p, dim=0)
#     J = torch.cat(sel_s, dim=0)
#     D = torch.cat(sel_d, dim=0)
#     return I, J, D

# # ---- Weighted Umeyama with IRLS (returns 4x4 Sim(3)) ----
# def weighted_umeyama_irls(X: torch.Tensor, Y: torch.Tensor, w0: torch.Tensor,
#                           iters: int = 3, huber_delta: float = 0.05, use_scale: bool = True):
#     """
#     X: (N,3) source, Y: (N,3) target correspondences, w0: (N,) initial weights (confidences).
#     Returns T ∈ R^{4x4} (Sim(3) if use_scale else SE(3)).
#     """
#     dev, dt = X.device, X.dtype
#     w = (w0 + 1e-12)
#     w = w / (w.sum() + 1e-12)

#     R = torch.eye(3, device=dev, dtype=dt)
#     s = torch.tensor(1.0, device=dev, dtype=dt)
#     t = torch.zeros(3, device=dev, dtype=dt)

#     for _ in range(iters):
#         wx = (w.view(-1,1) * X)
#         wy = (w.view(-1,1) * Y)
#         x_bar = wx.sum(dim=0) / (w.sum()+1e-12)
#         y_bar = wy.sum(dim=0) / (w.sum()+1e-12)

#         Xc = X - x_bar
#         Yc = Y - y_bar

#         # Weighted cross-covariance
#         Sigma = (w.view(-1,1) * Yc).T @ Xc  # 3x3
#         U, S, Vt = torch.linalg.svd(Sigma, full_matrices=False)
#         R_ = U @ torch.diag(torch.tensor([1.0, 1.0, torch.sign(torch.linalg.det(U@Vt))], device=dev, dtype=dt)) @ Vt
#         if use_scale:
#             var_x = (w.view(-1,1) * (Xc*Xc)).sum()
#             s_ = (S.sum()) / (var_x + 1e-12)
#         else:
#             s_ = torch.tensor(1.0, device=dev, dtype=dt)
#         t_ = y_bar - s_ * (R_ @ x_bar)

#         # residuals for IRLS
#         r = (s_* (X @ R_.T) + t_ - Y).norm(dim=1)
#         w = (w0 * _huber_w(r, huber_delta)).clamp(min=1e-8)
#         w = w / (w.sum() + 1e-12)

#         R, s, t = R_, s_, t_

#     T = torch.eye(4, device=dev, dtype=dt)
#     T[:3,:3] = s * R
#     T[:3, 3] = t
#     return T


# def solve_scale_from_depths(d0: torch.Tensor, d_map: torch.Tensor, w: torch.Tensor,
#                             s_min: float = 0.6, s_max: float = 1.4) -> torch.Tensor:
#     """
#     Closed-form LS: s* = sum w * d_map * d0 / sum w * d0^2, clamped.
#     d0:   (M,) ray lengths ||P-C|| before scaling (precomputed once)
#     d_map:(M,) raycast depths in meters (NaN where no hit)
#     w:    (M,) weights (e.g., confidences)
#     """
#     ok = torch.isfinite(d_map) & (d0 > 1e-6)
#     if not ok.any():
#         return torch.tensor(1.0, device=d0.device, dtype=d0.dtype)
#     d0_ok = d0[ok]; dm_ok = d_map[ok]; w_ok = w[ok]
#     num = (w_ok * dm_ok * d0_ok).sum()
#     den = (w_ok * (d0_ok ** 2)).sum() + 1e-12
#     return torch.clamp(num / den, s_min, s_max)

# def raycast_sdf_first_hit(SDF, block_origin, vs_m, DHW, C, Vhat,
#                           max_steps=64, start_step=0.5, hit_thresh_vox=0.6):
#     """
#     Sphere-trace along r(t)=C + t*Vhat (WORLD). SDF is in meters.
#     Returns distances t (meters); NaN if no hit.
#     """
#     dev, dt = C.device, C.dtype
#     t = torch.full((C.shape[0],), start_step*vs_m, device=dev, dtype=dt)
#     pos = C + Vhat * t.unsqueeze(-1)
#     hit = torch.zeros(C.shape[0], dtype=torch.bool, device=dev)
#     dist = torch.full((C.shape[0],), float('nan'), device=dev, dtype=dt)
#     for _ in range(max_steps):
#         g = world_to_block_coords(pos, block_origin, vs_m, DHW)
#         phi = sample_trilinear(SDF, g)  # meters
#         done = (phi <= hit_thresh_vox*vs_m) & (~hit)
#         if done.any():
#             dist[done] = t[done]
#             hit |= done
#         cont = ~hit
#         if not cont.any(): break
#         step = torch.clamp(phi[cont], min=0.25*vs_m, max=2.0*vs_m)
#         t[cont] = t[cont] + step
#         pos[cont] = C[cont] + Vhat[cont] * t[cont].unsqueeze(-1)
#     return dist

# def robust_weighted_median(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#     ok = torch.isfinite(x) & torch.isfinite(w) & (w > 0)
#     if not ok.any():
#         return torch.tensor(1.0, device=x.device, dtype=x.dtype)
#     x = x[ok]; w = w[ok]
#     xs, order = torch.sort(x)
#     ws = w[order]
#     c = torch.cumsum(ws, dim=0)
#     idx = torch.searchsorted(c, 0.5*(c[-1] + 1e-12))
#     return xs[min(int(idx.item()), xs.numel()-1)]


# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def pca_frame(X: torch.Tensor):
#     """Return centroid, eigenvectors (3x3, right-handed), and eigenvalues."""
#     c = X.mean(0)
#     Y = X - c
#     C = (Y.T @ Y) / max(1, X.shape[0]-1)
#     S,U = torch.linalg.eigh(C)     # S asc, U cols are eigenvectors
#     # sort descending
#     idx = torch.argsort(S, descending=True)
#     S = S[idx]; U = U[:, idx]
#     # enforce right-handed
#     if torch.det(U) < 0:
#         U[:, -1] *= -1
#     return c, U, S

# def compose_T(R, t, s, device, dtype):
#     T = torch.eye(4, device=device, dtype=dtype)
#     T[:3,:3] = s * R
#     T[:3, 3] = t
#     return T

# def apply_T(T, X):
#     Xh = torch.cat([X, torch.ones(X.shape[0],1, device=X.device, dtype=X.dtype)], dim=1)
#     Yh = (T @ Xh.T).T
#     return Yh[:,:3]

# def nn_correspondences(X, Y, max_dist):
#     # brute-force for clarity; replace with your kNN if available
#     d2 = torch.cdist(X, Y)  # (Nx, Ny)
#     dmin, j = d2.min(dim=1)
#     mask = dmin <= (max_dist**2)
#     i = torch.nonzero(mask, as_tuple=False).squeeze(1)
#     return i, j[mask], torch.sqrt(dmin[mask] + 1e-12)

# def weighted_umeyama_sim3(X, Y, w=None):
#     # X -> Y, solve Sim(3)
#     if w is None: w = torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
#     w = w / (w.sum() + 1e-12)
#     mx = (w[:,None]*X).sum(0); my = (w[:,None]*Y).sum(0)
#     Xc = X - mx; Yc = Y - my
#     Sigma = Xc.T @ (w[:,None]*Yc)
#     U,S,Vt = torch.linalg.svd(Sigma, full_matrices=True)
#     R = Vt.T @ U.T
#     if torch.det(R) < 0:
#         Vt[:, -1] *= -1
#         R = Vt.T @ U.T
#     var = (w * (Xc*Xc).sum(-1)).sum()
#     s = (S.sum() / (var + 1e-12))
#     t = my - s * (R @ mx)
#     return s, R, t

# def global_sim3_bootstrap(
#     P_part: torch.Tensor, C_part: torch.Tensor, grid,
#     *, occ_thr=0.5, stride=2, max_block_vox=192,
#     icp_iters=8, icp_huber=0.05, max_nn_dist=0.8
# ):
#     """
#     Returns T_boot (4x4). Robust Sim(3) init from PCA + Sim(3)-ICP to LT surface.
#     """
#     dev, dt = grid.device, grid.dtype
#     vs = grid.p.voxel_size

#     # 1) sample LT surface broadly around partial (large metric pad)
#     pc_min = torch.minimum(P_part.min(0).values, C_part.min(0).values)
#     pc_max = torch.maximum(P_part.max(0).values, C_part.max(0).values)
#     diag = torch.linalg.norm(pc_max - pc_min).item()
#     pad = max(6*vs, 0.75*diag)
#     aabb_min = pc_min - pad
#     aabb_max = pc_max + pad
#     S_world = extract_voxel_surface_points(grid, aabb_min, aabb_max,
#                                            max_block_vox=min(max_block_vox,192),
#                                            occ_thr=occ_thr, stride=stride)
#     if S_world.shape[0] < 500:
#         # expand more if too few points
#         pad = max(pad, 3.0)
#         aabb_min = pc_min - pad
#         aabb_max = pc_max + pad
#         S_world = extract_voxel_surface_points(grid, aabb_min, aabb_max,
#                                                max_block_vox=min(max_block_vox,192),
#                                                occ_thr=occ_thr, stride=stride)
#     if S_world.shape[0] < 200:
#         # fallback: take all LT occupied vox centers
#         ijk = grid._unhash_keys(grid.occupied_indices())
#         S_world = (grid.origin + (ijk.to(dt) + 0.5) * vs).to(dev, dt)

#     # 2) PCA alignment (axes + centroid) for rotation/translation guess
#     #    Also gives a scale ratio (diagonal length) as coarse s
#     cP, UP, SP = pca_frame(P_part)
#     cS, US, SS = pca_frame(S_world)
#     R0 = US @ UP.T   # align axes
#     if torch.det(R0) < 0:  # ensure right-handed
#         R0[:, -1] *= -1
#     # scale: match RMS extents (coarse)
#     lenP = torch.sqrt(SP.sum() + 1e-12)
#     lenS = torch.sqrt(SS.sum() + 1e-12)
#     s0 = (lenS / (lenP + 1e-12)).clamp(0.2, 5.0)
#     t0 = cS - s0 * (R0 @ cP)
#     T = compose_T(R0, t0, float(s0), dev, dt)

#     # 3) Sim(3) ICP (point-to-plane via small SVDs; here we do point-to-point with Huber for simplicity)
#     X = apply_T(T, P_part)
#     for _ in range(icp_iters):
#         # NN correspondences with robust cutoff
#         Ii, Jj, Dj = nn_correspondences(X, S_world, max_dist=max_nn_dist)
#         if Ii.numel() < 500: break
#         w = torch.clamp(icp_huber / (icp_huber + Dj), 0.15, 1.0)  # Huber-like weight

#         s, R, t = weighted_umeyama_sim3(X[Ii], S_world[Jj], w)
#         # compose incremental on the LEFT in world frame
#         T_inc = compose_T(R, t, float(s), dev, dt)
#         T = T_inc @ T
#         X = apply_T(T_inc, X)

#     return T


# def sim3_scale_about(anchor: torch.Tensor, s: float, device, dtype):
#     """
#     Return a 4x4 that scales about 'anchor' (no rotation/translation drift).
#     X' = anchor + s * (X - anchor).
#     """
#     T = torch.eye(4, device=device, dtype=dtype)
#     T[:3,:3] = torch.eye(3, device=device, dtype=dtype) * s
#     T[:3, 3] = anchor - s * anchor
#     return T

# def shrink_from_nn_projection(P, C, W, grid, aabb_min, aabb_max, voxel_size,
#                               max_block_vox=192, occ_thr=0.5, stride=2,
#                               p_percentile=0.30, bias=0.95, smin=0.3, smax=0.95):
#     """
#     Ray-FREE shrink: match P to voxel surface S, project onto each ray (P-C),
#     ratio = d_map_along_ray / d_obs → robust lower-percentile scale.
#     Does NOT move the partial; only returns a scalar s0 (<=1).
#     """
#     dev, dt = P.device, P.dtype
#     S_world = extract_voxel_surface_points(
#         grid, aabb_min, aabb_max,
#         max_block_vox=min(max_block_vox, 192), occ_thr=occ_thr, stride=stride
#     )
#     if S_world.numel() == 0:
#         return torch.tensor(1.0, device=dev, dtype=dt), 0

#     Ii, Jj, Dj = match_nn(P, S_world, max_dist=2.5*voxel_size)
#     nmatch = Ii.numel()
#     if nmatch < 200:
#         return torch.tensor(1.0, device=dev, dtype=dt), nmatch

#     X = P[Ii]; Y = S_world[Jj]; Csel = C[Ii]
#     V  = X - Csel
#     d0 = torch.linalg.norm(V, dim=-1)
#     ok = d0 > 1e-6
#     if not ok.any():
#         return torch.tensor(1.0, device=dev, dtype=dt), nmatch

#     Vhat = F.normalize(V[ok], dim=-1, eps=1e-8)
#     d_map_ray = (Y[ok] - Csel[ok]).mul(Vhat).sum(-1).clamp(min=0.0)
#     ratio = d_map_ray / (d0[ok] + 1e-6)

#     w = W[Ii][ok] * torch.clamp(1.0/(d0[ok]+1e-6), max=5.0)
#     xs, order = torch.sort(ratio); ws = w[order]
#     c = torch.cumsum(ws, dim=0); target = p_percentile * (c[-1] + 1e-12)
#     idx = torch.searchsorted(c, target)
#     s_raw = xs[min(int(idx.item()), xs.numel()-1)]

#     s0 = torch.clamp(s_raw * bias, smin, smax)
#     return s0, nmatch


# def sim3_align_to_voxel_robust(
#     pts_world: torch.Tensor,              # (S,H,W,3) or (N,3)
#     cams_world: torch.Tensor,             # same shape as pts_world (...,3)
#     grid,                                  # must have .device, .dtype, .p.voxel_size
#     cfg,                                   # AlignCfg-like object
#     conf_world: Optional[torch.Tensor] = None
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Returns (T_4x4, pts_aligned, cams_aligned).

#     Pipeline:
#       1) flatten, filter, optional downsample
#       2) global Sim(3) bootstrap (PCA + Sim(3)-ICP) → T_accum
#       3) coarse→fine refinement (SE(3) only by default; optionally tiny Sim(3) on last stage)
#     """
#     dev = grid.device; dt = grid.dtype
#     vs  = grid.p.voxel_size

#     # ---- flatten + filter ----
#     P_full = pts_world.reshape(-1, 3).to(dev, dt)
#     C_full = cams_world.reshape(-1, 3).to(dev, dt)
#     finite = torch.isfinite(P_full).all(dim=1) & torch.isfinite(C_full).all(dim=1)
#     P = P_full[finite]
#     C = C_full[finite]
#     if P.numel() == 0:
#         T = torch.eye(4, device=dev, dtype=dt)
#         return T, pts_world, cams_world

#     # confidences
#     if conf_world is not None:
#         W = conf_world.reshape(-1).to(dev, dt)[finite]
#         if getattr(cfg, "conf_filter", True):
#             thr = torch.quantile(W, getattr(cfg, "conf_quantile", 0.10))
#             keep = W >= thr
#             P, C, W = P[keep], C[keep], W[keep]
#     else:
#         W = torch.ones(P.shape[0], device=dev, dtype=dt)

#     # ---- optional stratified downsample in world ----
#     if getattr(cfg, "downsample_points", None) is not None and P.shape[0] > cfg.downsample_points:
#         cell = 2.0 * vs
#         keys = torch.floor(P / cell).to(torch.int64)
#         _, inv = torch.unique(keys, dim=0, return_inverse=True)
#         M = inv.numel()
#         ar = torch.arange(M, device=dev, dtype=torch.long)
#         comp = inv * M + ar
#         comp_sorted, order = torch.sort(comp)
#         inv_sorted = comp_sorted // M
#         change = torch.ones_like(inv_sorted, dtype=torch.bool); change[1:] = inv_sorted[1:] != inv_sorted[:-1]
#         starts = torch.nonzero(change, as_tuple=False).squeeze(1)
#         idx = order[starts]
#         if idx.numel() > cfg.downsample_points:
#             idx = idx[torch.randperm(idx.numel(), device=dev)[:cfg.downsample_points]]
#         P, C, W = P[idx], C[idx], W[idx]

#     # ---- global Sim(3) bootstrap (NO raycast/shrink needed) ----
#     # T_accum = global_sim3_bootstrap(
#     #     P, C, grid,
#     #     occ_thr=0.0, stride=2,
#     #     max_block_vox=getattr(cfg, "max_block_vox", 192),
#     #     icp_iters=getattr(cfg, "bootstrap_icp_iters", 8),
#     #     icp_huber=getattr(cfg, "bootstrap_huber", 0.05),
#     #     max_nn_dist=getattr(cfg, "bootstrap_nn_max_m", 1.2*vs)
#     # )
    
#     T_accum = torch.eye(4, device=dev, dtype=dt)

    
#     # ---- anchored SHRINK-ONLY bootstrap (no recentering) ----
#     # Choose an anchor to preserve current position (mean camera is a good default)
#     # anchor = C.mean(0)

#     # # Build a local AABB (don’t pull in the whole room)
#     # pc_min = torch.minimum(P.min(0).values, C.min(0).values)
#     # pc_max = torch.maximum(P.max(0).values, C.max(0).values)
#     # pad_m  = 4.0 * vs  # modest pad; keep it local
#     # aabb_min = pc_min - pad_m
#     # aabb_max = pc_max + pad_m

#     # # Estimate shrink factor from NN projection (robust, ray-free)
#     # s0, nmatch = shrink_from_nn_projection(
#     #     P, C, W, grid, aabb_min, aabb_max, voxel_size=vs,
#     #     max_block_vox=getattr(cfg, "max_block_vox", 192),
#     #     occ_thr=0.0, stride=2,
#     #     p_percentile=getattr(cfg, "shrink_percentile", 0.1),  # lower -> more aggressive
#     #     bias=getattr(cfg, "shrink_bias", 0.95),
#     #     smin=getattr(cfg, "s0_min", 0.25),                     # allow smaller if often too fat
#     #     smax=getattr(cfg, "s0_max", 0.98)
#     # )
#     # print(f"[anchored-shrink] matches={nmatch}  s0={float(s0):.3f}")

#     # # Scale about the anchor (keeps translation)
#     # T_accum = sim3_scale_about(anchor, float(s0), dev, dt)

        

#     # ---- build dense block once (cover P & C with metric pad) for refinement ----
#     pc_min = torch.minimum(P.min(0).values, C.min(0).values)
#     pc_max = torch.maximum(P.max(0).values, C.max(0).values)
#     diag  = torch.linalg.norm(pc_max - pc_min).item()
#     pad_m = max(6*vs, 0.5*diag)
#     aabb_min = pc_min - pad_m
#     aabb_max = pc_max + pad_m

#     occ_L, block_origin, vs_m, DHW = build_dense_block_from_sparse(
#         grid, aabb_min, aabb_max, getattr(cfg, "max_block_vox", 192)
#     )

#     # refinement schedules
#     sigmas = list(getattr(cfg, "sigmas", [3.0, 2.0, 1.0]))
#     lam_free_sched = list(getattr(cfg, "lambda_free_schedule", [0.25, 0.5, 1.0]))
#     if len(lam_free_sched) < len(sigmas):
#         lam_free_sched += [lam_free_sched[-1]]*(len(sigmas)-len(lam_free_sched))

#     # ---- coarse→fine refinement ----
#     allow_tiny_sim3_last = getattr(cfg, "allow_sim3_last_stage", False)
#     tiny_scale_clamp     = getattr(cfg, "tiny_scale_clamp", 0.01)

#     for stage, sigma in enumerate(sigmas):
#         k = max(1, int(2*sigma+1))
#         SDF = build_signed_distance_from_logit_occ(smooth3d(occ_L, k=k), vs_m)

#         # SE(3) for all stages, except optionally tiny Sim(3) on last
#         use_log_s = allow_tiny_sim3_last and (stage == len(sigmas)-1)
#         omega = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
#         t     = torch.zeros(3, device=dev, dtype=dt, requires_grad=True)
#         log_s = (torch.zeros(1, device=dev, dtype=dt, requires_grad=True) if use_log_s else None)

#         opt = torch.optim.Adam([omega, t] + ([log_s] if log_s is not None else []),
#                                lr=getattr(cfg,"step",1e-2) * (0.5 ** stage))
#         iters = max(1, getattr(cfg,"iters",12) // max(1, len(sigmas)))
#         W_sub = W

#         for _ in range(iters):
#             opt.zero_grad()
#             T_local = sim3_from_params(omega, t, log_s if log_s is not None else None)
#             T = T_local @ T_accum

#             X  = apply_sim3(T, P)
#             Ct = apply_sim3(T, C)

#             # optional overlap trim on coarse
#             g_end = world_to_block_coords(X, block_origin, vs_m, DHW)
#             phi_e = sample_trilinear(SDF, g_end)
#             if stage == 0 and getattr(cfg, "overlap_trim_vox", 2.0) is not None:
#                 mask = (phi_e.abs() < getattr(cfg, "overlap_trim_vox", 2.0)*vs_m)
#                 if mask.any():
#                     X, Ct, g_end, phi_e, W_sub = X[mask], Ct[mask], g_end[mask], phi_e[mask], W_sub[mask]

#             # hit term
#             w_hit = huber_weight(phi_e.abs(), getattr(cfg,"huber_delta",0.05))
#             L_hit = torch.mean((W_sub*w_hit) * (phi_e.abs() / (getattr(cfg,"sdf_eps",1e-3)+phi_e.abs())))

#             # free term
#             V    = X - Ct
#             dist = torch.linalg.norm(V, dim=-1)
#             Vhat = F.normalize(V, dim=-1, eps=1e-8)
#             semi = torch.clamp(dist - getattr(cfg,"free_margin_m",2.0*vs), min=getattr(cfg,"min_ray_m",2e-2))
#             K    = torch.clamp((semi / getattr(cfg,"free_step_m",vs)).ceil().to(torch.int64), min=1)
#             Kmax = int(K.max().item())
#             ts   = torch.linspace(0.05, 0.95, steps=Kmax, device=dev, dtype=dt).view(1,-1,1)
#             Y    = Ct.unsqueeze(1) + ts * semi.view(-1,1,1) * Vhat.unsqueeze(1)

#             g_free = world_to_block_coords(Y.reshape(-1,3), block_origin, vs_m, DHW)
#             phi_f  = sample_trilinear(SDF, g_free).view(-1, Kmax)
#             neg    = (phi_f < 0.0)
#             maskfs = ~(torch.cumsum(neg.int(), dim=1) > 0)
#             hinge  = torch.relu(getattr(cfg,"free_margin_m",2.0*vs) - phi_f) * maskfs.float()

#             w_free = huber_weight(hinge, getattr(cfg,"huber_delta",0.05))
#             Wk     = W_sub.view(-1,1).expand(-1, Kmax) * w_free
#             L_free = torch.mean((Wk * (hinge / max(getattr(cfg,"free_margin_m",2.0*vs),1e-6))).view(-1, Kmax).mean(dim=1))

#             # normals term
#             n_end = sample_trilinear_grad(SDF, g_end)
#             v_end = F.normalize(V, dim=-1, eps=1e-8)
#             L_norm = getattr(cfg,"lambda_norm",0.0) * torch.mean(W_sub * (1.0 - (v_end * n_end).sum(-1).pow(2)))

#             # priors
#             L_prior = getattr(cfg,"lambda_prior_t",1e-4)*(t*t).sum() + getattr(cfg,"lambda_prior_r",1e-4)*(omega*omega).sum()
#             if log_s is not None:
#                 L_prior = L_prior + getattr(cfg,"lambda_prior_s",1e-3)*(log_s*log_s).sum()  # stronger than before

#             loss = L_hit + lam_free_sched[stage]*L_free + L_norm + L_prior
#             loss.backward()
#             opt.step()

#             # clamps
#             with torch.no_grad():
#                 th = torch.linalg.norm(omega)
#                 th_max = (getattr(cfg,"rot_clamp_deg",15.0)/180.0)*torch.pi
#                 if th > th_max: omega *= (th_max/(th+1e-8))
#                 tt = torch.linalg.norm(t)
#                 if tt > getattr(cfg,"trans_clamp_m",0.30): t *= (getattr(cfg,"trans_clamp_m",0.30)/(tt+1e-8))
#                 if log_s is not None:
#                     # shrink-only tiny Sim(3) on last stage if enabled
#                     log_s.clamp_(min=-getattr(cfg,"scale_clamp", tiny_scale_clamp), max=0.0)

#         with torch.no_grad():
#             T_accum = (sim3_from_params(omega, t, log_s if log_s is not None else None)) @ T_accum

#     with torch.no_grad():
#         T_final = T_accum
#         P_out = apply_sim3(T_final, pts_world.reshape(-1,3).to(dev, dt)).view_as(pts_world)
#         C_out = apply_sim3(T_final, cams_world.reshape(-1,3).to(dev, dt)).view_as(cams_world)

#         # debug: report final scale
#         SR = T_final[:3,:3]; s_est = torch.det(SR).abs().pow(1/3).item()
#         print(f"[align] final scale det^(1/3) = {s_est:.5f}")

#     return T_final, P_out, C_out
