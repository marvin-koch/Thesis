#!/usr/bin/env python3
"""
nav_eval.py  –  Point-Goal Navigation with Dynamic Obstacle Avoidance
=====================================================================
Downstream robotics task for evaluating dynamic voxel-grid mapping.

The robot plans shortest paths on a 2-D occupancy grid extracted from different
map representations (Model / Baseline / Static / GT).  As dynamic objects move
between timesteps the map is updated and the robot **re-plans**.  Collisions are
checked against the ground-truth occupancy to measure safety.

Metrics reported (per-episode, per-method, and dataset-wide):
  • Success Rate  (SR)
  • SPL  (Success weighted by inverse Path Length)
  • Collision Rate  (% of steps that enter GT-occupied cells)
  • Re-plan Count  (how often the planned path became blocked)
  • Path Efficiency  (geodesic shortest / actual distance)

Usage:
    python nav_eval.py \
        --ckpt   /path/to/checkpoint.ckpt \
        --root   /path/to/dataset/ \
        --out    /path/to/nav_results/ \
        --episodes_per_seq  20 \
        --steps_per_episode 60
"""

import os, sys, json, copy, heapq, argparse, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ---- project imports (adjust if your repo layout differs) ----
from train import (
    TrainConfig, HabitatDataModule, VoxelUpdaterSystem,
    load_sparse_voxel_grid,
)
from voxel.voxel import TorchSparseVoxelGrid, VoxelParams
from voxel.utils import to_torch, rotate_points, build_maps_from_points_and_centers_torch,BevSpec, bev_from_voxels


# ═══════════════════════════════════════════════════════════════════
#  A* Path Planner on 2-D Occupancy Grid
# ═══════════════════════════════════════════════════════════════════

# 8-connected moves: (dy, dx, cost)
_MOVES_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
]


def astar(occ_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          max_expansions: int = 200_000) -> Optional[List[Tuple[int, int]]]:
    """
    A* on a 2-D boolean grid (True = occupied / blocked).
    Returns list of (row, col) waypoints from start→goal, or None if no path.
    """
    H, W = occ_grid.shape
    if occ_grid[start[0], start[1]] or occ_grid[goal[0], goal[1]]:
        return None

    def h(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    open_set = [(h(start, goal), 0.0, start)]
    came_from: Dict[Tuple, Tuple] = {}
    g_score = {start: 0.0}
    closed = set()
    expansions = 0

    while open_set and expansions < max_expansions:
        _, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dy, dx, cost in _MOVES_8:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < H and 0 <= nx < W and not occ_grid[ny, nx]:
                ng = g + cost
                nb = (ny, nx)
                if nb not in g_score or ng < g_score[nb]:
                    g_score[nb] = ng
                    came_from[nb] = current
                    heapq.heappush(open_set, (ng + h(nb, goal), ng, nb))

    return None  # no path found


def path_length(path: List[Tuple[int, int]]) -> float:
    if path is None or len(path) < 2:
        return 0.0
    d = 0.0
    for i in range(len(path) - 1):
        dy = path[i + 1][0] - path[i][0]
        dx = path[i + 1][1] - path[i][1]
        d += (dy ** 2 + dx ** 2) ** 0.5
    return d


# ═══════════════════════════════════════════════════════════════════
#  Voxel Grid  →  2-D Occupancy Grid
# ═══════════════════════════════════════════════════════════════════

def voxel_to_occ2d(
    vox,
    voxel_size: float,
    bev_origin: Tuple[float, float] = (-25.0, -25.0),
    bev_size: Tuple[float, float] = (50.0, 50.0),
    z_band: Tuple[float, float] = (-2.0, 5.0),
    occ_thresh: float = 0.0,
    is_latent: bool = False,
) -> np.ndarray:
    """
    Project a 3-D voxel grid onto a 2-D binary occupancy grid.
    Returns bool array (H, W) where True = occupied.
    """
    W_cells = int(round(bev_size[0] / voxel_size))
    H_cells = int(round(bev_size[1] / voxel_size))
    grid = np.zeros((H_cells, W_cells), dtype=bool)

    n = vox.keys.shape[0]
    if n == 0:
        return grid

    centers = vox.voxel_centers()  # (N, 3)
    if is_latent:
        logits = vox.decode_occupancy(with_xyz_cond=False)
        occ_mask = (logits > occ_thresh)
    else:
        vals = vox._display_vals().clamp(-10.0, 10.0)
        occ_mask = (vals > occ_thresh)

    # z-band filter
    z = centers[:, 2]
    z_mask = (z >= z_band[0]) & (z <= z_band[1])
    active = occ_mask & z_mask

    if not active.any():
        return grid

    cx = centers[active, 0].cpu().numpy()
    cy = centers[active, 1].cpu().numpy()

    # map world coords → grid indices
    col = ((cx - bev_origin[0]) / voxel_size).astype(int)
    row = ((cy - bev_origin[1]) / voxel_size).astype(int)

    valid = (row >= 0) & (row < H_cells) & (col >= 0) & (col < W_cells)
    grid[row[valid], col[valid]] = True
    return grid


# ═══════════════════════════════════════════════════════════════════
#  Episode Sampling & Running
# ═══════════════════════════════════════════════════════════════════

def sample_free_positions(
    occ_grid: np.ndarray, n: int, max_dist_cells: int = 20, min_dist_cells: int = 5, rng: np.random.Generator = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Sample n (start, goal) pairs on free cells with minimum geodesic separation."""
    if rng is None:
        rng = np.random.default_rng(42)

    free_ys, free_xs = np.where(~occ_grid)
    if len(free_ys) < 2:
        return []

    episodes = []
    attempts = 0
    while len(episodes) < n and attempts < n * 50:
        attempts += 1
        i, j = rng.choice(len(free_ys), size=2, replace=False)
        s = (int(free_ys[i]), int(free_xs[i]))
        g = (int(free_ys[j]), int(free_xs[j]))
        dist = ((s[0] - g[0]) ** 2 + (s[1] - g[1]) ** 2) ** 0.5
        if dist >= min_dist_cells and dist <= max_dist_cells:
            episodes.append((s, g))
    return episodes


@dataclass
class EpisodeResult:
    success: bool = False
    path_length_actual: float = 0.0
    path_length_optimal: float = 0.0
    collisions: int = 0
    steps: int = 0
    replans: int = 0

    @property
    def spl(self) -> float:
        if not self.success or self.path_length_actual == 0:
            return 0.0
        return self.path_length_optimal / max(self.path_length_optimal, self.path_length_actual)

    @property
    def collision_rate(self) -> float:
        return self.collisions / max(1, self.steps)


def run_episode(
    occ_grids_over_time: List[np.ndarray],   # occupancy per timestep (from a single method)
    gt_grids_over_time: List[np.ndarray],     # GT occupancy per timestep
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 200,
    goal_radius: int = 2,
) -> EpisodeResult:
    """
    Simulate one navigation episode.
    The robot re-plans whenever its planned path is invalidated by a map update.
    Collisions are counted against the GT map.
    """
    res = EpisodeResult()
    pos = start
    plan = None
    plan_idx = 0
    T = len(occ_grids_over_time)

    # Compute optimal path length on initial GT grid
    opt_path = astar(gt_grids_over_time[0], start, goal)
    if opt_path is None:
        return res  # unreachable on GT → skip
    res.path_length_optimal = path_length(opt_path)

    for step in range(max_steps):
        t = min(step, T - 1)  # clamp to last available timestep

        # Check if we reached the goal
        dy, dx = pos[0] - goal[0], pos[1] - goal[1]
        if (dy ** 2 + dx ** 2) ** 0.5 <= goal_radius:
            res.success = True
            break

        # Check if current plan is still valid
        need_replan = False
        if plan is None or plan_idx >= len(plan):
            need_replan = True
        elif plan_idx < len(plan):
            # Check if upcoming cells on the plan are now blocked
            look_ahead = min(plan_idx + 5, len(plan))
            for k in range(plan_idx, look_ahead):
                r, c = plan[k]
                if occ_grids_over_time[t][r, c]:
                    need_replan = True
                    break

        if need_replan:
            plan = astar(occ_grids_over_time[t], pos, goal)
            plan_idx = 1  # index 0 is current position
            if step > 0:
                res.replans += 1
            if plan is None:
                # No path available from current map → stay in place
                res.steps += 1
                continue

        # Move one step along the plan
        if plan is not None and plan_idx < len(plan):
            next_pos = plan[plan_idx]
            plan_idx += 1
        else:
            next_pos = pos  # stuck

        # Record collision if GT says this cell is occupied
        if gt_grids_over_time[t][next_pos[0], next_pos[1]]:
            res.collisions += 1

        dy = next_pos[0] - pos[0]
        dx = next_pos[1] - pos[1]
        res.path_length_actual += (dy ** 2 + dx ** 2) ** 0.5
        pos = next_pos
        res.steps += 1

    return res


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_episode(
    occ_grid: np.ndarray,
    gt_grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    plan_path: Optional[List[Tuple[int, int]]],
    save_path: str,
    title: str = "",
):
    """Save a top-down visualization of one episode."""
    H, W = occ_grid.shape
    img = np.ones((H, W, 3), dtype=np.uint8) * 240  # light grey background

    # Walls from the map the robot sees
    img[occ_grid] = [80, 80, 80]

    # GT walls the robot doesn't know about (dynamic diff)
    gt_only = gt_grid & ~occ_grid
    img[gt_only] = [255, 120, 120]  # light red = hidden obstacle

    # Planned path
    if plan_path:
        for r, c in plan_path:
            if 0 <= r < H and 0 <= c < W:
                img[r, c] = [100, 180, 255]

    # Start & goal
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            sr, sc = start[0] + dr, start[1] + dc
            gr, gc = goal[0] + dr, goal[1] + dc
            if 0 <= sr < H and 0 <= sc < W:
                img[sr, sc] = [0, 200, 0]
            if 0 <= gr < H and 0 <= gc < W:
                img[gr, gc] = [200, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, origin="lower")
    ax.set_title(title, fontsize=11)

    patches = [
        mpatches.Patch(color=[c / 255 for c in [80, 80, 80]], label="Map obstacle"),
        mpatches.Patch(color=[c / 255 for c in [255, 120, 120]], label="Hidden (GT only)"),
        mpatches.Patch(color=[c / 255 for c in [100, 180, 255]], label="Planned path"),
        mpatches.Patch(color=[c / 255 for c in [0, 200, 0]], label="Start"),
        mpatches.Patch(color=[c / 255 for c in [200, 0, 0]], label="Goal"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════════

def build_voxel_from_gt_direct(sim_data, voxel_size, device):
    """Identical to predict_step helper."""
    pts = sim_data["world_points"].to(device, dtype=torch.float32)
    extr = sim_data["extrinsic"].to(device, dtype=torch.float32)
    S, H, W, _ = pts.shape
    cam_centers = extr[:, :3, 3]
    pts_flat = pts.reshape(-1, 3)
    cams_flat = cam_centers[:, None, None, :].expand(S, H, W, 3).reshape(-1, 3)
    valid = torch.isfinite(pts_flat).all(dim=-1)
    vox = TorchSparseVoxelGrid(
        origin_xyz=[0, 0, 0],
        params=VoxelParams(voxel_size=voxel_size, promote_hits=1),
        device=device,
    )
    vox.integrate_points_with_cameras(
        pts_flat[valid], cams_flat[valid], carve_free=True, max_range=20.0,
        z_clip=None, samples_per_voxel=0.7, ray_stride=4, max_free_rays=10000,
    )
    return vox


def main():
    parser = argparse.ArgumentParser(description="Point-Goal Nav Evaluation on Dynamic Voxel Maps")
    parser.add_argument("--out", default="nav_results", help="Output directory")
    parser.add_argument("--episodes_per_seq", type=int, default=5)
    parser.add_argument("--steps_per_episode", type=int, default=10)
    parser.add_argument("--min_dist_cells", type=int, default=10,
                        help="Min start-goal distance in grid cells")
    parser.add_argument("--goal_radius", type=int, default=2)
    parser.add_argument("--voxel_size", type=float, default=0.2)
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--seq_file", default="seq_manifest.json")
    parser.add_argument("--gt_voxels_file", default="gt_voxels_per_timestep_new")
    parser.add_argument("--precomputed_cache_file", default="precomputed_cache")
    parser.add_argument("--pose_file", default="gt_poses_new")
    parser.add_argument("--real_gt_voxels_file", default="hm3d_voxels")
    parser.add_argument("--step", type=int, default=1,
                        help="Temporal stride for loading timesteps (matches STEP in training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_viz", type=int, default=5,
                        help="Number of episodes to visualize per sequence")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build config matching the checkpoint ──
    cfg = TrainConfig(
        dataset_root="/cluster/scratch/kochmar/hm3d_gt/",
        real_gt_voxels_file=args.real_gt_voxels_file,
        gt_voxels_file=args.gt_voxels_file,
        precomputed_cache_file=args.precomputed_cache_file,
        pose_file=args.pose_file,
        seq_file=args.seq_file,
        voxel_size=args.voxel_size,
        feature_dim=args.feature_dim,
        radius_m=1,
        topk=8,
        temp=0.5,
        occ_decoder_hidden=64,
        lr=3e-4,
        max_epochs=200,
        batch_size=1,
        num_workers=0,
        precision="bf16",
        skip=True,
        weight_decay=0.05,
        lambda_occ=1.0,
        lambda_temp=0.05,
        lambda_ent=1e-3,
        lambda_tv=1e-4,
    )

    # ── Load data module ──
    dm = HabitatDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=1,
        num_workers=cfg.num_workers,
        size=512,
        verbose=False,
        train_val_split=1.0,
        skip=True,
        seq_list=os.path.join(cfg.dataset_root, cfg.seq_file),
        step=args.step,
    )
    dm.setup("predict")

    # ── Load model ──
    ckpt_path = "/cluster/scratch/kochmar/checkpoints/full_ablation/voxup-epoch=03-val_loss_total=7.9336.ckpt"
    model = VoxelUpdaterSystem.load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
    model = model.to(device)
    model.eval()
    # ── BEV parameters (same as predict_step) ──
    bev_size=(25.0, 25.0)
    bev_origin=(-12.0, -12.0)
    z_band=(-2.0, 1.5)
    voxel_size = cfg.voxel_size

    # ── Paths ──
    gt_root         = os.path.join(cfg.dataset_root, cfg.gt_voxels_file)
    precomp_root    = os.path.join(cfg.dataset_root, cfg.precomputed_cache_file)
    pose_root       = os.path.join(cfg.dataset_root, cfg.pose_file)
    real_gt_root    = os.path.join(cfg.dataset_root, cfg.real_gt_voxels_file) if cfg.real_gt_voxels_file else None

    STEP = args.step
    threshold = 1.0

    # ── Method names we'll compare ──
    METHODS = ["model", "baseline", "static", "gt"]

    # ── Aggregate results ──
    all_results = {m: [] for m in METHODS}  # method -> list of EpisodeResult

    val_loader = dm.val_dataloader()
    if val_loader is None:
        print("No validation data. Exiting.")
        return

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        seq_id = batch["seq_id"]
        T = batch["timesteps"]
        print(f"\n{'='*70}")
        print(f"Sequence {batch_idx}: {seq_id}  (T={T})")
        print(f"{'='*70}")

        # ── Load alignment ──
        pose_path = os.path.join(pose_root, f"{seq_id}_t0000_align.npz")
        if not os.path.exists(pose_path):
            print(f"  Pose missing: {pose_path}, skipping.")
            continue
        d = np.load(pose_path, allow_pickle=True)
        Rmw = to_torch(d["Rmw"], device=device)
        tmw = to_torch(d["tmw"], device=device)
        scale_factor = float(d["scale"].item() if hasattr(d["scale"], "item") else d["scale"])
        tmw_scaled = tmw * scale_factor

        # ── Load GT voxel grids for all timesteps ──
        gt_seq = []
        for t in range(T):
            p = t * STEP
            gt_path = os.path.join(gt_root, f"{seq_id}_t{p:04d}_gt.npz")
            if os.path.exists(gt_path):
                gt_seq.append(load_sparse_voxel_grid(gt_path, device))
            else:
                gt_seq.append(None)

        # Skip if not enough GT
        valid_gt_count = sum(1 for g in gt_seq if g is not None)
        if valid_gt_count < 3:
            print(f"  Only {valid_gt_count} GT timesteps, skipping.")
            continue

        # ── Reset model voxel grids ──
        model.vox.reset_state()
        model.vox = model.vox.to(device)
        model.vox_baseline = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
            device=device,
        )
        model.vox_gt = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
            device=device,
        )

        # ── Build occupancy grids for each method at each timestep ──
        method_grids = {m: [] for m in METHODS}
        gt_grids = []
        mst = False
        static_vox = None

        # Helper for extracting 2D grid
        def extract_grid(vox, is_latent=False):
            return voxel_to_occ2d(
                vox, voxel_size, bev_origin, bev_size, z_band,
                occ_thresh=0.0, is_latent=is_latent,
            )

        R_w2m = to_torch(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float32), device=device)
        t_w2m = to_torch(np.zeros(3, dtype=np.float32), device=device)

        for t in range(T):
            model.vox_gt = gt_seq[t]
            if gt_seq[t] is None:
                # Duplicate previous grids
                for m in METHODS:
                    if method_grids[m]:
                        method_grids[m].append(method_grids[m][-1])
                    else:
                        H_cells = int(round(bev_size[1] / voxel_size))
                        W_cells = int(round(bev_size[0] / voxel_size))
                        method_grids[m].append(np.zeros((H_cells, W_cells), dtype=bool))
                gt_grids.append(gt_grids[-1] if gt_grids else np.zeros_like(method_grids["gt"][-1]))
                continue

            p = t * STEP
            cache_path = os.path.join(precomp_root, seq_id, f"t{p:04d}.pt")
            if not os.path.exists(cache_path):
                for m in METHODS:
                    method_grids[m].append(method_grids[m][-1] if method_grids[m] else
                        np.zeros((int(round(bev_size[1]/voxel_size)), int(round(bev_size[0]/voxel_size))), dtype=bool))
                gt_grids.append(gt_grids[-1] if gt_grids else np.zeros_like(method_grids["gt"][-1]))
                continue

            predictions = torch.load(cache_path, map_location=device)

            # ── Ensure float32 everywhere (cache may be bf16) ──
            for k in list(predictions.keys()):
                v = predictions[k]
                if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float32:
                    predictions[k] = v.float()
                elif isinstance(v, list):
                    predictions[k] = [
                        x.float() if torch.is_tensor(x) and x.is_floating_point() and x.dtype != torch.float32 else x
                        for x in v
                    ]

            R_w2m_f = R_w2m.float()
            t_w2m_f = t_w2m.float()
            Rmw_f   = Rmw.float()
            tmw_f   = tmw.float()

            # ── Standard alignment (same as predict_step) ──
            WPTS_m = rotate_points(predictions["world_points"], R_w2m_f, t_w2m_f)
            predictions["world_points"] = rotate_points(WPTS_m, Rmw_f, tmw_f) * scale_factor

            if isinstance(predictions["extrinsic"], torch.Tensor):
                predictions["extrinsic"][:, :3, 3] *= scale_factor
            elif isinstance(predictions["extrinsic"], list):
                for i in range(len(predictions["extrinsic"])):
                    predictions["extrinsic"][i][:3, 3] *= scale_factor

            stride = 1 if t == 0 else cfg.stride
            if "world_points_conf" in predictions:
                predictions["world_points_conf"] = predictions["world_points_conf"][..., ::stride, ::stride]
                conf_tensor = predictions["world_points_conf"]
                target_hw = conf_tensor.shape[-2:] if not isinstance(conf_tensor, list) else (128, 128)
            else:
                target_hw = (512 // stride, 512 // stride)
            if "world_points" in predictions:
                predictions["world_points"] = predictions["world_points"][..., ::stride, ::stride, :]
            if "images" in predictions:
                img = predictions["images"]
                if img.shape[-3] == 3 and img.shape[-1] != 3:
                    img = img.permute(0, 2, 3, 1)
                predictions["images"] = img[..., ::stride, ::stride, :]
                del img

            # Process features for the model
            if "view_feats" in predictions:
                raw_feats = predictions["view_feats"]
                proj_feats = []
                for f_raw in raw_feats:
                    proj_feats.append(model.apply_projector_to_map(f_raw, target_hw=target_hw))
                predictions["view_feats"] = proj_feats

            tmw_scaled_f = tmw_f * scale_factor

            # ── Model inference ──
            with torch.no_grad():
                if not mst and t != 0:
                    bev, mst, _, _ = model.inference(
                        1, batch["imgs_t"][t], mst, Rmw_f, tmw_scaled_f,
                        predictions, scale_factor, threshold=threshold, flip=False
                    )
                else:
                    bev, mst, _, _ = model.inference(
                        t, batch["imgs_t"][t], mst, Rmw_f, tmw_scaled_f,
                        predictions, scale_factor, threshold=threshold, flip=False
                    )

            # ── Baseline inference ──
            with torch.no_grad():
                bev_base, meta_base, _ = model.run_baseline_inference(
                    predictions, Rmw_f, tmw_scaled_f, scale_factor, threshold=threshold
                )

            # ── Static baseline (freeze after t=0) ──
            if t == 0:
                static_vox = copy.deepcopy(model.vox)

            # ── Real GT voxel (if available) ──
            if real_gt_root:
                real_gt_path = os.path.join(real_gt_root, f"{seq_id}_t{p:04d}.pt").replace(".glb", "")
                if os.path.exists(real_gt_path):
                    real_gt_data = torch.load(real_gt_path, map_location=device)
                    vox_real = build_voxel_from_gt_direct(real_gt_data, voxel_size, device)
                    gt_grids.append(extract_grid(vox_real, is_latent=False))
                    del vox_real, real_gt_data
                else:
                    gt_grids.append(extract_grid(gt_seq[t], is_latent=False))
            else:
                gt_grids.append(extract_grid(gt_seq[t], is_latent=False))

            # ── Extract 2D grids ──
            method_grids["model"].append(extract_grid(model.vox, is_latent=True))
            method_grids["baseline"].append(extract_grid(model.vox_baseline, is_latent=False))
            method_grids["static"].append(extract_grid(static_vox, is_latent=True))
            method_grids["gt"].append(gt_grids[-1])

            # Detach latent to prevent memory growth
            model.vox.z_latent = model.vox.z_latent.detach()

            del predictions
            torch.cuda.empty_cache()

        if not gt_grids:
            print("  No grids built, skipping.")
            continue

        # ── Sample navigation episodes ──
        # Use the first GT grid for sampling start/goal (ensures they're initially reachable)
        episodes = sample_free_positions(
            gt_grids[0], args.episodes_per_seq,
            min_dist_cells=args.min_dist_cells, rng=rng,
        )
        print(f"  Sampled {len(episodes)} episodes")

        if not episodes:
            print("  Could not sample episodes (too few free cells), skipping.")
            continue

        # ── Run episodes for each method ──
        seq_out_dir = os.path.join(args.out, seq_id)
        os.makedirs(seq_out_dir, exist_ok=True)

        for method in METHODS:
            grids = method_grids[method]
            if not grids:
                continue

            for ep_idx, (start, goal) in enumerate(episodes):
                result = run_episode(
                    grids, gt_grids, start, goal,
                    max_steps=args.steps_per_episode,
                    goal_radius=args.goal_radius,
                )
                all_results[method].append(result)

                # Visualize a few
                if ep_idx < args.num_viz and method in ["model", "gt"]:
                    plan = astar(grids[0], start, goal)
                    visualize_episode(
                        grids[0], gt_grids[0], start, goal, plan,
                        os.path.join(seq_out_dir, f"ep{ep_idx}_{method}.png"),
                        title=f"{method.upper()} – ep {ep_idx}",
                    )

        # Per-sequence summary
        for method in METHODS:
            results = all_results[method][-len(episodes):]
            if not results:
                continue
            sr = np.mean([r.success for r in results])
            spl = np.mean([r.spl for r in results])
            cr = np.mean([r.collision_rate for r in results])
            rp = np.mean([r.replans for r in results])
            print(f"  {method:10s}  SR={sr:.3f}  SPL={spl:.3f}  ColRate={cr:.4f}  Replans={rp:.1f}")

    # ═══════════════════════════════════════════════════════════════
    #  Dataset-Wide Summary
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("DATASET-WIDE NAVIGATION RESULTS")
    print("=" * 80)
    header = f"{'Method':>12s} | {'SR':>7s} | {'SPL':>7s} | {'ColRate':>8s} | {'Replans':>8s} | {'PathEff':>8s} | {'Episodes':>8s}"
    print(header)
    print("-" * len(header))

    summary = {}
    for method in METHODS:
        results = all_results[method]
        if not results:
            continue
        sr   = np.mean([r.success for r in results])
        spl  = np.mean([r.spl for r in results])
        cr   = np.mean([r.collision_rate for r in results])
        rp   = np.mean([r.replans for r in results])
        pe   = np.mean([
            r.path_length_optimal / max(r.path_length_actual, 1e-8)
            for r in results if r.success
        ]) if any(r.success for r in results) else 0.0

        print(f"{method:>12s} | {sr:7.3f} | {spl:7.3f} | {cr:8.4f} | {rp:8.1f} | {pe:8.3f} | {len(results):>8d}")
        summary[method] = {"SR": sr, "SPL": spl, "CollisionRate": cr, "Replans": rp, "PathEfficiency": pe, "N": len(results)}

    # ── Save JSON ──
    json_path = os.path.join(args.out, "nav_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {json_path}")

    # ── Bar chart ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_to_plot = [("SR", "Success Rate ↑"), ("SPL", "SPL ↑"), ("CollisionRate", "Collision Rate ↓"), ("Replans", "Re-plans")]
    colors = {"model": "#2196F3", "baseline": "#FF9800", "static": "#9E9E9E", "gt": "#4CAF50"}

    for ax, (key, label) in zip(axes, metrics_to_plot):
        vals = [summary.get(m, {}).get(key, 0) for m in METHODS]
        bars = ax.bar(METHODS, vals, color=[colors.get(m, "#ccc") for m in METHODS])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Point-Goal Navigation – Dynamic Obstacle Avoidance", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "nav_summary.png"), dpi=150)
    plt.close(fig)
    print(f"Summary plot saved to {os.path.join(args.out, 'nav_summary.png')}")


if __name__ == "__main__":
    main()
