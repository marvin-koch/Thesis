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
from scipy import ndimage

# ---- project imports (adjust if your repo layout differs) ----
from train import (
    TrainConfig, HabitatDataModule, VoxelUpdaterSystem,
    load_sparse_voxel_grid,
)
from voxel.voxel import TorchSparseVoxelGrid, VoxelParams
from voxel.utils import to_torch, rotate_points, build_maps_from_points_and_centers_torch,BevSpec, bev_from_voxels
from pytorch3d.ops import knn_points



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


def kabsch_umeyama_sim3(src, dst):

    # 1. Centroid subtraction
    mu_src = src.mean(dim=0)
    mu_dst = dst.mean(dim=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 2. Compute Scale (s)
    # Scale is the ratio of root-mean-square deviations from centroids
    var_src = (src_c**2).sum(dim=-1).mean()
    var_dst = (dst_c**2).sum(dim=-1).mean()
    scale = torch.sqrt(var_dst / var_src)

    # 3. Compute Rotation (R) using SVD
    # Correlation matrix
    H = src_c.T @ dst_c
    U, S, Vh = torch.linalg.svd(H)

    # Correction for reflection
    d = torch.sign(torch.linalg.det(U @ Vh))
    diag = torch.ones(3, device=src.device)
    diag[2] = d

    R = Vh.T @ torch.diag(diag) @ U.T

    # 4. Compute Translation (t)
    # t = mu_dst - s * (R @ mu_src)
    translation = mu_dst - scale * (R @ mu_src)

    return R, translation, scale

def icp_sim3(src, dst, init_R, init_t, init_s, max_iters=20, tolerance=1e-5):

    # Apply initial Kabsch alignment
    src_curr = init_s * (src @ init_R.T) + init_t

    # Accumulators for the total transformation
    R_acc = init_R.clone()
    t_acc = init_t.clone()
    s_acc = init_s.clone()

    prev_loss = float('inf')

    for i in range(max_iters):
        # 1. Find true nearest neighbors in 3D space
        # knn_points expects batched tensors: (1, N, 3)
        res = knn_points(src_curr.unsqueeze(0), dst.unsqueeze(0), K=1)

        # Squeeze out the batch dimension
        idx = res.idx.squeeze(0).squeeze(-1)      # (N,)
        dists = res.dists.squeeze(0).squeeze(-1)  # (N,)

        # 2. Filter Outliers (Crucial for Dust3R noise!)
        # Only keep the 85% closest points so floating artifacts don't warp the room
        q85 = torch.quantile(dists, 0.85)
        valid = dists < q85

        src_matched = src_curr[valid]
        dst_matched = dst[idx[valid]]

        # 3. Estimate new transformation on the matched pairs
        R, t, s = kabsch_umeyama_sim3(src_matched, dst_matched)

        # 4. Apply transformation for the next iteration
        src_curr = s * (src_curr @ R.T) + t

        # 5. Update global accumulators
        # (Sim3 Composition: S2 * R2 * (S1 * R1 * X + T1) + T2)
        t_acc = s * (R @ t_acc) + t
        R_acc = R @ R_acc
        s_acc = s * s_acc

        # Check for convergence
        mean_dist = dists[valid].mean().item()
        if abs(prev_loss - mean_dist) < tolerance:
            break
        prev_loss = mean_dist

    return R_acc, t_acc, s_acc
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

def _get_interior_free_mask(occ_grid: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask where True = cell is free AND all 8 neighbours are free.
    This avoids spawning right next to obstacles.
    """
    free = ~occ_grid  # True where not occupied
    H, W = free.shape
    # Erode the free mask by 1 cell in every direction (8-connected)
    interior = free.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            # Shift and AND: a cell is only interior if all shifted neighbours are also free
            shifted = np.zeros_like(free)
            src_r = slice(max(0, -dy), H + min(0, -dy))
            src_c = slice(max(0, -dx), W + min(0, -dx))
            dst_r = slice(max(0, dy), H + min(0, dy))
            dst_c = slice(max(0, dx), W + min(0, dx))
            shifted[dst_r, dst_c] = free[src_r, src_c]
            interior &= shifted
    return interior


def sample_free_positions(
    occ_grid: np.ndarray, n: int, max_dist_cells: int = 30, min_dist_cells: int = 10, rng: np.random.Generator = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Sample n (start, goal) pairs on free cells that have all 8 neighbours free,
    with minimum/maximum Euclidean separation.
    Falls back to any free cell if no interior cells are available.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Prefer interior free cells (free + all neighbours free)
    interior = _get_interior_free_mask(occ_grid)
    free_ys, free_xs = np.where(interior)

    # Fallback: if too few interior cells, use all free cells
    if len(free_ys) < 2:
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
    method="gt",
    episode=0,
    seq="scene"
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
    trajectory = [start]       # full trail of positions visited
    collision_history = []     # list of (step, attempted_pos) for every collision

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

        # Check if current plan is still valid (using the method's own map)
        need_replan = False
        if plan is None or plan_idx >= len(plan):
            need_replan = True
        elif plan_idx < len(plan):
            # Check if ANY upcoming cell on the remaining plan is now blocked
            for k in range(plan_idx, len(plan)):
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
        collision_this_step = False
        attempted_pos = next_pos
        if gt_grids_over_time[t][next_pos[0], next_pos[1]]:
            res.collisions += 1
            collision_this_step = True
            collision_history.append((step, attempted_pos))
            # Robot bumped into an obstacle it didn't see — force a replan
            # next step so it routes around it. Stay at current position.
            plan = None
            next_pos = pos

        # Visualize AFTER collision logic so the viz reflects the replan
        visualize_episode(
            occ_grids_over_time[t],
            gt_grids_over_time[t],
            start, goal, plan,
            os.path.join(seq, f"{episode}_{method}_{step:03d}.png"),
            title=f"{method.upper()} – step {step}",
            collision_pos=attempted_pos if collision_this_step else None,
            robot_pos=pos,
            trajectory=trajectory,
            collision_history=collision_history,
        )

        dy = next_pos[0] - pos[0]
        dx = next_pos[1] - pos[1]
        res.path_length_actual += (dy ** 2 + dx ** 2) ** 0.5
        pos = next_pos
        trajectory.append(pos)
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
    collision_pos: Optional[Tuple[int, int]] = None,
    robot_pos: Optional[Tuple[int, int]] = None,
    trajectory: Optional[List[Tuple[int, int]]] = None,
    collision_history: Optional[List[Tuple[int, Tuple[int, int]]]] = None,
):
    """Save a paper-quality top-down visualization of one episode frame."""
    H, W = occ_grid.shape

    # --- build RGB canvas ---
    img = np.ones((H, W, 3), dtype=np.float32) * 0.94  # off-white background

    # Walls from the method's map
    img[occ_grid] = [0.30, 0.30, 0.30]

    # GT-only walls (hidden dynamic obstacles)
    gt_only = gt_grid & ~occ_grid
    img[gt_only] = [0.90, 0.42, 0.42]          # muted red

    # Trajectory trail (drawn before plan so plan overlays it)
    if trajectory and len(trajectory) > 1:
        for i, (r, c) in enumerate(trajectory):
            if 0 <= r < H and 0 <= c < W:
                # Fade from light to dark along the trail
                alpha = 0.3 + 0.7 * (i / len(trajectory))
                img[r, c] = [0.40 * alpha, 0.75 * alpha, 0.40 * alpha]

    # Planned path (thin blue)
    if plan_path:
        for r, c in plan_path:
            if 0 <= r < H and 0 <= c < W:
                img[r, c] = [0.30, 0.60, 0.95]

    # --- plot with matplotlib for markers & legend ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, origin="lower", interpolation="nearest")

    # Start marker
    ax.plot(start[1], start[0], marker="s", color="#2ca02c", markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5, label="Start")

    # Goal marker
    ax.plot(goal[1], goal[0], marker="*", color="#d62728", markersize=12,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5, label="Goal")

    # Robot current position
    if robot_pos is not None:
        ax.plot(robot_pos[1], robot_pos[0], marker="o", color="#1f77b4",
                markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                zorder=6, label="Robot")

    # All past collisions (small red ×)
    if collision_history:
        for _, (cr, cc) in collision_history:
            ax.plot(cc, cr, marker="x", color="#ff7f0e", markersize=7,
                    markeredgewidth=2.0, zorder=7)
        # One legend entry for collisions
        ax.plot([], [], marker="x", color="#ff7f0e", markersize=7,
                markeredgewidth=2.0, linestyle="None", label="Collision")

    # Current-step collision highlight (larger)
    if collision_pos is not None:
        ax.plot(collision_pos[1], collision_pos[0], marker="x", color="#ff7f0e",
                markersize=12, markeredgewidth=2.5, zorder=8)

    # Legend patches for grid colours
    grid_patches = [
        mpatches.Patch(color=[0.30, 0.30, 0.30], label="Map obstacle"),
        mpatches.Patch(color=[0.90, 0.42, 0.42], label="Hidden obstacle (GT)"),
        mpatches.Patch(color=[0.30, 0.60, 0.95], label="Planned path"),
        mpatches.Patch(color=[0.40, 0.75, 0.40], label="Trajectory"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=grid_patches + handles, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="0.6")

    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Frontier-Based Exploration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExplorationResult:
    coverage: float = 0.0          # fraction of GT-free cells visited
    collisions: int = 0
    steps: int = 0
    distance_traveled: float = 0.0
    frontiers_visited: int = 0

    @property
    def collision_rate(self) -> float:
        return self.collisions / max(1, self.steps)

    @property
    def efficiency(self) -> float:
        """Coverage per unit distance — higher is better."""
        return self.coverage / max(self.distance_traveled, 1e-8)


def _find_frontiers(occ_grid: np.ndarray, explored: np.ndarray) -> np.ndarray:
    """
    Frontier cells = explored FREE cells adjacent to at least one UNEXPLORED cell.
    Returns boolean mask (H, W).
    """
    free_explored = explored & (~occ_grid)
    unexplored = ~explored
    unexplored_border = ndimage.binary_dilation(unexplored, iterations=1)
    return free_explored & unexplored_border


def _select_frontier_goal(
    frontier_mask: np.ndarray, pos: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Pick the nearest frontier cell to navigate to."""
    ys, xs = np.where(frontier_mask)
    if len(ys) == 0:
        return None
    dists = (ys - pos[0]) ** 2 + (xs - pos[1]) ** 2
    idx = np.argmin(dists)
    return (int(ys[idx]), int(xs[idx]))


def visualize_exploration_step(
    occ_grid: np.ndarray,
    gt_grid: np.ndarray,
    explored: np.ndarray,
    frontier: np.ndarray,
    pos: Tuple[int, int],
    plan_path: Optional[List[Tuple[int, int]]],
    goal: Optional[Tuple[int, int]],
    save_path: str,
    title: str = "",
):
    """Save a top-down visualization of one exploration step."""
    H, W = occ_grid.shape
    img = np.ones((H, W, 3), dtype=np.uint8) * 40  # dark = unexplored

    # Explored free space
    explored_free = explored & (~occ_grid)
    img[explored_free] = [180, 220, 255]  # light blue

    # Walls from the map the robot sees
    img[occ_grid & explored] = [80, 80, 80]

    # GT-only obstacles (hidden from robot)
    gt_only = gt_grid & ~occ_grid & explored
    img[gt_only] = [255, 120, 120]  # light red

    # Frontier cells
    img[frontier] = [255, 220, 50]  # yellow

    # Planned path
    if plan_path:
        for r, c in plan_path:
            if 0 <= r < H and 0 <= c < W:
                img[r, c] = [100, 180, 255]

    # Current frontier goal
    if goal is not None:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                gr, gc = goal[0] + dr, goal[1] + dc
                if 0 <= gr < H and 0 <= gc < W:
                    img[gr, gc] = [200, 0, 0]

    # Robot position
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            pr, pc = pos[0] + dr, pos[1] + dc
            if 0 <= pr < H and 0 <= pc < W:
                img[pr, pc] = [0, 255, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, origin="lower")
    ax.set_title(title, fontsize=11)

    patches = [
        mpatches.Patch(color=[c / 255 for c in [180, 220, 255]], label="Explored free"),
        mpatches.Patch(color=[c / 255 for c in [40, 40, 40]], label="Unexplored"),
        mpatches.Patch(color=[c / 255 for c in [80, 80, 80]], label="Map wall"),
        mpatches.Patch(color=[c / 255 for c in [255, 120, 120]], label="Hidden (GT)"),
        mpatches.Patch(color=[c / 255 for c in [255, 220, 50]], label="Frontier"),
        mpatches.Patch(color=[c / 255 for c in [100, 180, 255]], label="Planned path"),
        mpatches.Patch(color=[c / 255 for c in [0, 255, 0]], label="Robot"),
        mpatches.Patch(color=[c / 255 for c in [200, 0, 0]], label="Frontier goal"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def run_exploration(
    occ_grids_over_time: List[np.ndarray],
    gt_grids_over_time: List[np.ndarray],
    start: Tuple[int, int],
    sensor_radius: int = 10,
    max_steps: int = 300,
    viz_dir: Optional[str] = None,
) -> ExplorationResult:
    """
    Simulate frontier-based exploration.

    The robot iteratively:
      1. Marks cells within sensor_radius as explored
      2. Finds frontiers (explored-free adjacent to unexplored)
      3. Plans A* to the nearest frontier on the method's occ grid
      4. Walks one step along that plan
      5. Collisions checked against GT

    If viz_dir is set, saves a per-step image to that directory.
    """
    T = len(occ_grids_over_time)
    H, W = occ_grids_over_time[0].shape
    res = ExplorationResult()

    gt_free_t0 = ~gt_grids_over_time[0]
    total_free = gt_free_t0.sum()
    if total_free == 0:
        return res

    explored = np.zeros((H, W), dtype=bool)
    pos = start
    plan = None
    plan_idx = 0
    current_goal = None
    frontier = np.zeros((H, W), dtype=bool)

    # Pre-compute disc mask for sensor
    yy, xx = np.ogrid[-sensor_radius:sensor_radius + 1, -sensor_radius:sensor_radius + 1]
    disc = (yy ** 2 + xx ** 2) <= sensor_radius ** 2

    if viz_dir is not None:
        os.makedirs(viz_dir, exist_ok=True)

    for step in range(max_steps):
        t = min(step, T - 1)
        occ = occ_grids_over_time[t]
        gt = gt_grids_over_time[t]

        # 1. Sense — mark cells within radius as explored
        r0 = max(0, pos[0] - sensor_radius)
        r1 = min(H, pos[0] + sensor_radius + 1)
        c0 = max(0, pos[1] - sensor_radius)
        c1 = min(W, pos[1] + sensor_radius + 1)

        dr0 = r0 - (pos[0] - sensor_radius)
        dr1 = disc.shape[0] - ((pos[0] + sensor_radius + 1) - r1)
        dc0 = c0 - (pos[1] - sensor_radius)
        dc1 = disc.shape[1] - ((pos[1] + sensor_radius + 1) - c1)

        explored[r0:r1, c0:c1] |= disc[dr0:dr1, dc0:dc1]

        # 2. Check if we need to replan
        need_replan = False
        if plan is None or plan_idx >= len(plan):
            need_replan = True
        elif plan_idx < len(plan):
            for k in range(plan_idx, min(plan_idx + 5, len(plan))):
                r, c = plan[k]
                if occ[r, c]:
                    need_replan = True
                    break

        if need_replan:
            frontier = _find_frontiers(occ, explored)
            current_goal = _select_frontier_goal(frontier, pos)

            if current_goal is None:
                # Save final frame before breaking
                if viz_dir is not None:
                    visualize_exploration_step(
                        occ, gt, explored, frontier, pos, plan, current_goal,
                        os.path.join(viz_dir, f"{step:04d}.png"),
                        title=f"Step {step} — no frontiers left",
                    )
                break  # no more reachable frontiers

            plan = astar(occ, pos, current_goal)
            plan_idx = 1
            if plan is None:
                # Try another frontier
                frontier[current_goal[0], current_goal[1]] = False
                current_goal = _select_frontier_goal(frontier, pos)
                if current_goal is not None:
                    plan = astar(occ, pos, current_goal)
                    plan_idx = 1
                if plan is None:
                    res.steps += 1
                    continue

            res.frontiers_visited += 1

        # 3. Per-step visualization
        if viz_dir is not None:
            visualize_exploration_step(
                occ, gt, explored, frontier, pos, plan, current_goal,
                os.path.join(viz_dir, f"{step:04d}.png"),
                title=f"Step {step}  cov={explored[gt_free_t0].sum()/total_free:.1%}",
            )

        # 4. Move one step
        if plan is not None and plan_idx < len(plan):
            next_pos = plan[plan_idx]
            plan_idx += 1
        else:
            next_pos = pos

        # 5. Collision check against GT
        if gt[next_pos[0], next_pos[1]]:
            res.collisions += 1

        dy = next_pos[0] - pos[0]
        dx = next_pos[1] - pos[1]
        res.distance_traveled += (dy ** 2 + dx ** 2) ** 0.5
        pos = next_pos
        res.steps += 1

    # Final coverage
    explored_free = explored & gt_free_t0
    res.coverage = float(explored_free.sum()) / float(total_free)
    return res


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
    parser.add_argument("--episodes_per_seq", type=int, default=20)
    parser.add_argument("--steps_per_episode", type=int, default=100)
    parser.add_argument("--min_dist_cells", type=int, default=15,
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
    parser.add_argument("--exploration_starts", type=int, default=10,
                        help="Number of exploration episodes per sequence")
    parser.add_argument("--exploration_steps", type=int, default=300,
                        help="Max steps per exploration episode")
    parser.add_argument("--sensor_radius", type=int, default=10,
                        help="Exploration sensor radius in grid cells")
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
    bev_size=(12.0, 12.0)
    bev_origin=(-7.0, -7.0)
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
    all_exploration = {m: [] for m in METHODS}  # method -> list of ExplorationResult

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
                    real_gt = torch.load(real_gt_path, map_location=device)
                    
                    with torch.cuda.amp.autocast(enabled=False):
                        gt_pts = real_gt["world_points"].to(device)   # (S, H, W, 3)
                        gt_ex = real_gt["extrinsic"].to(device)
                        gt_cams = gt_ex[:, :3, 3]

                        pred_pts = predictions["world_points"]  # already strided (S, H', W', 3)

                        if t == 0:
                            # Match stride to predictions
                            _, pH, pW, _ = pred_pts.shape
                            _, gH, gW, _ = gt_pts.shape

                            # Downsample GT to match prediction resolution
                            gt_pts_strided = gt_pts[:, ::max(1, gH//pH), ::max(1, gW//pW), :]
                            # Ensure exact shape match
                            gt_pts_strided = gt_pts_strided[:, :pH, :pW, :]

                            # Corresponding points for Kabsch
                            gt_flat = gt_pts_strided.reshape(-1, 3)
                            pred_flat = pred_pts.reshape(-1, 3)

                            both_valid = torch.isfinite(gt_flat).all(-1) & torch.isfinite(pred_flat).all(-1)

                            gt_corr = gt_flat[both_valid]
                            pred_corr = pred_flat[both_valid]

                            n = min(50000, gt_corr.shape[0])
                            idx = torch.randperm(gt_corr.shape[0], device=device)[:n]

                            R_k, t_k, s_k = kabsch_umeyama_sim3(gt_corr[idx], pred_corr[idx])
                            R_k, t_k, s_k = icp_sim3(
                                gt_corr[idx],
                                pred_corr[idx],
                                init_R=R_k,
                                init_t=t_k,
                                init_s=s_k,
                                max_iters=50
                            )

                        # Apply to FULL resolution GT (not strided)
                        aligned_pts = s_k * (gt_pts @ R_k.T) + t_k
                        aligned_cams = s_k * (gt_cams @ R_k.T) + t_k

                    real_gt["world_points"] = aligned_pts
                    new_ex = gt_ex.clone()
                    new_ex[:, :3, 3] = aligned_cams
                    real_gt["extrinsic"] = new_ex

                    if real_gt["images"].dim() == 4 and real_gt["images"].shape[1] == 3:
                        real_gt["images"] = real_gt["images"].permute(0, 2, 3, 1)

                    vox_real = build_voxel_from_gt_direct(real_gt, voxel_size, device)
                    gt_grids.append(extract_grid(vox_real, is_latent=False))
                    del vox_real, real_gt
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

        # ── Keep only timesteps where GT changed (dynamics present) ──
        # Always keep t=0; then keep any t where GT differs from previous
        dynamic_idx = [0]
        for i in range(1, len(gt_grids)):
            if not np.array_equal(gt_grids[i], gt_grids[i - 1]):
                dynamic_idx.append(i)

        n_total = len(gt_grids)
        n_dyn = len(dynamic_idx)
        print(f"  Dynamic timesteps: {n_dyn}/{n_total}  (skipping {n_total - n_dyn} static)")

        if n_dyn < 2:
            print("  No dynamic changes detected, skipping sequence.")
            continue

        # Filter all grid lists to only dynamic timesteps
        gt_grids = [gt_grids[i] for i in dynamic_idx]
        for m in METHODS:
            method_grids[m] = [method_grids[m][i] for i in dynamic_idx]

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
                    method=method,
                    episode=ep_idx,
                    seq=seq_out_dir,
                )
                all_results[method].append(result)

                # Visualize a few
                """
                if ep_idx < args.num_viz and method in ["model", "gt", "baseline"]:
                    plan = astar(grids[0], start, goal)
                    visualize_episode(
                        grids[0], gt_grids[0], start, goal, plan,
                        os.path.join(seq_out_dir, f"ep{ep_idx}_{method}.png"),
                        title=f"{method.upper()} – ep {ep_idx}",
                    )
                """

        # ── Frontier-Based Exploration ──
        explore_starts = sample_free_positions(
            gt_grids[0], args.exploration_starts,
            min_dist_cells=5, max_dist_cells=999, rng=rng,
        )
        explore_starts = [s for s, _ in explore_starts]  # only need start points

        if explore_starts:
            print(f"  Running {len(explore_starts)} exploration episodes...")
            for method in METHODS:
                grids = method_grids[method]
                if not grids:
                    continue
                for exp_idx, start_pos in enumerate(explore_starts):
                    # Visualize only the first episode per method
                    if exp_idx == 0:
                        viz_path = os.path.join(seq_out_dir, f"explore_{method}")
                    else:
                        viz_path = None

                    exp_result = run_exploration(
                        grids, gt_grids, start_pos,
                        sensor_radius=args.sensor_radius,
                        max_steps=args.exploration_steps,
                        viz_dir=viz_path,
                    )
                    all_exploration[method].append(exp_result)

            # Per-sequence exploration summary
            print(f"  --- Exploration ---")
            for method in METHODS:
                results = all_exploration[method][-len(explore_starts):]
                if not results:
                    continue
                cov = np.mean([r.coverage for r in results])
                col = np.mean([r.collision_rate for r in results])
                eff = np.mean([r.efficiency for r in results])
                fv  = np.mean([r.frontiers_visited for r in results])
                print(f"  {method:10s}  Coverage={cov:.3f}  ColRate={col:.4f}  "
                      f"Efficiency={eff:.5f}  Frontiers={fv:.1f}")

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

    # ═══════════════════════════════════════════════════════════════
    #  Dataset-Wide Exploration Summary
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("DATASET-WIDE FRONTIER EXPLORATION RESULTS")
    print("=" * 80)
    exp_header = (f"{'Method':>12s} | {'Coverage':>8s} | {'ColRate':>8s} | "
                  f"{'Efficien':>8s} | {'Frontiers':>9s} | {'Steps':>6s} | {'N':>5s}")
    print(exp_header)
    print("-" * len(exp_header))

    exp_summary = {}
    for method in METHODS:
        results = all_exploration[method]
        if not results:
            continue
        cov  = float(np.mean([r.coverage for r in results]))
        cr   = float(np.mean([r.collision_rate for r in results]))
        eff  = float(np.mean([r.efficiency for r in results]))
        fv   = float(np.mean([r.frontiers_visited for r in results]))
        st   = float(np.mean([r.steps for r in results]))
        exp_summary[method] = {
            "Coverage": cov, "CollisionRate": cr, "Efficiency": eff,
            "Frontiers": fv, "AvgSteps": st, "N": len(results),
        }
        print(f"{method:>12s} | {cov:8.3f} | {cr:8.4f} | {eff:8.5f} | {fv:9.1f} | {st:6.0f} | {len(results):>5d}")

    print("=" * 80)

    # Save combined JSON
    combined_json = {"navigation": summary, "exploration": exp_summary}
    json_path = os.path.join(args.out, "all_metrics.json")
    with open(json_path, "w") as f:
        json.dump(combined_json, f, indent=2)
    print(f"All metrics saved to {json_path}")

    # ── Combined bar chart: Navigation + Exploration ──
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Row 1: Navigation
    nav_metrics = [("SR", "Success Rate ↑"), ("SPL", "SPL ↑"),
                   ("CollisionRate", "Collision Rate ↓"), ("Replans", "Re-plans ↓")]
    colors = {"model": "#2196F3", "baseline": "#FF9800", "static": "#9E9E9E", "gt": "#4CAF50"}

    for ax, (key, label) in zip(axes[0], nav_metrics):
        vals = [summary.get(m, {}).get(key, 0) for m in METHODS]
        bars = ax.bar(METHODS, vals, color=[colors.get(m, "#ccc") for m in METHODS])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0][0].set_title("Point-Goal Navigation", fontsize=11, fontweight="bold", loc="left")

    # Row 2: Exploration
    exp_plot_metrics = [("Coverage", "Coverage ↑"), ("CollisionRate", "Collision Rate ↓"),
                        ("Efficiency", "Efficiency ↑"), ("Frontiers", "Frontiers Reached")]

    for ax, (key, label) in zip(axes[1], exp_plot_metrics):
        vals = [exp_summary.get(m, {}).get(key, 0) for m in METHODS]
        bars = ax.bar(METHODS, vals, color=[colors.get(m, "#ccc") for m in METHODS])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1][0].set_title("Frontier Exploration", fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Downstream Evaluation – Dynamic Voxel Mapping", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "combined_summary.png"), dpi=150)
    plt.close(fig)
    print(f"Combined plot saved to {os.path.join(args.out, 'combined_summary.png')}")


if __name__ == "__main__":
    main()