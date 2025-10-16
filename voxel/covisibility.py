from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def covisibility_from_world_proximity(
    P_world: torch.Tensor,           # (S, H, W, 3), world-aligned per-frame point clouds
    conf_map: torch.Tensor,          # (S, H, W) or (S, H, W, 1) or (S,H,W,C)
    *,
    stride: int = 4,                 # subsample pixels: use every `stride`th row/col
    max_points: int = 20_000,        # cap points per frame after subsampling (random downsample)
    eps: float = 0.5,               # distance threshold for "same surface"
    sym: str = "mean",               # "mean" | "min" | "max"
    normalize: bool = True,          # return overlap ratio in [0,1] (else raw counts)
    chunk_size: int = 10_000,        # cdist chunking to save memory
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    conf_percentile: float | None = 50.0,  # global percentile over confidences to filter points; None = no percentile
    conf_min: float = 1e-5,          # absolute floor on confidence
):
    """
    Returns:
    W: (S, S) symmetric covisibility matrix (dtype), based on 3D proximity only.
    """
    assert sym in ("mean", "min", "max")
    assert P_world.ndim == 4 and P_world.shape[-1] == 3, f"P_world must be (S,H,W,3), got {tuple(P_world.shape)}"


    # Normalize conf_map to (S,H,W) tensor
    if conf_map.ndim == 4:
        conf_map = conf_map[..., 0]
    elif conf_map.ndim != 3:
        raise ValueError(f"conf_map must be (S,H,W) or (S,H,W,1[+]), got {tuple(conf_map.shape)}")

    P_world = torch.from_numpy(P_world)
    conf_map = torch.from_numpy(conf_map)

    # Keep everything in torch; decide device
    device = device or P_world.device
    P_world = P_world.to(device)
    conf_map = conf_map.to(device)

    S, H, Wimg, _ = P_world.shape

    # Build subsampling grid once
    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, Wimg, stride, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')       # (H/stride, W/stride)
    sub_v = grid_y.reshape(-1)                                    # (M,)
    sub_u = grid_x.reshape(-1)                                    # (M,)

    # Optionally compute a GLOBAL percentile threshold over confidences (only finite points)
    if conf_percentile is not None:
        # gather confidence over all frames at subsampled pixels
        conf_all = conf_map[:, sub_v, sub_u].reshape(-1)          # (S*M,)
        # keep finite confs only (some pipelines use NaNs/Infs for invalid)
        finite = torch.isfinite(conf_all)
        if finite.any():
            # torch.quantile is nan-agnostic only if you filter first
            conf_threshold = torch.quantile(conf_all[finite], conf_percentile / 100.0)
            conf_threshold = torch.maximum(conf_threshold, torch.as_tensor(conf_min, device=device))
        else:
            conf_threshold = torch.as_tensor(conf_min, device=device)
    else:
        conf_threshold = torch.as_tensor(conf_min, device=device)

    # Prepack subsampled, confidence-filtered points per frame
    frames_pts: list[torch.Tensor] = []
    # For deterministic downsampling on CPU, keep generator on CPU but use indices on device
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(S):
        Pi = P_world[i, sub_v, sub_u, :]                          # (M, 3)
        Ci = conf_map[i, sub_v, sub_u]                            # (M,)

        # validity mask: finite 3D & finite conf & conf >= thresholds
        valid3d = torch.isfinite(Pi).all(dim=-1)
        validc  = torch.isfinite(Ci) & (Ci >= conf_threshold) & (Ci > conf_min)
        mask = valid3d & validc

        Pi = Pi[mask]                                             # (m_i, 3)
        if Pi.numel() == 0:
            frames_pts.append(torch.empty((0, 3), device=device, dtype=dtype))
            continue

        # optional per-frame cap
        if Pi.shape[0] > max_points:
            # randperm on CPU generator → bring indices to device
            idx = torch.randperm(Pi.shape[0], generator=rng)[:max_points]
            idx = idx.to(device=device)
            Pi = Pi.index_select(0, idx)

        frames_pts.append(Pi.to(dtype=dtype))

    # Pairwise proximity with chunking
    W = torch.zeros((S, S), dtype=dtype, device=device)

    def nn_consistent(A: torch.Tensor, B: torch.Tensor, eps: float, chunk: int):
        """
        Count of points in A that have at least one NN in B within eps, and total count |A|.
        """
        if A.numel() == 0 or B.numel() == 0:
            return 0, 0
        nA = A.shape[0]
        ok_total = 0
        # Pre-chunk B too if it's very big to reduce peak memory
        # (Here we only chunk A; chunking B as well is easy if needed.)
        for start in range(0, nA, chunk):
            end = min(start + chunk, nA)
            a = A[start:end]                                      # (m,3)
            dmin = torch.cdist(a, B, p=2).min(dim=1).values       # (m,)
            ok_total += (dmin <= eps).sum().item()
        return ok_total, nA

    for i in range(S):
        Ai = frames_pts[i]
        for j in range(i, S):
            if i == j:
                w = 1.0
            else:
                Bj = frames_pts[j]
                ok_ij, n_i = nn_consistent(Ai, Bj, eps=eps, chunk=chunk_size)
                ok_ji, n_j = nn_consistent(Bj, Ai, eps=eps, chunk=chunk_size)

                if normalize:
                    s_ij = ok_ij / max(1, n_i)
                    s_ji = ok_ji / max(1, n_j)
                else:
                    s_ij, s_ji = float(ok_ij), float(ok_ji)

                if sym == "mean":
                    w = 0.5 * (s_ij + s_ji)
                elif sym == "min":
                    w = min(s_ij, s_ji)
                else:  # "max"
                    w = max(s_ij, s_ji)

            W[i, j] = W[j, i] = w

    return W


def covis_to_adj(
    W: torch.Tensor,                 # (S,S) float
    tau: float = 0.5,                # base threshold for "good" edges
    kmin: int = 2,                   # ensure each node has at least kmin neighbors
    window: int | None = None,       # optional |i-j| <= window constraint
    mutual: str = "or",              # 'or' (weakly-undirected) or 'and' (mutual)
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build an adjacency with no isolated nodes:
      - keep edges with W >= tau
      - add top-kmin edges per node as a safety net (k-NN)
      - symmetrize and set diagonal True
    """
    assert mutual in ("or", "and")
    device = device or W.device
    S = W.shape[0]
    W = W.to(device)

    # base thresholded edges
    adj = (W >= tau)

    # optional temporal/spatial band
    if window is not None:
        i = torch.arange(S, device=device)
        band = (i[:, None] - i[None, :]).abs() <= window
        adj = adj & band

    # k-NN backup (exclude self)
    W_knn = W.clone()
    W_knn.fill_diagonal_(-float("inf"))
    if window is not None:
        # forbid outside window in the KNN too
        W_knn = W_knn.masked_fill(~band, -float("inf"))

    if kmin > 0:
        # take top-kmin neighbors per row
        k = min(kmin, max(0, S - 1))
        if k > 0:
            idx = torch.topk(W_knn, k=k, dim=1).indices                        # (S,k)
            knn = torch.zeros((S, S), dtype=torch.bool, device=device)
            knn.scatter_(1, idx, True)                                         # mark top-k per row
            adj = adj | knn                                                     # ensure degree ≥ kmin

    # symmetrize
    adj = (adj | adj.t()) if mutual == "or" else (adj & adj.t())

    # diagonal = True
    adj.fill_diagonal_(True)
    return adj

def compute_tight_components(adj_bool_2d: torch.Tensor,
                            mutual: bool = True,
                            window: int | None = None,
                            jaccard_tau: float | None = None,
                            k_core_k: int | None = None):
    """
    adj_bool_2d: (S,S) bool, True=edge (directed or undirected)
    Returns: list[list[int]] connected components after pruning.
    """

    A = adj_bool_2d.clone()

    S = A.shape[0]
    # 1) symmetrize
    A = A | A.T                # weakly undirected (either direction)
    if mutual:
        A = A & A.T            # require i<->j (mutual)

    # 2) force self-edges
    i = torch.arange(S, device=A.device)
    A[i, i] = True

    # 3) optional: local window
    if window is not None:
        # keep |i-j| <= window
        idx = torch.arange(S, device=A.device)
        band = (idx[:, None] - idx[None, :]).abs() <= window
        A = A & band

    # 4) optional: Jaccard prune (neighborhood similarity)
    if jaccard_tau is not None:
        # avoid diag in sim (but keep diag in A)
        N = A.clone()
        N[i, i] = False
        inter = (N[:, None, :] & N[None, :, :]).sum(dim=-1)                  # (S,S)
        union = (N[:, None, :] | N[None, :, :]).sum(dim=-1).clamp_min_(1)
        jacc = inter.float() / union.float()
        keep = jacc >= jaccard_tau
        A = (A & keep) | torch.diag(torch.ones(S, dtype=torch.bool, device=A.device))

    # 5) optional: k-core prune
    if k_core_k is not None:
        deg = A.sum(dim=1) - 1  # exclude self
        active = torch.ones(S, dtype=torch.bool, device=A.device)
        changed = True
        while changed:
            drop = active & (deg < k_core_k)
            changed = bool(drop.any().item())
            active = active & ~drop
            if changed:
                # zero edges for dropped nodes
                A[drop, :] = False
                A[:, drop] = False
                A[i, i] = True
                deg = A.sum(dim=1) - 1

    # Connected components on the pruned, undirected graph
    S_ = S
    seen = torch.zeros(S_, dtype=torch.bool, device=A.device)
    comps = []
    for s in range(S_):
        if seen[s]:
            continue
        stack = [int(s)]
        seen[s] = True
        comp = [int(s)]
        while stack:
            u = stack.pop()
            nbrs = torch.where(A[u])[0]
            for v in nbrs.tolist():
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
                    comp.append(int(v))
        comps.append(comp)
    return comps


def merge_singletons_to_next(groups, S):
    """
    groups: list[list[int]]  (components over nodes 0..S-1)
    S: total number of frames/nodes

    Rule: if a component has size 1 (singleton {i}), merge i into the group
    that contains (i+1) % S. If *all* components are singletons, merge them all
    into a single group [0,1,...,S-1].
    """
    # keep only non-singletons first
    new_groups = [set(g) for g in groups if len(g) > 1]
    if not new_groups:
        # all singletons -> one big group
        return [list(range(S))]

    # map node -> group index for the current (non-singleton) groups
    node_to_gid = {}
    for gid, g in enumerate(new_groups):
        for v in g:
            node_to_gid[v] = gid

    # collect singletons in sorted order for determinism
    singles = sorted([g[0] for g in groups if len(g) == 1])

    for i in singles:
        # find the next index (wrap) that already sits in a non-singleton group
        t = (i + 1) % S
        hopped = 0
        while t not in node_to_gid and hopped < S:
            t = (t + 1) % S
            hopped += 1
        if hopped >= S:
            # should not happen because we handled "all singletons" above,
            # but just in case, attach to the first group
            target_gid = 0
        else:
            target_gid = node_to_gid[t]

        # merge i into that group
        new_groups[target_gid].add(i)
        node_to_gid[i] = target_gid

    # return as sorted lists (optional: keep original ordering of groups)
    return [sorted(list(g)) for g in new_groups]



import heapq
import math

def dijkstra_to_any(adj, targets, start=0):
    """
    adj: NxN matrix of non-negative weights; 0 or math.inf means no edge.
    targets: target node indices (set/list)
    Returns (target_node, path, dist) or (None, None, inf).
    """
    
    if start in targets:
        return start, [], 0
    
    covis = np.asarray(adj, dtype=float)
    cost = np.full_like(covis, np.inf)
    mask = covis > 0
    # cost[mask] = 1.0 / (covis[mask] + 1e-6)
    
    cost[mask] = -np.log(covis[mask] + 1e-6)

    np.fill_diagonal(cost, np.inf)  # no self-edges
    adj = cost
    
    N = len(adj); targets = set(targets)
    dist = [math.inf]*N
    parent = [-1]*N
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = [False]*N
    
    
    while pq:
        d,u = heapq.heappop(pq)
        print(d,u)
        if visited[u]: continue
        visited[u] = True
        if u in targets:
            # reconstruct path
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return u, path[:-1], d
        for v in range(N):
            w = adj[u][v]
            if w and w < math.inf:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
    return None, None, math.inf

import math, heapq
import numpy as np

def _covis_to_cost(covis, eps=1e-6):
    """Flip covisibility (bigger=better) to cost (smaller=better). 0 -> inf (no edge)."""
    A = np.asarray(covis, dtype=float)
    cost = np.full_like(A, np.inf)
    mask = A > 0
    cost[mask] = 1.0 / (A[mask] + eps)
    np.fill_diagonal(cost, np.inf)
    return cost

def _dijkstra_path(cost, src, dst):
    """Shortest path on dense cost matrix (np.inf = no edge). Returns list of nodes (src..dst) or [] if no path."""
    N = cost.shape[0]
    dist = [math.inf]*N; parent = [-1]*N; vis = [False]*N
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if vis[u]: continue
        vis[u] = True
        if u == dst:
            # reconstruct
            path = [u]
            while path[-1] != src:
                path.append(parent[path[-1]])
            return list(reversed(path))
        # relax
        row = cost[u]
        # iterate only finite neighbors
        nbrs = np.isfinite(row).nonzero()[0]
        for v in nbrs:
            nd = d + row[v]
            if nd < dist[v]:
                dist[v] = nd; parent[v] = u
                heapq.heappush(pq, (nd, v))
    return []  # unreachable

def _coverage_score(covis, S):
    """
    Coverage = mean over all nodes j of max_{i in S} covis[j,i].
    Returns (score in [0,1], per-node coverage vector).
    """
    A = np.asarray(covis, dtype=float)
    if len(S) == 0:
        return 0.0, np.zeros(A.shape[0], dtype=float)
    cov_to_S = A[:, S]                  # (N, |S|)
    per_node = cov_to_S.max(axis=1)     # (N,)
    score = float(per_node.mean())
    return score, per_node

def grow_set_until_coverage(
    covis,
    thresh=0.80,                 # target mean coverage in [0,1]
    start=None,                  # initial set (list/array); default: argmax row-sum
    max_steps=100,
    verbose=False
):
    """
    Greedy growth:
    - Source each round: node in current set with the LOWEST covis-to-set (sum).
    - Candidate to add: outside node giving largest coverage gain if included.
    - Connectivity: add shortest path (on cost=1/(covis+eps)) from source -> candidate.
    Returns: selected (list), paths_added (list of lists), cov_history (list of floats)
    """
    # prep
    A = np.asarray(covis, dtype=float)
    N = A.shape[0]
    cost = _covis_to_cost(A)

    # init set
    if start is None or len(start) == 0:
        row_sum = A.sum(axis=1)
        seed = int(np.argmax(row_sum))
        S = {seed}
    else:
        S = set(int(x) for x in start)

    cov, per_node = _coverage_score(A, list(S))
    cov_hist = [cov]
    paths = []

    if verbose:
        print(f"[init] |S|={len(S)}  cov={cov:.3f}")

    steps = 0
    while cov < thresh and steps < max_steps:
        steps += 1
        S_list = list(S)

        # ----- choose source: node in S with LOWEST covis-to-set (sum) -----
        # (you could also use mean or max; sum is fine and simple)
        sum_to_S = A[np.ix_(S_list, S_list)].sum(axis=1)  # |S|-vector
        src = S_list[int(np.argmin(sum_to_S))]

        # ----- pick best candidate outside S by marginal coverage gain -----
        outside = [i for i in range(N) if i not in S]
        if not outside:
            break

        best_gain = -1.0
        best_cand = None

        # current per-node coverage to S
        cur_cov = per_node  # shape (N,)

        # evaluate marginal gain if we add each candidate c
        # new per-node coverage would be: max(cur_cov, A[:,c])
        for c in outside:
            new_per = np.maximum(cur_cov, A[:, c])
            new_score = float(new_per.mean())
            gain = new_score - cov
            if gain > best_gain:
                best_gain = gain
                best_cand = c

        if best_cand is None or best_gain <= 0.0:
            if verbose:
                print("[stop] no positive marginal gain; breaking")
            break

        # ----- connect src -> best_cand with shortest path and add nodes -----
        path = _dijkstra_path(cost, src, best_cand)
        if not path:
            # if disconnected, just add the candidate (last resort)
            path = [best_cand]
            if verbose:
                print(f"[warn] no path {src}->{best_cand}; adding candidate alone")

        # add path nodes
        new_nodes = [p for p in path if p not in S]
        S.update(new_nodes)
        paths.append(path)

        # update coverage
        cov, per_node = _coverage_score(A, list(S))
        cov_hist.append(cov)

        if verbose:
            print(f"[iter {steps}] added {new_nodes} via path {path} |S|={len(S)} cov={cov:.3f} (+{best_gain:.3f})")

    return sorted(S), paths, cov_hist