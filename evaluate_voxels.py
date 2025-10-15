import numpy as np
from scipy.ndimage import map_coordinates

def world_extent(meta, H, W):
    res = float(meta.get("resolution", 0.10))
    ox, oy = meta.get("origin_xy", (0.0, 0.0))
    return ox, oy, ox + W * res, oy + H * res  # xmin, ymin, xmax, ymax

def resample_to_canvas(bev, meta, canvas, out_res):
    # canvas = (xmin, ymin, xmax, ymax); out_res in meters
    xmin, ymin, xmax, ymax = canvas
    H = int(np.ceil((ymax - ymin) / out_res))
    W = int(np.ceil((xmax - xmin) / out_res))
    # world coords of output grid centers
    xs = xmin + (np.arange(W) + 0.5) * out_res
    ys = ymin + (np.arange(H) + 0.5) * out_res
    X, Y = np.meshgrid(xs, ys)  # shape HxW

    # map world -> source pixel indices
    s_res = float(meta.get("resolution", 0.10))
    sox, soy = meta.get("origin_xy", (0.0, 0.0))
    # source pixel centers at (sox + (j+0.5)*s_res, soy + (i+0.5)*s_res)
    # invert to (i,j):
    J = (X - sox) / s_res - 0.5
    I = (Y - soy) / s_res - 0.5
    # sample with bilinear (order=1), fill outside with 0 (or np.nan if you prefer masking)
    sampled = map_coordinates(bev, [I, J], order=1, mode='constant', cval=0.0)
    return sampled

def bev_metrics(bevA, metaA, bevB, metaB, thresholds=np.linspace(0.05, 0.95, 19)):
    # 1) common canvas
    H_A, W_A = bevA.shape
    H_B, W_B = bevB.shape
    axmin, aymin, axmax, aymax = world_extent(metaA, H_A, W_A)
    bxmin, bymin, bxmax, bymax = world_extent(metaB, H_B, W_B)
    xmin, ymin = min(axmin, bxmin), min(aymin, bymin)
    xmax, ymax = max(axmax, bxmax), max(aymax, bymax)
    canvas = (xmin, ymin, xmax, ymax)

    out_res = min(float(metaA.get("resolution", 0.10)),
                  float(metaB.get("resolution", 0.10)))

    A = resample_to_canvas(bevA, metaA, canvas, out_res)
    B = resample_to_canvas(bevB, metaB, canvas, out_res)

    # Flatten for simple metrics
    a = A.ravel()
    b = B.ravel()

    # Probability metrics (Brier, etc.)
    brier = np.mean((a - b) ** 2)

    # Threshold sweeps
    out = {"brier": float(brier), "operating_points": []}
    eps = 1e-9
    for t in thresholds:
        ta = (a >= t).astype(np.uint8)
        tb = (b >= t).astype(np.uint8)
        tp = np.sum((ta == 1) & (tb == 1))
        fp = np.sum((ta == 1) & (tb == 0))
        fn = np.sum((ta == 0) & (tb == 1))
        tn = np.sum((ta == 0) & (tb == 0))
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        iou       = tp / (tp + fp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        out["operating_points"].append({
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    # best F1
    best = max(out["operating_points"], key=lambda d: d["f1"])
    out["best_f1"] = best
    return out


import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


# def chamfer_and_fscore(P: np.ndarray, Q: np.ndarray, tau=0.2,
#                        z_window=None,       # tuple (zmin, zmax) or None
#                        z_gate=None,         # float meters or None; gate TP by |Δz|<=z_gate
#                        nn_space="3d"        # "3d" or "xy" for nearest-neighbor search
#                        ):
#     """
#     P, Q: (N,3) and (M,3) centers in meters.
#     tau: match radius (meters) for precision/recall/F.
#     z_window: if given, keep only points with z in [zmin, zmax] for BOTH clouds (pre-filter).
#     z_gate: if given, a matched pair only counts as TP if |Δz| <= z_gate (post NN gate).
#     nn_space: "3d" uses full 3D KDTree; "xy" uses XY only (top-down distances).

#     Returns dict with:
#       chamfer_L2_sq (3D, symmetric), precision/recall/fscore, counts,
#       and z_gate_rejected counts (pairs that failed the vertical gate).
#     """
#     P = np.asarray(P, dtype=np.float32)
#     Q = np.asarray(Q, dtype=np.float32)

#     # Pre-filter by Z window if requested
#     if z_window is not None:
#         zmin, zmax = z_window
#         if P.size:
#             P = P[(P[:, 2] >= zmin) & (P[:, 2] <= zmax)]
#         if Q.size:
#             Q = Q[(Q[:, 2] >= zmin) & (Q[:, 2] <= zmax)]

#     if P.size == 0 and Q.size == 0:
#         return {"chamfer_L2_sq": 0.0, "tau": float(tau),
#                 "precision": 1.0, "recall": 1.0, "fscore": 1.0,
#                 "tp": 0, "fp": 0, "fn": 0, "nP": 0, "nQ": 0,
#                 "z_gate": float(z_gate) if z_gate is not None else None,
#                 "z_gate_rejected_P": 0, "z_gate_rejected_Q": 0}

#     # Build trees for chosen NN space
#     def make_data(X):
#         if nn_space == "xy":
#             return X[:, :2]
#         elif nn_space == "3d":
#             return X
#         else:
#             raise ValueError("nn_space must be '3d' or 'xy'")

#     P_nn = make_data(P) if P.size else P
#     Q_nn = make_data(Q) if Q.size else Q

#     treeQ = cKDTree(Q_nn) if Q.size else None
#     treeP = cKDTree(P_nn) if P.size else None

#     # Distances for Chamfer are always 3D Euclidean (symmetric)
#     if treeQ and P.size:
#         # indexes for Q nearest to each P in NN space
#         dPQ_nn, idxPQ = treeQ.query(P_nn, k=1, workers=-1)
#         # 3D distances for chamfer
#         dPQ = np.linalg.norm(P - Q[idxPQ], axis=1)
#         dzPQ = np.abs(P[:, 2] - Q[idxPQ, 2])
#     else:
#         dPQ_nn = np.full((len(P),), np.inf, dtype=np.float32)
#         dPQ    = np.full((len(P),), np.inf, dtype=np.float32)
#         dzPQ   = np.full((len(P),), np.inf, dtype=np.float32)
#         idxPQ  = np.full((len(P),), -1, dtype=int)

#     if treeP and Q.size:
#         dQP_nn, idxQP = treeP.query(Q_nn, k=1, workers=-1)
#         dQP = np.linalg.norm(Q - P[idxQP], axis=1)
#         dzQP = np.abs(Q[:, 2] - P[idxQP, 2])
#     else:
#         dQP_nn = np.full((len(Q),), np.inf, dtype=np.float32)
#         dQP    = np.full((len(Q),), np.inf, dtype=np.float32)
#         dzQP   = np.full((len(Q),), np.inf, dtype=np.float32)
#         idxQP  = np.full((len(Q),), -1, dtype=int)

#     # Symmetric Chamfer^2 (3D)
#     chamfer = float(np.mean(dPQ**2) + np.mean(dQP**2))

#     # Apply z_gate for TP counting (if set). Distances for TP are in the chosen NN space via tau.
#     if z_gate is not None:
#         gateP = (dzPQ <= z_gate)
#         gateQ = (dzQP <= z_gate)
#     else:
#         gateP = np.ones_like(dPQ_nn, dtype=bool)
#         gateQ = np.ones_like(dQP_nn, dtype=bool)

#     # True positives: P->Q matches within tau AND passing z_gate
#     tp_mask = (dPQ_nn <= tau) & gateP
#     tp = int(np.sum(tp_mask))

#     # False positives: P with no acceptable match
#     fp = int(len(P) - tp)

#     # False negatives: Q not covered by P within tau (and passing z_gate from Q's perspective)
#     # We count a Q as covered if its NN P is within tau AND passes z_gate.
#     covered_Q = (dQP_nn <= tau) & gateQ
#     fn = int(np.sum(~covered_Q))

#     precision = tp / (tp + fp + 1e-9)
#     recall    = tp / (tp + fn + 1e-9)
#     fscore    = 2 * precision * recall / (precision + recall + 1e-9)

#     # Diagnostics: how many pairs failed only due to Z gate
#     zrej_P = int(np.sum((dPQ_nn <= tau) & (~gateP)))
#     zrej_Q = int(np.sum((dQP_nn <= tau) & (~gateQ)))

#     return {"chamfer_L2_sq": chamfer, "tau": float(tau),
#             "precision": float(precision), "recall": float(recall), "fscore": float(fscore),
#             "tp": tp, "fp": fp, "fn": fn, "nP": int(len(P)), "nQ": int(len(Q)),
#             "z_window": tuple(z_window) if z_window is not None else None,
#             "z_gate": float(z_gate) if z_gate is not None else None,
#             "nn_space": nn_space,
#             "z_gate_rejected_P": zrej_P, "z_gate_rejected_Q": zrej_Q}


import numpy as np
from scipy.spatial import cKDTree

def chamfer_and_fscore(P: np.ndarray, Q: np.ndarray, tau=0.2,
                       z_window=None, z_gate=None, nn_space="3d",
                       mutual=True):
    """
    Density-robust F1: precision = fraction of P covered within tau,
                       recall    = fraction of Q covered within tau.
    Optional: z window/gate and mutual-NN matches.
    """
    P = np.asarray(P, np.float32); Q = np.asarray(Q, np.float32)

    # Optional Z prefilter
    if z_window is not None:
        zmin, zmax = z_window
        if P.size: P = P[(P[:,2] >= zmin) & (P[:,2] <= zmax)]
        if Q.size: Q = Q[(Q[:,2] >= zmin) & (Q[:,2] <= zmax)]

    if P.size == 0 and Q.size == 0:
        return {"chamfer_L2_sq": 0.0, "tau": float(tau),
                "precision": 1.0, "recall": 1.0, "fscore": 1.0,
                "tp": 0, "fp": 0, "fn": 0, "nP": 0, "nQ": 0,
                "z_gate": z_gate, "mutual": mutual}

    def proj(X):
        if nn_space == "xy": return X[:, :2]
        if nn_space == "3d": return X
        raise ValueError("nn_space must be '3d' or 'xy'")

    Pn = proj(P) if P.size else P
    Qn = proj(Q) if Q.size else Q
    treeQ = cKDTree(Qn) if Q.size else None
    treeP = cKDTree(Pn) if P.size else None

    # indices and dists in chosen NN space
    if treeQ and P.size:
        dPQ_tau, idxPQ = treeQ.query(Pn, k=1, workers=-1)
        dzPQ = np.abs(P[:,2] - Q[idxPQ,2]) if (Q.size and z_gate is not None) else None
    else:
        dPQ_tau = np.full((len(P),), np.inf, np.float32); idxPQ = np.full((len(P),), -1, int); dzPQ=None

    if treeP and Q.size:
        dQP_tau, idxQP = treeP.query(Qn, k=1, workers=-1)
        dzQP = np.abs(Q[:,2] - P[idxQP,2]) if (P.size and z_gate is not None) else None
    else:
        dQP_tau = np.full((len(Q),), np.inf, np.float32); idxQP = np.full((len(Q),), -1, int); dzQP=None

    # Chamfer^2 always in full 3D
    if P.size and Q.size:
        dPQ_3d = np.linalg.norm(P - Q[idxPQ], axis=1)
        dQP_3d = np.linalg.norm(Q - P[idxQP], axis=1)
    else:
        dPQ_3d = np.full((len(P),), np.inf, np.float32)
        dQP_3d = np.full((len(Q),), np.inf, np.float32)
    chamfer = float(np.mean(dPQ_3d**2) + np.mean(dQP_3d**2))

    # Optional mutual-NN gating
    mutual_mask_P = np.ones(len(P), dtype=bool)
    mutual_mask_Q = np.ones(len(Q), dtype=bool)
    if mutual and P.size and Q.size:
        back = np.full(len(Q), -1, int); back[idxPQ] = np.arange(len(P))
        mutual_mask_P = (back[idxPQ] == np.arange(len(P)))
        fwd = np.full(len(P), -1, int); fwd[idxQP] = np.arange(len(Q))
        mutual_mask_Q = (fwd[idxQP] == np.arange(len(Q)))

    # Z gate
    gateP = (dzPQ <= z_gate) if (z_gate is not None and dzPQ is not None) else np.ones(len(P), bool)
    gateQ = (dzQP <= z_gate) if (z_gate is not None and dzQP is not None) else np.ones(len(Q), bool)

    # Coverage on each side (this fixes the recall bug)
    covered_P = (dPQ_tau <= tau) & gateP & mutual_mask_P
    covered_Q = (dQP_tau <= tau) & gateQ & mutual_mask_Q

    precision = float(np.mean(covered_P)) if len(P) else (1.0 if len(Q)==0 else 0.0)
    recall    = float(np.mean(covered_Q)) if len(Q) else (1.0 if len(P)==0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # For reporting tp/fp/fn in a density-agnostic way:
    tp = int(np.sum(covered_P))                 # P-side coverage count
    fp = int(len(P) - tp)
    fn = int(np.sum(~covered_Q))                # Q-side uncovered count

    return {"chamfer_L2_sq": chamfer, "tau": float(tau),
            "precision": precision, "recall": recall, "fscore": f1,
            "tp": tp, "fp": fp, "fn": fn, "nP": int(len(P)), "nQ": int(len(Q)),
            "z_window": tuple(z_window) if z_window else None,
            "z_gate": float(z_gate) if z_gate is not None else None,
            "nn_space": nn_space, "mutual": bool(mutual)}


# # ---------- helpers ----------
def _to_tuple3(voxel_size):
    """Accept scalar or len-3 iterable -> (sx,sy,sz) floats."""
    arr = np.asarray(voxel_size, dtype=float).reshape(-1)
    if arr.size == 1:
        v = float(arr.item())
        return (v, v, v)
    assert arr.size == 3, f"voxel_size must be scalar or len-3, got {arr.shape}"
    return (float(arr[0]), float(arr[1]), float(arr[2]))

def ijk_to_centers_world(ijk: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert (N,3) IJK to world XYZ using meta: {origin_xyz, voxel_size}
    """
    ijk = np.asarray(ijk, dtype=np.float32)
    ox, oy, oz = meta["origin_xyz"]
    sx, sy, sz = _to_tuple3(meta["voxel_size"])
    centers = np.empty((ijk.shape[0], 3), dtype=np.float32)
    centers[:, 0] = ox + (ijk[:, 0] + 0.5) * sx
    centers[:, 1] = oy + (ijk[:, 1] + 0.5) * sy
    centers[:, 2] = oz + (ijk[:, 2] + 0.5) * sz
    return centers

def natural_key(p: Path):
    """Sort helper: 'frame2' before 'frame10'."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', p.stem)]

def find_pairs(dir_gt: Path, dir_pred: Path, npy_suffix="_ijk.npy", meta_suffix="_meta.json"):
    """
    Pair files by stem: expects e.g. foo_ijk.npy + foo_meta.json in each dir.
    Returns list of tuples: (gt_npy, gt_meta, pred_npy, pred_meta)
    """
    gt_npys = sorted(dir_gt.glob(f"*{npy_suffix}"), key=natural_key)
    pred_npys = sorted(dir_pred.glob(f"*{npy_suffix}"), key=natural_key)

    if len(gt_npys) != len(pred_npys):
        raise RuntimeError(f"Count mismatch: GT {len(gt_npys)} vs Pred {len(pred_npys)}")

    pairs = []
    for a, b in zip(gt_npys, pred_npys):
        if a.stem != b.stem:
            # If names differ but order is guaranteed, we still pair by order.
            # Otherwise, you can switch to dict lookup by base token.
            pass
        gt_meta = a.with_name(a.stem.replace(npy_suffix[:-4], "") + meta_suffix)
        pred_meta = b.with_name(b.stem.replace(npy_suffix[:-4], "") + meta_suffix)
        if not gt_meta.exists():
            raise FileNotFoundError(f"Missing GT meta for {a.name}: {gt_meta.name}")
        if not pred_meta.exists():
            raise FileNotFoundError(f"Missing Pred meta for {b.name}: {pred_meta.name}")
        pairs.append((a, gt_meta, b, pred_meta))
    return pairs


import numpy as np
from scipy.spatial import cKDTree

def apply_sim3_to_points(P, sim3):
    """P: (N,3). sim3: dict with s (float), R (3x3), t (3,) mapping src->tgt."""
    if sim3 is None:
        return P
    s = float(sim3["s"])
    R = np.asarray(sim3["R"], float).reshape(3,3)
    t = np.asarray(sim3["t"], float).reshape(3)
    return s * (P @ R.T) + t

def umeyama_sim3(Q, P, weights=None):
    """
    Estimate Sim(3) mapping Q->P:  x_P = s*R*x_Q + t
    Q, P: (N,3) with 1-1 correspondences.
    """
    Q = np.asarray(Q, float); P = np.asarray(P, float)
    assert Q.shape == P.shape and Q.shape[0] >= 3

    if weights is None:
        w = np.ones((Q.shape[0], 1))
    else:
        w = np.asarray(weights, float).reshape(-1,1)
    w /= (w.sum() + 1e-12)

    mu_Q = (w * Q).sum(axis=0, keepdims=True)
    mu_P = (w * P).sum(axis=0, keepdims=True)
    Qc = Q - mu_Q
    Pc = P - mu_P

    Sigma = (Qc * w).T @ Pc
    U, S, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # enforce proper rotation
        U[:, -1] *= -1
        R = U @ Vt

    var_Q = (w * (Qc**2)).sum()
    s = float(S.sum() / (var_Q + 1e-12))
    t = (mu_P.ravel() - s * (R @ mu_Q.ravel()))
    return {"s": s, "R": R, "t": t}

def sim3_icp_umeyama(src, tgt, max_iters=10, inlier_radius=None, trim_frac=0.0, init=None):
    """
    Estimate Sim(3) mapping src->tgt via NN correspondences + Umeyama.
    - inlier_radius: only keep pairs with dist <= radius (meters). If None, keep all.
    - trim_frac: 0..0.5 ; drop the worst fraction of pairs by distance (robust).
    - init: optional initial sim3 dict.
    Returns sim3 dict and RMS of final inliers.
    """
    if src.size == 0 or tgt.size == 0:
        return {"s":1.0, "R":np.eye(3), "t":np.zeros(3)}, np.nan

    sim3 = init or {"s":1.0, "R":np.eye(3), "t":np.zeros(3)}
    tree = cKDTree(tgt)

    for _ in range(max_iters):
        src_w = apply_sim3_to_points(src, sim3)
        d, idx = tree.query(src_w, k=1, workers=-1)
        pairs_src = src
        pairs_tgt = tgt[idx]

        # Inlier mask
        mask = np.ones(len(d), dtype=bool)
        if inlier_radius is not None:
            mask &= (d <= inlier_radius)

        # Trimming
        if trim_frac > 0:
            k = int((1.0 - trim_frac) * mask.sum())
            if k >= 3:
                keep_idx = np.argsort(d[mask])[:k]
                tmp = np.where(mask)[0][keep_idx]
                mask = np.zeros_like(mask); mask[tmp] = True

        if mask.sum() < 3:
            break

        sim3 = umeyama_sim3(pairs_src[mask], pairs_tgt[mask])

    # final RMS on inliers
    src_w = apply_sim3_to_points(src, sim3)
    d, _ = tree.query(src_w, k=1, workers=-1)
    if inlier_radius is not None:
        d = d[d <= inlier_radius]
    if trim_frac > 0 and len(d) >= 3:
        k = int((1.0 - trim_frac) * len(d))
        d = np.sort(d)[:max(k,0)]
    rms = float(np.sqrt(np.mean(d**2))) if len(d) else np.nan
    return sim3, rms

def _random_downsample(P, max_n=5000, seed=0):
    if len(P) <= max_n:
        return P
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=max_n, replace=False)
    return P[idx]

def viz_points_with_umeyama(P_gt, P_pr, estimate_sim3=True, init_sim3=None,
                            max_iters=15, inlier_radius=0.5, trim_frac=0.2,
                            marker_size=2.0, title_prefix=""):
    """
    Visualize two point clouds (GT vs Pred) before/after Sim(3) alignment.
    Optionally estimates Sim(3) mapping Pred->GT via NN+Umeyama (scale, rotation, translation).

    Args:
        P_gt, P_pr: (N,3) float arrays (meters).
        estimate_sim3: if True, run a tiny Sim(3)-ICP (Umeyama) to align Pred->GT.
        init_sim3: optional {"s":float, "R":(3,3), "t":(3,)} prior (e.g., scale from voxel sizes).
        max_iters: ICP iterations.
        inlier_radius: keep pairs with NN distance <= radius (meters). None = keep all.
        trim_frac: drop this fraction of worst pairs each iter (0..0.5) for robustness.
        marker_size: matplotlib scatter size.
        title_prefix: string prefix for figure titles.

    Returns:
        sim3: {"s":float, "R":(3,3) ndarray, "t":(3,) ndarray}
        figs: dict with matplotlib Figure objects:
              {"before_3d", "before_xy", "after_3d", "after_xy"}
    """
    import numpy as np
    from scipy.spatial import cKDTree
    import matplotlib.pyplot as plt

    def apply_sim3(P, s, R, t):
        return s * (P @ R.T) + t

    def umeyama_sim3(Q, P):
        # Estimate Sim(3) mapping Q->P given correspondences (Q_i ↔ P_i)
        Q = np.asarray(Q, float); P = np.asarray(P, float)
        assert Q.shape == P.shape and Q.shape[0] >= 3
        muQ = Q.mean(axis=0, keepdims=True)
        muP = P.mean(axis=0, keepdims=True)
        Qc, Pc = Q - muQ, P - muP
        Sigma = Qc.T @ Pc / Q.shape[0]
        U, S, Vt = np.linalg.svd(Sigma)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        varQ = (Qc * Qc).sum() / Q.shape[0]
        s = float(S.sum() / (varQ + 1e-12))
        t = (muP.ravel() - s * (R @ muQ.ravel()))
        return s, R, t

    def sim3_icp(src, tgt, init):
        if src.size == 0 or tgt.size == 0:
            return {"s":1.0, "R":np.eye(3), "t":np.zeros(3)}, np.nan
        if init is None:
            sim3 = {"s":1.0, "R":np.eye(3), "t":np.zeros(3)}
        else:
            sim3 = {"s":float(init["s"]),
                    "R":np.asarray(init["R"], float).reshape(3,3),
                    "t":np.asarray(init["t"], float).reshape(3)}
        tree = cKDTree(tgt)
        for _ in range(max_iters):
            src_w = apply_sim3(src, sim3["s"], sim3["R"], sim3["t"])
            d, idx = tree.query(src_w, k=1, workers=-1)
            Q = src.copy()
            P = tgt[idx]
            mask = np.ones(len(d), dtype=bool)
            if inlier_radius is not None:
                mask &= (d <= inlier_radius)
            if trim_frac > 0 and mask.sum() >= 3:
                k = int((1.0 - trim_frac) * mask.sum())
                if k >= 3:
                    keep = np.argsort(d[mask])[:k]
                    full_idx = np.where(mask)[0][keep]
                    mask = np.zeros_like(mask); mask[full_idx] = True
            if mask.sum() < 3:
                break
            s, R, t = umeyama_sim3(Q[mask], P[mask])
            sim3 = {"s": s, "R": R, "t": t}
        # final RMS
        src_w = apply_sim3(src, sim3["s"], sim3["R"], sim3["t"])
        d, _ = tree.query(src_w, k=1, workers=-1)
        if inlier_radius is not None:
            d = d[d <= inlier_radius]
        if trim_frac > 0 and len(d) >= 3:
            k = int((1.0 - trim_frac) * len(d))
            d = np.sort(d)[:max(k, 0)]
        rms = float(np.sqrt(np.mean(d**2))) if len(d) else np.nan
        return sim3, rms

    P_gt = np.asarray(P_gt, float)
    P_pr = np.asarray(P_pr, float)

    # ---- BEFORE plots ----
    figs = {}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_gt[:,0], P_gt[:,1], P_gt[:,2], s=marker_size, label="GT")
    ax.scatter(P_pr[:,0], P_pr[:,1], P_pr[:,2], s=marker_size, label="Pred (raw)")
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(f'{title_prefix}3D (before)')
    ax.legend(loc='upper left')
    figs["before_3d"] = fig

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(P_gt[:,0], P_gt[:,1], s=marker_size, label="GT")
    ax.scatter(P_pr[:,0], P_pr[:,1], s=marker_size, label="Pred (raw)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f'{title_prefix}Top-down (before)')
    ax.legend(loc='upper right')
    figs["before_xy"] = fig

    # ---- Estimate/apply Sim(3) ----
    if estimate_sim3:
        sim3, rms = sim3_icp(P_pr, P_gt, init_sim3)
    else:
        sim3 = {"s":1.0, "R":np.eye(3), "t":np.zeros(3)} if init_sim3 is None else init_sim3
        rms = np.nan
        return sim3, figs

    P_pr_aligned = apply_sim3(P_pr, sim3["s"], sim3["R"], sim3["t"])

    # ---- AFTER plots ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_gt[:,0], P_gt[:,1], P_gt[:,2], s=marker_size, label="GT")
    ax.scatter(P_pr_aligned[:,0], P_pr_aligned[:,1], s=marker_size, label="Pred (aligned)")
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(f'{title_prefix}3D (after)  RMS={rms:.3f} m')
    ax.legend(loc='upper left')
    figs["after_3d"] = fig

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(P_gt[:,0], P_gt[:,1], s=marker_size, label="GT")
    ax.scatter(P_pr_aligned[:,0], P_pr_aligned[:,1], s=marker_size, label="Pred (aligned)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title(f'{title_prefix}Top-down (after)')
    ax.legend(loc='upper right')
    figs["after_xy"] = fig

    return sim3, figs


def nn_diagnostics(P, Q, tau):
    P = np.asarray(P); Q = np.asarray(Q)
    treeP = cKDTree(P); treeQ = cKDTree(Q)
    dPQ, _ = treeQ.query(P, k=1, workers=-1)
    dQP, _ = treeP.query(Q, k=1, workers=-1)
    def stats(d):
        qs = np.percentile(d, [5,10,25,50,75,90,95])
        return dict(m=float(np.mean(d)), med=float(qs[3]),
                    p25=float(qs[2]), p75=float(qs[4]),
                    within_tau=float(np.mean(d <= tau)))
    return {"P->Q": stats(dPQ), "Q->P": stats(dQP)}

# ---------- main evaluation ----------
def evaluate_voxel_dirs(gt_dir, pred_dir, tau=0.2, save_csv=None):
    """
    gt_dir: folder with GT * _ijk.npy and *_meta.json
    pred_dir: folder with Pred * _ijk.npy and *_meta.json
    tau: match radius (meters) for F-score
    save_csv: optional path to save a CSV summary
    """
    dir_gt = Path(gt_dir)
    dir_pred = Path(pred_dir)
    pairs = find_pairs(dir_gt, dir_pred)[20:]

    rows = []
    for enum, (gt_npy, gt_meta, pr_npy, pr_meta) in enumerate(pairs):
        ijk_gt = np.load(gt_npy)
        ijk_pr = np.load(pr_npy)
        meta_gt = json.loads(Path(gt_meta).read_text())
        meta_pr = json.loads(Path(pr_meta).read_text())

        # Convert to world XYZ (so grids with different origins/voxel sizes compare correctly)
        P_gt = ijk_to_centers_world(ijk_gt, meta_gt)
        P_pr = ijk_to_centers_world(ijk_pr, meta_pr)
        
        
        
        theta = np.deg2rad(70)  # convert degrees to radians

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        ds_pr = 12 * _random_downsample((P_pr @ R.T), max_n=10000, seed=42) + [3,-7,0]
        

        N = ds_pr.shape[0]
        ds_gt = _random_downsample(P_gt, max_n=N, seed=42)

        # If you have a prior on scale from voxel_size, you can seed 'init' here.
        init = None
        # Example prior:
        # s0 = (np.mean(np.atleast_1d(meta_gt["voxel_size"])) /
        #       np.mean(np.atleast_1d(meta_pr["voxel_size"])))
        # init = {"s": s0, "R": np.eye(3), "t": np.zeros(3)}

      
        
        sim3, rms = sim3_icp_umeyama(
            src=ds_pr, tgt=ds_gt,
            max_iters=20,
            inlier_radius=2,   # meters; tune to your scene scale
            trim_frac=0.2,       # drop 20% worst matches each iter
            init=init
        )
        
      
      

        

        # Apply alignment and score
        P_pr_aligned = apply_sim3_to_points(ds_pr, sim3)
    
        # sim3, figs = viz_points_with_umeyama(ds_gt, P_pr_aligned, estimate_sim3=False,
        #                             inlier_radius=2, trim_frac=0.2,
        #                             title_prefix=f"{gt_npy.stem} - ")
        
        # from matplotlib import pyplot as plt
        # plt.show()
        
        
        
        # ---------- Evaluate F1 curve over multiple taus ----------
        # taus = np.linspace(0.01, 0.10, 10)  # 1–10 cm
        # curve = []
        
        # #ds_gt = _random_downsample(P_gt, max_n=100000, seed=42)

        # for t in taus:
        #     r = chamfer_and_fscore(ds_gt, P_pr_aligned, tau=t, z_window=(0.0,20))
        #     print(r)
        #     curve.append(r)

        # # extract arrays for plotting / summary
        # f1s = [r["fscore"] for r in curve]
        # prs = [r["precision"] for r in curve]
        # rcs = [r["recall"] for r in curve]

        # # AUC over τ (robust summary)
        # auc_f1 = np.trapz(f1s, taus) / (taus[-1] - taus[0])

        # # store per-scene summary + AUC
        # rows.append({
        #     "name": gt_npy.stem.replace("_ijk", ""),
        #     "f1_auc": float(auc_f1),
        #     "best_f1": float(max(f1s)),
        #     "best_tau": float(taus[int(np.argmax(f1s))]),
        #     "mean_chamfer_L2_sq": float(np.mean([r["chamfer_L2_sq"] for r in curve])),
        #     "nP": int(curve[0]["nP"]),
        #     "nQ": int(curve[0]["nQ"]),
        # })

        res = chamfer_and_fscore(ds_gt, P_pr_aligned,tau=tau, z_window=(0.2,20))  # (pred vs GT)
        
        diag = nn_diagnostics(ds_gt, P_pr_aligned, tau=tau)
  
        rows.append({
            "name": gt_npy.stem.replace("_ijk", ""),
            **res
        })

    
    # print(f"\nEvaluated {len(rows)} pairs (F1–τ curve)")
    # print(f"{'scene':25s}  {'AUC_F1':>8}  {'BestF1':>7}  {'τ*':>5}")
    # for r in rows:
    #     print(f"{r['name'][:25]:25s}  {r['f1_auc']:8.3f}  {r['best_f1']:7.3f}  {r['best_tau']:5.3f}")

    # # Macro averages
    # f1_auc_mean = np.mean([r["f1_auc"] for r in rows])
    # best_f1_mean = np.mean([r["best_f1"] for r in rows])
    # best_tau_mean = np.mean([r["best_tau"] for r in rows])
    # mean_chamfer = np.mean([r["mean_chamfer_L2_sq"] for r in rows])

    # print(f"\nAverages:  AUC_F1={f1_auc_mean:.3f},  mean BestF1={best_f1_mean:.3f}, mean tau={best_tau_mean:.3f}, mean chamfer={mean_chamfer:.3f}")


    #Print a tidy summary
    print(f"\nEvaluated {len(rows)} pairs @ tau={tau} m")
    print(f"{'scene':25s}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'Chamfer^2':>11}  {'nP':>6}  {'nQ':>6}")
    for r in rows:
        print(f"{r['name'][:25]:25s}  {r['fscore']:7.3f}  {r['precision']:7.3f}  {r['recall']:7.3f}  {r['chamfer_L2_sq']:11.4f}  {r['nP']:6d}  {r['nQ']:6d}")
       
    # Macro averages
    f1_mean = float(np.mean([r["fscore"] for r in rows])) if rows else float('nan')
    pr_mean = float(np.mean([r["precision"] for r in rows])) if rows else float('nan')
    rc_mean = float(np.mean([r["recall"] for r in rows])) if rows else float('nan')
    ch_mean = float(np.mean([r["chamfer_L2_sq"] for r in rows])) if rows else float('nan')
    print("\nAverages:")
    print(f"F1={f1_mean:.3f}, Precision={pr_mean:.3f}, Recall={rc_mean:.3f}, Chamfer^2={ch_mean:.4f}")

    # # Optional CSV
    # if save_csv:
    #     import csv
    #     with open(save_csv, "w", newline="") as f:
    #         w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    #         w.writeheader()
    #         w.writerows(rows)
    #     print(f"\nSaved per-scene metrics → {save_csv}")

    return rows

# ---------- example ----------
if __name__ == "__main__":
    # change these to your folders
    GT_DIR = "bedroom_habitat"
    PR_DIR = "bedroom_habitat_dust3r_fast_2"
    evaluate_voxel_dirs(GT_DIR, PR_DIR, tau=0.1, save_csv=None)
