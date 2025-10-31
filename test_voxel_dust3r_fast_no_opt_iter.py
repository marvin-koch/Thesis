import torch
import os
import os.path as osp
import sys

import tempfile
import matplotlib.pyplot as pl
import copy
import pow3r2.tools.path_to_dust3r

from dust3r.utils.device import todevice, to_numpy
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images as li
from dust3r.utils.image import rgb

from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo

from dust3r.cloud_opt.commons import i_j_ij, compute_edge_scores, edge_str


import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from voxel.covisibility import *
from voxel.viz_utils import *
from preprocess_images.filter_images import changed_images
import os, shutil, json


GA_CACHE = {
    "Twc": None,          # (N,4,4) cam->world from last full/partial run
    "K": None,            # (N,3,3)
    "pp": None,           # (N,2) principal points (optional)
    "depth": None,        # list of depth maps (HxW) used by optimizer (optional)
    "edges": None,        # list of ordered (i,j) used last time
    "per_edge": None,     # dict keyed by (i,j): {"view1":..., "view2":..., "pred1":..., "pred2":..., "conf1":..., "conf2":...}
    "anchor": 0,          # which pose we fix to pin world frame
    "pw_poses": None,
    "pw_adaptors": None
}


def to_torch(x, device="cuda"):
    """Convert to PyTorch tensor if not already."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.float32)
    else:
        return torch.tensor(x, device=device, dtype=torch.float32)
    
def _clone(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

def ensure_dir_for_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ------------------- helpers -------------------
def _reindex_local(imgs):
    """Ensure local idx = 0..M-1. (We rely on imgs[k]['gid'] to map to global)."""
    for k, d in enumerate(imgs):
        d["idx"] = k
        d["instance"] = str(k)
    return imgs

def _require_gids(imgs):
    for d in imgs:
        if "gid" not in d:
            raise ValueError("Each image dict must carry a stable global id in d['gid'] (int).")

def _find_local_index_by_gid(imgs, gid):
    for d in imgs:
        if d["gid"] == gid:
            return d["idx"]
    return None

def _sanitize_for_inference(imgs):
    cleaned = []
    for d in imgs:
        dd = {
            "img": d["img"],                    # (1,3,H,W) tensor
            "true_shape": d["true_shape"],      # np.int32[[H,W]]
            "idx": d["idx"],                    # int
            "instance": d.get("instance", str(d["idx"])),
        }
        cleaned.append(dd)
    return cleaned

def _apply_se3_to_pointmap(P, T):
    # P: (H,W,3) float; T: (4,4)
    H, W = P.shape[:2]
    X = P.reshape(-1, 3)
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    Xh = np.concatenate([X, ones], axis=1)                  # (N,4)
    Xw = (Xh @ T.T)[:, :3]                                   # (N,3)
    return Xw.reshape(H, W, 3)

def _median_scale_from_depths(depth_ref, depth_cur):
    # depth_ref, depth_cur: (H,W) positive floats
    m = np.isfinite(depth_ref) & np.isfinite(depth_cur) & (depth_ref > 0) & (depth_cur > 0)
    if m.sum() < 100:   # too few pixels -> fall back to 1.0
        return 1.0
    r = (depth_ref[m] / depth_cur[m])
    return np.median(r)

def _sim3_about_center_matrix(C, s):
    S = np.eye(4, dtype=np.float32)
    S[:3, :3] *= s
    S[:3, 3] = (1.0 - s) * C  # ensures X' = s X + (1-s) C
    return S

# Helpers you provided (already defined above in your code)
def unproject_depth(depth_hw, K_3x3):
    H,W = depth_hw.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pp = K_3x3[:2,2]         # (cx, cy)
    f  = K_3x3[0,0]          # fx=fy
    X = (xs - pp[0]) / f * depth_hw
    Y = (ys - pp[1]) / f * depth_hw
    Z = depth_hw
    P_cam = np.stack([X,Y,Z], axis=-1).astype(np.float32)     # (H,W,3)
    return P_cam

def cam_to_world(P_cam_hw3, Twc_4x4):
    H,W,_ = P_cam_hw3.shape
    X = P_cam_hw3.reshape(-1,3)
    Xh = np.concatenate([X, np.ones((X.shape[0],1), np.float32)], 1)
    Xw = (Xh @ Twc_4x4.T)[:, :3]
    return Xw.reshape(H,W,3)

def make_edge_from_image_cache(i, j, depth_list, K_list, Twc_list, conf_list=None):
    Di, Dj = depth_list[i], depth_list[j]
    Ki, Kj = K_list[i],   K_list[j]
    Tiw, Tjw = Twc_list[i], Twc_list[j]

    Pi_world = cam_to_world(unproject_depth(Di, Ki), Tiw)   # (H,W,3)
    Pj_world = cam_to_world(unproject_depth(Dj, Kj), Tjw)   # (H,W,3)

    conf_i = conf_list[i] if conf_list is not None else (np.ones(Di.shape, np.float32))
    conf_j = conf_list[j] if conf_list is not None else (np.ones(Dj.shape, np.float32))

    view1 = {"idx": i, "true_shape": np.int32([[Di.shape[0], Di.shape[1]]]), "img": np.int32([[Di.shape[0], Di.shape[1]]])}
    view2 = {"idx": j, "true_shape": np.int32([[Dj.shape[0], Dj.shape[1]]]), "img": np.int32([[Dj.shape[0], Dj.shape[1]]])}
    pred1 = {"pts3d": Pi_world,               "conf": conf_i}
    pred2 = {"pts3d_in_other_view": Pj_world, "conf": conf_j}
    return view1, view2, pred1, pred2



# def make_edge_from_image_cache(i, j, depth_list, K_list, Twc_list, conf_list=None):
#     Di, Dj = depth_list[i], depth_list[j]

#     # Replace heavy computations with dummy data (ones of matching shape)
#     Pi_world = np.ones((*Di.shape, 3), dtype=np.float32)
#     Pj_world = np.ones((*Dj.shape, 3), dtype=np.float32)

#     conf_i = np.ones(Di.shape, np.float32)
#     conf_j = np.ones(Dj.shape, np.float32)

#     view1 = {"idx": i, "true_shape": np.int32([[Di.shape[0], Di.shape[1]]]), "img": np.int32([[Di.shape[0], Di.shape[1]]])}
#     view2 = {"idx": j, "true_shape": np.int32([[Dj.shape[0], Dj.shape[1]]]), "img": np.int32([[Dj.shape[0], Dj.shape[1]]])}
#     pred1 = {"pts3d": Pi_world,               "conf": conf_i}
#     pred2 = {"pts3d_in_other_view": Pj_world, "conf": conf_j}

#     return view1, view2, pred1, pred2


# Pack into a DUSt3R-like output dict
def pack_edges(edge_list, device):
    view1 = {"idx":[], "true_shape":[], "img": []}
    view2 = {"idx":[], "true_shape":[], "img": []}
    pred1 = {"pts3d":[], "conf":[]}
    pred2 = {"pts3d_in_other_view":[], "conf":[]}
    for v1,v2,p1,p2 in edge_list:
        view1["idx"].append(v1["idx"])
        view1["true_shape"].append(v1["true_shape"])
        view2["idx"].append(v2["idx"])
        view2["true_shape"].append(v2["true_shape"])
        view1["img"].append(v1["img"])
        view2["img"].append(v2["img"])

        pred1["pts3d"].append(torch.as_tensor(p1["pts3d"], device=device))
        pred1["conf"].append(torch.as_tensor(p1["conf"],  device=device))
        pred2["pts3d_in_other_view"].append(torch.as_tensor(p2["pts3d_in_other_view"], device=device))
        pred2["conf"].append(torch.as_tensor(p2["conf"], device=device))
    return {"view1": view1, "view2": view2, "pred1": pred1, "pred2": pred2}



def robust_scores(edge_scores, squash=False):
    # edge_scores: dict[(i,j)] -> float
    vals = np.array(list(edge_scores.values()), dtype=np.float32)
    med  = np.median(vals)
    mad  = np.median(np.abs(vals - med)) + 1e-8
    zrob = (vals - med) / (1.4826 * mad)  # 1.4826: MAD -> std for Gaussian

    if squash:
        # map to (0,1), centered at 0 with slope=1
        w = 1 / (1 + np.exp(-zrob))
        return {k: float(v) for k, v in zip(edge_scores.keys(), w)}
    else:
        return {k: float(v) for k, v in zip(edge_scores.keys(), zrob)}


# Round-robin pointers are implicit via deque rotation.
def _good_edge(e, tau):
    k = i_j_ij(e)[1]
    return GA_CACHE["edge_scores"][k] >= tau  # (no age)

def _pick_from_list(lst, ptr, tau):
    n = len(lst)
    for ofs in range(n):
        e = lst[(ptr + ofs) % n]
        if _good_edge(e, tau):
            # advance pointer to the element AFTER the one we used
            return e, (ptr + ofs + 1) % n
    return None, ptr  # nothing passes the threshold

def pick_for_image_mst(i, tau_mst, tau_extra):
    # 1) try MST
    lst = GA_CACHE["incident_mst"].get(i)
    return lst[0]
    # if lst:
    #     ptr = 0
    #     e, ptr_new = _pick_from_list(lst, ptr, tau_mst)
    #     if e is not None:
    #         return e
    
def pick_one_for_image_extra(i, tau_mst, tau_extra):

    # 2) fallback to non-MST
    lst = GA_CACHE["incident_extra"].get(i)
    if lst:
        ptr = 0
        e, ptr_new = _pick_from_list(lst, ptr, tau_extra)
        if e is not None:
            return e
    return None

def schedule_pairs(changed_gids, local2gid, pairs, budget,
                tau_mst=0.0, tau_extra=0.4, max_count=4, refresh_one_stale_mst=True):
    """
    returns: pairs_to_run, run_mask (aligned with `pairs`), edge_lut (indices into `pairs`)
    """
    # which local image indices changed?
    changed_local = {i for i, gid in local2gid.items() if gid in changed_gids}

    run_set = set()     # edges as (i,j) with this direction
    edge_lut = []       # indices into `pairs` list for what we run

    # ≤1 edge per changed image
    for i in changed_local:
        if len(run_set) > budget: break
        
        e = pick_for_image_mst(i, tau_mst, tau_extra)
        if e is None: 
            continue
        run_set.add(e)
        
    # for i in changed_local:
    #     if len(run_set) > budget: break
    #     e = pick_one_for_image_extra(i, tau_mst, tau_extra)
    #     if e is None: 
    #         continue
    #     run_set.add(e)


        # # Optional: refresh the single stalest MST edge globally (drift control)
        # if refresh_one_stale_mst and len(run_set) < budget:
        #     mst_set = set(canonical(i,j) for i,j in GA_CACHE["mst_edges"])
        #     # pick the MST edge with largest counter
        #     def age_of(e_can):
        #         k = i_j_ij(e_can)[1]
        #         return GA_CACHE["edge_update_counter"][k]
        #     stale = None
        #     if mst_set:
        #         stale = max(mst_set, key=age_of)
        #     if stale is not None:
        #         # choose a direction that matches `pairs` if possible; default to (i,j)
        #         run_set.add(stale)

        # reset counters for edges we run; increment otherwise
    run_keys = set()
    for e in run_set:
        k = i_j_ij(e)[1]
        run_keys.add(k)
        GA_CACHE["edge_update_counter"][k] = 0

    for (i,j_k), age in list(GA_CACHE["edge_update_counter"].items()):
        if (i,j_k) not in GA_CACHE["edge_update_counter"]:
            continue
    for k in GA_CACHE["edge_update_counter"]:
        if k not in run_keys:
            GA_CACHE["edge_update_counter"][k] += 1

    # Build outputs aligned with `pairs`
    pairs_to_run = []
    run_mask = []
    for idx, (vi, vj) in enumerate(pairs):
        i, j = vi["idx"], vj["idx"]
        # if direction (i,j) not in run_set, accept (j,i) too
        want = (i, j) in run_set #or (j, i) in run_set
        run_mask.append(want)
        if want:
            pairs_to_run.append((vi, vj))
            edge_lut.append(idx)

    return pairs_to_run, run_mask, edge_lut

import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
def canonical(u, v):
        return (u, v) if u <= v else (v, u)

def build_mst_from_edge_scores(edge_scores):
    """
    edge_scores: dict[(i,j)] -> float   (higher = better)
                may contain (i,j) and/or (j,i)
    Returns: list of undirected edges in the MST as (i,j) with i<j
    """
    # 1) collect nodes
    nodes = set()
    for (i, j) in edge_scores.keys():
        nodes.add(i); nodes.add(j)
    n = max(nodes) + 1

    # 2) make undirected weights: take the best score per undirected edge
    und = {}
    for (i,j), s in edge_scores.items():
        ij = canonical(i,j)
        und[ij] = max(und.get(ij, -np.inf), s)

    # 3) build sparse matrix of "costs" = -score (so MST == maximum spanning tree on scores)
    X = sp.dok_array((n, n), dtype=np.float32)
    for (i,j), s in und.items():
        if i == j: continue
        c = -float(s)
        X[i, j] = c
        X[j, i] = c

    # 4) MST (on costs)
    T = minimum_spanning_tree(X).tocoo()

    # 5) extract edges
    mst_edges = [(int(i), int(j)) if i < j else (int(j), int(i))
                for i, j in zip(T.row, T.col)]
    mst_edges = sorted(set(mst_edges))
    return mst_edges
    
# ------------------- main -------------------
def get_reconstructed_scene(
    itr, outdir, imgs, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
    as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    scenegraph_type, winsize, refid, changed_gids=None, tau=0.45
):
    """
    Iter 0: pass ALL images, each with d['gid'] = its global index (0..N-1).
            We solve globally and cache Twc/K/pp and the anchor gid (+ per-image depths).
    Iter >0: pass ONLY CHANGED images (subset), each with d['gid'] pointing to the original image.
             Subset must include the anchor image (gid == GA_CACHE['anchor']).
             We warm-start from cached Twc/K/pp/depth for these gids, seed per-edge vars, and refine.
    """
    # ---- helpers expected: _reindex_local, _require_gids, _sanitize_for_inference, _find_local_index_by_gid
    imgs = _reindex_local(imgs)
    _require_gids(imgs)
    imgs_clean = _sanitize_for_inference(imgs)  # strip custom fields for inference()

    # scene graph
    if scenegraph_type == "swin":
        scenegraph = f"swin-{winsize}"
    elif scenegraph_type == "oneref":
        scenegraph = "oneref"  # we’ll set the local anchor below
    else:
        scenegraph = scenegraph_type

    # ---------- iter 0: full run ----------
    if itr == 0:
        GA_CACHE["anchor"] = int(refid)  # choose the anchor by gid

        if scenegraph.startswith("oneref"):
            anchor_local = _find_local_index_by_gid(imgs, GA_CACHE["anchor"])
            if anchor_local is None:
                raise ValueError(f"Anchor gid {GA_CACHE['anchor']} not present in imgs at iter 0.")
            sg = f"oneref-{anchor_local}"
        else:
            sg = scenegraph

        # Stronger constraints: symmetrize=True helps stability
        pairs = make_pairs(imgs_clean, scene_graph='complete', prefilter=None, symmetrize=True)

        # pairs = [(imgs_clean[0], imgs_clean[1]), (imgs_clean[0], imgs_clean[2])]

        output = inference(pairs, model, device, batch_size=16, verbose=not silent)

        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            _ = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=1e-2)

        # Cache cameras (full set)
        Twc = scene.get_im_poses().detach().cpu().numpy()   # (N,4,4)
        K   = scene.get_intrinsics().detach().cpu().numpy() # (N,3,3)
        try:
            pp = scene.get_principal_points().detach().cpu().numpy()
        except Exception:
            pp = None

        # NEW: cache per-image depths (raw, NOT normalized)
        depth_list = [d.detach().cpu().numpy() if torch.is_tensor(d) else d for d in scene.get_depthmaps()]

        GA_CACHE["Twc"]  = Twc
        GA_CACHE["K"]    = K
        GA_CACHE["pp"]   = pp
        GA_CACHE["depth"] = depth_list  # NEW
        GA_CACHE["pw_poses"] = scene.pw_poses.detach().cpu().numpy()    
        GA_CACHE["pw_adaptors"] = scene.pw_adaptors.detach().cpu().numpy() 
        GA_CACHE["conf_i"] = scene.conf_i
        GA_CACHE["conf_j"] = scene.conf_j

        GA_CACHE["base_scale"] = scene.base_scale
        GA_CACHE["pw_break"] = scene.pw_break
        GA_CACHE["norm_pw"] = scene.norm_pw_scale

        

        edges = scene.edges
        conf_i = scene.conf_i
        conf_j = scene.conf_j

        edge_scores = compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j)
        
        GA_CACHE["edge_scores"] = robust_scores(edge_scores, squash=True)
        GA_CACHE["edge_update_counter"] = {edge: 0 for edge in GA_CACHE["edge_scores"]}
        
        # Outputs (stack as float arrays if all same size)
        pts       = [p.detach().cpu().numpy() if torch.is_tensor(p) else p for p in scene.get_pts3d()]
        confs_raw = [c.detach().cpu().numpy() if torch.is_tensor(c) else c for c in scene.im_conf]
        Tcw       = [np.linalg.inv(Twc[i]) for i in range(len(Twc))]

        # Stack safely if same (H,W)
        h0, w0 = pts[0].shape[:2]
        if all(p.shape[:2] == (h0, w0) for p in pts):
            P = np.stack(pts, axis=0).astype(np.float32)
            C = np.stack(confs_raw, axis=0).astype(np.float32)
        else:
            P = np.array(pts, dtype=object)        # fallback
            C = np.array(confs_raw, dtype=object)  # fallback


        predictions = {
            "world_points":       P,
            "world_points_conf":  C,
            "images":             np.array(scene.imgs, dtype=object),
            "extrinsic":          np.stack(Tcw, axis=0),   # (N,4,4) world->camera
            "intrinsic_K":        K,
            "gids":               np.array([d["gid"] for d in imgs]),
        }
        return predictions

    # ---------- iter > 0: subset run ----------
    if GA_CACHE.get("Twc") is None or GA_CACHE.get("K") is None:
        raise RuntimeError("GA_CACHE is empty—run itr=0 with the full set first.")

    # Require anchor in subset
    anchor_gid = GA_CACHE["anchor"]
    anchor_local = _find_local_index_by_gid(imgs, anchor_gid)
    if anchor_local is None:
        raise ValueError(
            f"Subset must include the anchor image (gid={anchor_gid}). "
            "Add that image to imgs for iter > 0 so we can pin the world frame."
        )

    # Build star (consider union with logwin-3 if weak overlap)
    sg = f"oneref-{anchor_local}" if scenegraph.startswith("oneref") else scenegraph
    
    pairs = make_pairs(imgs_clean, scene_graph='complete', prefilter=None, symmetrize=True)
    
    

    # ---------- MIXED EDGE CONSTRUCTION ----------
    depth_all = GA_CACHE["depth"]
    K_all     = GA_CACHE["K"]
    Twc_all   = GA_CACHE["Twc"]
    conf_i = GA_CACHE["conf_i"]
    conf_j = GA_CACHE["conf_j"]

    # Map local idx -> global gid for this subset
    local2gid = {d["idx"]: d["gid"] for d in imgs}

    # Decide which edges need fresh DUSt3R vs can be synthesized
    run_mask = []            # per edge in 'pairs'
    pairs_to_run = []        # the subset we actually send to inference()
    edge_lut = []            # back-map from compact run-list index -> edge index in 'pairs'

    start = time.time()


    if itr == 1:
        
        mst_edges = build_mst_from_edge_scores(GA_CACHE["edge_scores"])
        
        GA_CACHE["mst_edges"] = mst_edges

        from collections import defaultdict, deque

        # Build per-image lists of incident edges, split into MST vs non-MST.
        incident_mst   = defaultdict(list)   # i -> list of (i,j) in MST
        incident_extra = defaultdict(list)   # i -> list of (i,j) not in MST

        for (vi, vj) in pairs:
            i = vi["idx"]; j = vj["idx"]

            if (i, j) in mst_edges:
                incident_mst[i].append((i, j))
                incident_mst[j].append((i, j))
            else:
                incident_extra[i].append((i, j))
                incident_extra[j].append((i, j))

        # Order each list by a stable priority (e.g., cached score descending)
        def sort_by_score(lst):
            return sorted(lst, key=lambda e: GA_CACHE["edge_scores"][i_j_ij(e)[1]], reverse=True)

        GA_CACHE["incident_mst"]   = {i: deque(sort_by_score(v)) for i, v in incident_mst.items()}
        GA_CACHE["incident_extra"] = {i: deque(sort_by_score(v)) for i, v in incident_extra.items()}
        
        

    B = max(1, len(imgs))  # hard cap
    pairs_to_run, run_mask, edge_lut = schedule_pairs(
        changed_gids=changed_gids,
        local2gid=local2gid,
        pairs=pairs,          # full pair list (vi, vj)
        budget=B,
        tau_mst=0.0,          # be lenient on MST edges
        tau_extra=0.5,        # stricter on non-MST
        max_count=4,
        refresh_one_stale_mst=True
    )
    
    print("running pairs")
    
    pairs_to_run_tuples = [(vi["idx"], vj["idx"]) for (vi, vj) in pairs_to_run]
    for (vi, vj) in pairs_to_run:
        print(vi["idx"], vj["idx"])

 
    # Run the network ONLY for needed edges
    
    start = time.time()

    if len(pairs_to_run):
        out_delta = inference(pairs_to_run, model, device, batch_size=16, verbose=not silent)
        
    else:
        # fabricate an empty structure with lists
        out_delta = {"view1":{"idx":[],"true_shape":[]},
                    "view2":{"idx":[],"true_shape":[]},
                    "pred1":{"pts3d":[],"conf":[]},
                    "pred2":{"pts3d_in_other_view":[],"conf":[]}}

    end = time.time()
    
    print("ONLY INFERENCE", end - start)




    # # Extract per-edge outputs in the SAME order as 'pairs'
    # mixed_edges = []
    # run_k = 0
    # for e, (vi, vj) in enumerate(pairs):
    #     i = vi["idx"]; j = vj["idx"]
    #     gi = local2gid[i]; gj = local2gid[j]
    #     if run_mask[e]:
    #         # take kth slice from out_delta
    #         v1 = {"idx": i, "true_shape": out_delta["view1"]["true_shape"][run_k], "img": out_delta["view1"]["img"][run_k]}
    #         v2 = {"idx": j, "true_shape": out_delta["view2"]["true_shape"][run_k], "img": out_delta["view2"]["img"][run_k]}
    #         p1 = {"pts3d": out_delta["pred1"]["pts3d"][run_k],
    #             "conf":  out_delta["pred1"]["conf"][run_k]}
    #         p2 = {"pts3d_in_other_view": out_delta["pred2"]["pts3d_in_other_view"][run_k],
    #             "conf":                out_delta["pred2"]["conf"][run_k]}
    #         mixed_edges.append((v1,v2,p1,p2))
    #         run_k += 1
    #     else:
    #         # synthesize from cache (use GLOBAL ids to fetch per-image data)
    #         v1,v2,p1,p2 = make_edge_from_image_cache(
    #             gi, gj, depth_all, K_all, Twc_all, conf_list=None
    #         )
    #         # fix view indices to LOCAL indices expected by the optimizer
    #         v1["idx"] = i
    #         v2["idx"] = j
    #         mixed_edges.append((v1,v2,p1,p2))
    
  
    # E = len(pairs)
    # ones_dtype = out_delta["pred1"]["pts3d"].dtype if torch.is_tensor(out_delta["pred1"]["pts3d"]) else torch.float32

    # # Preallocate lists at the correct length (keeps indexing stable)
    # view1 = {"idx": [None]*E, "true_shape": [None]*E, "img": [None]*E}
    # view2 = {"idx": [None]*E, "true_shape": [None]*E, "img": [None]*E}
    # pred1 = {"pts3d": [None]*E, "conf": [None]*E}
    # pred2 = {"pts3d_in_other_view": [None]*E, "conf": [None]*E}

    # # Normalize mask and compute k for each TRUE edge via cumsum
    # run_mask_t = torch.as_tensor(run_mask, dtype=torch.bool)
    # run_idx = (run_mask_t.cumsum(0) - 1)  # tensor of same length; valid only where mask True

    # # Hoist out_delta references
    # od_v1_ts = out_delta["view1"]["true_shape"]
    # od_v1_im = out_delta["view1"]["img"]
    # od_v2_ts = out_delta["view2"]["true_shape"]
    # od_v2_im = out_delta["view2"]["img"]
    # od_p1_p  = out_delta["pred1"]["pts3d"]
    # od_p1_c  = out_delta["pred1"]["conf"]
    # od_p2_p  = out_delta["pred2"]["pts3d_in_other_view"]
    # od_p2_c  = out_delta["pred2"]["conf"]

    # # Single pass
    # for e, (vi, vj) in enumerate(pairs):
    #     i = vi["idx"]; j = vj["idx"]
    #     view1["idx"][e] = i
    #     view2["idx"][e] = j

    #     if run_mask_t[e]:
    #         k = int(run_idx[e].item())
    #         # Copy metadata straight from out_delta
    #         view1["true_shape"][e] = od_v1_ts[k]
    #         view1["img"][e]        = od_v1_im[k]
    #         view2["true_shape"][e] = od_v2_ts[k]
    #         view2["img"][e]        = od_v2_im[k]

    #         # Ensure tensors are on the requested device
    #         pred1["pts3d"][e] = torch.as_tensor(od_p1_p[k], device=device)
    #         pred1["conf"][e]  = torch.as_tensor(od_p1_c[k], device=device)
    #         pred2["pts3d_in_other_view"][e] = torch.as_tensor(od_p2_p[k], device=device)
    #         pred2["conf"][e]                = torch.as_tensor(od_p2_c[k], device=device)

    #     else:
    #         # Make "ones" placeholders for non-run edges, sized from the global images
    #         gi = local2gid[i]; gj = local2gid[j]
    #         Di = depth_all[gi]; Dj = depth_all[gj]
    #         Hi, Wi = int(Di.shape[0]), int(Di.shape[1])
    #         Hj, Wj = int(Dj.shape[0]), int(Dj.shape[1])

    #         # Match your earlier dict shape convention (1x2 int32 arrays)
    #         ts_i = np.int32([[Hi, Wi]])
    #         ts_j = np.int32([[Hj, Wj]])
    #         view1["true_shape"][e] = ts_i
    #         view1["img"][e]        = ts_i
    #         view2["true_shape"][e] = ts_j
    #         view2["img"][e]        = ts_j

    #         # Ones tensors with correct shapes
    #         pred1["pts3d"][e] = torch.ones((Hi, Wi, 3), dtype=ones_dtype, device=device)
    #         pred1["conf"][e]  = torch.ones((Hi, Wi),     dtype=ones_dtype, device=device)
    #         pred2["pts3d_in_other_view"][e] = torch.ones((Hj, Wj, 3), dtype=ones_dtype, device=device)
    #         pred2["conf"][e]                = torch.ones((Hj, Wj),     dtype=ones_dtype, device=device)

    # output = {"view1": view1, "view2": view2, "pred1": pred1, "pred2": pred2}


    start = time.time()




    E = len(pairs)
    use_cache_for_nonrun = False
    ones_dtype = out_delta["pred1"]["pts3d"].dtype if torch.is_tensor(out_delta["pred1"]["pts3d"]) else torch.float32

    # Preallocate lists for stable ordering
    view1 = {"idx": [None]*E, "true_shape": [None]*E, "img": [None]*E}
    view2 = {"idx": [None]*E, "true_shape": [None]*E, "img": [None]*E}
    pred1 = {"pts3d": [None]*E, "conf": [None]*E}
    pred2 = {"pts3d_in_other_view": [None]*E, "conf": [None]*E}

    # Build per-edge k index for run edges
    run_mask_t = torch.as_tensor(run_mask, dtype=torch.bool)
    run_idx = (run_mask_t.cumsum(0) - 1)

    # Hoist out_delta refs
    od_v1_ts = out_delta["view1"]["true_shape"]
    od_v1_im = out_delta["view1"]["img"]
    od_v2_ts = out_delta["view2"]["true_shape"]
    od_v2_im = out_delta["view2"]["img"]
    od_p1_p  = out_delta["pred1"]["pts3d"]
    od_p1_c  = out_delta["pred1"]["conf"]
    od_p2_p  = out_delta["pred2"]["pts3d_in_other_view"]
    od_p2_c  = out_delta["pred2"]["conf"]

    # Optional memo for cache path to avoid recomputing repeated (gi, gj)
    cache_memo = {}

    for e, (vi, vj) in enumerate(pairs):
        i = vi["idx"]; j = vj["idx"]
        view1["idx"][e] = i
        view2["idx"][e] = j

        if run_mask_t[e]:
            k = int(run_idx[e].item())

            view1["true_shape"][e] = od_v1_ts[k]
            view1["img"][e]        = od_v1_im[k]
            view2["true_shape"][e] = od_v2_ts[k]
            view2["img"][e]        = od_v2_im[k]

            pred1["pts3d"][e] = torch.as_tensor(od_p1_p[k], device=device)
            pred1["conf"][e]  = torch.as_tensor(od_p1_c[k], device=device)
            pred2["pts3d_in_other_view"][e] = torch.as_tensor(od_p2_p[k], device=device)
            pred2["conf"][e]                = torch.as_tensor(od_p2_c[k], device=device)

        else:
            gi = local2gid[i]; gj = local2gid[j]

            if use_cache_for_nonrun and make_edge_from_image_cache is not None:
                key = (gi, gj)
                if key in cache_memo:
                    v1c, v2c, p1c, p2c = cache_memo[key]
                else:
                    v1c, v2c, p1c, p2c = make_edge_from_image_cache(
                        gi, gj, depth_all, K_all, Twc_all, conf_list=None
                    )
                    cache_memo[key] = (v1c, v2c, p1c, p2c)

                # Keep LOCAL indices in packed output, but use cached tensors
                Hi, Wi = int(v1c["true_shape"][0,0]), int(v1c["true_shape"][0,1])
                Hj, Wj = int(v2c["true_shape"][0,0]), int(v2c["true_shape"][0,1])

                # metadata (use the same 1x2 int32 arrays convention)
                view1["true_shape"][e] = np.int32([[Hi, Wi]])
                view1["img"][e]        = np.int32([[Hi, Wi]])
                view2["true_shape"][e] = np.int32([[Hj, Wj]])
                view2["img"][e]        = np.int32([[Hj, Wj]])

                # tensors
                pred1["pts3d"][e] = torch.as_tensor(p1c["pts3d"], device=device)
                pred1["conf"][e]  = torch.as_tensor(p1c["conf"],  device=device)
                pred2["pts3d_in_other_view"][e] = torch.as_tensor(p2c["pts3d_in_other_view"], device=device)
                pred2["conf"][e]                = torch.as_tensor(p2c["conf"], device=device)

            else:
                # Fast placeholder path: ones with correct shapes from depth_all
                Di = depth_all[gi]; Dj = depth_all[gj]
                Hi, Wi = int(Di.shape[0]), int(Di.shape[1])
                Hj, Wj = int(Dj.shape[0]), int(Dj.shape[1])

                ts_i = np.int32([[Hi, Wi]])
                ts_j = np.int32([[Hj, Wj]])
                view1["true_shape"][e] = ts_i
                view1["img"][e]        = ts_i
                view2["true_shape"][e] = ts_j
                view2["img"][e]        = ts_j

                pred1["pts3d"][e] = torch.ones((Hi, Wi, 3), dtype=ones_dtype, device=device)
                pred1["conf"][e]  = torch.ones((Hi, Wi),     dtype=ones_dtype, device=device)
                pred2["pts3d_in_other_view"][e] = torch.ones((Hj, Wj, 3), dtype=ones_dtype, device=device)
                pred2["conf"][e]                = torch.ones((Hj, Wj),     dtype=ones_dtype, device=device)

    output = {"view1": view1, "view2": view2, "pred1": pred1, "pred2": pred2}
    
    # output = pack_edges(mixed_edges, device)

    # packed, remap = compact_packed_edges(out_delta)

      
    
    # output = inference(pairs, model, device, batch_size=1, verbose=not silent)


    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

    # Warm-start from cache for this subset
    Twc0, K0, pp0 = GA_CACHE["Twc"], GA_CACHE["K"], GA_CACHE["pp"]
    depth0_all = GA_CACHE.get("depth")  # NEW
    M = len(imgs)
    
    # M = len(remap)
    # inv = {new: old for old, new in remap.items()}          # new_local -> old_local
    # gid_by_new = [imgs[inv[n]]['gid'] for n in range(M)]    # global ids in new-local order

    # Optionally, replace imgs with a remapped version so everything else stays consistent
    # imgs = [{'gid': gid, 'lid': n} for n, gid in enumerate(gid_by_new)]

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        # 1) poses (match new-local order)
        subset_Twc = [torch.tensor(Twc0[d['gid']], dtype=torch.float32) for d in imgs]
        scene.preset_pose(subset_Twc, pose_msk=None)

        # 2) intrinsics (match new-local order)
        subset_f = [float(K0[d['gid'], 0, 0]) for d in imgs]
        scene.preset_focal(subset_f, msk=None)

        if pp0 is not None:
            was_pp = scene.im_pp.requires_grad
            scene.im_pp.requires_grad_(True)
            subset_pp = [pp0[d['gid']] for d in imgs]
            scene.preset_principal_point(subset_pp, msk=None)
            scene.im_pp.requires_grad_(was_pp)

        # 4) freeze intrinsics during subset refine
        scene.im_focals.requires_grad_(False)
        scene.im_pp.requires_grad_(False)
        scene.im_depthmaps.requires_grad_(True)

        # 5) enable pose grads (preset_pose froze them)
        scene.im_poses.requires_grad_(False)
        scene.pw_poses.requires_grad_(True)
        scene.pw_adaptors.requires_grad_(True)

        with torch.no_grad():
            scene.pw_poses.copy_(torch.from_numpy(GA_CACHE["pw_poses"]).to(device))
            scene.pw_adaptors.copy_(torch.from_numpy(GA_CACHE["pw_adaptors"]).to(device))
     
        scene.base_scale   = GA_CACHE["base_scale"]
        scene.pw_break     = GA_CACHE["pw_break"]
        scene.norm_pw_scale = GA_CACHE["norm_pw"]
        # scene.conf_i = conf_i
        # scene.conf_j = conf_j

        # 6) NEW: seed per-edge latents using current poses (PnP-like), but no steps yet
        # try:
        #     _ = scene.compute_global_alignment(init='known', niter=0, schedule=schedule, lr=1e-2)
        # except Exception:
        #     print("Exception")
        #     # Fallback: a 0-iter 'mst' or a tiny run to seed edge vars
        #     _ = scene.compute_global_alignment(init='mst', niter=0)

        # print("pw_poses", scene.get_pw_poses())

        # if depth0_all is not None and len(depth0_all) > 0:
        #     for i, dct in enumerate(imgs):
        #         gi = dct['gid']
        #         di_np = depth0_all[gi]
        #         di_t  = torch.as_tensor(di_np)
        #         if i not in changed_gids:
        #             scene._set_depthmap(i, di_t, force=True)
                    
                    
        best_depthmaps = {}
        # init all pairwise poses
        for e, (i, j) in enumerate(scene.edges):
            i_j = edge_str(i, j)
            # remember if this is a good depthmap
            score = float(scene.conf_i[i_j].mean())
            s = scene.get_pw_scale()[e]
            if score > best_depthmaps.get(i, (0,))[0] and (i,j) in pairs_to_run_tuples:
                best_depthmaps[i] = score, i_j, s
      

        # init all image poses
        for n in range(scene.n_imgs):
            #assert known_poses_msk[n]
            # score, i_j, scale = best_depthmaps[n]

            item = best_depthmaps.get(n)
            if item is None:
                continue  # skip if not found
            score, i_j, scale = item

            depth = scene.pred_i[i_j][:, :, 2]
            scene._set_depthmap(n, depth * scale)

        # 7) short refine without re-init
        # _ = scene.compute_global_alignment(init=None, niter=niter/20, schedule=schedule, lr=1e-2) #5e-2


    # Read subset outputs
    Twc_sub = scene.get_im_poses().detach().cpu().numpy()    # (M,4,4)
    K_sub   = scene.get_intrinsics().detach().cpu().numpy()

    # # --- Align to original world via anchor (exact gauge match) ---
    # anchor_local = next(d['idx'] for d in imgs if d['gid'] == anchor_gid)
    # T_align = Twc0[anchor_gid] @ np.linalg.inv(Twc_sub[anchor_local])
    # Twc_sub = np.einsum('ab,sbc->sac', T_align, Twc_sub)

    # Points / conf (raw)
    pts  = [p.detach().cpu().numpy() if torch.is_tensor(p) else p for p in scene.get_pts3d()]
    conf = [c.detach().cpu().numpy() if torch.is_tensor(c) else c for c in scene.im_conf]
    scene_imgs = [img.detach().cpu().numpy() if torch.is_tensor(img) else img for img in scene.imgs]

    # 3) slice cameras/intrinsics to changed frames only
    Twc_sub = Twc_sub[changed_gids]                 # (Mchg,4,4) cam->world
    K_sub   = K_sub[changed_gids]                   # (Mchg,3,3)

    pts  = [pts[i]  for i in changed_gids]
    conf = [conf[i] for i in changed_gids]
    scene_imgs = [scene_imgs[i] for i in changed_gids]

    # --- SIM(3) scale snap (about anchor camera center) ---
    # get depths
    depth_ref = GA_CACHE["depth"][anchor_gid]                              # from iter-0 (cached)
    depth_cur = scene.get_depthmaps()[anchor_local].detach().cpu().numpy() # current subset

    # median depth ratio → global scale
    s = _median_scale_from_depths(depth_ref, depth_cur)

    # if s is NaN/inf or extreme, clamp/fallback
    if not np.isfinite(s) or s < 0.25 or s > 4.0:
        s = 1.0  # safe fallback; tune bounds as you prefer

    # similarity matrix about the (already aligned) anchor camera center
    C_anchor = Twc_sub[anchor_local][:3, 3].copy()       # after SE(3) snap, equals Twc0[anchor_gid][:3,3]
    S_C = _sim3_about_center_matrix(C_anchor, s)

    # apply to all subset cameras
    Twc_sub = np.einsum('ab,sbc->sac', S_C, Twc_sub)

    # ... after you computed Twc_sub (aligned cameras) and have T_align:
    # transform the *subset* 3D pointmaps into the original world frame
    
    
    pts = [_apply_se3_to_pointmap(p, S_C) for p in pts]


    err = np.linalg.norm(Twc0[anchor_gid][:3,3] - Twc_sub[anchor_local][:3,3])
    print(f"[snap] anchor translation error after align: {err:.6f}")  # should be ~0

   
    
    # If all same size, stack (nice for downstream)
    h0, w0 = pts[0].shape[:2]
    if all(p.shape[:2] == (h0, w0) for p in pts):
        P = np.stack(pts, axis=0).astype(np.float32)
        C = np.stack(conf, axis=0).astype(np.float32)
    else:
        P = np.array(pts, dtype=object)
        C = np.array(conf, dtype=object)

    Tcw_sub = [np.linalg.inv(k) for k in Twc_sub]

    # Write back ALIGNED cameras (+ keep cached depths as-is, or replace if you wish)
    
    # for i, dct in enumerate(imgs):
    #     # GA_CACHE["Twc"][dct["gid"]] = Twc_sub[i]
    #     # GA_CACHE["K"][dct["gid"]]   = K_sub[i]
    
    
    #     # Optional: refresh cached depth with the refined one
    #     refined_depth_i = scene.get_depthmaps()[i].detach().cpu().numpy()
    #     GA_CACHE["depth"][dct["gid"]] = refined_depth_i

    scene_imgs = [im["img"].detach().cpu().numpy() for im in imgs_clean if im["idx"] in changed_gids]
    
    end = time.time()

    print("global alignment", end - start)

    predictions = {
        "world_points":       P,
        "world_points_conf":  C,
        "images":             np.array(scene_imgs, dtype=object).squeeze(1),
        "extrinsic":          np.stack(Tcw_sub, axis=0),   # (M,4,4)
        "intrinsic_K":        K_sub,
        "gids":               np.array([d["gid"] for d in imgs]),
    }
    return predictions



sys.path.append("Thesis/")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
POINTS = "world_points"
CONF = "world_points_conf"
VIZ = False
threshold = 1.0     
coverage_threshold = 0.7       
z_clip_map = (-0.1, 0.3)   

R_w2m = np.array([[0, 0, -1],
                [-1, 0, 0],
                [0, -1, 0]], dtype=np.float32)

# R_w2m = np.array([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, -1]], dtype=np.float32)

t_w2m = np.zeros(3, dtype=np.float32)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
print("Loading DUST3R")


weights_path = "naver/" + "DUSt3R_ViTLarge_BaseDecoder_512_dpt"
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)

model.eval()
model = model.to(device)
# model = model.to(dtype)


voxel_size = 0.01
vox = TorchSparseVoxelGrid(
    origin_xyz=np.zeros(3, dtype=np.float32),
    params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
    device=device, dtype=torch.float32
)

save_root = "kitchen_habitat_dust3r_fast_2/"
os.makedirs(save_root, exist_ok=True)

# target_dir = "/Users/marvin/Documents/Thesis/vggt/examples/fishbowl1/"
# sub_dirs = ["images","images", "images1", "images2", "images3"]
# sub_dirs = ["00000000", "00000050","00000100", "00000150", "00000200", "00000250"]
# sub_dirs = ["00000000", "00000300",  "00000350", "00000400"]

target_dir = "../frames_bedroom/"  # folder that contains intrinsics.json and time_XXXXX/

sub_dirs = sorted([d for d in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, d))])


# sub_dirs = [d for d in os.listdir(target_dir) 
#             if os.path.isdir(os.path.join(target_dir, d))]

keyframes = []
changed_idx = []
last_save_paths = None  # global-ish tracker for previous iter’s artifact paths

for i, images in enumerate(sub_dirs):

    print(f"Iteration {i}")
   
    imgs = li(target_dir + images, size=512, verbose=False)
    
    image_tensors = torch.stack([d["img"] for d in imgs])

    image_tensors = []
    for d in imgs:
        t = d["img"]                        # (1,3,H,W), likely float in [0,1]
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]                        # -> (3,H,W)
        t = t.detach().cpu()
        if not t.dtype.is_floating_point:
            t = t.float()
        if t.max() > 1.0:                   # in case values are 0..255
            t = t / 255.0
        image_tensors.append(t.clamp(0,1))
        
    image_tensors = torch.stack(image_tensors, dim=0).to(device)


    if i < 1:
        
        start = time.time()

        predictions = get_reconstructed_scene(i, ".", imgs, model, device, False, 512, target_dir + images, "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0)
        
        keyframes = image_tensors.clone()
        
        end = time.time()
        length = end - start

        print("Running inference took", length, "seconds!")
        
        stacked_predictions = [predictions]

    else:
    
        start = time.time()

        changed_idx = changed_images(image_tensors, keyframes, thresh=0.000005)
        
        print(changed_idx)

        end = time.time()
        length = end - start
        
        if len(changed_idx) < 2:
            if last_save_paths is None:
                print("No previous outputs to clone (i==0 case). Running normally.")
            else:
                print("No images have changed; cloning previous outputs for this iter.")
                # Build this iter's destination paths
                dst_paths = {
                    "bev_png":  os.path.join(save_root, f"bev_{i}.png"),
                    "bev_npy":  os.path.join(save_root, f"bev_{i}_np.npy"),
                    "bev_meta": os.path.join(save_root, f"bev_{i}_meta.json"),
                    "vox_ply":  os.path.join(save_root, f"voxels{i}.ply"),
                    "vox_ijk":  os.path.join(save_root, f"voxels{i}_ijk.npy"),
                    "vox_meta": os.path.join(save_root, f"voxels{i}_meta.json"),
                }
                # Clone each previous artifact to the new iter name
                _clone(last_save_paths["bev_png"],  dst_paths["bev_png"])
                _clone(last_save_paths["bev_npy"],  dst_paths["bev_npy"])
                _clone(last_save_paths["bev_meta"], dst_paths["bev_meta"])
                _clone(last_save_paths["vox_ply"],  dst_paths["vox_ply"])
                _clone(last_save_paths["vox_ijk"],  dst_paths["vox_ijk"])
                _clone(last_save_paths["vox_meta"], dst_paths["vox_meta"])

                # Optional: write a tiny provenance note
                with open(os.path.join(save_root, f"iter_{i}_provenance.json"), "w") as f:
                    json.dump({"copied_from_iter": i-1,
                               "reason": "no image change"}, f)

                # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                vox.next_epoch()

                # Update last_save_paths to the newly written copies (so a run of no-op iters still works)
                last_save_paths = dst_paths
            continue

        print("Finding changed images took", length, "seconds!")

        changed_idx = [0] + [x for x in changed_idx if x != 0]
        
        index_map = {new: old for new, old in enumerate(changed_idx)}
                
        idx_t = torch.tensor(changed_idx, device=device, dtype=torch.long)
        keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
        
        
        print("final changed idx:", changed_idx)
        new_vggt_input = [[imgs[g] for g in changed_idx]]

        start = time.time()

        # stacked_predictions = []
        # for input_frames in [imgs]:
        predictions = get_reconstructed_scene(i, ".", imgs, model, device, False, 512, target_dir + images, "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx)
        
        # predictions = run_model(model, vggt_input, attn_mask=adj)s
            
        end = time.time()
        length = end - start

        print("Running inference took", length, "seconds!")


    # Keep tensors; only extract what we need later.
    # If you truly need NumPy later, convert specific keys then.
    needed = {
        "images","extrinsic", POINTS, CONF
    }
    for k in list(predictions.keys()):
        if k not in needed:
            del predictions[k]  # drop unneeded heavy stuff early


    # print("Align Point Cloud")

    start = time.time()
    
    WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)

    # if VIZ:
    #     visualize_vggt_pointcloud(predictions, key=POINTS, conf_key=CONF, threshold=threshold)

    Rmw, tmw, info = align_pointcloud_torch(WPTS_m, inlier_dist=voxel_size*0.75)
    WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
    predictions[POINTS] = WPTS_m

    end = time.time()
    length = end - start

    print("Aligning frames took", length, "seconds!")
    
    if VIZ:
        visualize_vggt_pointcloud(predictions, key=POINTS, conf_key=CONF, threshold=threshold)

    # print("Building Voxels and BEV")
    start = time.time()


    R_w2m = to_torch(R_w2m, device=device)
    t_w2m = to_torch(t_w2m, device=device)

    camera_R = R_w2m @ Rmw
    camera_t = t_w2m + tmw
    frames_map, cam_centers_map, conf_map, (S,H,W), frame_ids = build_frames_and_centers_vectorized_torch(
        predictions,
        POINTS=POINTS,
        CONF=CONF,
        threshold=threshold,
        Rmw=camera_R, tmw=camera_t,
        z_clip_map=z_clip_map,   # or None
        return_flat=True
       # dtype=dtype
    )   
    end = time.time()
    length = end - start

    print("Building frames/camera centers took", length, "seconds!")

    start = time.time()

    align_to_voxel = False #(i > 0)
    
    # if i == 0:
    #     vox.begin_bootstrap()

    vox, bev, meta = build_maps_from_points_and_centers_torch(
        frames_map,
        cam_centers_map,
        conf_map,
        vox,
        align_to_voxel=align_to_voxel,
        voxel_size=voxel_size,           # 10 cm
        bev_window_m=(5.0, 5.0), # local 20x20 m
        bev_origin_xy=(-2.0, -2.0),
        z_clip_vox=(-np.inf, np.inf),
        z_band_bev=(-0.04, 0.5),
        max_range_m=None,
        carve_free=True,
        samples_per_voxel=0.7,#1,
        ray_stride=6,#2,
        max_free_rays=10000,
        frame_ids=frame_ids
    )

    # if i == 0:
    #     vox.end_bootstrap()   # promotes current occupied to LT and clears ST
    #     vox.lock_long_term()  # optional: forbids any future LT change
        
    end = time.time()
    length = end - start

    print("Building Voxel and BEV took", length, "seconds!")

    # print("BEV:", bev.shape, meta)    

    num_cells = vox.keys.numel() if hasattr(vox, "keys") else len(vox.logodds)
    num_occ = int((vox.vals > vox.p.occ_thresh).sum().item()) if hasattr(vox, "vals") else \
            sum(1 for L in vox.logodds.values() if L > vox.p.occ_thresh)

    # print("Voxels (stored cells):", num_cells)
    # print("Voxels (occupied):", num_occ)

    # On-screen plots
    if VIZ:
        visualize_bev(bev, meta)
        # visualize_voxels(vox, z_band=(-0.5, 3), max_dim=200)
        visualize_points_and_voxels_open3d(frames_map, vox,cam_centers_map,max_points=150_000, max_voxels=25_000)

    # Files
    # save_bev_png(bev, meta, f"bev_{i}_{j}.png")
    # export_occupied_voxels_as_ply(vox, "voxels.ply")
        

    save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
    export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

    bev_png  = save_root + f"bev_{i}.png"
    bev_npy  = save_root + f"bev_{i}_np.npy"
    bev_meta = save_root + f"bev_{i}_meta.json"
    vox_ply  = save_root + f"voxels{i}.ply"
    vox_ijk  = save_root + f"voxels{i}_ijk.npy"
    vox_meta = save_root + f"voxels{i}_meta.json"
    
    for p in [bev_png, bev_npy, bev_meta, vox_ply, vox_ijk, vox_meta]:
        ensure_dir_for_file(p)

    save_bev(bev, meta, bev_png, bev_npy, bev_meta)
    export_occupied_voxels(vox, vox_ply, vox_ijk, vox_meta, z_clip_map)

    # >>> after-save bookkeeping so we can clone next time
    last_save_paths = {
        "bev_png": bev_png, "bev_npy": bev_npy, "bev_meta": bev_meta,
        "vox_ply": vox_ply, "vox_ijk": vox_ijk, "vox_meta": vox_meta,
    }

        
    vox.next_epoch()
