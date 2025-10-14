import torch, torch.nn.functional as F
import torchvision
import numpy as np
from typing import List, Union
from PIL import Image

# ---------------------------
# Utilities
# ---------------------------

def _to_rgb_tensor_01(img) -> torch.Tensor:
    """
    Accept PIL.Image, np.ndarray (H,W[,3/4]), or torch.Tensor (3,H,W in [0,1]).
    Returns torch.FloatTensor (3,H,W) in [0,1].
    """
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        assert t.ndim == 3 and t.shape[0] == 3, "Tensor must be (3,H,W)"
        return t.float().clamp(0,1)

    if isinstance(img, Image.Image):
        im = img
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (255,255,255,255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)  # H,W,3
    else:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:
            rgb = arr[..., :3].astype(np.float32)
            a = (arr[..., 3:4].astype(np.float32) / 255.0)
            arr = (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
        elif arr.shape[-1] == 3:
            arr = arr.astype(np.uint8)
        else:
            raise ValueError("Unsupported numpy image shape")
    t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    return t

def _as_batched_tensor(
    x: Union[List[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Accept list of images/tensors or a single batched tensor -> (N,3,H,W) in [0,1]."""
    if isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3:
        return x.float().clamp(0,1)
    if isinstance(x, torch.Tensor) and x.ndim == 3 and x.shape[0] == 3:
        return x.unsqueeze(0).float().clamp(0,1)
    assert isinstance(x, (list, tuple)) and len(x) > 0, "Provide a list or a batched tensor."
    return torch.stack([_to_rgb_tensor_01(i) for i in x], dim=0)

# ---------------------------
# MobileNet feature extractor
# ---------------------------

def _build_mobilenet_feature_extractor(device=None):
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    except Exception:
        model = torchvision.models.mobilenet_v2(pretrained=True)
    # Remove classifier → returns 1280-d embeddings after global pool
    model.classifier = torch.nn.Identity()
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

@torch.no_grad()
def _mobilenet_preprocess(batched: torch.Tensor, size: int = 160, device: str = "cpu"):
    """
    batched: (N,3,H,W) in [0,1], returns normalized tensor for MobileNet on device.
    Uses simple resize to size×size for speed.
    """
    x = batched.to(device=device, dtype=torch.float32)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std
    return x

@torch.no_grad()
def _mobilenet_embeddings(
    images_batched: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    images_batched: (N,3,H,W) in [0,1]
    returns L2-normalized embeddings (N,1280)
    """
    N = images_batched.shape[0]
    embs = []
    for i in range(0, N, batch_size):
        xb = _mobilenet_preprocess(images_batched[i:i+batch_size], device=device)
        eb = model(xb)                  # (B,1280)
        eb = F.normalize(eb, p=2, dim=1)
        embs.append(eb)
    return torch.cat(embs, dim=0) if len(embs) > 1 else embs[0]

# ---------------------------
# Main API
# ---------------------------

# @torch.no_grad()
# def changed_images(
#     new_images: Union[List[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor],
#     kept_images: Union[List[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor],
#     *,
#     thresh: float = 0.83,   # >= thresh → unchanged ; < thresh → changed
#     batch_size: int = 64,
#     device: str = None
# ) -> List[int]:
#     """
#     Returns indices in `new_images` that are **changed** based on MobileNet-V2 embeddings.
#     """
#     # Prep inputs
#     new_b  = _as_batched_tensor(new_images)
#     kept_b = _as_batched_tensor(kept_images) if (kept_images is not None and \
#               (isinstance(kept_images, torch.Tensor) or len(kept_images) > 0)) else None

#     N = new_b.shape[0]
#     if kept_b is None or kept_b.shape[0] == 0:
#         return list(range(N))  # no reference → everything is new

#     # Build model, compute embeddings
#     model, device = _build_mobilenet_feature_extractor(device)
#     new_embs  = _mobilenet_embeddings(new_b,  model, device=device, batch_size=batch_size)
#     kept_embs = _mobilenet_embeddings(kept_b, model, device=device, batch_size=batch_size)

#     keep = []

#     K = kept_embs.shape[0]
#     upto = min(N, K)
#     # 1) compare overlapping range
#     sims = (new_embs[:upto] * kept_embs[:upto]).sum(dim=1)  # cosine
#     changed_mask = sims < thresh
#     keep.extend(torch.nonzero(changed_mask, as_tuple=False).view(-1).tolist())
#     # 2) extra new frames beyond kept length are new by definition
#     if N > K:
#         keep.extend(range(K, N))



#     return keep




import cv2
import numpy as np
import torch
from typing import List, Union
from PIL import Image

# Re-use your existing helpers:
# - _to_rgb_tensor_01
# - _as_batched_tensor

def _to_gray_small_np01(t3chw: torch.Tensor, dscale: float = 0.5) -> np.ndarray:
    """
    t3chw: torch float tensor (3,H,W) in [0,1]
    Returns grayscale np.uint8 (hs, ws) after downscaling by dscale.
    """
    c, h, w = t3chw.shape
    img = (t3chw.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)  # H,W,3
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if dscale != 1.0:
        hs, ws = int(round(gray.shape[0]*dscale)), int(round(gray.shape[1]*dscale))
        gray = cv2.resize(gray, (ws, hs), interpolation=cv2.INTER_AREA)
    return gray

def _median_klt_flow_px(gray_ref: np.ndarray, gray_cur: np.ndarray,
                        max_corners: int = 400,
                        quality: float = 0.01,
                        min_dist: int = 7,
                        win: tuple = (21,21),
                        levels: int = 3,
                        term=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)) -> float:
    """
    Returns the median Lucas–Kanade displacement (in pixels) from ref -> cur.
    If not enough tracks, returns +inf to force 'changed'.
    """
    pts0 = cv2.goodFeaturesToTrack(gray_ref, maxCorners=max_corners,
                                   qualityLevel=quality, minDistance=min_dist)
    if pts0 is None or len(pts0) < 10:
        return float('inf')

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(gray_ref, gray_cur, pts0, None,
                                           winSize=win, maxLevel=levels, criteria=term)
    if pts1 is None or st is None:
        return float('inf')

    good = st.reshape(-1).astype(bool)
    if not np.any(good):
        return float('inf')

    disp = np.linalg.norm(pts1[good] - pts0[good], axis=2).reshape(-1)  # px at downscaled res
    if disp.size == 0:
        return float('inf')

    # robust stat beats mean (suppresses a few wild tracks)
    return float(np.median(disp))

def _klt_motion_score(
    gray_ref, gray_cur,
    max_corners=800, quality=0.01, min_dist=5,
    win=(21,21), levels=3,
    term=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    pctl=90,          # focus on the fast-moving tail (your single object)
    move_thr=0.5      # pixels at downscaled resolution to call a track "moving"
):
    pts0 = cv2.goodFeaturesToTrack(gray_ref, maxCorners=max_corners,
                                   qualityLevel=quality, minDistance=min_dist)
    if pts0 is None or len(pts0) < 10:
        return float('inf'), 0  # treat as "changed"

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(
        gray_ref, gray_cur, pts0, None,
        winSize=win, maxLevel=levels, criteria=term
    )
    if pts1 is None or st is None:
        return float('inf'), 0

    good = st.reshape(-1).astype(bool)
    if not np.any(good):
        return float('inf'), 0

    disp = np.linalg.norm(pts1[good] - pts0[good], axis=2).reshape(-1)
    if disp.size == 0:
        return float('inf'), 0

    score = float(np.percentile(disp, pctl))     # emphasizes the moving object
    n_moving = int((disp >= move_thr).sum())     # how many tracks agree it's moving
    return score, n_moving

@torch.no_grad()
def changed_images(
    new_images: Union[List[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor],
    kept_images: Union[List[Union[Image.Image, np.ndarray, torch.Tensor]], torch.Tensor],
    *,
    thresh: float = 1.2,    # REINTERPRETED: median flow threshold in pixels (downscaled space)
    batch_size: int = 64,           # (unused; kept for API compatibility)
    device: str = None              # (unused; LK runs on CPU)
) -> List[int]:
    """
    Returns indices in `new_images` that are **changed** based on Lucas–Kanade motion.

    Decision rule:
      - Compute median KLT displacement (in pixels) between reference and new image (grayscale, downscaled).
      - If median >= `cos_sim_thresh` → changed.

   
    """
    # ---- Tunables (good defaults for room cams) ----
    DSCALE = 1.0        # downscale for speed; if your frames are small, set to 1.0
    MAX_CORNERS = 400
    QUALITY = 0.01
    MIN_DIST = 7
    WIN = (21,21)
    LEVELS = 3
    TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    # Prep inputs to (N,3,H,W) tensors in [0,1]
    new_b  = _as_batched_tensor(new_images)
    kept_b = _as_batched_tensor(kept_images) if (kept_images is not None and \
              (isinstance(kept_images, torch.Tensor) or len(kept_images) > 0)) else None

    N = new_b.shape[0]
    if kept_b is None or kept_b.shape[0] == 0:
        # No reference → everything is new
        return list(range(N))

    # Convert all frames to downscaled grayscale NumPy once (fast path)
    new_gray  = [ _to_gray_small_np01(new_b[i], dscale=DSCALE) for i in range(N) ]
    kept_gray = [ _to_gray_small_np01(kept_b[i], dscale=DSCALE) for i in range(kept_b.shape[0]) ]

    changed_idxs: List[int] = []
    K = len(kept_gray)
    upto = min(N, K)

    # 1) overlapping range
    for i in range(upto):
        # med_px = _median_klt_flow_px(kept_gray[i], new_gray[i],
        #                                 max_corners=MAX_CORNERS, quality=QUALITY,
        #                                 min_dist=MIN_DIST, win=WIN, levels=LEVELS, term=TERM)
        
        med_px, n_moving = _klt_motion_score(kept_gray[i], new_gray[i], pctl=90, move_thr=thresh, min_dist=MIN_DIST, levels=LEVELS, win=WIN)

        if med_px >= thresh:
            changed_idxs.append(i)
    # 2) extra new frames are new by definition
    if N > K:
        changed_idxs.extend(range(K, N))


    return changed_idxs
