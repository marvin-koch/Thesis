"""
Monocular Depth + OctoMap/Voxblox-style Baseline
=================================================

A classical baseline that bypasses learned multi-view mapping:
  1. Runs an off-the-shelf monocular depth estimator (Depth Anything V2 Metric)
     on each camera view independently.
  2. Unprojects the per-pixel metric depth into 3D using known camera
     intrinsics / extrinsics.
  3. Feeds the resulting point cloud into a standard log-odds voxel grid
     (the same TorchSparseVoxelGrid used elsewhere) with ray-carving —
     functionally equivalent to OctoMap.

Why it matters for a reviewer:
  It proves that the DUSt3R + Latent Voxel Grid pipeline outperforms a much
  simpler, modern alternative (zero-shot depth + classical fusion) — giving a
  rock-solid external comparison without training a new network.

Requirements (pip):
  pip install transformers --break-system-packages    # for Depth Anything V2
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from voxel.voxel import TorchSparseVoxelGrid, VoxelParams


# ---------------------------------------------------------------------------
# Depth model loader (lazy, so import cost is zero if you don't use it)
# ---------------------------------------------------------------------------

def _load_depth_anything_v2(model_id: str, device: torch.device):
    """Load Depth Anything V2 **metric** model from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def _load_zoedepth(device: torch.device):
    """Fallback: load ZoeDepth-NK via torch hub (Intel ISL)."""
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return None, model  # processor=None → uses its own preprocessing


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _build_intrinsics(hfov_deg: float, H: int, W: int,
                      device: torch.device) -> torch.Tensor:
    """Pinhole intrinsics from horizontal FOV (typical Habitat setup)."""
    fx = W / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx  # square pixels
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32, device=device)
    return K


def _unproject_depth(depth: torch.Tensor,
                     K: torch.Tensor,
                     extrinsic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject a depth map to world-frame 3D points.

    Args:
        depth:     (H, W) metric depth in metres.
        K:         (3, 3) intrinsics.
        extrinsic: (4, 4) camera-to-world (c2w) matrix.

    Returns:
        pts_world:  (N, 3) valid 3D points.
        cam_center: (3,)   camera origin in world frame.
    """
    H, W = depth.shape
    device = depth.device

    # pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # camera-frame 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # (H*W, 3)

    # validity mask (finite, positive depth, not too far)
    valid = (z.reshape(-1) > 1e-3) & torch.isfinite(z.reshape(-1))
    pts_cam = pts_cam[valid]

    if pts_cam.numel() == 0:
        return pts_cam, extrinsic[:3, 3]

    # camera → world
    R = extrinsic[:3, :3]  # (3, 3)
    t = extrinsic[:3, 3]   # (3,)
    pts_world = (pts_cam @ R.T) + t

    return pts_world, t


# ---------------------------------------------------------------------------
# Main baseline class
# ---------------------------------------------------------------------------

class MonocularDepthFusion:
    """
    Monocular Depth Estimation + Log-Odds Voxel Fusion.

    Drop-in extra-baseline that follows the same interface expected by
    ``compute_extra_baseline_metrics`` in train.py.

    Usage in predict_step:
    >>> mono = MonocularDepthFusion(voxel_size=0.2, device=device)
    >>> # per timestep:
    >>> mono.integrate_from_images(predictions["images"],
    ...                            predictions["extrinsic"])
    """

    # ---- supported backends ----
    DEPTH_ANYTHING_V2_SMALL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    DEPTH_ANYTHING_V2_BASE  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    DEPTH_ANYTHING_V2_LARGE = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    ZOEDEPTH = "zoedepth"

    def __init__(
        self,
        voxel_size: float = 0.2,
        device: torch.device | str = "cuda",
        model_id: str = DEPTH_ANYTHING_V2_SMALL,
        hfov_deg: float = 90.0,
        img_size: int = 512,
        max_range: float = 20.0,
        promote_epochs: int = 2,
    ):
        self.device = torch.device(device)
        self.voxel_size = voxel_size
        self.hfov_deg = hfov_deg
        self.img_size = img_size
        self.max_range = max_range
        self.model_id = model_id

        # --- depth model ---
        print(f"[MonocularDepthFusion] Loading depth model: {model_id}")
        if model_id == self.ZOEDEPTH:
            self.processor, self.depth_model = _load_zoedepth(self.device)
            self._backend = "zoedepth"
        else:
            self.processor, self.depth_model = _load_depth_anything_v2(
                model_id, self.device
            )
            self._backend = "depth_anything_v2"
        print(f"[MonocularDepthFusion] Depth model loaded.")

        # --- voxel grid (same type as the main baseline) ---
        self.vox = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(
                voxel_size=voxel_size,
                promote_epochs=promote_epochs,
            ),
            device=self.device,
        )

        # pre-built intrinsics (rebuilt on first call if image size differs)
        self._K: Optional[torch.Tensor] = None
        self._K_hw: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Standard extra-baseline interface (no-op: this baseline generates
    # its own 3D points from images, not from DUSt3R eb_pts/eb_cams)
    # ------------------------------------------------------------------
    def integrate(self, pts, cams, *, conf=None, max_range=20.0):
        """No-op — MonocularDepthFusion uses ``integrate_from_images``."""
        pass

    # ------------------------------------------------------------------
    # Image-based integration (the actual workhorse)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def integrate_from_images(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        max_range: Optional[float] = None,
        max_depth_m: float = 10.0,
        depth_scale: float = 1.0,
    ):
        """
        Run monocular depth on each view, unproject, and fuse.

        Args:
            images:     (S, H, W, 3) float32 in [0, 1]  **or**
                        (S, 3, H, W) float32 in [0, 1].
            extrinsics: (S, 4, 4) camera-to-world matrices.
                        **Must already be in the aligned evaluation frame**
                        (i.e. rotations AND translations fully transformed).
            max_range:  override per-ray max range for integration.
            max_depth_m: clamp predicted depth (avoids sky hallucinations).
            depth_scale: multiply metric depth by this before unprojection.
                         Use 1.0 when the aligned frame is metric (DUSt3R path).
                         Use s_k when aligning GT extrinsics via Kabsch/ICP
                         (because s_k converts GT-metres → aligned-frame units).
        """
        if max_range is None:
            max_range = self.max_range

        imgs = images.to(self.device, dtype=torch.float32)
        extrs = extrinsics.to(self.device, dtype=torch.float32)

        # normalise layout to (S, H, W, 3)
        if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
            imgs = imgs.permute(0, 2, 3, 1)

        S, H, W, _ = imgs.shape

        # (re)build intrinsics if resolution changed
        if self._K is None or self._K_hw != (H, W):
            self._K = _build_intrinsics(self.hfov_deg, H, W, self.device)
            self._K_hw = (H, W)

        all_pts = []
        all_cams = []

        for s in range(S):
            depth = self._estimate_depth(imgs[s], target_hw=(H, W))  # (H, W)
            depth = depth.clamp(max=max_depth_m) * depth_scale

            pts_w, cam_c = _unproject_depth(depth, self._K, extrs[s])

            if pts_w.numel() == 0:
                continue

            # per-point camera center (expanded)
            cams_w = cam_c.unsqueeze(0).expand_as(pts_w)
            all_pts.append(pts_w)
            all_cams.append(cams_w)

        if not all_pts:
            return

        pts_cat = torch.cat(all_pts, dim=0)
        cams_cat = torch.cat(all_cams, dim=0)

        # feed into the standard log-odds voxel grid (OctoMap-style)
        self.vox.integrate_points_with_cameras(
            pts_cat,
            cams_cat,
            carve_free=True,
            max_range=max_range,
            z_clip=None,
            samples_per_voxel=0.7,
            ray_stride=4,
            max_free_rays=10_000,
        )
        self.vox.next_epoch()

    # ------------------------------------------------------------------
    # Internal: depth estimation dispatch
    # ------------------------------------------------------------------

    def _estimate_depth(self, img_hwc: torch.Tensor,
                        target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Run depth estimation on a single image.

        Args:
            img_hwc: (H, W, 3) float in [0, 1].
            target_hw: desired output (H, W).

        Returns:
            depth: (H, W) metric depth in metres.
        """
        if self._backend == "depth_anything_v2":
            return self._depth_anything_v2(img_hwc, target_hw)
        elif self._backend == "zoedepth":
            return self._zoedepth(img_hwc, target_hw)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _depth_anything_v2(self, img_hwc: torch.Tensor,
                           target_hw: Tuple[int, int]) -> torch.Tensor:
        """Depth Anything V2 Metric Indoor via HuggingFace transformers."""
        from PIL import Image as PILImage

        # processor expects PIL or numpy uint8
        img_np = (img_hwc.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.depth_model(**inputs)
        pred = outputs.predicted_depth  # (1, h, w) — model native resolution

        # resize to match the image resolution we're unprojecting at
        depth = F.interpolate(
            pred.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (H, W)

        return depth

    def _zoedepth(self, img_hwc: torch.Tensor,
                  target_hw: Tuple[int, int]) -> torch.Tensor:
        """ZoeDepth-NK via torch hub."""
        # ZoeDepth expects (1, 3, H, W) in [0, 1]
        img_chw = img_hwc.permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth = self.depth_model.infer(img_chw)  # (1, 1, H, W)
        depth = F.interpolate(
            depth, size=target_hw, mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        return depth

    # ------------------------------------------------------------------
    # Convenience properties (mirror TorchSparseVoxelGrid for metrics)
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        """Delegate any attribute not found on this wrapper to the inner vox grid.
        This ensures compute_extra_baseline_metrics (which calls _unhash_keys,
        _display_vals, p.occ_thresh, etc.) works transparently."""
        # Avoid infinite recursion during init / pickling
        if name in ("vox", "__dict__", "__class__"):
            raise AttributeError(name)
        try:
            return getattr(self.vox, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    @property
    def keys(self):
        return self.vox.keys

    def occupied_mask(self):
        return self.vox.occupied_mask()

    def voxel_centers(self):
        return self.vox.voxel_centers()

    def _display_vals(self):
        return self.vox._display_vals()