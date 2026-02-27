"""
Monocular Depth + OctoMap/Voxblox-style Baseline
=================================================

A classical baseline that bypasses learned multi-view mapping:
  1. Runs an off-the-shelf monocular depth estimator (Depth Anything V2 Metric)
     on each camera view independently.
  2. Unprojects the depth maps into 3D using known camera intrinsics and
     extrinsics that are auto-calibrated against a reference point cloud.
  3. Feeds the resulting point cloud into a standard log-odds voxel grid
     (the same TorchSparseVoxelGrid used elsewhere) with ray-carving —
     functionally equivalent to OctoMap.

CRITICAL: call ``calibrate()`` once at t=0 before ``integrate_from_images()``.
  The existing pipeline never uses the rotation part of DUSt3R extrinsics
  (only camera centers), so the rotation convention is unknown.  calibrate()
  solves for the correct camera→world rotation via Procrustes alignment
  against DUSt3R's reference point cloud.

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
    return None, model


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


def _unproject_depth_to_cam(depth: torch.Tensor,
                            K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject a depth map to **camera-frame** 3D points.

    Standard pinhole convention: x-right, y-down, z-forward.

    Returns:
        pts_cam: (N, 3)  valid camera-frame 3D points.
        valid:   (H*W,)  bool mask over the flattened depth map.
    """
    H, W = depth.shape
    device = depth.device

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    valid = (z.reshape(-1) > 1e-3) & torch.isfinite(z.reshape(-1))

    return pts_cam[valid], valid


# ---------------------------------------------------------------------------
# Main baseline class
# ---------------------------------------------------------------------------

class MonocularDepthFusion:
    """
    Monocular Depth Estimation + Log-Odds Voxel Fusion.

    Call ``calibrate()`` once at t=0 with DUSt3R world_points as reference,
    then ``integrate_from_images()`` at every timestep.
    """

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

        # --- voxel grid ---
        self.vox = TorchSparseVoxelGrid(
            origin_xyz=np.zeros(3, dtype=np.float32),
            params=VoxelParams(
                voxel_size=voxel_size,
                promote_epochs=promote_epochs,
            ),
            device=self.device,
        )

        self._K: Optional[torch.Tensor] = None
        self._K_hw: Optional[Tuple[int, int]] = None

        # Rotation correction: maps pinhole camera-space → extrinsic convention
        self._R_cam_correction: Optional[torch.Tensor] = None
        self._calibrated = False

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        ref_world_points: torch.Tensor,
        depth_scale: float = 1.0,
        n_views: int = 3,
        n_sample: int = 20_000,
    ):
        """
        Determine the correct cam→world rotation by comparing
        mono-depth unprojection with DUSt3R world_points (Procrustes).

        Call once at t=0. Only used for calibration — the baseline
        itself does not depend on DUSt3R at inference time.

        Args:
            images:           (S, H, W, 3) or (S, 3, H, W) float [0, 1].
            extrinsics:       (S, 4, 4) aligned extrinsic matrices.
            ref_world_points: (S, H, W, 3) DUSt3R points in aligned frame.
            depth_scale:      same as will be used for integration.
            n_views:          how many views to use (averaged).
            n_sample:         subsample per view for speed.
        """
        dev = self.device

        imgs = images.to(dev, dtype=torch.float32)
        if isinstance(extrinsics, list):
            extrinsics = torch.stack(extrinsics, dim=0)
        extrs = extrinsics.to(dev, dtype=torch.float32)
        ref_pts = ref_world_points.to(dev, dtype=torch.float32)

        if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
            imgs = imgs.permute(0, 2, 3, 1)

        S, H, W, _ = imgs.shape

        if self._K is None or self._K_hw != (H, W):
            self._K = _build_intrinsics(self.hfov_deg, H, W, dev)
            self._K_hw = (H, W)

        all_cam_pts = []
        all_ref_cam = []

        for s in range(min(n_views, S)):
            cam_center = extrs[s, :3, 3]
            R_ext = extrs[s, :3, :3]

            # 1) Mono depth → camera-space points
            depth = self._estimate_depth(imgs[s], target_hw=(H, W))
            depth = depth.clamp(max=10.0) * depth_scale
            pts_cam, valid_mask = _unproject_depth_to_cam(depth, self._K)

            # 2) Corresponding DUSt3R world points → camera-space
            #    cam_ref = R_ext^{-T} @ (ref_world - cam_center)
            #    We try both R_ext.T and R_ext to see which convention is used
            ref_flat = ref_pts[s].reshape(-1, 3)
            ref_valid = ref_flat[valid_mask]
            ref_cam_centered = ref_valid - cam_center  # world_pt - cam_center

            # Both must be finite
            both_ok = (torch.isfinite(pts_cam).all(-1) &
                       torch.isfinite(ref_cam_centered).all(-1))
            pts_cam_ok = pts_cam[both_ok]
            ref_cam_ok = ref_cam_centered[both_ok]

            if pts_cam_ok.shape[0] < 100:
                continue

            n = min(n_sample // max(1, n_views), pts_cam_ok.shape[0])
            idx = torch.randperm(pts_cam_ok.shape[0], device=dev)[:n]
            all_cam_pts.append(pts_cam_ok[idx])
            all_ref_cam.append(ref_cam_ok[idx])

        if not all_cam_pts:
            print("[MonocularDepthFusion] WARNING: calibration failed, "
                  "using identity")
            self._R_cam_correction = torch.eye(3, device=dev)
            self._calibrated = True
            return

        cam_pts = torch.cat(all_cam_pts, dim=0)   # our camera-space pts
        ref_cam = torch.cat(all_ref_cam, dim=0)    # ref pts centred on camera

        # The ref points in camera space depend on R_ext convention:
        #   If R_ext is c2w: ref_cam_actual = R_ext^T @ (ref_world - cam_center)
        #   If R_ext is w2c: ref_cam_actual = R_ext   @ (ref_world - cam_center)
        # We don't know which. But we can solve for the full transform
        # from cam_pts → ref_cam via Procrustes. The resulting R_correction
        # absorbs the convention ambiguity.

        # Procrustes: solve  ref_cam ≈ R_correction @ cam_pts
        # i.e.  H = cam_pts^T @ ref_cam, then SVD
        H_mat = cam_pts.T @ ref_cam
        U, Sv, Vh = torch.linalg.svd(H_mat)
        d = torch.det(Vh.T @ U.T)
        diag = torch.ones(3, device=dev)
        diag[2] = d.sign()
        R_correction = Vh.T @ torch.diag(diag) @ U.T

        # Check residuals
        corrected = cam_pts @ R_correction.T
        residual_corrected = (corrected - ref_cam).norm(dim=-1).median().item()
        residual_identity = (cam_pts - ref_cam).norm(dim=-1).median().item()

        print(f"[MonocularDepthFusion] Calibration ({cam_pts.shape[0]} pts):")
        print(f"  Median residual WITH correction:    {residual_corrected:.4f}")
        print(f"  Median residual WITHOUT correction: {residual_identity:.4f}")
        print(f"  R_correction diagonal: "
              f"[{R_correction[0,0]:.3f}, {R_correction[1,1]:.3f}, {R_correction[2,2]:.3f}]")

        if residual_identity < residual_corrected * 0.8:
            print(f"  → No correction needed (identity)")
            self._R_cam_correction = torch.eye(3, device=dev)
        else:
            improvement = (1.0 - residual_corrected / max(residual_identity, 1e-6)) * 100
            print(f"  → Using Procrustes correction ({improvement:.1f}% improvement)")
            self._R_cam_correction = R_correction

        self._calibrated = True

    # ------------------------------------------------------------------
    # Standard extra-baseline interface (no-op)
    # ------------------------------------------------------------------
    def integrate(self, pts, cams, *, conf=None, max_range=20.0):
        """No-op — MonocularDepthFusion uses ``integrate_from_images``."""
        pass

    # ------------------------------------------------------------------
    # Image-based integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def integrate_from_images(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        max_range: Optional[float] = None,
        max_depth_m: float = 10.0,
        depth_scale: float = 1.0,
        pixel_stride: int = 4,
    ):
        """
        Run monocular depth on each view, unproject, and fuse.

        Args:
            images:      (S, H, W, 3) float32 in [0, 1] or (S, 3, H, W).
            extrinsics:  (S, 4, 4) aligned extrinsic matrices.
            max_range:   per-ray max range for integration.
            max_depth_m: clamp predicted depth.
            depth_scale: scale metric depth to aligned frame units.
            pixel_stride: subsample depth map before unprojection.
        """
        if not self._calibrated:
            print("[MonocularDepthFusion] WARNING: calibrate() not called! "
                  "Using identity rotation (results may be wrong).")
            self._R_cam_correction = torch.eye(3, device=self.device)
            self._calibrated = True

        if max_range is None:
            max_range = self.max_range

        imgs = images.to(self.device, dtype=torch.float32)
        if isinstance(extrinsics, list):
            extrinsics = torch.stack(extrinsics, dim=0)
        extrs = extrinsics.to(self.device, dtype=torch.float32)

        if imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[-1] != 3:
            imgs = imgs.permute(0, 2, 3, 1)

        S, H, W, _ = imgs.shape

        if self._K is None or self._K_hw != (H, W):
            self._K = _build_intrinsics(self.hfov_deg, H, W, self.device)
            self._K_hw = (H, W)

        R_corr = self._R_cam_correction  # (3, 3)

        all_pts = []
        all_cams = []

        for s in range(S):
            depth = self._estimate_depth(imgs[s], target_hw=(H, W))
            depth = depth.clamp(max=max_depth_m) * depth_scale

            if pixel_stride > 1:
                depth = depth[::pixel_stride, ::pixel_stride]

            K_eff = self._K.clone()
            if pixel_stride > 1:
                K_eff[:2, :] /= pixel_stride

            pts_cam, _ = _unproject_depth_to_cam(depth, K_eff)
            if pts_cam.numel() == 0:
                continue

            # Apply calibrated rotation correction in camera space
            pts_cam_corrected = pts_cam @ R_corr.T

            # Camera → world using extrinsic
            R_ext = extrs[s, :3, :3]
            cam_center = extrs[s, :3, 3]
            pts_world = (pts_cam_corrected @ R_ext.T) + cam_center

            cams_w = cam_center.unsqueeze(0).expand_as(pts_world)
            all_pts.append(pts_world)
            all_cams.append(cams_w)

        if not all_pts:
            return

        pts_cat = torch.cat(all_pts, dim=0)
        cams_cat = torch.cat(all_cams, dim=0)

        # Diagnostic
        n = pts_cat.shape[0]
        dists = torch.linalg.norm(pts_cat - cams_cat, dim=-1)
        print(f"  [MonoDepth] {n} pts | depth_scale={depth_scale:.3f} | "
              f"range [{dists.min():.2f}, {dists.median():.2f}, {dists.max():.2f}] | "
              f"xyz mean={pts_cat.mean(0).tolist()}")

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
    # Internal: depth estimation
    # ------------------------------------------------------------------
    def _estimate_depth(self, img_hwc: torch.Tensor,
                        target_hw: Tuple[int, int]) -> torch.Tensor:
        if self._backend == "depth_anything_v2":
            return self._depth_anything_v2(img_hwc, target_hw)
        elif self._backend == "zoedepth":
            return self._zoedepth(img_hwc, target_hw)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _depth_anything_v2(self, img_hwc: torch.Tensor,
                           target_hw: Tuple[int, int]) -> torch.Tensor:
        from PIL import Image as PILImage
        img_np = (img_hwc.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.depth_model(**inputs)
        pred = outputs.predicted_depth
        depth = F.interpolate(
            pred.unsqueeze(0), size=target_hw,
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        return depth

    def _zoedepth(self, img_hwc: torch.Tensor,
                  target_hw: Tuple[int, int]) -> torch.Tensor:
        img_chw = img_hwc.permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth = self.depth_model.infer(img_chw)
        depth = F.interpolate(
            depth, size=target_hw, mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        return depth

    # ------------------------------------------------------------------
    # Delegate to inner voxel grid for metrics compatibility
    # ------------------------------------------------------------------
    def __getattr__(self, name):
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