"""Camera geometry for the evaluation grid.

Everything here operates on the *GT depth grid* (800x648), not the native RGB
grid (1600x1296). Three conventions are load-bearing and choosing wrong is
silent -- the depth map still looks right and only an external length
reference exposes the error (see ballbar_validation/README.md):

1. The stored depth is Z along the optical axis, not range along the ray.
   Back-projection is `t = Z / ray_z`.
2. `camera_info` intrinsics describe the native grid. fx/fy/cx/cy must be
   scaled by the resolution ratio; `xi` and `alpha` are dimensionless and must
   not be.
3. `depth_gt_info` already carries intrinsics on the depth grid, so when it is
   available it is used directly and no scaling is applied.

The DS (double sphere) projection/unprojection follows
ego_depth_pipeline/src/ego_depth/stereo/camera_model.py so that the benchmark
and the GT pipeline agree on the same camera.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DSCamera:
    """Double-sphere fisheye camera on a specific pixel grid."""

    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    alpha: float
    width: int
    height: int

    @classmethod
    def from_depth_info(cls, depth_info: dict, shape_hw: tuple) -> "DSCamera":
        """Build from a `depth_gt_info` payload, which is already on the depth grid."""
        cam = depth_info.get("camera", depth_info)
        K = cam["K"]
        fx, cx, fy, cy = K[0], K[2], K[4], K[5]
        h, w = shape_hw
        return cls(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            xi=float(cam.get("xi", 0.0)), alpha=float(cam.get("alpha", 0.5)),
            width=int(w), height=int(h),
        )

    @classmethod
    def from_camera_info(cls, cam_calib: dict, shape_hw: tuple) -> "DSCamera":
        """Build from a native-resolution `camera_info` entry, rescaling to shape_hw.

        `D` for the DS model is [fx, fy, cx, cy, xi, alpha].
        """
        native_w = int(cam_calib.get("width", 0))
        native_h = int(cam_calib.get("height", 0))
        h, w = shape_hw

        D = cam_calib.get("D", [])
        if len(D) >= 6:
            fx, fy, cx, cy, xi, alpha = D[:6]
        else:
            K = cam_calib["K"]
            if isinstance(K[0], list):
                fx, cx, fy, cy = K[0][0], K[0][2], K[1][1], K[1][2]
            else:
                fx, cx, fy, cy = K[0], K[2], K[4], K[5]
            xi, alpha = 0.0, 0.5

        # Scale focal length and principal point only; xi/alpha are dimensionless.
        sx = w / native_w if native_w else 1.0
        sy = h / native_h if native_h else 1.0
        return cls(
            fx=float(fx) * sx, fy=float(fy) * sy,
            cx=float(cx) * sx, cy=float(cy) * sy,
            xi=float(xi), alpha=float(alpha),
            width=int(w), height=int(h),
        )

    def unproject(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """Pixels -> unit ray direction (X, Y, Z). Invalid rays come back as zeros."""
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx**2 + my**2

        alpha = self.alpha
        if alpha > 0.5:
            valid = r2 <= 1.0 / (2.0 * alpha - 1.0)
        else:
            valid = np.ones_like(r2, dtype=bool)

        disc = np.maximum(1.0 - (2.0 * alpha - 1.0) * r2, 1e-10)
        mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(disc) + 1.0 - alpha)

        xi = self.xi
        coeff = (mz * xi + np.sqrt(np.maximum(mz**2 + (1.0 - xi**2) * r2, 1e-10))) / (
            mz**2 + r2 + 1e-12
        )
        X = coeff * mx
        Y = coeff * my
        Z = coeff * mz - xi

        norm = np.maximum(np.sqrt(X**2 + Y**2 + Z**2), 1e-10)
        X, Y, Z = X / norm, Y / norm, Z / norm
        X = np.where(valid, X, 0.0)
        Y = np.where(valid, Y, 0.0)
        Z = np.where(valid, Z, 0.0)
        return X, Y, Z

    def field_angle_map(self) -> np.ndarray:
        """Per-pixel angle from the optical axis, in degrees.

        This is the physically meaningful "how far off-axis is this pixel"
        measure for a fisheye, unlike pixel radius which mixes in the
        projection's own nonlinearity.
        """
        vs, us = np.mgrid[: self.height, : self.width].astype(np.float64)
        X, Y, Z = self.unproject(us, vs)
        r = np.sqrt(X**2 + Y**2)
        ang = np.degrees(np.arctan2(r, Z))
        # Rays flagged invalid unproject to (0,0,0) and would read as angle 0.
        ang[(X == 0) & (Y == 0) & (Z == 0)] = np.nan
        return ang.astype(np.float32)


def radial_norm_map(shape_hw: tuple) -> np.ndarray:
    """Normalized fisheye radius, 0 at image centre and 1 at the mid-edge.

    Defined as hypot(u - w/2, v - h/2) / (0.5 * w), i.e. image centre and
    half-width -- deliberately identical to
    ballbar_validation/ballbar/measure.py::_radial_norm. Section V's radial
    bins are compared against Section IV's per-stratum ball-bar error, so the
    two must be the same quantity; using the principal point or the corner
    distance here would make those columns incomparable.
    """
    h, w = shape_hw
    vs, us = np.mgrid[:h, :w].astype(np.float32)
    return (np.hypot(us - w / 2.0, vs - h / 2.0) / (0.5 * w)).astype(np.float32)


def center_crop_mask(
    camera: DSCamera,
    fov_deg: float = 60.0,
) -> np.ndarray:
    """Pixels inside a centred `fov_deg` cone, for the pinhole ablation.

    The paper's ablation isolates fisheye projection error by restricting to a
    roughly pinhole-like central region. Selecting by field angle (rather than
    by a pixel-space box) is what makes "60 degrees FoV" mean the same thing
    regardless of the distortion profile. fov_deg is the *full* cone angle, so
    the half-angle threshold is fov_deg/2.
    """
    ang = camera.field_angle_map()
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(ang, nan=1e9) <= (fov_deg / 2.0)


def load_eval_camera(
    calib: dict,
    depth_info: Optional[dict],
    shape_hw: tuple,
    cam_key: str = "cam2",
) -> Optional[DSCamera]:
    """Best available camera for the depth grid.

    Prefers `depth_gt_info` (already on the depth grid, and it is the
    intrinsics the GT was actually rendered with); falls back to rescaling
    `camera_info`.
    """
    if depth_info:
        try:
            return DSCamera.from_depth_info(depth_info, shape_hw)
        except (KeyError, IndexError, TypeError):
            pass
    cam_calib = calib.get(cam_key, {})
    if cam_calib:
        try:
            return DSCamera.from_camera_info(cam_calib, shape_hw)
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            pass
    return None
