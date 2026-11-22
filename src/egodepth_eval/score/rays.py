"""Per-camera double-sphere ray fields and the Z/range/angle relations.

Built on `evaluate.geometry.DSCamera` (the existing unprojection, which matches
the GT pipeline's camera model) rather than a second implementation, so the
benchmark and the GT production agree by construction.

THREE RELATIONS USED THROUGHOUT
  Z = range * u_z              optical-axis depth from range along the ray
  range = Z / u_z              the inverse; undefined as u_z -> 0
  theta = acos(u_z)            field angle for the E2 angular bins

`u_z <= 0` means the ray is not forward-facing. Those pixels are excluded from
the angular strata rather than folded into the 80-90 degree bin: at u_z = 0 the
range is infinite, and a bin that mixes in unbounded values stops being a
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from egodepth_eval.data.contract import DSCalibration
from egodepth_eval.evaluate.geometry import DSCamera

# E2 main strata. The Z upper edge is closed so exactly 2.0 m is scored.
Z_BINS = ((0.25, 0.8), (0.8, 1.2), (1.2, 2.0))
ANGLE_BINS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 90))


def ds_camera_from_calibration(calibration: DSCalibration) -> DSCamera:
    return DSCamera(
        fx=calibration.fx, fy=calibration.fy,
        cx=calibration.cx, cy=calibration.cy,
        xi=calibration.xi, alpha=calibration.alpha,
        width=int(calibration.width), height=int(calibration.height),
    )


@dataclass
class RayField:
    """Unit ray directions on one camera's grid, with derived quantities.

    `valid` marks pixels whose DS unprojection succeeded. `forward` additionally
    requires u_z > 0.
    """

    directions: np.ndarray          # (H, W, 3) float32 unit vectors
    valid: np.ndarray               # (H, W) bool -- unprojection succeeded
    forward: np.ndarray             # (H, W) bool -- valid and u_z > 0
    camera_id: int
    grid_hw: tuple

    @property
    def u_z(self) -> np.ndarray:
        return self.directions[..., 2]

    def field_angle_deg(self) -> np.ndarray:
        """theta = acos(u_z) in degrees; NaN where the ray is not forward-facing."""
        u_z = np.clip(self.u_z, -1.0, 1.0)
        angle = np.degrees(np.arccos(u_z))
        return np.where(self.forward, angle, np.nan).astype(np.float32)

    def range_to_z(self, range_map: np.ndarray) -> np.ndarray:
        """Z = range * u_z. Non-forward rays become NaN, not zero.

        Zero would read as "predicted 0 m", which the scorer would treat as an
        invalid-but-present prediction; NaN keeps it out of the prediction
        support entirely, which is what an unconvertible ray actually means.
        """
        range_map = np.asarray(range_map, dtype=np.float32)
        z = range_map * self.u_z
        return np.where(self.forward, z, np.nan).astype(np.float32)

    def z_to_range(self, z_map: np.ndarray) -> np.ndarray:
        """range = Z / u_z, NaN where u_z is not safely positive."""
        z_map = np.asarray(z_map, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(self.forward, z_map / self.u_z, np.nan)
        return result.astype(np.float32)


def build_ray_field(
    calibration: DSCalibration, grid_hw: Optional[tuple] = None
) -> RayField:
    """Unit rays for one camera. `grid_hw` rescales the intrinsics if it differs.

    Each camera gets its own field. In the 336L snapshot the six cameras' principal
    points differ by up to 9 px, so reusing camera 2's rays for the other five is
    a real error that no model reports.
    """
    if grid_hw is not None:
        grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        if (calibration.height, calibration.width) != grid_hw:
            calibration = calibration.scaled_to(*grid_hw)
    height, width = int(calibration.height), int(calibration.width)

    camera = ds_camera_from_calibration(calibration)
    vs, us = np.mgrid[:height, :width].astype(np.float64)
    x, y, z = camera.unproject(us, vs)
    directions = np.stack([x, y, z], axis=-1).astype(np.float32)

    # DSCamera.unproject returns exact zeros for rays outside the model's valid
    # radius; that is the sentinel it uses.
    norm = np.linalg.norm(directions, axis=-1)
    valid = norm > 1e-6
    forward = valid & (directions[..., 2] > 0)

    return RayField(
        directions=directions, valid=valid, forward=forward,
        camera_id=calibration.camera_id, grid_hw=(height, width),
    )


class RayFieldCache:
    """Ray fields keyed by (camera_id, grid, DS parameters).

    Keying on the parameters and not just the camera id is deliberate: two
    recordings with different calibration for the same camera must not share a
    cached field. Returned arrays are read-only views, because a caller
    normalizing in place would corrupt every later sample -- the exact failure
    mode already documented for UniDepth's cached Camera object.
    """

    def __init__(self):
        self._fields: dict = {}

    def get(self, calibration: DSCalibration, grid_hw: Optional[tuple] = None) -> RayField:
        target = tuple(grid_hw) if grid_hw is not None else (calibration.height, calibration.width)
        key = (calibration.camera_id, target, tuple(round(v, 9) for v in calibration.ds_params))
        if key not in self._fields:
            field = build_ray_field(calibration, grid_hw=target)
            field.directions.flags.writeable = False
            field.valid.flags.writeable = False
            field.forward.flags.writeable = False
            self._fields[key] = field
        return self._fields[key]

    def directions_copy(self, calibration: DSCalibration, grid_hw: Optional[tuple] = None) -> np.ndarray:
        """A writable copy, for handing to a model that normalizes in place."""
        return np.array(self.get(calibration, grid_hw).directions, copy=True)

    def __len__(self) -> int:
        return len(self._fields)


def z_bin_index(z: np.ndarray, bins=Z_BINS) -> np.ndarray:
    """Bin index per pixel, -1 outside every bin.

    Edges are [lo, hi) except the last bin, which closes at hi so that exactly
    2.0 m is scored and no pixel lands in two bins.
    """
    z = np.asarray(z, dtype=np.float32)
    index = np.full(z.shape, -1, dtype=np.int8)
    for position, (lo, hi) in enumerate(bins):
        last = position == len(bins) - 1
        upper = (z <= hi) if last else (z < hi)
        index[np.isfinite(z) & (z >= lo) & upper & (index < 0)] = position
    return index


def angle_bin_index(angle_deg: np.ndarray, bins=ANGLE_BINS) -> np.ndarray:
    """Bin index per pixel from field angle in degrees, -1 outside every bin.

    NaN (non-forward rays) stays -1, so unusable rays are counted separately
    rather than accumulating in the outermost bin.
    """
    angle_deg = np.asarray(angle_deg, dtype=np.float32)
    index = np.full(angle_deg.shape, -1, dtype=np.int8)
    for position, (lo, hi) in enumerate(bins):
        last = position == len(bins) - 1
        upper = (angle_deg <= hi) if last else (angle_deg < hi)
        with np.errstate(invalid="ignore"):
            selected = np.isfinite(angle_deg) & (angle_deg >= lo) & upper & (index < 0)
        index[selected] = position
    return index


def bin_label(bounds: tuple) -> str:
    lo, hi = bounds
    return f"[{lo},{hi})"


def z_bin_labels(bins=Z_BINS) -> list:
    labels = [bin_label(b) for b in bins[:-1]]
    labels.append(f"[{bins[-1][0]},{bins[-1][1]}]")
    return labels


def angle_bin_labels(bins=ANGLE_BINS) -> list:
    labels = [bin_label(b) for b in bins[:-1]]
    labels.append(f"[{bins[-1][0]},{bins[-1][1]}]")
    return labels
