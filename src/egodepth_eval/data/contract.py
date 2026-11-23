"""Synchronized six-camera sample and double-sphere calibration on the release grid.

Only camera 2 has reference optical-axis Z-depth, in meters. Each input view
uses its own intrinsics. Camera-to-rig transforms are provided to a model
only in the corresponding pose-conditioned experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

NUM_CAMERAS = 6
TARGET_CAMERA_ID = 2

# Main scoring grid, (H, W). The array shape is (416, 512).
SCORING_GRID_HW = (416, 512)

@dataclass(frozen=True)
class DSCalibration:
    """Double-sphere fisheye intrinsics on a specific pixel grid, plus extrinsics.

    `T_body_cam` transforms camera coordinates into rig coordinates. Each
    method configuration declares whether the model receives this transform.
    """

    camera_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    alpha: float
    width: int
    height: int
    T_body_cam: Optional[list] = None
    source: str = "unspecified"

    @property
    def ds_params(self) -> list:
        """[fx, fy, cx, cy, xi, alpha], the order the DS camera model expects."""
        return [self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha]

    def scaled_to(self, height: int, width: int) -> "DSCalibration":
        """Same camera on a different grid.

        fx/fy/cx/cy scale with resolution; xi and alpha are dimensionless and
        must not. This is the only sanctioned way to change grids -- an
        unscaled K on a resized image is the classic silent failure here.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"cam{self.camera_id}: invalid source grid")
        sx, sy = width / self.width, height / self.height
        return DSCalibration(
            camera_id=self.camera_id,
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            xi=self.xi, alpha=self.alpha,
            width=int(width), height=int(height),
            T_body_cam=self.T_body_cam,
            source=f"{self.source}+scaled_{self.height}x{self.width}->{height}x{width}",
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DSCalibration":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class CanonicalSample:
    """One synchronized timestamp: six RGB paths plus camera-2 reference depth.

    Paths are relative to the dataset's `asset_root` so the manifest survives a
    root relocation (a read-only cache, another host) without rewriting.
    `physical_site_id` is deliberately Optional: a recording ID is not a site,
    and a missing site mapping must block a formal run rather than be invented.
    """

    sample_token: str
    recording_id: str
    timestamp_ns: int
    rgb_rel_paths: dict            # {camera_id: relative path}
    depth_rel_path: str            # camera 2 only
    depth_shape_hw: tuple
    calibrations: dict             # {camera_id: DSCalibration}
    valid_mask_rel_path: Optional[str] = None
    packed_valid_mask_rel_path: Optional[str] = None
    mask_encoding: Optional[dict] = None
    physical_site_id: Optional[str] = None
    split: Optional[str] = None
    prev_token: Optional[str] = None
    next_token: Optional[str] = None
    selection_policy: Optional[str] = None
    depth_gt_type: Optional[str] = None
    data_version: Optional[str] = None
    target_camera_id: int = TARGET_CAMERA_ID
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["calibrations"] = {
            str(k): (v.to_dict() if isinstance(v, DSCalibration) else v)
            for k, v in self.calibrations.items()
        }
        payload["rgb_rel_paths"] = {str(k): v for k, v in self.rgb_rel_paths.items()}
        payload["depth_shape_hw"] = list(self.depth_shape_hw)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "CanonicalSample":
        payload = dict(payload)
        payload["calibrations"] = {
            int(k): DSCalibration.from_dict(v)
            for k, v in payload.get("calibrations", {}).items()
        }
        payload["rgb_rel_paths"] = {
            int(k): v for k, v in payload.get("rgb_rel_paths", {}).items()
        }
        payload["depth_shape_hw"] = tuple(payload.get("depth_shape_hw", SCORING_GRID_HW))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def validate(self) -> None:
        """Structural checks that must hold before a sample is usable.

        Deliberately strict about the six-view contract: a sample missing one
        camera would otherwise silently become a five-view MapAnything call
        with a different information budget from every other sample.
        """
        missing = [c for c in range(NUM_CAMERAS) if c not in self.rgb_rel_paths]
        if missing:
            raise ValueError(f"{self.sample_token}: missing RGB for cameras {missing}")
        missing_calib = [c for c in range(NUM_CAMERAS) if c not in self.calibrations]
        if missing_calib:
            raise ValueError(
                f"{self.sample_token}: missing calibration for cameras {missing_calib}"
            )
        if self.target_camera_id != TARGET_CAMERA_ID:
            raise ValueError(
                f"{self.sample_token}: target camera is {self.target_camera_id}, "
                f"but this benchmark's reference depth exists only for camera "
                f"{TARGET_CAMERA_ID}"
            )
        if not self.depth_rel_path:
            raise ValueError(f"{self.sample_token}: no camera-2 reference depth path")
        if len(self.depth_shape_hw) != 2 or any(int(v) <= 0 for v in self.depth_shape_hw):
            raise ValueError(f"{self.sample_token}: bad depth shape {self.depth_shape_hw}")
        if int(self.timestamp_ns) <= 0:
            raise ValueError(f"{self.sample_token}: non-positive timestamp")
