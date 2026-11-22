"""MapAnything inference for E1/E3/E4, with the DS rays actually reaching the model.

THE BUG THIS MODULE EXISTS TO AVOID
`mapanything.utils.image.preprocess_inputs` accepts `ray_directions`, then calls
`recover_pinhole_intrinsics_from_ray_directions` and forwards the *fitted pinhole
K* instead. For a double-sphere lens reaching 90 degrees off-axis, no 3x3 K
describes the camera, so the whole point of supplying real rays is lost -- and
nothing warns. This module builds the network's input tensors itself:
image normalization and resize applied identically to the image and the ray field,
and `ray_directions_cam` handed straight to `forward`. `verify_rays_reached_model`
compares what the model received against what was built.

VIEW COUNT AND TARGET SELECTION
E3 runs 1, 2 and 6 views. The camera-2 output is located by its *position in the
supplied view list*, not a fixed index -- with `[2, 3]` the target is at position
0, and reading index 2 would either fail or silently read a different camera.

PROTOCOL FLAGS
`use_multiview_confidence` is off and no extra confidence/edge filter is added
beyond the model's own support, per the E3 contract: filtering by multi-view
consistency would preferentially delete the moving hand and object -- exactly the
region being measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.data.contract import TARGET_CAMERA_ID, DSCalibration
from egodepth_eval.score.rays import RayFieldCache

# DINOv2 patch size: MapAnything's encoder requires H and W to be multiples of 14.
PATCH_SIZE = 14
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


def snap_to_patch(value: int, patch: int = PATCH_SIZE) -> int:
    return max(patch, int(round(value / patch)) * patch)


@dataclass
class ViewTensors:
    """What was actually built for one view, kept for verification."""

    image: np.ndarray               # (3, H, W) float32 normalized
    rays: np.ndarray                # (H, W, 3) float32 unit
    camera_id: int
    source_hw: tuple
    target_hw: tuple


def build_view_tensors(
    image: np.ndarray,
    calibration: DSCalibration,
    height: int,
    width: int,
    ray_cache: Optional[RayFieldCache] = None,
) -> ViewTensors:
    """Resize the image and build rays on the SAME grid from rescaled intrinsics.

    Rays are generated at the target resolution rather than resized: interpolating
    a direction field denormalizes it and bends the directions near the rim, where
    the fisheye changes fastest.
    """
    import cv2

    source_hw = tuple(np.asarray(image).shape[:2])
    resized = (
        image if source_hw == (height, width)
        else cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array(DINOV2_MEAN, dtype=np.float32)) / np.array(
        DINOV2_STD, dtype=np.float32
    )
    normalized = np.ascontiguousarray(array.transpose(2, 0, 1))

    cache = ray_cache or RayFieldCache()
    rays = cache.directions_copy(calibration, grid_hw=(height, width))
    return ViewTensors(
        image=normalized, rays=rays, camera_id=calibration.camera_id,
        source_hw=source_hw, target_hw=(height, width),
    )


def to_model_views(view_tensors: Sequence[ViewTensors], device: str, dtype=None) -> list:
    """Assemble `forward`-ready view dicts. No depth key, ever.

    Note the key is `ray_directions_cam`: `forward` reads that name.
    `ray_directions` (the `infer` name) would be ignored here and the model would
    run images-only while the log claimed otherwise.
    """
    import torch

    views = []
    for tensors in view_tensors:
        image = torch.from_numpy(tensors.image)[None].to(device)
        if dtype is not None:
            image = image.to(dtype)
        rays = torch.from_numpy(tensors.rays)[None].to(device)
        rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-8)
        views.append({
            "img": image,
            "ray_directions_cam": rays.to(torch.float32),
            "data_norm_type": ["dinov2"],
            "is_metric_scale": torch.ones(1, dtype=torch.bool, device=device),
        })
    return views


def verify_rays_reached_model(
    model_views: Sequence[dict], view_tensors: Sequence[ViewTensors],
    expect_poses: bool = False,
) -> dict:
    """Confirm the tensors the model got are the DS rays we built.

    Checks the key name, the shape, unit norm, and that the six views are
    genuinely different (a copied camera-2 field would make them identical). This
    is the guard against the upstream pinhole-refit path silently reappearing.
    """
    import torch

    report = {"n_views": len(model_views), "checks": {}, "ok": True}

    has_rays = all("ray_directions_cam" in view for view in model_views)
    report["checks"]["ray_key_present"] = has_rays
    report["checks"]["no_intrinsics_key"] = not any(
        "intrinsics" in view for view in model_views
    )
    report["checks"]["no_depth_key"] = not any(
        key in view for view in model_views
        for key in ("depth_z", "depth_along_ray", "sparse_depth")
    )
    has_pose = any(
        key in view for view in model_views
        for key in ("camera_poses", "camera_pose_quats", "camera_pose_trans")
    )
    if expect_poses:
        report["checks"]["all_rig_pose_keys_present"] = all(
            "camera_pose_quats" in v and "camera_pose_trans" in v
            and bool(v["is_metric_scale"].all()) for v in model_views
        )
    else:
        report["checks"]["no_pose_key"] = not has_pose

    max_deviation = 0.0
    max_difference = 0.0
    for position, (view, tensors) in enumerate(zip(model_views, view_tensors)):
        rays = view["ray_directions_cam"][0].detach().float().cpu().numpy()
        expected = tensors.rays / (
            np.linalg.norm(tensors.rays, axis=-1, keepdims=True) + 1e-8
        )
        max_difference = max(max_difference, float(np.abs(rays - expected).max()))
        norms = np.linalg.norm(rays, axis=-1)
        max_deviation = max(max_deviation, float(np.abs(norms - 1.0).max()))

    report["checks"]["rays_match_built_ds_field"] = max_difference < 1e-4
    report["max_abs_difference_from_built_rays"] = max_difference
    report["checks"]["rays_unit_norm"] = max_deviation < 1e-3
    report["max_unit_norm_deviation"] = max_deviation

    if len(model_views) > 1:
        signatures = {
            view["ray_directions_cam"][0, 100, 100].detach().float().cpu().numpy().tobytes()
            for view in model_views
        }
        report["n_distinct_ray_fields"] = len(signatures)
        report["checks"]["per_camera_rays_distinct"] = len(signatures) == len(model_views)

    report["ok"] = all(report["checks"].values())
    report["note"] = (
        "Guards against mapanything.utils.image.preprocess_inputs, which replaces "
        "supplied ray_directions with a pinhole K refitted from them."
    )
    return report


def rig_camera_poses(calibrations, view_ids: Sequence[int]) -> list:
    """Camera-to-rig SE(3), with the release's metre translations unchanged.

    MapAnything expects camera-to-world. The rigid sensor body is a valid world
    frame for a synchronized timestamp; the model computes inv(T_0) @ T_i.
    Do not invert each T_body_cam or substitute an identity for a missing pose.
    """
    poses = []
    for camera_id in view_ids:
        cal = calibrations[camera_id]
        if cal.camera_id != camera_id or cal.T_body_cam is None:
            raise ValueError(f'cam{camera_id}: calibrated camera-to-rig pose missing or mismatched')
        pose = np.asarray(cal.T_body_cam, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f'cam{camera_id}: invalid rig pose array')
        rotation = pose[:3, :3]
        if not (np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError(f'cam{camera_id}: rig pose is not a rigid SE(3) transform')
        poses.append(pose.astype(np.float32))
    return poses


def to_rig_model_views(view_tensors, calibrations, device: str) -> list:
    """Use the official direct-ray/pose preprocessor, preserving metric baselines."""
    import torch
    from mapanything.utils.inference import preprocess_input_views_for_inference

    poses = rig_camera_poses(calibrations, [t.camera_id for t in view_tensors])
    external = []
    for tensors, pose in zip(view_tensors, poses):
        external.append({
            'img': torch.from_numpy(tensors.image)[None].to(device),
            # Official preprocessing normalizes these exactly once.
            'ray_directions': torch.from_numpy(tensors.rays)[None].to(device),
            'camera_poses': torch.from_numpy(pose)[None].to(device),
            'data_norm_type': ['dinov2'],
            'is_metric_scale': torch.ones(1, dtype=torch.bool, device=device),
        })
    return preprocess_input_views_for_inference(external)


def extract_depth_z(prediction: dict) -> np.ndarray:
    """Camera-frame optical-axis Z from one view's prediction.

    `pts3d_cam[..., 2]` is Z. `depth_along_ray` is range and reads long off-axis
    by up to 1/cos(theta) -- about 8x at this lens's rim.
    """
    import torch

    if "pts3d_cam" in prediction:
        value = prediction["pts3d_cam"][..., 2]
    elif "depth_z" in prediction:
        value = prediction["depth_z"]
        if value.dim() == 4 and value.shape[-1] == 1:
            value = value[..., 0]
    elif "depth_along_ray" in prediction and "ray_directions" in prediction:
        depth_along_ray = prediction["depth_along_ray"]
        if depth_along_ray.dim() == 4 and depth_along_ray.shape[-1] == 1:
            depth_along_ray = depth_along_ray[..., 0]
        value = depth_along_ray * prediction["ray_directions"][..., 2]
    else:
        raise KeyError(
            f"no usable depth output; keys: {sorted(prediction.keys())}"
        )
    return value.squeeze().detach().float().cpu().numpy()


def extract_support_mask(prediction: dict, shape_hw: tuple) -> np.ndarray:
    """The model's own declared support, without extra filtering.

    `non_ambiguous_mask` is the model's statement about where its output means
    something. A learned `conf` threshold is deliberately NOT applied: the E3
    contract keeps the model's legal support and does not add a confidence filter
    that would remove the moving hand and object.
    """
    import torch

    for key in ("non_ambiguous_mask", "mask"):
        if key in prediction:
            mask = prediction[key]
            if mask.dim() == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            array = mask.squeeze().detach().float().cpu().numpy()
            if array.shape == tuple(shape_hw):
                return array > 0.5
    return np.ones(tuple(shape_hw), dtype=bool)


def resample_to_scoring_grid(
    depth_z: np.ndarray, support: np.ndarray, grid_hw: tuple,
    require_positive: bool = True,
) -> tuple:
    """Bring a prediction onto the scoring grid using ONE consistent kernel.

    THE BUG THIS REPLACES
    Depth used INTER_LINEAR while support used INTER_AREA. Different kernels see
    different neighbourhoods, so an output pixel could be marked valid by the area
    test while its depth had been blended with a neighbour the model never
    supported. The audit's counter-example: a 406x518 map of all 1.0 with row 41
    unsupported and set to 101.0 produced 512 pixels still marked valid whose depth
    had become 1.2403846.

    The fix is a weighted resample where the value and the weight go through the
    SAME kernel: unsupported pixels are zeroed, both the masked values and the mask
    are resampled with INTER_AREA, and the value is normalized by the resampled
    mask. An output pixel's depth is then a weighted average of *supported* inputs
    only. Support is additionally required to be near-complete, so a pixel that
    only partly overlaps supported input is dropped rather than half-guessed.

    Non-finite and non-positive inputs are folded into the support before
    resampling, because an inf or a negative in the numerator would poison every
    output pixel whose kernel touches it.
    """
    import cv2

    grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
    depth_z = np.asarray(depth_z, dtype=np.float32)
    support = np.asarray(support, dtype=bool)

    # A value the model declared but that is not usable is not support.
    # Affine depth/disparity can legitimately be zero (or negative for affine
    # depth) before alignment. Physical Z requires positivity; a representation
    # awaiting alignment must preserve its own support through resampling.
    usable = support & np.isfinite(depth_z)
    if require_positive:
        usable &= depth_z > 0
    if depth_z.shape == grid_hw:
        return depth_z, usable

    values = np.where(usable, depth_z, 0.0).astype(np.float32)
    weights = usable.astype(np.float32)

    resized_values = cv2.resize(
        values, (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_AREA
    )
    resized_weights = cv2.resize(
        weights, (grid_hw[1], grid_hw[0]), interpolation=cv2.INTER_AREA
    )

    # Near-complete support only. Below this the output pixel straddles the support
    # boundary and its value would be a partial average.
    keep = resized_weights >= 0.999
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(keep, resized_values / np.maximum(resized_weights, 1e-8), np.nan)
    return normalized.astype(np.float32), keep


class MapAnythingRunner:
    """Six/two/one-view MapAnything inference on the canonical sample contract."""

    def __init__(
        self,
        model,
        device: str = "cuda",
        view_ids: Sequence[int] = tuple(range(6)),
        target_camera_id: int = TARGET_CAMERA_ID,
        height: int = 406,
        width: int = 518,
        amp_dtype: str = "bf16",
        scale_factor: float = 1.0,
        pose_mode: str = "none",
    ):
        if target_camera_id not in tuple(view_ids):
            raise ValueError(
                f"target camera {target_camera_id} is not among the input views "
                f"{list(view_ids)}: the scored camera must be one the model sees"
            )
        self.model = model
        self.device = device
        self.view_ids = tuple(int(v) for v in view_ids)
        self.target_camera_id = int(target_camera_id)
        self.height = snap_to_patch(height)
        self.width = snap_to_patch(width)
        self.amp_dtype = amp_dtype
        # For the E4 Scale-only arm: one fixed positive multiplier from training.
        self.scale_factor = float(scale_factor)
        if pose_mode not in ("none", "rig"):
            raise ValueError(f'Unsupported MapAnything pose_mode: {pose_mode}')
        self.pose_mode = pose_mode
        self.ray_cache = RayFieldCache()
        self.last_verification: Optional[dict] = None
        self._configure_geometric_inputs()

    def _configure_geometric_inputs(self) -> None:
        """Rays on, depth off; metric rig poses only in the declared rig arm."""
        self.model.geometric_input_config.update({
            "overall_prob": 1.0, "dropout_prob": 0.0,
            "ray_dirs_prob": 1.0,       # calibrated rays always supplied
            "depth_prob": 0.0,          # no depth hints, ever
            "cam_prob": float(self.pose_mode == "rig"),
            "sparse_depth_prob": 0.0,
            "depth_scale_norm_all_prob": 0.0,
            "pose_scale_norm_all_prob": 0.0,
        })

    def predict(self, loaded) -> dict:
        """One synchronized timestamp -> camera-2 Z on the scoring grid."""
        import torch

        view_tensors = [
            build_view_tensors(
                loaded.images[camera_id], loaded.calibrations[camera_id],
                self.height, self.width, self.ray_cache,
            )
            for camera_id in self.view_ids
        ]
        model_views = (to_rig_model_views(view_tensors, loaded.calibrations, self.device)
                       if self.pose_mode == 'rig' else to_model_views(view_tensors, self.device))
        self.last_verification = verify_rays_reached_model(
            model_views, view_tensors, expect_poses=self.pose_mode == 'rig')
        if not self.last_verification["ok"]:
            raise RuntimeError(
                "ray verification failed before forward: "
                f"{self.last_verification['checks']}"
            )

        # A shared model may have been used by another runner/API call.
        self._configure_geometric_inputs()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[self.amp_dtype]
        with torch.no_grad():
            with torch.autocast(
                "cuda" if str(self.device).startswith("cuda") else "cpu",
                dtype=amp_dtype, enabled=self.amp_dtype != "fp32",
            ):
                predictions = self.model(model_views)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Position, not a fixed index: with view_ids=[2,3] the target is at 0.
        position = self.view_ids.index(self.target_camera_id)
        prediction = predictions[position]
        depth_z = extract_depth_z(prediction)
        support = extract_support_mask(prediction, depth_z.shape)

        grid_hw = loaded.grid_hw
        depth_on_grid, support_on_grid = resample_to_scoring_grid(
            depth_z, support, grid_hw
        )
        if self.scale_factor != 1.0:
            depth_on_grid = depth_on_grid * self.scale_factor

        return {
            "depth_z": depth_on_grid,
            "valid": support_on_grid,
            # `raw` is the output on the MODEL's own grid with the model's own
            # support, before any resampling or scale factor. Saving the already
            # transformed array as "raw" would make an oracle refit impossible to
            # reproduce from the file.
            "raw": depth_z,
            "raw_valid": support,
            "raw_grid_hw": list(depth_z.shape),
            "raw_space": "metric_depth_z_model_grid",
            "raw_transform": (
                "support-weighted INTER_AREA to the scoring grid; support requires "
                ">=0.999 coverage by usable input"
            ),
            "timing_ms": elapsed_ms,
            "peak_gpu_gb": (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else None
            ),
            "model_input_hw": [self.height, self.width],
            "scoring_grid_hw": list(grid_hw),
            "view_ids": list(self.view_ids),
            "target_view_position": position,
            "scale_factor_applied": self.scale_factor,
            "pose_mode": self.pose_mode,
            "ray_verification": self.last_verification,
            "output_keys": sorted(prediction.keys()),
        }
