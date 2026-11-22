"""Training dataset: six RGB views plus per-camera rays, camera-2 depth as target.

FOUR CONTRACT POINTS ENFORCED HERE

1. The reference depth is a *target*, never an input. It is returned under
   `target_depth_z` / `target_valid` and the view dicts handed to the model carry
   no depth key at all. `assert_no_reference_in_inputs` re-checks this at the call
   site, because a `depth_z` key would silently switch MapAnything into
   depth-conditioned mode and the loss would then be predicting its own input.

2. Rays are per camera. The six cameras' principal points differ, so camera 2's
   rays are wrong for the other five.

3. Camera-2 depth is never copied to the other five views. There is no reference
   for them; a zero-filled GT would train the model to predict zero depth.

4. Images and rays go through the SAME transform. A resize applied to the image
   but not to the rays leaves the model with rays describing a different camera
   than the pixels it sees.

SAMPLING is site-balanced: a site is drawn uniformly, then a recording within it,
then a timestamp. A large site would otherwise dominate the gradient exactly as it
would dominate a globally pooled score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.data.contract import NUM_CAMERAS, TARGET_CAMERA_ID, CanonicalSample
from egodepth_eval.data.mask_profile import MaskProfile
from egodepth_eval.data.reader import SampleReader
from egodepth_eval.score.rays import RayFieldCache

# DINOv2 normalization, matching MapAnything's "dinov2" data_norm_type. Written
# out here rather than imported so the training transform does not depend on the
# upstream inference helper that also refits rays to a pinhole K.
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingBatchSpec:
    """Resolution and view contract for a training run. Must match evaluation."""

    height: int = 406            # multiple of 14 for DINOv2 patches
    width: int = 518
    view_ids: tuple = tuple(range(NUM_CAMERAS))
    target_camera_id: int = TARGET_CAMERA_ID
    data_norm_type: str = "dinov2"
    pose_mode: str = "none"

    def to_dict(self) -> dict:
        return {
            "height": self.height, "width": self.width,
            "view_ids": list(self.view_ids),
            "target_camera_id": self.target_camera_id,
            "data_norm_type": self.data_norm_type,
            "pose_mode": self.pose_mode,
        }


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW, DINOv2-normalized."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(DINOV2_MEAN, dtype=np.float32)) / np.array(
        DINOV2_STD, dtype=np.float32
    )
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def resize_reference(
    depth: np.ndarray, valid: np.ndarray, height: int, width: int
) -> tuple:
    """Resample the reference to the training grid.

    Depth uses nearest neighbour and validity is eroded by requiring the
    downsampled valid mask to be fully valid. Bilinear depth across a depth
    discontinuity invents a surface at an intermediate distance that exists
    nowhere, and the loss would then be fitting an artifact.
    """
    import cv2

    if depth.shape == (height, width):
        return depth.astype(np.float32), valid.astype(bool)
    resized_depth = cv2.resize(
        depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
    )
    # Area-average the mask and keep only fully-valid cells.
    resized_valid = cv2.resize(
        valid.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    return resized_depth, resized_valid >= 0.999


class SixViewTrainingDataset:
    """Site-balanced six-view samples with camera-2 supervision.

    Plain iterable rather than a torch Dataset subclass so it can be exercised
    without torch; `torch_dataset()` wraps it when torch is available.
    """

    def __init__(
        self,
        samples: Sequence[CanonicalSample],
        reader: SampleReader,
        spec: Optional[TrainingBatchSpec] = None,
        seed: int = 20260912,
    ):
        self.samples = list(samples)
        self.reader = reader
        self.spec = spec or TrainingBatchSpec()
        self.seed = int(seed)
        self.ray_cache = RayFieldCache()

        self.by_site: dict = {}
        for index, sample in enumerate(self.samples):
            site = sample.physical_site_id or f"unmapped::{sample.recording_id}"
            self.by_site.setdefault(site, {}).setdefault(sample.recording_id, []).append(index)
        self.sites = sorted(self.by_site)
        if not self.sites:
            raise ValueError("no samples supplied to the training dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def site_balanced_order(self, n_draws: int, epoch: int = 0) -> list:
        """Sample indices: uniform site, then uniform recording, then timestamp.

        Equal probability per site regardless of size, and spread over recordings
        within a site, so neither a large site nor one long recording dominates.
        """
        rng = np.random.default_rng(self.seed + 1009 * epoch)
        return [int(index) for index in self._draw(rng, n_draws)]

    def _draw(self, rng, n_draws: int):
        for _ in range(n_draws):
            site = self.sites[int(rng.integers(len(self.sites)))]
            recordings = sorted(self.by_site[site])
            recording = recordings[int(rng.integers(len(recordings)))]
            indices = self.by_site[site][recording]
            yield int(indices[int(rng.integers(len(indices)))])

    def site_balanced_stream(self, n_draws: int, epoch: int = 0, start: int = 0):
        """The same order as a generator, so a long budget is never materialized.

        `max_steps x grad_accum_steps` indices at 10k steps is 40,000 draws; the
        indices themselves are cheap, but the previous code turned each into a
        loaded six-view tensor batch up front. At ~31 MB per item that is about
        1.25 TB of RAM for a 10k-step run. Pixels are now loaded per batch by the
        DataLoader and released after the step.

        `start` skips the first N draws, which is how a resumed run continues from
        the sampling position it stopped at rather than restarting the stream.
        """
        rng = np.random.default_rng(self.seed + 1009 * epoch)
        for position, index in enumerate(self._draw(rng, n_draws)):
            if position >= start:
                yield index

    def build_item(self, index: int) -> dict:
        """One training item: normalized images, per-camera rays, camera-2 target."""
        sample = self.samples[index]
        loaded = self.reader.load(sample, cameras=list(self.spec.view_ids))
        height, width = self.spec.height, self.spec.width

        views: list = []
        if self.spec.pose_mode not in ('none', 'rig'):
            raise ValueError('Unknown training pose_mode')
        poses = None
        if self.spec.pose_mode == 'rig':
            from egodepth_eval.infer.mapanything_backend import rig_camera_poses
            poses = rig_camera_poses(loaded.calibrations, self.spec.view_ids)
        for camera_id in self.spec.view_ids:
            image = resize_image(loaded.images[camera_id], height, width)
            calibration = loaded.calibrations[camera_id]
            # The image was resized, so the rays are built on the SAME grid from
            # rescaled intrinsics. Building them on the source grid and resizing
            # the ray array would interpolate directions and denormalize them.
            rays = self.ray_cache.directions_copy(calibration, grid_hw=(height, width))
            views.append({
                "img": normalize_image(image),
                "ray_directions": rays,
                "data_norm_type": self.spec.data_norm_type,
                "camera_id": int(camera_id),
            })
            if poses is not None:
                views[-1]['camera_poses'] = poses[len(views) - 1]

        depth, valid = resize_reference(
            loaded.reference_depth, loaded.reference_valid, height, width
        )
        return {
            "views": views,
            "target_depth_z": depth,
            "target_valid": valid,
            "target_camera_id": self.spec.target_camera_id,
            "target_view_position": list(self.spec.view_ids).index(
                self.spec.target_camera_id
            ),
            "sample_token": sample.sample_token,
            "site_id": sample.physical_site_id,
            "recording_id": sample.recording_id,
            "mask_audit": loaded.mask_audit,
        }

    def torch_dataset(self, index_order: Optional[Sequence[int]] = None):
        """A torch Dataset over this sample set.

        `index_order` maps position -> sample index, which lets a DataLoader walk a
        precomputed site-balanced order (cheap: a list of ints) while loading
        pixels lazily. Without it the dataset is indexed directly.
        """
        import torch
        from torch.utils.data import Dataset

        outer = self
        order = list(index_order) if index_order is not None else None

        class _Wrapped(Dataset):
            def __len__(self):
                return len(order) if order is not None else len(outer)

            def __getitem__(self, index):
                real = order[int(index)] if order is not None else int(index)
                item = outer.build_item(real)
                item["views"] = [
                    {
                        "img": torch.from_numpy(view["img"]),
                        "ray_directions": torch.from_numpy(view["ray_directions"]),
                        "data_norm_type": view["data_norm_type"],
                        "camera_id": view["camera_id"],
                        **({'camera_poses': torch.from_numpy(view['camera_poses'])}
                           if 'camera_poses' in view else {}),
                    }
                    for view in item["views"]
                ]
                item["target_depth_z"] = torch.from_numpy(item["target_depth_z"])
                item["target_valid"] = torch.from_numpy(item["target_valid"])
                return item

        return _Wrapped()


def collate_views(items: Sequence[dict]) -> dict:
    """Batch a list of items into MapAnything's per-view batched format.

    MapAnything wants a list over views, each holding a (B, ...) tensor -- the
    transpose of the natural per-sample layout.
    """
    import torch

    if not items:
        raise ValueError("empty batch")
    n_views = len(items[0]["views"])
    views: list = []
    for position in range(n_views):
        images = torch.stack([
            item["views"][position]["img"]
            if torch.is_tensor(item["views"][position]["img"])
            else torch.from_numpy(item["views"][position]["img"])
            for item in items
        ])
        rays = torch.stack([
            item["views"][position]["ray_directions"]
            if torch.is_tensor(item["views"][position]["ray_directions"])
            else torch.from_numpy(item["views"][position]["ray_directions"])
            for item in items
        ])
        views.append({
            "img": images,
            "ray_directions": rays,
            "data_norm_type": [items[0]["views"][position]["data_norm_type"]] * len(items),
            "camera_id": items[0]["views"][position]["camera_id"],
        })
        pose_present = ['camera_poses' in item['views'][position] for item in items]
        if any(pose_present):
            if not all(pose_present): raise ValueError('Mixed pose contracts in one batch')
            views[-1]['camera_poses'] = torch.stack([
                torch.as_tensor(item['views'][position]['camera_poses']) for item in items])

    def _stack(key):
        return torch.stack([
            item[key] if torch.is_tensor(item[key]) else torch.from_numpy(item[key])
            for item in items
        ])

    return {
        "views": views,
        "target_depth_z": _stack("target_depth_z"),
        "target_valid": _stack("target_valid"),
        "target_view_position": items[0]["target_view_position"],
        "target_camera_id": items[0]["target_camera_id"],
        "sample_tokens": [item["sample_token"] for item in items],
        "site_ids": [item["site_id"] for item in items],
        "recording_ids": [item["recording_id"] for item in items],
    }


def assert_no_reference_in_inputs(views: Sequence[dict]) -> None:
    """Fail loudly if any reference-derived key reached the model inputs.

    MapAnything accepts `depth_z` / `depth_along_ray` as *inputs*. If one slipped
    into a view, the network would be handed the answer, the loss would drop, and
    nothing else in the run would look wrong. Cheap to check, so it is checked on
    every step.
    """
    forbidden = {"depth_z", "depth_along_ray", "sparse_depth", "target_depth_z",
                 "target_valid", "depth"}
    for position, view in enumerate(views):
        present = forbidden & set(view.keys())
        if present:
            raise AssertionError(
                f"view {position} carries reference-derived input keys {sorted(present)}: "
                "camera-2 depth is a target only and must never enter the model input"
            )


def load_samples_jsonl(path: Path) -> list:
    return [
        CanonicalSample.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines() if line.strip()
    ]
