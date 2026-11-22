"""Load pixels for a CanonicalSample: six RGB images plus camera-2 reference Z.

Used identically by inference, AI annotation, training and scoring, so a
resize/crop convention cannot drift between them.

ROOT RELOCATION. Relative asset paths are resolved against an ordered list of
roots, so a read-only cache holding a development subset can sit in front of the
privileged source directory without rewriting the manifest. The first root that
holds the file wins, and which root served each file is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import io

import numpy as np

from egodepth_eval.data.contract import (
    NUM_CAMERAS,
    TARGET_CAMERA_ID,
    CanonicalSample,
    DSCalibration,
)
from egodepth_eval.data.mask_profile import (
    MaskProfile,
    load_mask_array,
    resolve_reference_validity,
)


@dataclass
class LoadedSample:
    """Pixels for one synchronized timestamp.

    `reference_valid` is the reference-validity domain only. The benchmark's
    Z interval is applied in the scorer, so "no reference here" and "reference
    outside [0.25, 2.0]" stay distinguishable -- they mean different things for
    coverage.
    """

    sample: CanonicalSample
    images: dict                    # {camera_id: HxWx3 uint8 RGB}
    reference_depth: np.ndarray     # (H, W) float32, optical-axis Z in metres
    reference_valid: np.ndarray     # (H, W) bool
    calibrations: dict              # {camera_id: DSCalibration on its image grid}
    reference_calibration: DSCalibration
    mask_audit: dict = field(default_factory=dict)
    resolved_roots: dict = field(default_factory=dict)

    @property
    def grid_hw(self) -> tuple:
        return tuple(self.reference_depth.shape)


class AssetResolver:
    """Resolve manifest-relative paths against an ordered list of roots."""

    def __init__(self, roots: Sequence[Path]):
        self.roots = [Path(root) for root in roots]
        if not self.roots:
            raise ValueError("at least one asset root is required")
        self._misses: dict = {}

    def resolve(self, relative: str) -> tuple[Path, Path]:
        """Return `(absolute_path, root_that_served_it)`."""
        relative = str(relative).strip()
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe relative asset path {relative!r}")
        for root in self.roots:
            candidate = root / relative
            try:
                if candidate.exists():
                    return candidate, root
            except PermissionError:
                # A root this process cannot traverse is skipped, not fatal: the
                # usual setup is a readable cache in front of a privileged source.
                continue
        self._misses[relative] = self._misses.get(relative, 0) + 1
        raise FileNotFoundError(
            f"{relative} not found under any of: "
            + ", ".join(str(root) for root in self.roots)
        )

    @property
    def misses(self) -> dict:
        return dict(self._misses)

    def read_bytes(self, relative: str) -> tuple[bytes, Path]:
        """Read a file or a bounded member of an uncompressed release shard.

        Locator: relative/archive.tar::byte_offset:byte_length. Offsets are
        generated from the actual tar writer and covered by shard checksums.
        No extraction, privileged worker, or absolute source path is needed.
        """
        if '::' not in relative:
            path, root = self.resolve(relative)
            return path.read_bytes(), root
        archive, span = relative.rsplit('::', 1)
        offset, length = map(int, span.split(':'))
        if offset < 0 or length <= 0:
            raise ValueError('invalid shard byte range')
        path, root = self.resolve(archive)
        with path.open('rb') as handle:
            if offset + length > path.stat().st_size:
                raise ValueError('shard byte range exceeds archive')
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise IOError('truncated shard member')
        return payload, root


class SampleReader:
    """Turn a CanonicalSample into pixels under one fixed validity profile."""

    def __init__(
        self,
        asset_roots: Sequence[Path],
        mask_profile: MaskProfile,
        target_camera_id: int = TARGET_CAMERA_ID,
    ):
        self.resolver = AssetResolver(asset_roots)
        self.mask_profile = mask_profile
        self.target_camera_id = int(target_camera_id)

    def load(
        self,
        sample: CanonicalSample,
        load_images: bool = True,
        cameras: Optional[Sequence[int]] = None,
    ) -> LoadedSample:
        import cv2

        sample.validate()
        cameras = list(range(NUM_CAMERAS)) if cameras is None else [int(c) for c in cameras]

        depth_payload, depth_root = self.resolver.read_bytes(sample.depth_rel_path)
        depth_path = sample.depth_rel_path
        depth = np.load(io.BytesIO(depth_payload), allow_pickle=False)
        if depth.ndim != 2:
            raise ValueError(f"{depth_path}: expected a 2D depth array, got {depth.shape}")
        # Stored float16; every computation downstream is float32 or better.
        depth = depth.astype(np.float32)

        # NPY first, then packbits BIN. A missing NPY must fall through to the BIN
        # rather than raising here: a release that ships only packed masks is a
        # legitimate form, and raising at the NPY lookup made the BIN branch
        # unreachable. `require_mask_file` is enforced once, after BOTH forms have
        # been tried.
        declared = None
        packed_payload = None
        resolved_mask_root = None
        mask_sources: dict = {"npy": "not_declared", "bin": "not_declared"}

        if sample.valid_mask_rel_path:
            mask_sources["npy"] = "declared_missing"
            try:
                mask_bytes, mask_root = self.resolver.read_bytes(sample.valid_mask_rel_path)
                declared = np.load(io.BytesIO(mask_bytes), allow_pickle=False)
                if declared.ndim != 2 or not np.all((declared == 0) | (declared == 1)):
                    raise ValueError('invalid reference mask array')
                mask_sources["npy"] = "loaded"
                resolved_mask_root = str(mask_root)
            except FileNotFoundError:
                pass

        if declared is None and sample.packed_valid_mask_rel_path:
            mask_sources["bin"] = "declared_missing"
            try:
                packed_payload, packed_root = self.resolver.read_bytes(
                    sample.packed_valid_mask_rel_path
                )
                mask_sources["bin"] = "loaded"
                resolved_mask_root = str(packed_root)
            except FileNotFoundError:
                pass

        if (
            declared is None
            and packed_payload is None
            and self.mask_profile.require_mask_file
        ):
            raise FileNotFoundError(
                f"{sample.sample_token}: profile {self.mask_profile.name!r} requires a "
                f"validity mask, and neither form resolved. "
                f"npy={sample.valid_mask_rel_path!r} ({mask_sources['npy']}), "
                f"bin={sample.packed_valid_mask_rel_path!r} ({mask_sources['bin']}). "
                "A declared path is not evidence the file exists."
            )

        # Bit order comes from the sample's own mask_encoding, not a hardcoded
        # default: a batch written big-endian would otherwise decode to a mask
        # that is wrong per byte while still being the right shape.
        bitorder = "little"
        if isinstance(sample.mask_encoding, dict):
            bitorder = str(sample.mask_encoding.get("bitorder", "little"))

        reference_valid, mask_audit = resolve_reference_validity(
            depth, self.mask_profile, mask_array=declared,
            packed_payload=packed_payload, packed_bitorder=bitorder,
        )
        mask_audit["mask_sources"] = mask_sources
        mask_audit["declared_bitorder"] = bitorder

        images: dict = {}
        resolved_roots: dict = {"depth_cam2": str(depth_root)}
        if resolved_mask_root:
            resolved_roots["camera2_valid_mask"] = resolved_mask_root
        grid_hw = depth.shape
        calibrations: dict = {}

        for camera_id in cameras:
            calibration = sample.calibrations[camera_id]
            if load_images:
                image_bytes, image_root = self.resolver.read_bytes(
                    sample.rgb_rel_paths[camera_id]
                )
                bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"{sample.rgb_rel_paths[camera_id]}: unreadable image")
                images[camera_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                resolved_roots[f"rgb_cam{camera_id}"] = str(image_root)
                image_hw = images[camera_id].shape[:2]
            else:
                image_hw = (calibration.height, calibration.width)

            # The calibration is declared on ref_size. If the actual image is a
            # different grid -- which future release assets may well be -- scale
            # rather than assuming they match. Assuming is silent: the rays stay
            # plausible and the depth is wrong by the resolution ratio.
            if (calibration.height, calibration.width) != tuple(image_hw):
                calibration = calibration.scaled_to(image_hw[0], image_hw[1])
            calibrations[camera_id] = calibration

        reference_calibration = sample.calibrations[self.target_camera_id]
        if (reference_calibration.height, reference_calibration.width) != tuple(grid_hw):
            reference_calibration = reference_calibration.scaled_to(*grid_hw)

        return LoadedSample(
            sample=sample,
            images=images,
            reference_depth=depth,
            reference_valid=reference_valid,
            calibrations=calibrations,
            reference_calibration=reference_calibration,
            mask_audit=mask_audit,
            resolved_roots=resolved_roots,
        )


def stage_assets(
    samples: Sequence[CanonicalSample],
    source_roots: Sequence[Path],
    cache_root: Path,
    include_masks: bool = True,
) -> dict:
    """Copy a subset's assets into a user-readable cache, preserving relative paths.

    The source directory needs privileged access on this host, so development
    work runs against a cache. Relative paths are preserved exactly, which is
    what lets the same manifest resolve against either root.
    """
    import shutil

    resolver = AssetResolver(source_roots)
    cache_root = Path(cache_root)
    counters = {"copied": 0, "already_present": 0, "missing": 0,
                "missing_optional": 0}
    missing: list = []

    for sample in samples:
        wanted = [sample.depth_rel_path, *sample.rgb_rel_paths.values()]
        if include_masks:
            # Both mask forms: a BIN-only release must be cacheable too, and a
            # declared path that does not exist is skipped by the resolver below
            # rather than treated as a missing asset.
            for candidate in (sample.valid_mask_rel_path,
                             sample.packed_valid_mask_rel_path):
                if candidate:
                    wanted.append(candidate)
        for relative in wanted:
            destination = cache_root / relative
            if destination.exists():
                counters["already_present"] += 1
                continue
            try:
                source, _ = resolver.resolve(relative)
            except FileNotFoundError:
                # A declared-but-absent mask form is expected (the 336L batches
                # declare packed masks they do not ship), so it is counted
                # separately from a genuinely missing image or depth file.
                optional = relative in (
                    sample.valid_mask_rel_path, sample.packed_valid_mask_rel_path
                )
                counters["missing_optional" if optional else "missing"] += 1
                if not optional and len(missing) < 20:
                    missing.append(relative)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            counters["copied"] += 1

    return {"cache_root": str(cache_root), "counts": counters, "missing": missing}
