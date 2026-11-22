"""Release-manifest reads, mask direction, camera intrinsics and asset relocation."""

import json
import sqlite3

import numpy as np
import pytest

from egodepth_eval.data.contract import NUM_CAMERAS, CanonicalSample, DSCalibration
from egodepth_eval.data.manifest_store import ManifestStore
from egodepth_eval.data.mask_profile import (
    PROFILE_336L_LEGACY_20260828,
    PROFILE_RELEASE_DECLARED,
    decode_packed_mask,
    get_profile,
    resolve_reference_validity,
)
from egodepth_eval.data.reader import AssetResolver, SampleReader


def release_sample(shape):
    prefix = "rec/frames/tok"
    return CanonicalSample(
        sample_token="tok",
        recording_id="rec",
        timestamp_ns=1784188138985604865,
        rgb_rel_paths={i: f"{prefix}/camera{i}.jpg" for i in range(NUM_CAMERAS)},
        depth_rel_path=f"{prefix}/camera2.depth.npy",
        depth_shape_hw=shape,
        calibrations={
            i: DSCalibration(i, 164.25, 164.75, 260 + i, 214, 0.005, 0.57, 512, 416)
            for i in range(NUM_CAMERAS)
        },
    )


@pytest.mark.parametrize("shared_calibration", [True, False])
def test_indexed_manifest_roundtrip(tmp_path, shared_calibration):
    sample = release_sample((416, 512))
    record = sample.to_dict()
    if shared_calibration:
        calibrations = record.pop("calibrations")
        record["calibration_id"] = "rig"
        (tmp_path / "calibrations.json").write_text(json.dumps({"rig": calibrations}))
    payload = json.dumps(record).encode() + b"\n"
    (tmp_path / "samples.jsonl").write_bytes(payload)
    (tmp_path / "index_meta.json").write_text("{}")
    with sqlite3.connect(tmp_path / "index.sqlite") as connection:
        connection.execute(
            "CREATE TABLE samples (sample_token, recording_id, timestamp_ns, byte_offset, byte_length)"
        )
        connection.execute(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
            ("tok", "rec", sample.timestamp_ns, 0, len(payload)),
        )
    store = ManifestStore(tmp_path)
    try:
        assert store.tokens() == ["tok"]
        assert store.recording_samples("rec") == ["tok"]
        assert list(store.iter_samples()) == [sample]
        assert store.get("tok") == sample
        with pytest.raises(KeyError):
            store.get("missing")
    finally:
        store.close()


def test_odd_sized_depth_and_image_are_handled(tmp_path):
    """A future release may ship a different grid; the code must not hardcode 416x512."""
    shape = (208, 256)
    sample = release_sample(shape)
    assert sample.depth_shape_hw == shape

    root = tmp_path / "assets"
    frame_dir = root / "rec" / "frames" / "tok"
    frame_dir.mkdir(parents=True)
    depth = np.full(shape, 1.0, dtype=np.float16)
    np.save(frame_dir / "camera2.depth.npy", depth)
    import cv2

    for camera_id in range(NUM_CAMERAS):
        cv2.imwrite(str(frame_dir / f"camera{camera_id}.jpg"), np.zeros((*shape, 3), np.uint8))

    reader = SampleReader([root], get_profile("336l_legacy_20260828"))
    loaded = reader.load(sample)
    assert loaded.grid_hw == shape
    # The calibration was declared on 416x512 and must have been rescaled.
    assert loaded.reference_calibration.width == shape[1]
    assert loaded.reference_calibration.fx == pytest.approx(164.25 * shape[1] / 512)
    assert loaded.reference_calibration.xi == pytest.approx(0.005), (
        "xi is dimensionless and must not be rescaled"
    )
    assert loaded.reference_depth.dtype == np.float32, (
        "stored float16 must be widened before any computation"
    )


# ------------------------------------------------------------ mask profiles


def test_legacy_profile_detects_the_reversed_mask():
    """The snapshot's mask==1 never coincides with positive depth."""
    depth = np.array([[0.0, 1.0, 0.0, 1.5]], dtype=np.float32)
    stored_mask = np.array([[1, 0, 1, 0]], dtype=np.uint8)  # inverted

    valid, audit = resolve_reference_validity(
        depth, PROFILE_336L_LEGACY_20260828, mask_array=stored_mask
    )
    assert valid.ravel().tolist() == [False, True, False, True]
    assert audit["looks_inverted"] is True
    assert audit["n_mask_true_and_positive_depth"] == 0
    assert audit["n_finite_positive"] == 2
    assert audit["applied"] == "isfinite(depth) & (depth > 0)"


def test_release_profile_uses_the_declared_mask_direction():
    depth = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    declared = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    profile = PROFILE_RELEASE_DECLARED
    valid, audit = resolve_reference_validity(depth, profile, mask_array=declared)
    assert valid.ravel().tolist() == [True, True, False, False]
    assert audit["applied"] == "mask & finite & positive"
    # A quality mask legitimately drops finite positive pixels; that is recorded.
    assert audit["share_finite_positive_dropped_by_mask"] == pytest.approx(0.5)


def test_release_profile_requires_the_mask_file():
    depth = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(FileNotFoundError, match="requires a declared validity mask"):
        resolve_reference_validity(depth, PROFILE_RELEASE_DECLARED, mask_array=None)


def test_profiles_disagree_so_they_cannot_be_used_interchangeably():
    """The two profiles must give different answers on the snapshot's data."""
    depth = np.array([[0.0, 1.0, 0.0, 1.5]], dtype=np.float32)
    stored = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    legacy, _ = resolve_reference_validity(depth, PROFILE_336L_LEGACY_20260828, mask_array=stored)
    strict = PROFILE_RELEASE_DECLARED.__class__(
        name="test_release",
        validity_source="declared_mask",
        one_is_valid=True,
        require_mask_file=True,
        max_audit_disagreement=1.0,
    )
    release, _ = resolve_reference_validity(depth, strict, mask_array=stored)
    assert legacy.sum() == 2
    assert release.sum() == 0, (
        "the declared direction yields no valid pixels on this snapshot, which is "
        "exactly why the profile must be explicit"
    )


def test_packed_mask_roundtrip():
    """Odd image widths exercise byte padding."""
    height, width = 5, 11
    rng = np.random.default_rng(4)
    original = rng.integers(0, 2, size=(height, width)).astype(bool)
    payload = np.packbits(original.ravel(), bitorder="little").tobytes()
    decoded = decode_packed_mask(payload, height, width, "little")
    assert np.array_equal(decoded, original)


def test_packed_mask_rejects_a_shape_mismatch():
    """A wrong declared shape shifts every row; refuse rather than reshape."""
    payload = np.packbits(np.ones(64, dtype=bool), bitorder="little").tobytes()
    with pytest.raises(ValueError, match="more than one byte of padding"):
        decode_packed_mask(payload, 4, 4)
    with pytest.raises(ValueError, match="need"):
        decode_packed_mask(payload, 100, 100)


def test_packed_mask_rejects_nonzero_padding():
    payload = bytes([0b11111111])
    with pytest.raises(ValueError, match="padding bits are not zero"):
        decode_packed_mask(payload, 1, 3)


# --------------------------------------------------------- root relocation


def test_asset_root_relocation_gives_identical_samples(tmp_path):
    """A cache in front of the source must resolve to the same pixels."""
    import cv2

    shape = (16, 20)
    sample = release_sample(shape)

    depth = np.linspace(0.3, 1.8, shape[0] * shape[1]).reshape(shape).astype(np.float16)
    for root_name in ("source", "cache"):
        frame_dir = tmp_path / root_name / "rec" / "frames" / "tok"
        frame_dir.mkdir(parents=True)
        np.save(frame_dir / "camera2.depth.npy", depth)
        for camera_id in range(NUM_CAMERAS):
            cv2.imwrite(
                str(frame_dir / f"camera{camera_id}.jpg"), np.full((*shape, 3), 90, np.uint8)
            )

    profile = get_profile("336l_legacy_20260828")
    from_source = SampleReader([tmp_path / "source"], profile).load(sample)
    from_cache = SampleReader([tmp_path / "cache", tmp_path / "source"], profile).load(sample)
    assert np.array_equal(from_source.reference_depth, from_cache.reference_depth)
    assert np.array_equal(from_source.reference_valid, from_cache.reference_valid)
    assert "cache" in from_cache.resolved_roots["depth_cam2"]
    assert "source" in from_source.resolved_roots["depth_cam2"]


def test_resolver_rejects_escaping_paths():
    resolver = AssetResolver(["/tmp"])
    for bad in ("../etc/passwd", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError, match="unsafe relative asset path"):
            resolver.resolve(bad)


def test_missing_bin_does_not_break_the_legacy_profile(tmp_path):
    """The snapshot declares packed masks it does not ship."""
    import cv2

    shape = (8, 10)
    sample = release_sample(shape)
    frame_dir = tmp_path / "rec" / "frames" / "tok"
    frame_dir.mkdir(parents=True)
    np.save(frame_dir / "camera2.depth.npy", np.ones(shape, dtype=np.float16))
    for camera_id in range(NUM_CAMERAS):
        cv2.imwrite(str(frame_dir / f"camera{camera_id}.jpg"), np.zeros((*shape, 3), np.uint8))
    # No .valid.npy and no .valid.bin on disk at all.
    loaded = SampleReader([tmp_path], get_profile("336l_legacy_20260828")).load(sample)
    assert loaded.reference_valid.all()
    assert loaded.mask_audit["mask_file_present"] is False
