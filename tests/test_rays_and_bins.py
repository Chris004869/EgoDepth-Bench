"""Ray geometry, Z/range conversion, and bin-edge tests.

The headline test here is `test_reusing_camera2_rays_is_detectable`: the six
cameras in the 336L snapshot have genuinely different principal points, and the
whole point of per-camera ray fields is that reusing camera 2's is wrong. If that
error were undetectable at this scale, the E3/E4 contract would not be worth
enforcing.
"""

import numpy as np
import pytest

from egodepth_eval.data.contract import DSCalibration
from egodepth_eval.score.rays import (
    ANGLE_BINS,
    Z_BINS,
    RayFieldCache,
    angle_bin_index,
    angle_bin_labels,
    build_ray_field,
    z_bin_index,
    z_bin_labels,
)

# Real 336L values, from the 2026-08-28 snapshot's first record.
REAL_CALIBRATIONS = {
    0: (163.7047552330622, 164.23906197008628, 260.6878514589989, 217.98531023402256,
        0.001543306654191802, 0.5722890711633026),
    1: (164.0014352736744, 164.52632052621232, 253.98895415768573, 214.19718357741127,
        -0.0032657495149201027, 0.5713320193908591),
    2: (164.2574691977083, 164.80102674935074, 263.11646738421285, 214.06625416118294,
        0.004698253395971279, 0.5701189711160091),
    3: (164.73114550710046, 165.39218455533997, 263.2136453955822, 219.72847092068915,
        0.0035108239841461183, 0.5728004867256571),
    4: (165.16543383941, 165.67824231882, 255.39241946, 214.50894424,
        0.0000047, 0.5721578),
    5: (164.62206, 165.18332, 260.41207, 215.36703, -0.000465, 0.5732461),
}


def real_calibration(camera_id: int) -> DSCalibration:
    fx, fy, cx, cy, xi, alpha = REAL_CALIBRATIONS[camera_id]
    return DSCalibration(
        camera_id=camera_id, fx=fx, fy=fy, cx=cx, cy=cy, xi=xi, alpha=alpha,
        width=512, height=416,
    )


# ------------------------------------------------------- per-camera rays

def test_reusing_camera2_rays_is_detectable():
    """Six cameras with real, different intrinsics must give different ray fields.

    A silent reuse of camera 2's rays for camera 1 shifts the field angle by a
    measurable amount; if this assertion were loose, the E3 six-view contract
    could not be enforced.
    """
    field_1 = build_ray_field(real_calibration(1))
    field_2 = build_ray_field(real_calibration(2))

    shared = field_1.forward & field_2.forward
    assert shared.sum() > 10_000

    angle_1 = field_1.field_angle_deg()[shared]
    angle_2 = field_2.field_angle_deg()[shared]
    max_difference = float(np.nanmax(np.abs(angle_1 - angle_2)))
    # cx differs by 9.1 px between cam1 and cam2 at fx ~ 164.
    assert max_difference > 1.0, (
        f"cam1 and cam2 rays differ by only {max_difference:.3f} deg, so the "
        "per-camera contract would be untestable"
    )

    # A ray-direction mismatch also changes the metric depth a model would infer.
    range_map = np.full(field_1.grid_hw, 1.0, dtype=np.float32)
    z_correct = field_1.range_to_z(range_map)
    z_wrong = field_2.range_to_z(range_map)
    difference = np.abs(z_correct - z_wrong)[shared]
    assert float(np.nanmax(difference)) > 0.005, (
        "reusing the wrong camera's rays must change the converted Z"
    )


def test_all_six_cameras_have_distinct_ray_fields():
    fields = {c: build_ray_field(real_calibration(c)) for c in range(6)}
    signatures = set()
    for camera_id, field in fields.items():
        signature = np.round(field.directions[200, 250], 6).tobytes()
        signatures.add(signature)
    assert len(signatures) == 6, "two cameras produced identical rays at one pixel"


def test_ray_cache_keys_on_parameters_not_camera_id():
    """The same camera id with different calibration must not share a cached field."""
    cache = RayFieldCache()
    original = real_calibration(2)
    shifted = DSCalibration(
        camera_id=2, fx=original.fx, fy=original.fy,
        cx=original.cx + 12.0, cy=original.cy, xi=original.xi,
        alpha=original.alpha, width=512, height=416,
    )
    first = cache.get(original)
    second = cache.get(shifted)
    assert len(cache) == 2
    assert not np.allclose(first.directions, second.directions)


def test_cached_ray_directions_are_read_only():
    """A caller normalizing in place must not be able to poison the cache."""
    cache = RayFieldCache()
    field = cache.get(real_calibration(2))
    with pytest.raises(ValueError):
        field.directions[0, 0, 0] = 99.0
    # The sanctioned way to obtain a mutable array.
    writable = cache.directions_copy(real_calibration(2))
    writable[0, 0, 0] = 99.0
    assert cache.get(real_calibration(2)).directions[0, 0, 0] != 99.0


# ------------------------------------------------------ Z / range / angle

def test_z_range_roundtrip_on_known_rays():
    field = build_ray_field(real_calibration(2))
    z = np.full(field.grid_hw, 1.25, dtype=np.float32)
    range_map = field.z_to_range(z)
    recovered = field.range_to_z(range_map)
    forward = field.forward
    assert np.allclose(recovered[forward], z[forward], rtol=1e-5)
    # Range exceeds Z everywhere off-axis, by exactly 1/u_z.
    assert np.all(range_map[forward] >= z[forward] - 1e-6)


def test_range_to_z_scales_with_u_z_at_known_angle():
    """At the principal point Z == range; far off-axis Z is substantially smaller."""
    calibration = real_calibration(2)
    field = build_ray_field(calibration)
    centre = (int(round(calibration.cy)), int(round(calibration.cx)))
    assert field.u_z[centre] == pytest.approx(1.0, abs=1e-3)

    angle = field.field_angle_deg()
    far = np.nanmax(angle)
    assert far > 60.0, f"this fisheye should reach beyond 60 deg, got {far:.1f}"

    # Check the acos relation at a moderate angle rather than at the extreme
    # rim: near 90 degrees u_z is ~5e-5 and a relative tolerance there measures
    # float32 spacing, not the geometry.
    sampled = np.nan_to_num(angle, nan=-1.0)
    peripheral = np.unravel_index(np.argmin(np.abs(sampled - 65.0)), angle.shape)
    assert abs(float(angle[peripheral]) - 65.0) < 1.0
    expected = np.cos(np.radians(float(angle[peripheral])))
    assert field.u_z[peripheral] == pytest.approx(expected, rel=1e-4)

    # And confirm Z really does collapse relative to range off-axis.
    range_map = np.ones(field.grid_hw, dtype=np.float32)
    z = field.range_to_z(range_map)
    assert float(z[peripheral]) < 0.5, "Z should be well under range at 65 degrees"


def test_non_forward_rays_become_nan_not_zero():
    """u_z <= 0 must leave the angular strata, not land in the 80-90 bin."""
    field = build_ray_field(real_calibration(2))
    # Force a non-forward ray and confirm the conversion refuses it.
    directions = np.array(field.directions, copy=True)
    directions[0, 0] = [0.0, 1.0, 0.0]
    forward = np.array(field.forward, copy=True)
    forward[0, 0] = False
    field.directions, field.forward = directions, forward

    z = field.range_to_z(np.ones(field.grid_hw, dtype=np.float32))
    assert np.isnan(z[0, 0]), "an unconvertible ray must be NaN, not 0"
    assert np.isnan(field.field_angle_deg()[0, 0])
    assert angle_bin_index(field.field_angle_deg())[0, 0] == -1


# ------------------------------------------------------------ bin edges

def test_z_bins_partition_and_include_two_metres():
    z = np.array([[0.24, 0.25, 0.79, 0.8, 1.19, 1.2, 1.99, 2.0, 2.01]], dtype=np.float32)
    index = z_bin_index(z)
    assert index.ravel().tolist() == [-1, 0, 0, 1, 1, 2, 2, 2, -1], (
        "the 2.0 m upper bound must be included and no value may fall in two bins"
    )


def test_z_bin_membership_is_unique():
    rng = np.random.default_rng(3)
    z = rng.uniform(0.1, 2.5, size=(64, 64)).astype(np.float32)
    index = z_bin_index(z)
    total = 0
    for position, (lo, hi) in enumerate(Z_BINS):
        last = position == len(Z_BINS) - 1
        upper = (z <= hi) if last else (z < hi)
        selected = (z >= lo) & upper
        assert np.array_equal(selected, index == position)
        total += int(selected.sum())
    assert total == int((index >= 0).sum()), "bins overlap or leave a gap"


def test_angle_bins_partition_and_close_at_ninety():
    angle = np.array([[0.0, 19.9, 20.0, 39.9, 40.0, 59.9, 60.0, 79.9, 80.0, 90.0, 90.1, np.nan]],
                     dtype=np.float32)
    index = angle_bin_index(angle)
    assert index.ravel().tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1, -1]


def test_bin_labels_mark_the_closed_upper_edge():
    assert z_bin_labels()[-1] == "[1.2,2.0]"
    assert angle_bin_labels()[-1] == "[80,90]"
    assert z_bin_labels()[0] == "[0.25,0.8)"


# --------------------------------------------------- intrinsic rescaling

def test_scaling_calibration_leaves_xi_and_alpha_alone():
    calibration = real_calibration(2)
    scaled = calibration.scaled_to(208, 256)
    assert scaled.fx == pytest.approx(calibration.fx * 0.5)
    assert scaled.cx == pytest.approx(calibration.cx * 0.5)
    assert scaled.xi == calibration.xi, "xi is dimensionless and must not scale"
    assert scaled.alpha == calibration.alpha, "alpha is dimensionless and must not scale"


def test_field_angles_survive_a_resolution_change():
    """Rays must describe the same physical directions on a coarser grid."""
    calibration = real_calibration(2)
    full = build_ray_field(calibration)
    half = build_ray_field(calibration.scaled_to(208, 256))

    full_angle = full.field_angle_deg()
    half_angle = half.field_angle_deg()
    # Compare the same physical point: pixel (2i, 2j) at full res vs (i, j) at half.
    sampled_full = full_angle[::2, ::2][:200, :250]
    sampled_half = half_angle[:200, :250]
    difference = np.abs(sampled_full - sampled_half)
    assert float(np.nanmax(difference)) < 0.5, (
        "rescaling changed the physical ray directions"
    )
