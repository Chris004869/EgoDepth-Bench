"""Tests with discriminating power over the scoring contract.

Each test is written so that the obvious wrong implementation fails it. Tests
that merely mirror the implementation are not useful here -- the failures this
protocol is guarding against are silent.
"""

import numpy as np
import pytest

from egodepth_eval.score.accumulate import (
    DEPTH_MAX,
    DEPTH_MIN,
    TAU_GRID_M,
    ScoreAccumulator,
    accumulate_frame,
    coverage_breakdown,
    prediction_domain,
    reference_domain,
)
from egodepth_eval.score.aggregate import (
    paired_site_bootstrap,
    site_macro,
    standardize_by_strata,
)
from egodepth_eval.score.alignment import (
    fit_alignment,
    fit_shared_positive_scale,
    representation_valid,
)


def _frame(values):
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


# --------------------------------------------------------------- domains

def test_nonfinite_and_nonpositive_predictions_stay_invalid():
    """NaN, inf and <=0 must not be scored, and must not vanish from |G|."""
    reference = _frame([1.0, 1.0, 1.0, 1.0, 1.0])
    valid = np.ones_like(reference, dtype=bool)
    prediction = _frame([1.0, np.nan, np.inf, 0.0, -2.0])

    G = reference_domain(reference, valid)
    P = prediction_domain(prediction)
    assert G.sum() == 5
    assert P.sum() == 1

    accumulator = accumulate_frame(prediction, reference, G, P)
    assert accumulator.n_reference == 5, "invalid predictions must not shrink |G|"
    assert accumulator.n_scored == 1
    assert accumulator.n_pred_nonfinite_in_G == 2
    assert accumulator.n_pred_nonpositive_in_G == 2
    assert accumulator.finalize()["coverage"] == pytest.approx(0.2)


def test_out_of_range_prediction_keeps_full_error():
    """A finite 5 m prediction against 0.5 m reference is not clipped to 2.0 m."""
    reference = _frame([0.5])
    valid = np.ones_like(reference, dtype=bool)
    prediction = _frame([5.0])

    G = reference_domain(reference, valid)
    accumulator = accumulate_frame(prediction, reference, G, prediction_domain(prediction))
    metrics = accumulator.finalize()
    assert metrics["mae_z_m"] == pytest.approx(4.5), "prediction was clipped"
    assert metrics["abs_rel"] == pytest.approx(9.0)


def test_reference_outside_interval_is_not_in_G():
    reference = _frame([0.2, 0.25, 1.0, 2.0, 2.5])
    valid = np.ones_like(reference, dtype=bool)
    G = reference_domain(reference, valid)
    # Both interval endpoints are included; outside values are not.
    assert G.ravel().tolist() == [False, True, True, True, False]


def test_missing_prediction_and_missing_reference_differ():
    """Zero coverage (no prediction) must not look like not-applicable (no reference)."""
    reference = _frame([1.0, 1.0])
    empty_prediction = np.zeros_like(reference, dtype=bool)

    with_reference = accumulate_frame(
        np.zeros_like(reference), reference, reference_domain(reference, np.ones_like(reference, bool)),
        empty_prediction,
    ).finalize()
    without_reference = accumulate_frame(
        np.zeros_like(reference), reference,
        np.zeros_like(reference, dtype=bool), empty_prediction,
    ).finalize()

    assert with_reference["n_reference"] == 2
    assert with_reference["coverage"] == 0.0
    assert with_reference["abs_rel"] is None
    assert without_reference["n_reference"] == 0
    assert without_reference["coverage"] is None, (
        "no reference must give an undefined coverage, not zero"
    )


# ----------------------------------------------------------------- A_tau

def test_a_tau_is_monotone_and_bounded_by_coverage():
    reference = _frame([1.0] * 10)
    valid = np.ones_like(reference, dtype=bool)
    # Errors of 0, 1, ..., 9 cm; four pixels have no prediction at all.
    prediction = _frame([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, np.nan, np.nan, np.nan, np.nan])

    G = reference_domain(reference, valid)
    P = prediction_domain(prediction)
    metrics = accumulate_frame(prediction, reference, G, P).finalize()

    values = [metrics["a_tau"][f"{tau:.3f}"] for tau in TAU_GRID_M]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:])), "A_tau is not monotone"
    coverage = metrics["coverage"]
    assert all(value <= coverage + 1e-12 for value in values), (
        "A_tau exceeded prediction coverage, so the denominator is wrong"
    )
    # Denominator is |G| = 10, not the 6 predicted pixels.
    assert metrics["a_tau"]["0.050"] == pytest.approx(0.6)
    assert metrics["a_5cm"] == pytest.approx(0.6)


def test_a_tau_denominator_is_reference_not_prediction():
    """Halving the predictions must halve A_tau, not leave it unchanged."""
    reference = _frame([1.0] * 8)
    valid = np.ones_like(reference, dtype=bool)
    G = reference_domain(reference, valid)

    full = _frame([1.0] * 8)
    half = _frame([1.0] * 4 + [np.nan] * 4)
    a_full = accumulate_frame(full, reference, G, prediction_domain(full)).finalize()["a_5cm"]
    a_half = accumulate_frame(half, reference, G, prediction_domain(half)).finalize()["a_5cm"]
    assert a_full == pytest.approx(1.0)
    assert a_half == pytest.approx(0.5)


# ------------------------------------------------------------- coverage trio

def test_three_coverage_fractions_are_distinct():
    """Reference validity, in-range share and prediction coverage must differ."""
    reference = _frame([0.1, 0.5, 1.0, 1.5, np.nan, 3.0, 1.0, 1.0])
    reference_valid = _frame([1, 1, 1, 1, 0, 1, 1, 1]).astype(bool)
    prediction = _frame([1.0, 0.5, 1.0, np.nan, 1.0, 1.0, np.nan, np.nan])

    G = reference_domain(reference, reference_valid)
    P = prediction_domain(prediction)
    breakdown = coverage_breakdown(reference_valid, G, P)

    assert breakdown["n_visible"] == 8
    assert breakdown["n_reference_valid"] == 7
    # In range: 0.5, 1.0, 1.5, 1.0, 1.0 -> five pixels (0.1 and 3.0 are outside).
    assert breakdown["n_reference_in_range"] == 5
    assert breakdown["n_scored"] == 2
    assert breakdown["reference_validity_share"] == pytest.approx(7 / 8)
    assert breakdown["reference_in_range_share"] == pytest.approx(5 / 7)
    assert breakdown["prediction_coverage"] == pytest.approx(2 / 5)
    assert len({
        breakdown["reference_validity_share"],
        breakdown["reference_in_range_share"],
        breakdown["prediction_coverage"],
    }) == 3


# ---------------------------------------------------------------- alignment

def test_positive_scale_recovers_known_scale_analytically():
    reference = _frame([0.5, 1.0, 1.5, 2.0])
    prediction = reference / 3.0
    G = reference_domain(reference, np.ones_like(reference, dtype=bool))
    # 32-pixel minimum: tile the analytic example.
    reference = np.tile(reference, (1, 8))
    prediction = np.tile(prediction, (1, 8))
    G = np.tile(G, (1, 8))

    fit = fit_alignment(prediction, reference, G, "positive_scale")
    assert fit.ok
    assert fit.coefficients["scale"] == pytest.approx(3.0, rel=1e-6)
    assert np.allclose(fit.apply(prediction), reference, rtol=1e-5)


def test_affine_inverse_recovers_known_coefficients():
    reference = np.tile(_frame([0.4, 0.8, 1.2, 1.6]), (1, 8))
    # 1/z = 2 * pred + 0.5  =>  pred = (1/z - 0.5) / 2
    prediction = ((1.0 / reference) - 0.5) / 2.0
    G = reference_domain(reference, np.ones_like(reference, dtype=bool))

    fit = fit_alignment(prediction, reference, G, "affine_inverse")
    assert fit.ok
    assert fit.coefficients["a"] == pytest.approx(2.0, rel=1e-5)
    assert fit.coefficients["b"] == pytest.approx(0.5, rel=1e-5)
    assert np.allclose(fit.apply(prediction), reference, rtol=1e-4)


def test_affine_depth_recovers_known_coefficients():
    reference = np.tile(_frame([0.4, 0.8, 1.2, 1.6]), (1, 8))
    prediction = (reference - 0.3) / 1.7        # z = 1.7 * pred + 0.3
    G = reference_domain(reference, np.ones_like(reference, dtype=bool))

    fit = fit_alignment(prediction, reference, G, "affine_depth")
    assert fit.ok
    assert fit.coefficients["a"] == pytest.approx(1.7, rel=1e-5)
    assert fit.coefficients["b"] == pytest.approx(0.3, rel=1e-5)


def test_inverse_zero_endpoint_is_legal_but_positive_scale_zero_is_not():
    """Zero disparity means 'very far' and must survive to the fit."""
    prediction = _frame([0.0, 0.5, -0.1, np.nan])
    assert representation_valid(prediction, "affine_inverse").ravel().tolist() == [
        True, True, False, False
    ]
    # A scale cannot fix a sign, so only strictly positive values inform it.
    assert representation_valid(prediction, "positive_scale").ravel().tolist() == [
        False, True, False, False
    ]
    # An affine depth fit carries an unknown shift, so negatives are legal input.
    assert representation_valid(prediction, "affine_depth").ravel().tolist() == [
        True, True, True, False
    ]


def test_degenerate_fits_are_recorded_and_produce_no_depth():
    reference = np.tile(_frame([1.0, 1.0, 1.0, 1.0]), (1, 8))
    G = reference_domain(reference, np.ones_like(reference, dtype=bool))

    constant = np.full_like(reference, 0.5)
    fit = fit_alignment(constant, reference, G, "affine_depth")
    assert fit.status == "degenerate"
    assert np.isnan(fit.apply(constant)).all(), (
        "a failed fit must yield no depth rather than an arbitrary one"
    )

    tiny = np.tile(_frame([1.0, 1.0]), (1, 4))
    small_G = reference_domain(tiny, np.ones_like(tiny, dtype=bool))
    insufficient = fit_alignment(tiny / 2, tiny, small_G, "positive_scale")
    assert insufficient.status == "insufficient_support"


def test_alignment_is_not_refit_per_roi():
    """The same fit object must serve every ROI unchanged."""
    reference = np.tile(_frame([0.5, 1.0, 1.5, 2.0]), (1, 8))
    prediction = reference / 4.0
    G = reference_domain(reference, np.ones_like(reference, dtype=bool))
    fit = fit_alignment(prediction, reference, G, "positive_scale")

    roi = np.zeros_like(G)
    roi[0, :4] = True
    # A per-ROI refit on this ROI alone would give a different scale if the ROI
    # were biased; the contract is that `apply` takes no domain at all.
    aligned_full = fit.apply(prediction)
    roi_only_fit = fit_alignment(
        prediction, reference, G & roi, "positive_scale"
    )
    assert roi_only_fit.n_fit_pixels < fit.n_fit_pixels
    assert np.allclose(aligned_full[roi], reference[roi], rtol=1e-5)


def test_scale_only_matches_public_oracle_on_shared_support():
    """E4 cross-check: Scale-only's oracle scale must equal public's."""
    reference = np.tile(_frame([0.5, 1.0, 1.5, 2.0]), (1, 8))
    public = reference / 2.5
    scale_only = public * 1.37             # frozen network, one positive scalar
    shared = reference_domain(reference, np.ones_like(reference, dtype=bool))

    fit_public = fit_shared_positive_scale(public, reference, shared)
    fit_scale_only = fit_shared_positive_scale(scale_only, reference, shared)
    assert np.allclose(fit_public.apply(public), fit_scale_only.apply(scale_only), rtol=1e-5)


# ------------------------------------------------------------- site macro

def test_site_macro_weights_sites_equally_regardless_of_pixel_count():
    small = ScoreAccumulator(
        n_reference=10, n_scored=10, sum_abs_rel=1.0, sum_abs_err_m=1.0, n_frames=1
    )
    large = ScoreAccumulator(
        n_reference=10_000, n_scored=10_000, sum_abs_rel=3000.0,
        sum_abs_err_m=3000.0, n_frames=1,
    )
    result = site_macro({"a": small, "b": large})
    # AbsRel is 0.1 at site a and 0.3 at site b; equal weights give 0.2.
    # Global pixel pooling would give ~0.2998.
    assert result.metrics["abs_rel"] == pytest.approx(0.2)


def test_duplicating_a_sites_pixels_does_not_change_its_weight():
    site_a = ScoreAccumulator(n_reference=100, n_scored=100, sum_abs_rel=10.0, n_frames=1)
    site_b = ScoreAccumulator(n_reference=100, n_scored=100, sum_abs_rel=30.0, n_frames=1)
    before = site_macro({"a": site_a, "b": site_b}).metrics["abs_rel"]

    doubled = ScoreAccumulator(n_reference=200, n_scored=200, sum_abs_rel=20.0, n_frames=2)
    after = site_macro({"a": doubled, "b": site_b}).metrics["abs_rel"]
    assert before == pytest.approx(after)


def test_site_with_no_predictions_makes_complete_macro_undefined():
    good = ScoreAccumulator(n_reference=100, n_scored=100, sum_abs_rel=10.0, n_frames=1)
    silent = ScoreAccumulator(n_reference=100, n_scored=0, n_frames=1)
    result = site_macro({"good": good, "silent": silent})

    assert result.metrics["abs_rel"] is None, (
        "a site with reference but no prediction must not be averaged away"
    )
    assert result.available_site_metrics["abs_rel"] == pytest.approx(0.1)
    assert result.n_sites_with_predictions == 1
    assert result.n_sites_no_predictions == 1
    assert result.metrics["coverage"] == pytest.approx(0.5), (
        "the silent site must contribute zero coverage, not be dropped"
    )
    assert not result.complete_macro_defined


def test_site_without_reference_is_not_applicable_not_failure():
    good = ScoreAccumulator(n_reference=100, n_scored=100, sum_abs_rel=10.0, n_frames=1)
    absent = ScoreAccumulator(n_reference=0, n_scored=0, n_frames=1)
    result = site_macro({"good": good, "absent": absent})

    assert result.n_sites_not_applicable == 1
    assert result.n_sites_no_predictions == 0
    assert result.complete_macro_defined, (
        "a site with no reference in this stratum must not invalidate the macro"
    )
    assert result.metrics["abs_rel"] == pytest.approx(0.1)


# --------------------------------------------------------------- bootstrap

def test_paired_bootstrap_uses_one_shared_site_index():
    """A constant per-site offset must give a tight CI that excludes zero."""
    sites = {}
    other = {}
    rng = np.random.default_rng(7)
    for index in range(8):
        base = float(rng.uniform(0.1, 0.9))
        sites[f"site{index}"] = ScoreAccumulator(
            n_reference=100, n_scored=100, sum_abs_rel=100 * base, n_frames=1
        )
        other[f"site{index}"] = ScoreAccumulator(
            n_reference=100, n_scored=100, sum_abs_rel=100 * (base - 0.05), n_frames=1
        )
    result = paired_site_bootstrap(sites, other, metric="abs_rel", n_boot=500)
    assert result["status"] == "ok"
    assert result["n_sites_compared"] == 8
    assert result["difference"] == pytest.approx(-0.05, abs=1e-6)
    lo, hi = result["ci95"]
    assert hi < 0, "a constant per-site improvement must give a CI excluding zero"
    # Pairing removes the between-site spread, so the interval is far tighter
    # than the ~0.8-wide spread of the underlying site values.
    assert (hi - lo) < 0.01


def test_paired_bootstrap_reports_sites_it_could_not_compare():
    a = {
        "s1": ScoreAccumulator(n_reference=10, n_scored=10, sum_abs_rel=1.0, n_frames=1),
        "s2": ScoreAccumulator(n_reference=10, n_scored=10, sum_abs_rel=2.0, n_frames=1),
        "s3": ScoreAccumulator(n_reference=10, n_scored=10, sum_abs_rel=3.0, n_frames=1),
    }
    b = dict(a)
    b["s3"] = ScoreAccumulator(n_reference=10, n_scored=0, n_frames=1)   # no predictions
    result = paired_site_bootstrap(a, b, metric="abs_rel", n_boot=100)
    assert result["n_sites_common"] == 3
    assert result["n_sites_compared"] == 2
    assert result["n_sites_dropped"] == 1


# ------------------------------------------------------- standardization

def test_standardization_uses_the_same_retained_strata_for_both_numbers():
    values = {"a": 1.0, "b": 3.0, "c": None}
    weights = {"a": 0.9, "b": 0.1, "c": 5.0}
    result = standardize_by_strata(values, weights)

    assert result["n_strata_retained"] == 2
    assert result["strata_discarded_no_value"] == ["c"]
    assert result["unweighted"] == pytest.approx(2.0)
    assert result["standardized"] == pytest.approx(0.9 * 1.0 + 0.1 * 3.0)


def test_standardization_reports_strata_dropped_for_lacking_weight():
    values = {"a": 1.0, "b": 3.0}
    weights = {"a": 1.0}
    result = standardize_by_strata(values, weights)
    assert result["strata_discarded_no_weight"] == ["b"]
    assert result["n_strata_retained"] == 1


def test_sparse_strata_report_sites_without_any_eligible_frames():
    from egodepth_eval.score.scorer import SampleScore, aggregate_scores

    key = "roi::hand"
    sites = [
        SampleScore("a0", "a", {key: ScoreAccumulator(n_reference=10, n_scored=5, n_frames=1)}),
        SampleScore("b0", "b"),
    ]
    result = aggregate_scores(sites)[key]
    assert result["support"] == {
        "n_sites_total": 1, "n_sites_with_predictions": 1,
        "n_sites_no_predictions": 0, "n_sites_not_applicable": 1,
    }
    assert result["per_site"]["b"]["coverage"] is None
    assert result["per_site"]["b"]["n_frames"] == 0
    assert result["metrics"]["coverage"] == 0.5
    assert result["metrics"]["n_frames"] == 1
