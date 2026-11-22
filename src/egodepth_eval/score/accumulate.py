"""Sufficient statistics for the 2026-09-12 scoring contract.

DOMAINS (the whole protocol hinges on keeping these apart)
  G = reference_valid AND finite(z*) AND 0.25 <= z* <= 2.0
  P = finite(z_pred) AND z_pred > 0 AND inside the model's actual output support
  errors accumulate on G n P; coverage = |G n P| / |G|

Three different fractions are all called "coverage" in casual use, and collapsing
them into one field makes a model with no predictions indistinguishable from a
scene with no reference:

  reference_validity_share   |reference_valid| / |visible RGB pixels|
  reference_in_range_share   |G| / |reference_valid|
  prediction_coverage        |G n P| / |G|

`A_tau(R) = |{x in G n R n P : |z_pred - z*| <= tau}| / |G n R|` divides by the
reference count, not the predicted count, so a missing prediction is a failure
rather than an exclusion. That also makes A_tau monotone in tau and bounded above
by prediction coverage -- both asserted in tests.

Predictions are NOT clipped to [0.25, 2.0]. A finite 5 m prediction where the
reference says 0.6 m carries its full error. Clipping was in the previous
evaluator to tame RMSE; RMSE is no longer a headline metric, and clipping would
turn "predicted the sky" into a bounded 2.0 m mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

DEPTH_MIN = 0.25
DEPTH_MAX = 2.0
DELTA_1_THRESHOLD = 1.25

# A_tau curve, in metres: 0 to 10 cm in 0.5 cm steps. A_5cm is the tabled value.
TAU_GRID_M = tuple(round(0.005 * i, 4) for i in range(0, 21))
TAU_TABLE_M = 0.05


@dataclass
class ScoreAccumulator:
    """Additive statistics for one stratum. Pooled within a site, then averaged.

    Everything is a sum or a count so that accumulators combine exactly. AbsRel
    is the mean of |z_pred - z*| / z*, MAE_Z the mean of |z_pred - z*| in metres,
    and both must be finalized once at the end of pooling rather than averaged
    over frames.
    """

    n_reference: int = 0             # |G n R|
    n_scored: int = 0                # |G n R n P|
    sum_abs_rel: float = 0.0
    sum_abs_err_m: float = 0.0
    sum_signed_err_m: float = 0.0
    sum_sq_err_m2: float = 0.0
    n_delta_1: int = 0
    n_tau: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TAU_GRID_M), dtype=np.int64)
    )
    # Diagnostics that must not be inferred from the counts above.
    n_pred_finite_positive_outside_G: int = 0
    n_pred_nonfinite_in_G: int = 0
    n_pred_nonpositive_in_G: int = 0
    n_frames: int = 0

    def add(self, other: "ScoreAccumulator") -> "ScoreAccumulator":
        self.n_reference += other.n_reference
        self.n_scored += other.n_scored
        self.sum_abs_rel += other.sum_abs_rel
        self.sum_abs_err_m += other.sum_abs_err_m
        self.sum_signed_err_m += other.sum_signed_err_m
        self.sum_sq_err_m2 += other.sum_sq_err_m2
        self.n_delta_1 += other.n_delta_1
        self.n_tau = self.n_tau + other.n_tau
        self.n_pred_finite_positive_outside_G += other.n_pred_finite_positive_outside_G
        self.n_pred_nonfinite_in_G += other.n_pred_nonfinite_in_G
        self.n_pred_nonpositive_in_G += other.n_pred_nonpositive_in_G
        self.n_frames += other.n_frames
        return self

    def finalize(self) -> dict:
        """Metrics for this stratum.

        Error metrics are None (not zero) when nothing was scored: a site with
        reference but no predictions has undefined error and zero coverage, and
        a zero would silently read as perfect.
        """
        scored, reference = self.n_scored, self.n_reference
        result = {
            "n_reference": int(reference),
            "n_scored": int(scored),
            "coverage": float(scored / reference) if reference else None,
            "abs_rel": float(self.sum_abs_rel / scored) if scored else None,
            "mae_z_m": float(self.sum_abs_err_m / scored) if scored else None,
            "mae_z_cm": float(100.0 * self.sum_abs_err_m / scored) if scored else None,
            "signed_err_m": float(self.sum_signed_err_m / scored) if scored else None,
            "rmse_m": float(np.sqrt(self.sum_sq_err_m2 / scored)) if scored else None,
            "delta_1": float(self.n_delta_1 / scored) if scored else None,
            "n_frames": int(self.n_frames),
            "n_pred_finite_positive_outside_G": int(self.n_pred_finite_positive_outside_G),
            "n_pred_nonfinite_in_G": int(self.n_pred_nonfinite_in_G),
            "n_pred_nonpositive_in_G": int(self.n_pred_nonpositive_in_G),
        }
        # delta_1_all divides correct pixels by |G| instead of |G n P|, so a
        # method that predicts almost nothing cannot look accurate.
        result["delta_1_all"] = (
            float(self.n_delta_1 / reference) if reference else None
        )
        result["a_tau"] = {
            f"{tau:.3f}": (float(count / reference) if reference else None)
            for tau, count in zip(TAU_GRID_M, self.n_tau.tolist())
        }
        result["a_5cm"] = result["a_tau"].get(f"{TAU_TABLE_M:.3f}")
        return result


def reference_domain(
    reference_depth: np.ndarray,
    reference_valid: np.ndarray,
    depth_min: float = DEPTH_MIN,
    depth_max: float = DEPTH_MAX,
) -> np.ndarray:
    """G: reference-valid, finite, and inside the benchmark's Z interval."""
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    domain = (
        np.asarray(reference_valid, dtype=bool)
        & np.isfinite(reference_depth)
        & (reference_depth >= depth_min)
        & (reference_depth <= depth_max)
    )
    return domain


def prediction_domain(
    prediction_z: np.ndarray, support: Optional[np.ndarray] = None
) -> np.ndarray:
    """P: finite, positive, inside the model's declared output support.

    `support` is the model's own validity mask -- the rectified stereo overlap,
    a reconstruction's non-ambiguous mask. It is applied here and never
    reconstructed from the values, because a model can legitimately emit a
    plausible number outside its support.
    """
    prediction_z = np.asarray(prediction_z, dtype=np.float32)
    domain = np.isfinite(prediction_z) & (prediction_z > 0)
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != domain.shape:
            raise ValueError(
                f"support shape {support.shape} != prediction shape {domain.shape}"
            )
        domain = domain & support
    return domain


def accumulate_frame(
    prediction_z: np.ndarray,
    reference_depth: np.ndarray,
    reference_domain_mask: np.ndarray,
    prediction_domain_mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    tau_grid: Sequence[float] = TAU_GRID_M,
) -> ScoreAccumulator:
    """Statistics for one frame, optionally restricted to an ROI.

    `reference_domain_mask` is G and `prediction_domain_mask` is P, both computed
    by the callers above -- passing them in rather than recomputing keeps a
    single G per frame shared by every ROI and stratum, which is what makes the
    strata add up to the whole.
    """
    prediction_z = np.asarray(prediction_z, dtype=np.float32)
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    G = np.asarray(reference_domain_mask, dtype=bool)
    P = np.asarray(prediction_domain_mask, dtype=bool)
    if roi_mask is not None:
        G = G & np.asarray(roi_mask, dtype=bool)

    accumulator = ScoreAccumulator(n_frames=1)
    accumulator.n_reference = int(G.sum())

    # Diagnostics on the reference domain: why is a G pixel not scored?
    with np.errstate(invalid="ignore"):
        accumulator.n_pred_nonfinite_in_G = int((G & ~np.isfinite(prediction_z)).sum())
        accumulator.n_pred_nonpositive_in_G = int(
            (G & np.isfinite(prediction_z) & (prediction_z <= 0)).sum()
        )
        accumulator.n_pred_finite_positive_outside_G = int(
            (~G & np.isfinite(prediction_z) & (prediction_z > 0)).sum()
        )

    scored = G & P
    accumulator.n_scored = int(scored.sum())
    if accumulator.n_scored == 0:
        return accumulator

    pred = prediction_z[scored].astype(np.float64)
    reference = reference_depth[scored].astype(np.float64)
    abs_error = np.abs(pred - reference)

    accumulator.sum_abs_rel = float(np.sum(abs_error / reference))
    accumulator.sum_abs_err_m = float(np.sum(abs_error))
    accumulator.sum_signed_err_m = float(np.sum(pred - reference))
    accumulator.sum_sq_err_m2 = float(np.sum((pred - reference) ** 2))
    ratio = np.maximum(pred / reference, reference / pred)
    accumulator.n_delta_1 = int(np.sum(ratio < DELTA_1_THRESHOLD))
    accumulator.n_tau = np.array(
        [int(np.sum(abs_error <= tau)) for tau in tau_grid], dtype=np.int64
    )
    return accumulator


def coverage_breakdown(
    reference_valid: np.ndarray,
    reference_domain_mask: np.ndarray,
    prediction_domain_mask: np.ndarray,
    visible_mask: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> dict:
    """The three separate fractions, with explicit numerators and denominators.

    Exported as counts as well as ratios so a downstream reader can re-derive
    them and cannot mistake one for another.
    """
    reference_valid = np.asarray(reference_valid, dtype=bool)
    G = np.asarray(reference_domain_mask, dtype=bool)
    P = np.asarray(prediction_domain_mask, dtype=bool)
    if visible_mask is None:
        visible_mask = np.ones_like(reference_valid, dtype=bool)
    else:
        visible_mask = np.asarray(visible_mask, dtype=bool)
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        reference_valid = reference_valid & roi_mask
        G = G & roi_mask
        visible_mask = visible_mask & roi_mask

    n_visible = int(visible_mask.sum())
    n_reference_valid = int((reference_valid & visible_mask).sum())
    n_in_range = int(G.sum())
    n_predicted = int((G & P).sum())

    def ratio(numerator: int, denominator: int):
        return float(numerator / denominator) if denominator else None

    return {
        "n_visible": n_visible,
        "n_reference_valid": n_reference_valid,
        "n_reference_in_range": n_in_range,
        "n_scored": n_predicted,
        "reference_validity_share": ratio(n_reference_valid, n_visible),
        "reference_in_range_share": ratio(n_in_range, n_reference_valid),
        "prediction_coverage": ratio(n_predicted, n_in_range),
    }
