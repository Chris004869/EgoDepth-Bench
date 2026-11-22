"""Oracle alignment protocols, fit once per inference call and then reused.

    positive_scale   s = sum(p*g) / sum(p^2), applied as s * pred
                     VGGT, VGGT-Omega, MoGe-1, and the E4 shape diagnostic
    affine_inverse   fit 1/z* = a * pred_inverse + b, then invert
                     Depth Anything v2 (its head emits inverse depth)
    affine_depth     fit z* = a * pred_depth + b
                     Marigold (fitted in depth space, not inverse space)

THREE RULES THAT KEEP THIS FROM BECOMING SCORE-FITTING

1. ONE FIT PER INFERENCE CALL, on the full camera-2 target frame, reused
   unchanged for every ROI, Z stratum and angle stratum. Re-fitting per ROI would
   let each region absorb its own scale error, and hand/object numbers would
   stop being comparable to the whole-image number. `AlignmentFit.apply` is the
   only way to use a fit, and it takes no domain argument.

2. NO ITERATION. No outlier rejection, no re-fit after dropping residuals. Fit
   the declared support once; if it is degenerate, record the failure.

3. THE FIT SUPPORT IS DECLARED, NOT DISCOVERED. It is G intersected with the
   valid values of the model's *own representation*. For inverse-depth output
   that includes zero -- zero disparity legitimately means "very far" -- so the
   representation-legality check happens before the fit and the positivity check
   happens after conversion to Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ALIGNMENT_MODES = ("none", "positive_scale", "affine_inverse", "affine_depth")

# Reject a fit whose design matrix is this close to singular. A scale fitted from
# a near-zero denominator explodes, and the resulting depth map looks like a
# model failure rather than an alignment failure.
MIN_DENOMINATOR = 1e-12
MIN_FIT_PIXELS = 32


@dataclass
class AlignmentFit:
    """A frozen fit. `apply` maps a raw prediction to metric Z."""

    mode: str
    coefficients: dict = field(default_factory=dict)
    n_fit_pixels: int = 0
    status: str = "ok"                 # ok | degenerate | insufficient_support | skipped
    reason: str = ""
    fit_support_description: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def apply(self, prediction: np.ndarray) -> np.ndarray:
        """Raw model output -> metric Z, with unconvertible pixels as NaN.

        Takes no mask: a fit is applied to the whole frame exactly as fitted, so
        it cannot be silently re-derived per region.
        """
        prediction = np.asarray(prediction, dtype=np.float32)
        if self.mode in (None, "none"):
            return prediction
        if not self.ok:
            # A failed fit produces no depth at all rather than an arbitrary one.
            # The frame then scores as zero coverage, which is the honest result.
            return np.full_like(prediction, np.nan, dtype=np.float32)

        if self.mode == "positive_scale":
            return (prediction * float(self.coefficients["scale"])).astype(np.float32)

        a = float(self.coefficients["a"])
        b = float(self.coefficients["b"])
        if self.mode == "affine_depth":
            return (a * prediction + b).astype(np.float32)

        if self.mode == "affine_inverse":
            fitted_inverse = a * prediction.astype(np.float64) + b
            with np.errstate(divide="ignore", invalid="ignore"):
                # Non-positive fitted disparity means at-or-beyond infinity. NaN,
                # not a negative depth that a later `pred > 0` filter would drop
                # after it had already polluted an intermediate statistic.
                depth = np.where(fitted_inverse > 1e-8, 1.0 / fitted_inverse, np.nan)
            return depth.astype(np.float32)

        raise ValueError(f"unknown alignment mode {self.mode!r}")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "n_fit_pixels": int(self.n_fit_pixels),
            "status": self.status,
            "reason": self.reason,
            "fit_support_description": self.fit_support_description,
        }


def representation_valid(prediction: np.ndarray, mode: str) -> np.ndarray:
    """Pixels that are legal values *in the model's own output representation*.

    For inverse-depth output, 0 is legal (infinitely far) and negative is not.
    For depth-like output under an affine fit, an unknown shift means the raw
    values need not be positive, so only finiteness is required. Applying a
    blanket `pred > 0` before the fit would delete legitimate far-field pixels
    for exactly the models whose fit needs them.
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    finite = np.isfinite(prediction)
    if mode == "affine_inverse":
        return finite & (prediction >= 0)
    if mode == "affine_depth":
        return finite
    if mode == "positive_scale":
        # A scale cannot fix a sign, so only positive values inform the fit.
        return finite & (prediction > 0)
    return finite & (prediction > 0)


def fit_alignment(
    prediction: np.ndarray,
    reference_depth: np.ndarray,
    reference_domain_mask: np.ndarray,
    mode: str,
    prediction_support: Optional[np.ndarray] = None,
) -> AlignmentFit:
    """Fit one alignment on the full target frame.

    `reference_domain_mask` must be the whole-frame G. Passing an ROI here would
    violate rule 1 above.
    """
    if mode in (None, "none"):
        return AlignmentFit(mode="none", status="ok",
                            fit_support_description="metric output; no fitting")
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"unknown alignment mode {mode!r}; known: {ALIGNMENT_MODES}")

    prediction = np.asarray(prediction, dtype=np.float32)
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    support = representation_valid(prediction, mode)
    if prediction_support is not None:
        support = support & np.asarray(prediction_support, dtype=bool)
    fit_mask = (
        np.asarray(reference_domain_mask, dtype=bool)
        & support
        & np.isfinite(reference_depth)
        & (reference_depth > 0)
    )
    n_fit = int(fit_mask.sum())
    description = (
        "full camera-2 target frame: G n representation-valid output support"
    )

    if n_fit < MIN_FIT_PIXELS:
        return AlignmentFit(
            mode=mode, n_fit_pixels=n_fit, status="insufficient_support",
            reason=f"{n_fit} fit pixels, need at least {MIN_FIT_PIXELS}",
            fit_support_description=description,
        )

    p = prediction[fit_mask].astype(np.float64)
    g = reference_depth[fit_mask].astype(np.float64)

    if mode == "positive_scale":
        denominator = float(np.sum(p * p))
        if denominator <= MIN_DENOMINATOR:
            return AlignmentFit(
                mode=mode, n_fit_pixels=n_fit, status="degenerate",
                reason=f"sum(pred^2) = {denominator:.3e} is not usable",
                fit_support_description=description,
            )
        scale = float(np.sum(p * g) / denominator)
        if not np.isfinite(scale) or scale <= 0:
            return AlignmentFit(
                mode=mode, n_fit_pixels=n_fit, status="degenerate",
                reason=f"fitted scale {scale!r} is not positive and finite",
                fit_support_description=description,
            )
        return AlignmentFit(
            mode=mode, coefficients={"scale": scale}, n_fit_pixels=n_fit,
            status="ok", fit_support_description=description,
        )

    target = (1.0 / g) if mode == "affine_inverse" else g
    design = np.stack([p, np.ones_like(p)], axis=1)
    try:
        solution, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError as exc:
        return AlignmentFit(
            mode=mode, n_fit_pixels=n_fit, status="degenerate",
            reason=f"least squares failed: {exc}",
            fit_support_description=description,
        )
    if rank < 2:
        # A constant prediction cannot pin down a slope; the fit would be an
        # arbitrary split between a and b.
        return AlignmentFit(
            mode=mode, n_fit_pixels=n_fit, status="degenerate",
            reason=f"design matrix rank {rank} < 2 (prediction is near-constant)",
            fit_support_description=description,
        )
    a, b = float(solution[0]), float(solution[1])
    if not (np.isfinite(a) and np.isfinite(b)):
        return AlignmentFit(
            mode=mode, n_fit_pixels=n_fit, status="degenerate",
            reason=f"non-finite coefficients a={a!r} b={b!r}",
            fit_support_description=description,
        )
    return AlignmentFit(
        mode=mode, coefficients={"a": a, "b": b}, n_fit_pixels=n_fit,
        status="ok", fit_support_description=description,
    )


def fit_shared_positive_scale(
    prediction: np.ndarray,
    reference_depth: np.ndarray,
    shared_domain_mask: np.ndarray,
) -> AlignmentFit:
    """Positive scale on a caller-supplied shared domain, for the E4 diagnostic.

    Table VI's last two rows fit each of public and fine-tuned on
    `G n P_public n P_finetuned` -- the same pixels for both -- so the aligned
    comparison is not confounded by the two checkpoints predicting different
    pixels. Separate entry point because the domain here is deliberately
    narrower than either model's own support.
    """
    fit = fit_alignment(
        prediction=prediction,
        reference_depth=reference_depth,
        reference_domain_mask=shared_domain_mask,
        mode="positive_scale",
    )
    fit.fit_support_description = (
        "full camera-2 target frame: G n P_public n P_finetuned (shared support)"
    )
    return fit
