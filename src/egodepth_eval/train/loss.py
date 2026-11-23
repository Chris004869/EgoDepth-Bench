"""Masked log-Z Huber loss on camera 2 only, plus the frozen-scale calibration.

THE MASK COMES FROM THE REFERENCE, NOT FROM THE MODEL
The supervised set is `valid_GT and finite and 0.25 <= z* <= 2.0`. The model's own
confidence never removes a pixel: a learnable confidence that can drop pixels lets
the network lower the loss by declining to predict the hard ones, which is exactly
the failure this benchmark exists to detect.

NON-FINITE PREDICTIONS ARE A DIAGNOSED FAILURE, NOT A SILENT DROP
`log` needs a positive argument, so predictions are clamped to a floor before the
log -- and the clamp's effect is counted and returned. A step whose loss is
NaN/inf raises `NonFiniteLoss` rather than being skipped: silently skipping bad
steps and reporting "training completed" is a false pass. The count of clamped
pixels is part of the step record, so "the loss is low because half the
predictions were floored" is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Supervised depth interval, matching the scoring domain.
TRAIN_DEPTH_MIN = 0.25
TRAIN_DEPTH_MAX = 2.0
# Floor applied before log(). Two orders of magnitude below the shallowest
# supervised depth, so it never touches a plausible prediction.
POSITIVE_FLOOR_M = 1e-3
# Huber transition in log-depth units. 0.1 in log space is ~10% relative error.
DEFAULT_HUBER_DELTA = 0.1


class NonFiniteLoss(RuntimeError):
    """Raised when a step's loss is not finite. Never caught to skip the step."""


@dataclass
class LossOutputs:
    loss: object                     # torch scalar tensor
    n_supervised: int
    n_pred_nonfinite: int
    n_pred_floored: int
    mean_abs_log_residual: float
    status: str = "ok"

    def to_dict(self) -> dict:
        return {
            "loss": float(self.loss.detach().item()) if hasattr(self.loss, "detach") else float(self.loss),
            "n_supervised": self.n_supervised,
            "n_pred_nonfinite": self.n_pred_nonfinite,
            "n_pred_floored": self.n_pred_floored,
            "positive_floor_m": POSITIVE_FLOOR_M,
            "mean_abs_log_residual": self.mean_abs_log_residual,
            "status": self.status,
        }


def supervision_mask(
    target_depth_z,
    target_valid,
    depth_min: float = TRAIN_DEPTH_MIN,
    depth_max: float = TRAIN_DEPTH_MAX,
):
    """The supervised pixel set. Depends only on the reference."""
    import torch

    return (
        target_valid.bool()
        & torch.isfinite(target_depth_z)
        & (target_depth_z >= depth_min)
        & (target_depth_z <= depth_max)
    )


def huber_on_log_residual(residual, huber_delta: float):
    """The single Huber-on-log-residual expression used everywhere.

    Fine-tuning and scale-only calibration apply this loss with the same
    support mask and reduction.
    """
    absolute = residual.abs()
    return (
        (0.5 * residual.pow(2)).where(
            absolute <= huber_delta,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    )


def masked_log_huber_loss(
    prediction_z,
    target_depth_z,
    target_valid,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    depth_min: float = TRAIN_DEPTH_MIN,
    depth_max: float = TRAIN_DEPTH_MAX,
    positive_floor: float = POSITIVE_FLOOR_M,
    reduction: str = "mean",
) -> LossOutputs:
    """Huber(log z_pred - log z*) over the reference-valid camera-2 pixels.

    Reduction is the mean over supervised pixels. Note this weights a frame with
    many valid pixels more within a batch; that matches the pooled-within-site
    scoring convention and is recorded rather than silently chosen.
    """
    import torch

    mask = supervision_mask(target_depth_z, target_valid, depth_min, depth_max)
    n_supervised = int(mask.sum().item())
    if n_supervised == 0:
        # A batch with no supervised pixel is a data problem. Return a real zero
        # tensor that still carries grad_fn so the caller can detect it, and say so.
        zero = (prediction_z.sum() * 0.0)
        return LossOutputs(
            loss=zero, n_supervised=0, n_pred_nonfinite=0, n_pred_floored=0,
            mean_abs_log_residual=0.0, status="no_supervised_pixels",
        )

    pred = prediction_z[mask]
    target = target_depth_z[mask]

    finite = torch.isfinite(pred)
    n_nonfinite = int((~finite).sum().item())
    if n_nonfinite:
        raise NonFiniteLoss(f"{n_nonfinite} non-finite predictions on supervised pixels")

    needs_floor = pred < positive_floor
    n_floored = int(needs_floor.sum().item())
    pred = torch.clamp(pred, min=positive_floor)

    residual = torch.log(pred) - torch.log(target)
    absolute = residual.abs()
    # The same expression the Scale-only calibration uses, so the two arms differ
    # by exactly one scalar and not by the objective.
    per_pixel = huber_on_log_residual(residual, huber_delta)
    loss = per_pixel.mean() if reduction == "mean" else per_pixel.sum()

    if not torch.isfinite(loss):
        raise NonFiniteLoss(
            f"loss is {loss.item()!r} over {n_supervised} supervised pixels "
            f"({n_nonfinite} non-finite predictions, {n_floored} floored). "
            "Training fails here rather than skipping the step."
        )

    return LossOutputs(
        loss=loss,
        n_supervised=n_supervised,
        n_pred_nonfinite=n_nonfinite,
        n_pred_floored=n_floored,
        mean_abs_log_residual=float(absolute.mean().detach().item()),
        status="ok",
    )


def fit_calibration_scale(
    predictions,
    targets,
    valids,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    n_steps: int = 400,
    learning_rate: float = 0.05,
    depth_min: float = TRAIN_DEPTH_MIN,
    depth_max: float = TRAIN_DEPTH_MAX,
    positive_floor: float = POSITIVE_FLOOR_M,
    reduction: str = "per_frame_mean",
) -> dict:
    """Fit one positive scalar `s_C` on the training calibration subset C.

    THE OBJECTIVE MATCHES FINE-TUNING EXACTLY, which the previous version did not:

    - Mask: `supervision_mask` on the reference only. Predictions are NOT
      intersected with the model's support here. Intersecting made the calibration
      domain narrower than the training domain, so `s_C` was fitted on an easier
      pixel set than the fine-tuned arm ever saw.
    - Numerical protection: the same positive floor as training, with the number of
      floored pixels reported, instead of dropping non-positive predictions.
    - Reduction: `per_frame_mean` averages within a frame and then across frames,
      matching the micro-batch mean-of-means that gradient accumulation produces.
      Concatenating every frame's pixels and taking one mean, as before, weights a
      frame by its valid-pixel count and is a different estimator.

    C must come from train only, and the resulting scalar is fixed across every
    test sample and site -- it is a property of the training domain, not of the
    frame it is applied to.
    """
    import torch

    if reduction not in ("per_frame_mean", "pixel_pooled"):
        raise ValueError(f"unknown reduction {reduction!r}")

    log_scale = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale], lr=learning_rate)

    frames: list = []
    n_floored_total, n_nonfinite_total, n_pixels = 0, 0, 0
    for prediction, target, valid in zip(predictions, targets, valids):
        prediction = torch.as_tensor(prediction, dtype=torch.float64)
        target = torch.as_tensor(target, dtype=torch.float64)
        valid = torch.as_tensor(valid).bool()
        # Reference-only mask, exactly as in training.
        mask = supervision_mask(target, valid, depth_min, depth_max)
        if not mask.any():
            continue
        selected_pred = prediction[mask]
        selected_target = target[mask]

        finite = torch.isfinite(selected_pred)
        n_nonfinite_total += int((~finite).sum().item())
        if not finite.all():
            raise NonFiniteLoss("Non-finite predictions in the training calibration subset")
        n_floored_total += int((selected_pred < positive_floor).sum().item())
        # Scale first, then floor: clamping the unscaled prediction would change
        # the objective relative to fine-tuning on small or non-positive values.
        frames.append((selected_pred, torch.log(selected_target)))
        n_pixels += int(selected_pred.numel())

    if not frames:
        return {
            "status": "no_support", "scale": None, "n_pixels": 0, "n_frames_used": 0,
            "objective": "log-Huber, identical mask and reduction to fine-tuning",
        }

    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        if reduction == "per_frame_mean":
            per_frame = [
                huber_on_log_residual(torch.clamp(pred * log_scale.exp(), min=positive_floor).log() - log_target, huber_delta).mean()
                for pred, log_target in frames
            ]
            loss = torch.stack(per_frame).mean()
        else:
            residual = torch.cat([
                torch.clamp(pred * log_scale.exp(), min=positive_floor).log() - log_target
                for pred, log_target in frames
            ])
            loss = huber_on_log_residual(residual, huber_delta).mean()
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"scale calibration loss is {loss.item()!r} at step {step}")
        loss.backward()
        optimizer.step()
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            history.append({"step": step, "loss": float(loss.item()),
                            "scale": float(log_scale.detach().exp().item())})

    scale = float(log_scale.detach().exp().item())
    return {
        "status": "ok",
        "scale": scale,
        "log_scale": float(log_scale.detach().item()),
        "n_pixels": n_pixels,
        "n_frames_used": len(frames),
        "n_pred_floored": n_floored_total,
        "n_pred_nonfinite": n_nonfinite_total,
        "positive_floor_m": positive_floor,
        "huber_delta": huber_delta,
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "reduction": reduction,
        "supervision_mask": "reference-only (valid & finite & in [depth_min, depth_max])",
        "prediction_support_intersected": False,
        "objective": (
            "Huber(log(s * pred) - log(target)) with the same mask, positive floor "
            "and reduction as fine-tuning"
        ),
        "history": history,
        "note": (
            "Fitted on the training calibration subset C only. One scalar is "
            "fixed across all test samples and sites. Scale-only is therefore not "
            "fully zero-shot: it has seen training-domain labels."
        ),
    }
