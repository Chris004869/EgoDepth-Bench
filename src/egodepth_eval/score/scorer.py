"""The single scorer for E1-E4. One implementation, one set of domains.

There is deliberately no second scoring path: the bootstrap CLI, the table
exporter and the Fig. 4 exporter all consume the artifacts this module writes, so
a clip or an aggregation rule cannot diverge between them (the previous
bootstrap_ci.py re-read predictions and re-applied its own clip).

FLOW FOR ONE METHOD
  for each sample:
      G = reference_domain(reference, reference_valid)         # once per sample
      fit = fit_alignment(raw, reference, G, mode)             # once per sample
      z   = fit.apply(raw)                                     # whole frame
      P   = prediction_domain(z, declared_support)
      accumulate on G n P for: whole frame, each ROI, each Z x angle cell
  pool within site -> average sites equally

Every stratum shares the one G and the one fit for that sample.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.data.contract import CanonicalSample
from egodepth_eval.score.accumulate import (
    DEPTH_MAX,
    DEPTH_MIN,
    ScoreAccumulator,
    accumulate_frame,
    coverage_breakdown,
    prediction_domain,
    reference_domain,
)
from egodepth_eval.score.aggregate import site_macro
from egodepth_eval.score.alignment import AlignmentFit, fit_alignment
from egodepth_eval.score.prediction_io import LoadedPrediction, PredictionContract
from egodepth_eval.score.rays import (
    ANGLE_BINS,
    Z_BINS,
    RayFieldCache,
    angle_bin_index,
    angle_bin_labels,
    z_bin_index,
    z_bin_labels,
)

RESULT_SCHEMA = "egodepth-result-2"

# ROI names. `unknown` is a first-class region: it marks the un-annotated domain
# and is reported, never merged into background and never counted as a model
# being right or wrong.
ROI_NAMES = ("full", "hand", "object", "background", "unknown", "active_object")


@dataclass
class SampleScore:
    """Per-sample record: strata accumulators plus the coverage breakdown."""

    sample_token: str
    site_id: str
    accumulators: dict = field(default_factory=dict)     # stratum key -> ScoreAccumulator
    coverage: dict = field(default_factory=dict)         # roi -> coverage breakdown
    alignment: Optional[dict] = None
    status: str = "scored"                               # scored | missing_prediction | failed
    reason: str = ""


def _cell_key(roi: str, z_label: Optional[str] = None, angle_label: Optional[str] = None) -> str:
    if z_label is None and angle_label is None:
        return f"roi::{roi}"
    return f"cell::{roi}::z={z_label}::angle={angle_label}"


class MethodScorer:
    """Score one method over a manifest of samples."""

    def __init__(
        self,
        alignment_mode: str = "none",
        depth_min: float = DEPTH_MIN,
        depth_max: float = DEPTH_MAX,
        z_bins=Z_BINS,
        angle_bins=ANGLE_BINS,
        score_strata: bool = True,
    ):
        self.alignment_mode = alignment_mode
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.z_bins = z_bins
        self.angle_bins = angle_bins
        self.score_strata = score_strata
        self.ray_cache = RayFieldCache()
        self._z_labels = z_bin_labels(z_bins)
        self._angle_labels = angle_bin_labels(angle_bins)

    def score_sample(
        self,
        sample: CanonicalSample,
        site_id: str,
        reference_depth: np.ndarray,
        reference_valid: np.ndarray,
        prediction: Optional[LoadedPrediction],
        reference_calibration,
        roi_masks: Optional[dict] = None,
        visible_mask: Optional[np.ndarray] = None,
        shared_support: Optional[np.ndarray] = None,
        precomputed_fit: Optional[AlignmentFit] = None,
    ) -> SampleScore:
        """Score one sample. A missing prediction still contributes its |G|.

        Dropping the sample instead would remove it from the coverage
        denominator, so a method that failed on hard frames would show a coverage
        of 100% over the frames it managed.
        """
        result = SampleScore(sample_token=sample.sample_token, site_id=site_id)
        G = reference_domain(
            reference_depth, reference_valid, self.depth_min, self.depth_max
        )
        roi_masks = dict(roi_masks or {})
        roi_masks.setdefault("full", np.ones_like(G, dtype=bool))

        if prediction is None:
            # No prediction: |G| enters every denominator, |G n P| is zero.
            result.status = "missing_prediction"
            result.reason = "no prediction file for this sample"
            empty = np.zeros_like(G, dtype=bool)
            zeros = np.zeros_like(reference_depth, dtype=np.float32)
            for roi_name, roi_mask in roi_masks.items():
                key = _cell_key(roi_name)
                result.accumulators[key] = accumulate_frame(
                    zeros, reference_depth, G, empty, roi_mask=roi_mask
                )
                result.coverage[roi_name] = coverage_breakdown(
                    reference_valid, G, empty, visible_mask, roi_mask
                )
            if self.score_strata:
                self._accumulate_strata(
                    result, zeros, reference_depth, G, empty, roi_masks,
                    reference_calibration,
                )
            return result

        # An oracle fit must run on the scoring grid, against this frame's G. `raw`
        # may be on the model's own grid (that is the point of keeping it), so it is
        # only used when its shape matches; otherwise the already-resampled
        # metric array is used and the choice is recorded rather than silent.
        if prediction.raw is not None and prediction.raw_on_scoring_grid:
            raw = prediction.raw
            raw_source = f"raw ({prediction.raw_space})"
        else:
            raw = prediction.depth_z
            raw_source = (
                "depth_z (raw is on a different grid: "
                f"{None if prediction.raw is None else list(prediction.raw.shape)})"
            )
        if self.alignment_mode in (None, "none"):
            depth_z = prediction.depth_z
            fit = AlignmentFit(
                mode="none", status="ok",
                fit_support_description="metric output; no fitting",
            )
        else:
            fit = precomputed_fit or fit_alignment(
                prediction=raw,
                reference_depth=reference_depth,
                reference_domain_mask=G,
                mode=self.alignment_mode,
                prediction_support=prediction.valid,
            )
            depth_z = fit.apply(raw)
        result.alignment = fit.to_dict()
        result.alignment["fit_input"] = raw_source

        support = prediction.valid
        if shared_support is not None:
            support = support & np.asarray(shared_support, dtype=bool)
        P = prediction_domain(depth_z, support)

        for roi_name, roi_mask in roi_masks.items():
            key = _cell_key(roi_name)
            result.accumulators[key] = accumulate_frame(
                depth_z, reference_depth, G, P, roi_mask=roi_mask
            )
            result.coverage[roi_name] = coverage_breakdown(
                reference_valid, G, P, visible_mask, roi_mask
            )

        if self.score_strata:
            self._accumulate_strata(
                result, depth_z, reference_depth, G, P, roi_masks, reference_calibration
            )
        return result

    def _accumulate_strata(
        self,
        result: SampleScore,
        depth_z: np.ndarray,
        reference_depth: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        roi_masks: dict,
        reference_calibration,
    ) -> None:
        """Z x angle cells. Angle comes from calibrated rays, never pixel radius."""
        ray_field = self.ray_cache.get(reference_calibration, grid_hw=G.shape)
        angle = ray_field.field_angle_deg()
        z_index = z_bin_index(reference_depth, self.z_bins)
        angle_index = angle_bin_index(angle, self.angle_bins)

        # Non-forward rays cannot be placed in an angular bin. Counted here so
        # they are visible rather than absorbed into the outermost bin.
        unusable = G & (angle_index < 0)
        result.accumulators["strata_meta::non_forward_reference_pixels"] = ScoreAccumulator(
            n_reference=int(unusable.sum()), n_frames=1
        )

        for roi_name in ("full", "hand", "object"):
            roi_mask = roi_masks.get(roi_name)
            if roi_mask is None:
                continue
            for z_position, z_label in enumerate(self._z_labels):
                z_selected = z_index == z_position
                if not z_selected.any():
                    continue
                for angle_position, angle_label in enumerate(self._angle_labels):
                    cell = roi_mask & z_selected & (angle_index == angle_position)
                    if not (G & cell).any():
                        continue
                    result.accumulators[_cell_key(roi_name, z_label, angle_label)] = (
                        accumulate_frame(depth_z, reference_depth, G, P, roi_mask=cell)
                    )


def aggregate_scores(sample_scores: Sequence[SampleScore]) -> dict:
    """Pool within site, then average sites equally, for every stratum."""
    by_stratum: dict = {}
    sites = set()
    for score in sample_scores:
        sites.add(score.site_id)
        for key, accumulator in score.accumulators.items():
            by_stratum.setdefault(key, {}).setdefault(score.site_id, ScoreAccumulator())
            by_stratum[key][score.site_id].add(accumulator)

    aggregated: dict = {}
    for key, site_accumulators in by_stratum.items():
        # A cell may be absent from an entire site. Report that site as having
        # no reference in the cell, while leaving the applicable-site mean intact.
        for site in sites - site_accumulators.keys():
            site_accumulators[site] = ScoreAccumulator()
        aggregated[key] = site_macro(site_accumulators).to_dict()
    return aggregated


def aggregate_coverage(sample_scores: Sequence[SampleScore]) -> dict:
    """Site-macro of the three coverage fractions, keeping counts alongside."""
    by_roi: dict = {}
    for score in sample_scores:
        for roi_name, breakdown in score.coverage.items():
            site_bucket = by_roi.setdefault(roi_name, {}).setdefault(
                score.site_id,
                {"n_visible": 0, "n_reference_valid": 0, "n_reference_in_range": 0,
                 "n_scored": 0},
            )
            for field_name in site_bucket:
                site_bucket[field_name] += int(breakdown.get(field_name) or 0)

    output: dict = {}
    for roi_name, sites in by_roi.items():
        per_site = {}
        for site, counts in sites.items():
            per_site[site] = {
                **counts,
                "reference_validity_share": (
                    counts["n_reference_valid"] / counts["n_visible"]
                    if counts["n_visible"] else None
                ),
                "reference_in_range_share": (
                    counts["n_reference_in_range"] / counts["n_reference_valid"]
                    if counts["n_reference_valid"] else None
                ),
                "prediction_coverage": (
                    counts["n_scored"] / counts["n_reference_in_range"]
                    if counts["n_reference_in_range"] else None
                ),
            }
        macro = {}
        for share in ("reference_validity_share", "reference_in_range_share",
                      "prediction_coverage"):
            values = [
                site_value[share] for site_value in per_site.values()
                if site_value[share] is not None
            ]
            macro[share] = float(np.mean(values)) if values else None
        output[roi_name] = {
            "site_macro": macro,
            "per_site": per_site,
            "totals": {
                key: int(sum(counts[key] for counts in sites.values()))
                for key in ("n_visible", "n_reference_valid", "n_reference_in_range",
                            "n_scored")
            },
        }
    return output


def build_result_document(
    method: str,
    contract: PredictionContract,
    sample_scores: Sequence[SampleScore],
    population: dict,
    extra: Optional[dict] = None,
) -> dict:
    """The result file. Carries its own population and manifest identity."""
    statuses: dict = {}
    for score in sample_scores:
        statuses[score.status] = statuses.get(score.status, 0) + 1

    alignment_states: dict = {}
    for score in sample_scores:
        if score.alignment:
            state = score.alignment.get("status", "unknown")
            alignment_states[state] = alignment_states.get(state, 0) + 1

    return {
        "schema": RESULT_SCHEMA,
        "method": method,
        "aggregation": "site_macro_equal_weight",
        "scoring_domain": {
            "reference": "valid & finite & 0.25 <= z* <= 2.0 m (optical-axis Z, metres)",
            "prediction": "finite & positive & inside declared output support",
            "clipping": "none: predictions are not clipped to the reference interval",
        },
        "population": population,
        "prediction_contract": contract.to_dict(),
        "sample_status_counts": statuses,
        "alignment_status_counts": alignment_states,
        "n_samples": len(sample_scores),
        "strata": aggregate_scores(sample_scores),
        "coverage": aggregate_coverage(sample_scores),
        "per_sample_status": [
            {"sample_token": s.sample_token, "site_id": s.site_id,
             "status": s.status, "reason": s.reason}
            for s in sample_scores if s.status != "scored"
        ],
        **(extra or {}),
    }


def write_result(path: Path, document: dict, *, overwrite: bool = True) -> Path:
    """Publish complete JSON atomically, optionally requiring a new destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    stream = temporary.open("x", encoding="utf-8")
    try:
        with stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
        if overwrite:
            os.replace(temporary, path)
        else:
            # link(2) fails atomically if another writer has claimed the name.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
