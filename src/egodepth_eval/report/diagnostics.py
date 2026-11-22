"""E2 diagnostics that require re-reading predictions: motion contrast, common support.

These cannot be assembled from the per-method result documents alone, which is why
the previous export produced an empty `comparisons` list and a "common support"
comparison that never intersected any masks. Each function here re-reads the
prediction arrays and computes on the pixels it claims to.

THREE THINGS THAT ARE COMPUTED, NOT ASSERTED

  motion contrast   active minus low-motion MAE_Z on the SAME retained
                    site x Z x angle strata, unweighted and reference-weighted,
                    with a paired site bootstrap and the discarded support listed.

  common support    `G n P_A n P_B`, by loading both methods' masks. The pair's
                    own-support results are reported alongside, so the effect of
                    intersecting is visible rather than implied.

  aligned diagnostic  each arm's positive scale fitted on the shared domain from
                    its OWN predictions. Nothing is copied between arms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.annotate.pipeline import region_masks_from_record
from egodepth_eval.annotate.schema import MotionLabel
from egodepth_eval.score.accumulate import (
    ScoreAccumulator,
    accumulate_frame,
    prediction_domain,
    reference_domain,
)
from egodepth_eval.score.aggregate import (
    paired_site_bootstrap,
    site_macro,
    standardize_by_strata,
)
from egodepth_eval.score.alignment import fit_shared_positive_scale
from egodepth_eval.score.prediction_io import load_prediction
from egodepth_eval.score.rays import (
    ANGLE_BINS,
    Z_BINS,
    RayFieldCache,
    angle_bin_index,
    angle_bin_labels,
    z_bin_index,
    z_bin_labels,
)

# A stratum needs this many reference pixels in BOTH motion conditions before it
# can carry a difference. Below it the cell mean is noise.
MIN_STRATUM_REFERENCE_PX = 200


def standardize_motion_by_site(differences, reference_weights, raw_site_differences,
                               n_boot=2000, seed=20260912):
    """Match Z/angle composition within each site, then give sites equal weight.

    The raw and standardized estimates have different sampling distributions;
    each gets its own paired site-bootstrap interval.
    """
    by_site = {}
    for key, difference in differences.items():
        site = key.split('::', 1)[0]
        by_site.setdefault(site, []).append((difference, reference_weights[key]))
    sites = sorted(by_site)
    standardized = np.asarray([
        np.average([x[0] for x in by_site[s]], weights=[x[1] for x in by_site[s]])
        for s in sites
    ], dtype=float)
    raw = np.asarray([raw_site_differences[s] for s in sites], dtype=float)
    if not sites or not np.isfinite(standardized).all() or not np.isfinite(raw).all():
        raise ValueError('Motion contrast requires finite paired site estimates')
    intervals = [None, None]
    if len(sites) >= 2:
        draws = np.random.default_rng(seed).integers(0, len(sites), (n_boot, len(sites)))
        intervals = [np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()
                     for values in (raw, standardized)]
    return {'unweighted': float(raw.mean()), 'standardized': float(standardized.mean()),
        'unweighted_ci95': intervals[0], 'standardized_ci95': intervals[1],
        'n_sites': len(sites), 'per_site_raw': dict(zip(sites, raw.tolist())),
        'per_site_standardized': dict(zip(sites, standardized.tolist())),
        'weighting': 'reference composition within site; equal weight across sites',
        'bootstrap': {'unit': 'site', 'n_boot': n_boot, 'seed': seed}}


def _stratum_key(site: str, z_label: str, angle_label: str) -> str:
    return f"{site}::z={z_label}::angle={angle_label}"


def collect_motion_strata(
    population,
    reader,
    records: dict,
    prediction_dir: Path,
    roi: str = "object",
) -> dict:
    """Per-(site, Z, angle) accumulators split by motion condition.

    Only frames whose motion label was accepted after visual review contribute;
    an unknown label withdraws the frame from this comparison while leaving it in
    the main test set and in the ROI diagnostics.
    """
    ray_cache = RayFieldCache()
    z_labels = z_bin_labels()
    angle_labels = angle_bin_labels()
    by_condition: dict = {"active_manipulation": {}, "low_motion": {}}
    reference_weight: dict = {}
    counters = {
        "n_samples": 0, "n_with_prediction": 0, "n_motion_usable": 0,
        "n_active": 0, "n_low": 0, "n_roi_unusable": 0,
    }

    for sample in population.samples:
        counters["n_samples"] += 1
        record = records.get(sample.sample_token)
        if record is None or not record.motion_usable:
            continue
        condition = record.motion_label
        if condition not in by_condition:
            continue
        counters["n_motion_usable"] += 1
        counters["n_active" if condition == "active_manipulation" else "n_low"] += 1

        path = Path(prediction_dir) / f"{sample.sample_token}.npz"
        loaded = reader.load(sample, load_images=False)
        prediction = load_prediction(path) if path.exists() else None
        if prediction is not None:
            counters["n_with_prediction"] += 1
        roi_masks = region_masks_from_record(record, loaded.grid_hw)
        roi_mask = roi_masks.get(roi)
        if roi_mask is None or not roi_mask.any():
            counters["n_roi_unusable"] += 1
            continue

        G = reference_domain(loaded.reference_depth, loaded.reference_valid)
        depth = prediction.depth_z if prediction is not None else np.full(loaded.grid_hw, np.nan)
        P = prediction_domain(depth, prediction.valid) if prediction is not None else np.zeros(loaded.grid_hw, bool)
        field = ray_cache.get(loaded.reference_calibration, grid_hw=loaded.grid_hw)
        z_index = z_bin_index(loaded.reference_depth)
        angle_index = angle_bin_index(field.field_angle_deg())
        site = sample.physical_site_id or f"unmapped::{sample.recording_id}"

        for z_position, z_label in enumerate(z_labels):
            for angle_position, angle_label in enumerate(angle_labels):
                cell = (
                    roi_mask
                    & (z_index == z_position)
                    & (angle_index == angle_position)
                )
                if not (G & cell).any():
                    continue
                key = _stratum_key(site, z_label, angle_label)
                accumulator = accumulate_frame(
                    depth, loaded.reference_depth, G, P, roi_mask=cell
                )
                by_condition[condition].setdefault(key, ScoreAccumulator()).add(
                    accumulator
                )
                # Weights come from reference composition only -- never from any
                # model's score -- and are accumulated across both conditions.
                reference_weight[key] = (
                    reference_weight.get(key, 0) + accumulator.n_reference
                )

    return {
        "by_condition": by_condition,
        "reference_weight": reference_weight,
        "counters": counters,
        "roi": roi,
    }


def compute_motion_contrast(
    population,
    reader,
    records: dict,
    prediction_dir: Path,
    method: str,
    roi: str = "object",
    n_boot: int = 2000,
    seed: int = 20260912,
) -> dict:
    """Active minus low-motion MAE_Z, unweighted and standardized, with paired CI.

    The retained strata are exactly those with enough reference support in BOTH
    conditions, and both the unweighted and the standardized number come from that
    same retained set -- otherwise a change caused by dropping strata would read as
    an effect of reweighting.

    Prediction completeness is checked separately: raw needs finite pooled means
    at every retained site; standardization also needs every retained cell mean.
    """
    collected = collect_motion_strata(
        population, reader, records, prediction_dir, roi=roi
    )
    active = collected["by_condition"]["active_manipulation"]
    low = collected["by_condition"]["low_motion"]
    weights = collected["reference_weight"]

    shared = sorted(set(active) & set(low))
    retained, discarded, missing_prediction = [], [], []
    for key in shared:
        active_metrics = active[key].finalize()
        low_metrics = low[key].finalize()
        enough = (
            active_metrics["n_reference"] >= MIN_STRATUM_REFERENCE_PX
            and low_metrics["n_reference"] >= MIN_STRATUM_REFERENCE_PX
        )
        scored = (
            active_metrics["mae_z_cm"] is not None
            and low_metrics["mae_z_cm"] is not None
            and np.isfinite(active_metrics["mae_z_cm"])
            and np.isfinite(low_metrics["mae_z_cm"])
        )
        if enough:
            retained.append(key)
            if not scored:
                missing_prediction.append(key)
        else:
            discarded.append({
                "stratum": key,
                "reason": (
                    "insufficient reference support" if not enough
                    else "no scored pixels in one condition"
                ),
                "n_reference_active": active_metrics["n_reference"],
                "n_reference_low": low_metrics["n_reference"],
            })

    result = {
        "method": method,
        "roi": roi,
        "counters": collected["counters"],
        "n_strata_active_only": len(set(active) - set(low)),
        "n_strata_low_only": len(set(low) - set(active)),
        "n_strata_shared": len(shared),
        "n_strata_retained": len(retained),
        "n_strata_discarded": len(discarded),
        "discarded_strata": discarded[:20],
        "missing_prediction_strata": missing_prediction,
        "min_stratum_reference_px": MIN_STRATUM_REFERENCE_PX,
        "weight_source": "reference pixel composition, fixed before scoring",
        "comparisons": [],
        "raw_status": "insufficient_support",
        "standardized_status": "insufficient_support",
    }
    if not retained:
        result["status"] = "insufficient_support"
        result["reason"] = (
            f"{collected['counters']['n_active']} active and "
            f"{collected['counters']['n_low']} low-motion frames have accepted motion "
            f"labels, and {len(shared)} strata are shared, but none has at least "
            f"{MIN_STRATUM_REFERENCE_PX} reference pixels in both conditions. "
            "No static group is synthesized."
        )
        return result

    # Pool ALL reference-eligible strata before checking prediction completeness.
    # A missing cell prevents standardization, but need not prevent pooled raw MAE.
    active_by_site: dict = {}
    low_by_site: dict = {}
    for key in retained:
        site = key.split("::", 1)[0]
        active_by_site.setdefault(site, ScoreAccumulator()).add(active[key])
        low_by_site.setdefault(site, ScoreAccumulator()).add(low[key])
    active_macro = site_macro(active_by_site)
    low_macro = site_macro(low_by_site)
    sites = sorted(active_by_site)
    raw_site_differences = {}
    for site in sites:
        active_mae = active_macro.per_site[site]['mae_z_cm']
        low_mae = low_macro.per_site[site]['mae_z_cm']
        raw_site_differences[site] = (
            active_mae - low_mae
            if active_mae is not None and low_mae is not None
            and np.isfinite(active_mae) and np.isfinite(low_mae) else None
        )
    missing_raw_sites = [site for site in sites if raw_site_differences[site] is None]
    # paired_site_bootstrap permits available-site comparisons. Guard its call so
    # an unpredicted eligible site can never silently disappear from this estimand.
    paired = (paired_site_bootstrap(
        low_by_site, active_by_site, metric="mae_z_cm", n_boot=n_boot, seed=seed
    ) if not missing_raw_sites else None)
    supported_status = 'ok' if len(sites) >= 2 else 'ok_without_interval'
    raw_status = 'undefined_prediction_support' if missing_raw_sites else supported_status
    standardized_status = 'undefined_prediction_support' if missing_prediction else supported_status
    differences = {
        key: (active[key].finalize()['mae_z_cm'] - low[key].finalize()['mae_z_cm'])
        if key not in missing_prediction else None for key in retained
    }
    if not missing_prediction:
        # Keep the established point estimates and bootstrap draws unchanged.
        standardized = standardize_motion_by_site(
            differences, {key: weights[key] for key in retained},
            raw_site_differences, n_boot=n_boot, seed=seed,
        )
    else:
        raw = (np.asarray([raw_site_differences[site] for site in sites], dtype=float)
               if not missing_raw_sites else None)
        raw_ci = None
        if raw is not None and len(sites) >= 2:
            draws = np.random.default_rng(seed).integers(0, len(sites), (n_boot, len(sites)))
            raw_ci = np.quantile(raw[draws].mean(axis=1), [.025, .975]).tolist()
        standardized = {
            'unweighted': float(raw.mean()) if raw is not None else None,
            'standardized': None,
            'unweighted_ci95': raw_ci, 'standardized_ci95': None,
            'n_sites': len(sites), 'per_site_raw': raw_site_differences,
            'per_site_standardized': {site: None for site in sites},
            'weighting': 'reference composition within site; equal weight across sites',
            'bootstrap': {'unit': 'site', 'n_boot': n_boot, 'seed': seed},
        }
    standardized.update(raw_status=raw_status, standardized_status=standardized_status)

    # Preserve the top-level completeness status while exposing raw independently.
    result.update(status=standardized_status, raw_status=raw_status,
                  standardized_status=standardized_status,
                  n_sites_eligible=len(sites), missing_raw_sites=missing_raw_sites)
    if missing_prediction:
        result['reason'] = ('Reference-eligible strata lack predictions in a condition; '
                            'the complete standardized contrast is undefined. '
                            'These strata are retained in the raw pooling.')
    result["comparisons"] = [{
        "label": f"{method}\n{roi}",
        "method": method,
        "roi": roi,
        "unweighted": standardized["unweighted"],
        "standardized": standardized["standardized"],
        "unweighted_ci95": standardized['unweighted_ci95'],
        "standardized_ci95": standardized['standardized_ci95'],
        "raw_status": raw_status,
        "standardized_status": standardized_status,
        "ci95_applies_to": "separate paired site bootstraps for raw and standardized differences",
        "ci95_status": paired.get("status") if paired is not None else raw_status,
        "ci95_note": paired.get("ci95_note") if paired is not None else None,
        "n_strata_retained": len(retained),
        "n_strata_discarded": len(discarded),
        "n_sites_compared": len(sites) if not missing_raw_sites else 0,
        "n_sites_eligible": len(sites),
        "active_mae_z_cm": active_macro.metrics["mae_z_cm"],
        "low_motion_mae_z_cm": low_macro.metrics["mae_z_cm"],
        "per_stratum_difference": differences,
    }]
    if paired is not None:
        result["paired_bootstrap"] = paired
    result["standardization"] = standardized
    result["note"] = (
        "Observational comparison. Motion, occlusion and appearance may vary "
        "together; the difference is not attributed to motion alone."
    )
    return result


def compute_common_support_pair(
    population,
    reader,
    records: dict,
    predictions_root: Path,
    method_a: str,
    method_b: str,
    roi: str = "full",
    n_boot: int = 2000,
    seed: int = 20260912,
) -> dict:
    """Score a declared pair on their OWN supports and on `G n P_A n P_B`.

    The previous version rebuilt accumulators from each method's independent
    whole-image statistics and bootstrapped those, so no pixel intersection was
    ever taken. Here both prediction masks are loaded and intersected per frame.
    """
    dir_a = Path(predictions_root) / method_a
    dir_b = Path(predictions_root) / method_b
    if not (dir_a.exists() and dir_b.exists()):
        return {
            "pair": [method_a, method_b],
            "status": "unavailable",
            "reason": "one or both methods have no prediction directory",
        }

    own_a: dict = {}
    own_b: dict = {}
    shared_a: dict = {}
    shared_b: dict = {}
    counters = {"n_samples": 0, "n_both_present": 0, "n_shared_nonempty": 0}
    shared_pixels, own_pixels_a, own_pixels_b = 0, 0, 0

    for sample in population.samples:
        counters["n_samples"] += 1
        path_a = dir_a / f"{sample.sample_token}.npz"
        path_b = dir_b / f"{sample.sample_token}.npz"
        if not (path_a.exists() and path_b.exists()):
            continue
        counters["n_both_present"] += 1

        loaded = reader.load(sample, load_images=False)
        prediction_a = load_prediction(path_a)
        prediction_b = load_prediction(path_b)
        G = reference_domain(loaded.reference_depth, loaded.reference_valid)

        roi_mask = None
        if roi != "full":
            record = records.get(sample.sample_token)
            if record is None:
                continue
            roi_mask = region_masks_from_record(record, loaded.grid_hw).get(roi)
            if roi_mask is None or not roi_mask.any():
                continue

        P_a = prediction_domain(prediction_a.depth_z, prediction_a.valid)
        P_b = prediction_domain(prediction_b.depth_z, prediction_b.valid)
        shared = P_a & P_b                     # the actual intersection
        if (G & shared).any():
            counters["n_shared_nonempty"] += 1

        site = sample.physical_site_id or f"unmapped::{sample.recording_id}"
        own_a.setdefault(site, ScoreAccumulator()).add(
            accumulate_frame(prediction_a.depth_z, loaded.reference_depth, G, P_a,
                             roi_mask=roi_mask)
        )
        own_b.setdefault(site, ScoreAccumulator()).add(
            accumulate_frame(prediction_b.depth_z, loaded.reference_depth, G, P_b,
                             roi_mask=roi_mask)
        )
        shared_a.setdefault(site, ScoreAccumulator()).add(
            accumulate_frame(prediction_a.depth_z, loaded.reference_depth, G, shared,
                             roi_mask=roi_mask)
        )
        shared_b.setdefault(site, ScoreAccumulator()).add(
            accumulate_frame(prediction_b.depth_z, loaded.reference_depth, G, shared,
                             roi_mask=roi_mask)
        )
        selected = G if roi_mask is None else (G & roi_mask)
        own_pixels_a += int((selected & P_a).sum())
        own_pixels_b += int((selected & P_b).sum())
        shared_pixels += int((selected & shared).sum())

    if not counters["n_both_present"]:
        return {
            "pair": [method_a, method_b],
            "status": "no_overlapping_samples",
            "counters": counters,
        }

    def summarize(accumulators: dict) -> dict:
        macro = site_macro(accumulators)
        return {
            "abs_rel": macro.metrics["abs_rel"] or macro.available_site_metrics["abs_rel"],
            "mae_z_cm": macro.metrics["mae_z_cm"] or macro.available_site_metrics["mae_z_cm"],
            "coverage": macro.metrics["coverage"],
            "n_sites": macro.n_sites_total,
            "n_scored_px": macro.metrics["n_scored"],
            "complete_macro_defined": macro.complete_macro_defined,
        }

    return {
        "pair": [method_a, method_b],
        "roi": roi,
        "status": "ok",
        "counters": counters,
        "own_support": {method_a: summarize(own_a), method_b: summarize(own_b)},
        "common_support": {method_a: summarize(shared_a), method_b: summarize(shared_b)},
        "support_pixels": {
            f"{method_a}_own": own_pixels_a,
            f"{method_b}_own": own_pixels_b,
            "shared": shared_pixels,
            "shared_share_of_a": (
                shared_pixels / own_pixels_a if own_pixels_a else None
            ),
            "shared_share_of_b": (
                shared_pixels / own_pixels_b if own_pixels_b else None
            ),
        },
        "own_support_difference": paired_site_bootstrap(
            own_a, own_b, metric="abs_rel", n_boot=n_boot, seed=seed
        ),
        "common_support_difference": paired_site_bootstrap(
            shared_a, shared_b, metric="abs_rel", n_boot=n_boot, seed=seed
        ),
        "domain": "G n P_A n P_B, computed per frame from both prediction masks",
        "note": (
            "Only pre-specified pairs are intersected. Intersecting the whole model "
            "pool would shrink the domain to the weakest method's coverage."
        ),
    }


def compute_aligned_three_arm(
    population,
    reader,
    records: dict,
    predictions_root: Path,
    arms: Sequence[str] = ("mapanything_6view", "mapanything_scale_only", "mapanything_ft"),
    shared_arms: Sequence[str] = ("mapanything_6view", "mapanything_ft"),
    scale_pair: Sequence[str] = ("mapanything_6view", "mapanything_scale_only"),
) -> dict:
    """Table VI's aligned rows: every arm fitted from its OWN predictions.

    The previous version copied Public's aligned numbers into the Scale-only row,
    so the two were equal by assignment. The equality is the *implementation check*
    -- a frozen network times a positive scalar has the same shape -- and copying
    the field destroyed the only thing the check could have caught. Here each arm
    is loaded, fitted on the shared domain, and scored independently; the equality
    is then reported as a computed outcome with the residual difference.

    The shared domain is `G n P_public n P_finetuned` per the protocol; Scale-only
    is scored on that same domain so all three rows are comparable.
    """
    arm_dirs = {arm: Path(predictions_root) / arm for arm in arms}
    missing = [arm for arm, path in arm_dirs.items() if not path.exists()]
    if missing:
        return {
            "status": "unavailable",
            "reason": f"no predictions for {missing}",
            "arms_present": [arm for arm in arms if arm not in missing],
        }

    accumulators = {arm: {"full": {}, "object": {}} for arm in arms}
    fits: dict = {arm: [] for arm in arms}
    counters = {"n_samples": 0, "n_all_arms_present": 0, "n_shared_nonempty": 0}

    for sample in population.samples:
        counters["n_samples"] += 1
        paths = {arm: arm_dirs[arm] / f"{sample.sample_token}.npz" for arm in arms}
        if not all(path.exists() for path in paths.values()):
            continue
        counters["n_all_arms_present"] += 1

        loaded = reader.load(sample, load_images=False)
        predictions = {arm: load_prediction(paths[arm]) for arm in arms}
        G = reference_domain(loaded.reference_depth, loaded.reference_valid)

        shared = G.copy()
        for arm in shared_arms:
            shared &= prediction_domain(
                predictions[arm].depth_z, predictions[arm].valid
            )
        if not shared.any():
            continue
        counters["n_shared_nonempty"] += 1

        record = records.get(sample.sample_token)
        object_mask = (
            region_masks_from_record(record, loaded.grid_hw).get("object")
            if record else None
        )
        site = sample.physical_site_id or f"unmapped::{sample.recording_id}"

        for arm in arms:
            prediction = predictions[arm]
            # Fitted from THIS arm's own depth on the shared domain.
            fit = fit_shared_positive_scale(
                prediction.depth_z, loaded.reference_depth, shared
            )
            fits[arm].append(fit.to_dict())
            aligned = fit.apply(prediction.depth_z)
            P = prediction_domain(aligned, prediction.valid) & shared
            accumulators[arm]["full"].setdefault(site, ScoreAccumulator()).add(
                accumulate_frame(aligned, loaded.reference_depth, shared, P)
            )
            if object_mask is not None and object_mask.any():
                accumulators[arm]["object"].setdefault(site, ScoreAccumulator()).add(
                    accumulate_frame(
                        aligned, loaded.reference_depth, shared, P,
                        roi_mask=object_mask,
                    )
                )

    if not counters["n_shared_nonempty"]:
        return {"status": "no_shared_support", "counters": counters}

    output: dict = {
        "status": "ok",
        "counters": counters,
        "fit_domain": 'G intersect ' + ' intersect '.join('P_' + arm for arm in shared_arms)
                      + ' on the full camera-2 target frame; fit reused unchanged for the object ROI',
        "arms": list(arms),
        "shared_arms": list(shared_arms),
        "independence": (
            "each arm's scale is fitted from its own predictions; no field is "
            "copied between arms"
        ),
    }
    for arm in arms:
        full = site_macro(accumulators[arm]["full"])
        obj = site_macro(accumulators[arm]["object"])
        scales = [
            f["coefficients"].get("scale") for f in fits[arm]
            if f["status"] == "ok" and "scale" in f["coefficients"]
        ]
        output[arm] = {
            "abs_rel": full.metrics["abs_rel"],
            "object_mae_z_cm": obj.metrics["mae_z_cm"],
            "n_sites": full.n_sites_total,
            "object_n_sites": obj.n_sites_total,
            "n_fits": len(fits[arm]),
            "n_failed_fits": sum(1 for f in fits[arm] if f["status"] != "ok"),
            "median_fitted_scale": float(np.median(scales)) if scales else None,
        }

    # The scalar-invariance check, now on independently computed numbers.
    public, scale_only = scale_pair
    if public in output and scale_only in output:
        a = output[public]["abs_rel"]
        b = output[scale_only]["abs_rel"]
        difference = None if (a is None or b is None) else abs(a - b)
        output["scalar_invariance_check"] = {
            "public_aligned_abs_rel": a,
            "scale_only_aligned_abs_rel": b,
            "absolute_difference": difference,
            # Tolerance covers float32 storage and the least-squares solve, not a
            # real discrepancy.
            "tolerance": 1e-4,
            "passed": difference is not None and difference <= 1e-4,
            "median_scale_ratio": (
                output[public]["median_fitted_scale"] / output[scale_only]["median_fitted_scale"]
                if output[public]["median_fitted_scale"]
                and output[scale_only]["median_fitted_scale"] else None
            ),
            "expectation": (
                "a frozen network times one positive scalar has identical shape, so "
                "the aligned scores must match and the fitted scales must differ by "
                "exactly that scalar. Both sides are computed from their own "
                "prediction files."
            ),
        }
    return output
