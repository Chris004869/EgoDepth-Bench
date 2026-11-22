"""Site-macro aggregation and paired site bootstrap.

AGGREGATION ORDER IS PART OF THE PROTOCOL
Pixels pool *within* a site, then sites average with equal weight. Pooling all
pixels globally lets one large site dominate; averaging per frame gives a frame
with three far-field pixels the same weight as one with fifty thousand. Both were
considered and rejected; this module implements neither.

A SITE WITH REFERENCE BUT NO PREDICTIONS IS NOT DROPPED
Its coverage is 0 and its error is undefined. The complete macro error is then
undefined too, and an available-site macro is reported alongside with its own
site count. Quietly averaging the sites that did produce output is how a method
that failed on the hardest location comes out looking best.

A SITE WITH NO REFERENCE IN THE STRATUM IS SEPARATE AGAIN: not applicable, and
distinguished from prediction failure in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.score.accumulate import TAU_GRID_M, ScoreAccumulator

# Metrics for which a site contributes only when it actually scored pixels.
ERROR_METRICS = ("abs_rel", "mae_z_m", "mae_z_cm", "signed_err_m", "rmse_m", "delta_1")
# Metrics defined whenever the site has reference support, prediction or not.
COVERAGE_METRICS = ("coverage", "delta_1_all")


@dataclass
class SiteMacroResult:
    """Equal-weight site average with explicit support bookkeeping."""

    metrics: dict                     # metric -> value or None
    n_sites_total: int                # sites with reference support in this stratum
    n_sites_with_predictions: int
    n_sites_no_predictions: int
    n_sites_not_applicable: int       # no reference support here at all
    per_site: dict                    # site -> finalized stratum dict
    available_site_metrics: dict      # error metrics over predicting sites only
    complete_macro_defined: bool

    def to_dict(self) -> dict:
        return {
            "aggregation": "site_macro_equal_weight",
            "metrics": self.metrics,
            "available_site_metrics": self.available_site_metrics,
            "complete_macro_defined": self.complete_macro_defined,
            "support": {
                "n_sites_total": self.n_sites_total,
                "n_sites_with_predictions": self.n_sites_with_predictions,
                "n_sites_no_predictions": self.n_sites_no_predictions,
                "n_sites_not_applicable": self.n_sites_not_applicable,
            },
            "per_site": self.per_site,
        }


def pool_within_site(accumulators: Sequence[ScoreAccumulator]) -> ScoreAccumulator:
    """Add up every frame's statistics for one site."""
    pooled = ScoreAccumulator()
    for accumulator in accumulators:
        pooled.add(accumulator)
    return pooled


def site_macro(site_accumulators: dict) -> SiteMacroResult:
    """Equal-weight average over sites of within-site pooled metrics.

    `site_accumulators` maps site id -> ScoreAccumulator already pooled within
    that site. Sites are weighted equally regardless of pixel count, so
    duplicating every pixel of one site does not change its weight (asserted in
    tests).
    """
    per_site: dict = {}
    with_predictions: list = []
    no_predictions: list = []
    not_applicable: list = []

    for site, accumulator in sorted(site_accumulators.items()):
        finalized = accumulator.finalize()
        per_site[site] = finalized
        if finalized["n_reference"] == 0:
            not_applicable.append(site)
        elif finalized["n_scored"] == 0:
            no_predictions.append(site)
        else:
            with_predictions.append(site)

    applicable = with_predictions + no_predictions
    metrics: dict = {}
    available: dict = {}

    for metric in ERROR_METRICS:
        values = [per_site[s][metric] for s in with_predictions if per_site[s][metric] is not None]
        available[metric] = float(np.mean(values)) if values else None
        # Complete macro error needs every applicable site to have contributed.
        metrics[metric] = (
            float(np.mean(values))
            if values and not no_predictions and len(values) == len(applicable)
            else None
        )

    for metric in COVERAGE_METRICS:
        # Defined on every applicable site: a zero-coverage site contributes 0.
        values = [
            per_site[s][metric] if per_site[s][metric] is not None else 0.0
            for s in applicable
        ]
        metrics[metric] = float(np.mean(values)) if values else None
        available[metric] = (
            float(np.mean([
                per_site[s][metric] for s in with_predictions
                if per_site[s][metric] is not None
            ]))
            if with_predictions else None
        )

    tau_keys = [f"{tau:.3f}" for tau in TAU_GRID_M]
    macro_tau: dict = {}
    for key in tau_keys:
        values = [
            (per_site[s]["a_tau"].get(key) or 0.0) for s in applicable
        ]
        macro_tau[key] = float(np.mean(values)) if values else None
    metrics["a_tau"] = macro_tau
    metrics["a_5cm"] = macro_tau.get("0.050")

    metrics["n_reference"] = int(sum(per_site[s]["n_reference"] for s in applicable))
    metrics["n_scored"] = int(sum(per_site[s]["n_scored"] for s in applicable))
    metrics["n_frames"] = int(sum(per_site[s]["n_frames"] for s in per_site))

    return SiteMacroResult(
        metrics=metrics,
        n_sites_total=len(applicable),
        n_sites_with_predictions=len(with_predictions),
        n_sites_no_predictions=len(no_predictions),
        n_sites_not_applicable=len(not_applicable),
        per_site=per_site,
        available_site_metrics=available,
        complete_macro_defined=bool(applicable) and not no_predictions,
    )


def _site_metric_value(accumulator: ScoreAccumulator, metric: str) -> Optional[float]:
    finalized = accumulator.finalize()
    if metric in COVERAGE_METRICS:
        value = finalized.get(metric)
        return 0.0 if value is None and finalized["n_reference"] else value
    if metric == "a_5cm":
        return finalized.get("a_5cm") if finalized["n_reference"] else None
    return finalized.get(metric)


def paired_site_bootstrap(
    accumulators_a: dict,
    accumulators_b: dict,
    metric: str = "abs_rel",
    n_boot: int = 2000,
    seed: int = 20260912,
) -> dict:
    """95% CI on the B-minus-A difference, resampling sites with one shared index.

    The same resampled site index is applied to both methods on every draw, which
    is what makes the interval a paired one. Resampling each method independently
    would inflate the interval by the between-site variance the pairing removes.

    Only sites where BOTH methods have a defined value enter the comparison, and
    the count is reported: a difference computed over different site sets is not
    a paired difference.
    """
    common = sorted(set(accumulators_a) & set(accumulators_b))
    values_a, values_b, usable_sites = [], [], []
    for site in common:
        a = _site_metric_value(accumulators_a[site], metric)
        b = _site_metric_value(accumulators_b[site], metric)
        if a is None or b is None:
            continue
        values_a.append(float(a))
        values_b.append(float(b))
        usable_sites.append(site)

    result = {
        "metric": metric,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_sites_common": len(common),
        "n_sites_compared": len(usable_sites),
        "sites_compared": usable_sites,
        "n_sites_dropped": len(common) - len(usable_sites),
        "aggregation": "site_macro_equal_weight",
        "resampling": "shared site index applied to both methods per draw",
    }
    if len(usable_sites) < 2:
        # A site bootstrap over one site resamples the same value every draw, so the
        # interval collapses to zero width -- which would read as a precise
        # measurement. Report the difference with `ci95: None` and a status naming
        # the reason instead.
        result["status"] = "insufficient_sites"
        result["ci95_note"] = (
            f"{len(usable_sites)} site(s) compared; a site bootstrap needs at least 2. "
            "The difference is reported without an interval rather than with a "
            "zero-width one."
        )
        result["mean_a"] = float(np.mean(values_a)) if values_a else None
        result["mean_b"] = float(np.mean(values_b)) if values_b else None
        result["difference"] = (
            float(np.mean(values_b) - np.mean(values_a)) if values_a else None
        )
        result["ci95"] = None
        return result

    array_a = np.asarray(values_a, dtype=np.float64)
    array_b = np.asarray(values_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_sites = array_a.size
    draws = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        index = rng.integers(0, n_sites, size=n_sites)
        draws[draw] = float(array_b[index].mean() - array_a[index].mean())

    result["status"] = "ok"
    result["mean_a"] = float(array_a.mean())
    result["mean_b"] = float(array_b.mean())
    result["difference"] = float(array_b.mean() - array_a.mean())
    result["ci95"] = [
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    ]
    result["per_site_difference"] = {
        site: float(b - a) for site, a, b in zip(usable_sites, values_a, values_b)
    }
    return result


def common_support_mask(prediction_domains: Sequence[np.ndarray]) -> np.ndarray:
    """Intersection of several methods' prediction domains, for a declared pair.

    Used only for pre-specified comparisons. Intersecting the whole model pool
    would shrink the domain to the pixels the weakest fisheye method covers,
    which is a different (and much easier) benchmark.
    """
    if not prediction_domains:
        raise ValueError("no prediction domains supplied")
    shared = np.asarray(prediction_domains[0], dtype=bool)
    for domain in prediction_domains[1:]:
        domain = np.asarray(domain, dtype=bool)
        if domain.shape != shared.shape:
            raise ValueError(
                f"prediction domain shape {domain.shape} != {shared.shape}"
            )
        shared = shared & domain
    return shared


def standardize_by_strata(
    stratum_values: dict,
    stratum_weights: dict,
) -> dict:
    """Reference-weighted average over retained site-Z-angle strata.

    Both the unweighted and the standardized number must come from the SAME
    retained strata, otherwise a difference caused by dropping strata gets
    reported as an effect of reweighting. This returns both plus the retained and
    discarded support so the comparison is auditable.

    Weights are fixed from reference composition beforehand; they are never
    chosen from model scores.
    """
    retained = sorted(
        key for key in stratum_values
        if stratum_values[key] is not None and stratum_weights.get(key)
    )
    discarded_no_value = sorted(
        key for key in stratum_values if stratum_values[key] is None
    )
    discarded_no_weight = sorted(
        key for key in stratum_values
        if stratum_values[key] is not None and not stratum_weights.get(key)
    )

    if not retained:
        return {
            "unweighted": None, "standardized": None,
            "n_strata_retained": 0,
            "strata_retained": [],
            "strata_discarded_no_value": discarded_no_value,
            "strata_discarded_no_weight": discarded_no_weight,
            "weight_total_retained": 0.0,
        }

    values = np.array([float(stratum_values[key]) for key in retained], dtype=np.float64)
    weights = np.array([float(stratum_weights[key]) for key in retained], dtype=np.float64)
    return {
        "unweighted": float(values.mean()),
        "standardized": float(np.sum(values * weights) / np.sum(weights)),
        "n_strata_retained": len(retained),
        "strata_retained": retained,
        "strata_discarded_no_value": discarded_no_value,
        "strata_discarded_no_weight": discarded_no_weight,
        "weight_total_retained": float(weights.sum()),
        "weight_source": "fixed from reference composition, not from model scores",
    }
