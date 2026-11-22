"""Region-versus-whole-image comparisons matched by reference-bearing frames."""
import hashlib
import numpy as np
from egodepth_eval.score.accumulate import ScoreAccumulator
from egodepth_eval.score.aggregate import site_macro, paired_site_bootstrap


def standardized_region_comparison(sample_scores, roi, n_boot=2000, seed=20260912, min_reference=200):
    """Compare the same frames and site/Z/angle cells at a common composition.

    Eligible cells depend only on ROI reference support. A model with no scores
    in an eligible cell receives an undefined standardized estimate, not an easier
    model-specific selection. Raw estimates pool all eligible cells first and
    require finite operands at every eligible site. Reference ROI pixel counts
    fix the standardized within-site weights.
    """
    cells={}
    for row in sample_scores:
        region=row.accumulators.get('roi::'+roi)
        if region is None or not region.n_reference:continue
        for key,acc in row.accumulators.items():
            prefix='cell::'+roi+'::'
            if not key.startswith(prefix):continue
            full_key='cell::full::'+key[len(prefix):]
            pair=cells.setdefault((row.site_id,key[len(prefix):]),[ScoreAccumulator(),ScoreAccumulator()])
            pair[0].add(row.accumulators[full_key]);pair[1].add(acc)
    eligible={key:pair for key,pair in cells.items() if pair[1].n_reference>=min_reference}
    def finite_error(acc):
        value = acc.finalize()['mae_z_cm']
        return value is not None and np.isfinite(value)

    missing=[key for key,pair in eligible.items() if not all(finite_error(acc) for acc in pair)]
    support={'n_cells_eligible':len(eligible),'n_cells_below_reference_threshold':len(cells)-len(eligible),
        'min_reference_per_region_cell':min_reference,'missing_prediction_cells':missing,
        'weight_source':'ROI reference composition within site, then equal site weights',
        'retained_reference_pixels':sum(pair[1].n_reference for pair in eligible.values()),
        'discarded_reference_pixels':sum(pair[1].n_reference for key,pair in cells.items() if key not in eligible)}
    by_site={};raw_full={};raw_region={}
    for (site,cell),(full,region) in eligible.items():
        if (site,cell) not in missing:
            difference=region.finalize()['mae_z_cm']-full.finalize()['mae_z_cm']
            by_site.setdefault(site,[]).append((difference,region.n_reference))
        # Pool every reference-eligible cell, including cells with no predictions.
        # Raw MAE can remain defined after pooling, while a cellwise contrast cannot.
        raw_full.setdefault(site,ScoreAccumulator()).add(full);raw_region.setdefault(site,ScoreAccumulator()).add(region)
    sites=sorted(raw_full)
    missing_raw_sites=[s for s in sites if not finite_error(raw_full[s]) or not finite_error(raw_region[s])]
    raw_defined=bool(sites) and not missing_raw_sites
    standardized_defined=bool(sites) and not missing
    raw=(np.array([raw_region[s].finalize()['mae_z_cm']-raw_full[s].finalize()['mae_z_cm'] for s in sites])
         if raw_defined else None)
    standardized=(np.array([np.average([x[0] for x in by_site[s]],weights=[x[1] for x in by_site[s]]) for s in sites])
                  if standardized_defined else None)
    intervals=[None,None]
    if len(sites)>=2:
        draws=np.random.default_rng(seed).integers(0,len(sites),(n_boot,len(sites)))
        intervals=[np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist() if x is not None else None
                   for x in (raw,standardized)]
    supported_status='ok' if len(sites)>=2 else 'ok_without_interval'
    raw_status=supported_status if raw_defined else ('undefined_prediction_support' if sites else 'insufficient_support')
    standardized_status=supported_status if standardized_defined else ('undefined_prediction_support' if sites else 'insufficient_support')
    return {**support,'status':standardized_status,'n_sites':len(sites),
        'raw_status':raw_status,'standardized_status':standardized_status,'missing_raw_sites':missing_raw_sites,
        'raw_retained_difference_cm':float(raw.mean()) if raw is not None else None,
        'standardized_difference_cm':float(standardized.mean()) if standardized is not None else None,
        'raw_retained_ci95':intervals[0],'standardized_ci95':intervals[1],
        'per_site':{s:{'raw':(raw_region[s].finalize()['mae_z_cm']-raw_full[s].finalize()['mae_z_cm'])
                            if s not in missing_raw_sites else None,
                       'standardized':float(standardized[i]) if standardized is not None else None}
                    for i,s in enumerate(sites)},
        'interpretation':'whole-frame error reweighted to the region\'s Z/angle distribution on the same reference-selected cells'}


def paired_region_comparisons(sample_scores, n_boot=2000, seed=20260912):
    output = {}
    for roi in ('hand', 'object', 'background'):
        full, region, tokens = {}, {}, []
        for row in sample_scores:
            acc = row.accumulators.get('roi::' + roi)
            # Selection depends on accepted labels and reference, never predictions.
            if acc is None or acc.n_reference == 0:
                continue
            tokens.append(row.sample_token)
            full.setdefault(row.site_id, ScoreAccumulator()).add(row.accumulators['roi::full'])
            region.setdefault(row.site_id, ScoreAccumulator()).add(acc)
        output[roi] = {
            'n_frames': len(tokens),
            'token_set_id': hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest()[:16],
            'full': site_macro(full).to_dict(),
            'region': site_macro(region).to_dict(),
            'paired_difference': paired_site_bootstrap(full, region, metric='mae_z_cm', n_boot=n_boot, seed=seed),
            'selection': 'same frames with accepted ROI and reference support, independent of model predictions',
            'standardized': standardized_region_comparison(sample_scores,roi,n_boot,seed),
        }
    return output
