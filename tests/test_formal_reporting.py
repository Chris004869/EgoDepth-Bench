import pytest

from egodepth_eval.report.diagnostics import standardize_motion_by_site


def test_motion_standardization_preserves_site_macro_and_its_own_uncertainty():
    # Site B has 100 times more pixels. It must still have half the final weight.
    # Composition adjustment reverses A's sign but leaves B unchanged: using the
    # raw interval for the standardized estimate would conceal that difference.
    differences = {'A::near': -10., 'A::far': 10., 'B::near': 4., 'B::far': 4.}
    weights = {'A::near': 9, 'A::far': 1, 'B::near': 500, 'B::far': 500}
    result = standardize_motion_by_site(differences, weights, {'A': 10., 'B': 4.})
    assert result['standardized'] == pytest.approx(-2.)
    assert result['unweighted'] == pytest.approx(7.)
    assert result['standardized_ci95'] == pytest.approx([-8., 4.])
    assert result['unweighted_ci95'] == pytest.approx([4., 10.])


def test_one_site_motion_difference_does_not_claim_a_confidence_interval():
    result = standardize_motion_by_site({'A::near': 3.}, {'A::near': 200}, {'A': 3.})
    assert result['standardized'] == 3.
    assert result['standardized_ci95'] is None


def test_region_standardization_removes_depth_composition_only_effect():
    from types import SimpleNamespace
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    from egodepth_eval.score.accumulate import ScoreAccumulator
    def acc(n,cm):return ScoreAccumulator(n_reference=n,n_scored=n,sum_abs_err_m=n*cm/100)
    rows=[]
    for site in ['A','B']:
        rows.append(SimpleNamespace(site_id=site,accumulators={
            'roi::object':acc(1200,3.25),
            'cell::full::near':acc(1000,1),'cell::full::far':acc(9000,10),
            'cell::object::near':acc(900,1),'cell::object::far':acc(300,10)}))
    result=standardized_region_comparison(rows,'object')
    assert result['raw_retained_difference_cm']==pytest.approx(-5.85)
    assert result['standardized_difference_cm']==pytest.approx(0.)
    assert result['n_cells_eligible']==4
    # Missing predictions must not silently remove a difficult eligible cell.
    rows[0].accumulators['cell::object::far'].n_scored=0
    assert standardized_region_comparison(rows,'object')['status']=='undefined_prediction_support'


def test_motion_cannot_drop_reference_eligible_cell_with_missing_predictions(monkeypatch, tmp_path):
    import egodepth_eval.report.diagnostics as diagnostics
    from egodepth_eval.score.accumulate import ScoreAccumulator
    key = 'A::near::central'
    active = ScoreAccumulator(n_reference=1000, n_scored=0)
    low = ScoreAccumulator(n_reference=1000, n_scored=1000, sum_abs_err_m=10)
    monkeypatch.setattr(diagnostics, 'collect_motion_strata', lambda *a, **k: {
        'by_condition': {'active_manipulation': {key: active}, 'low_motion': {key: low}},
        'reference_weight': {key: 2000}, 'counters': {'n_active': 1, 'n_low': 1}})
    result = diagnostics.compute_motion_contrast(None, None, {}, tmp_path, 'example')
    assert result['status'] == 'undefined_prediction_support'
    assert result['n_strata_retained'] == 1
    assert result['n_strata_discarded'] == 0


def _contrast_acc(reference, cm=0., scored=None):
    from egodepth_eval.score.accumulate import ScoreAccumulator
    scored = reference if scored is None else scored
    return ScoreAccumulator(n_reference=reference, n_scored=scored,
                            sum_abs_err_m=scored * cm / 100)


def _region_row(site, cells):
    from types import SimpleNamespace
    return SimpleNamespace(site_id=site, accumulators={
        'roi::object': _contrast_acc(sum(region.n_reference for _, region in cells.values())),
        **{'cell::full::' + key: full for key, (full, _) in cells.items()},
        **{'cell::object::' + key: region for key, (_, region) in cells.items()},
    })


def _motion_result(monkeypatch, tmp_path, active, low, **kwargs):
    import egodepth_eval.report.diagnostics as diagnostics
    monkeypatch.setattr(diagnostics, 'collect_motion_strata', lambda *a, **k: {
        'by_condition': {'active_manipulation': active, 'low_motion': low},
        'reference_weight': {
            key: active.get(key, _contrast_acc(0)).n_reference
                 + low.get(key, _contrast_acc(0)).n_reference
            for key in active.keys() | low.keys()},
        'counters': {'n_active': 3, 'n_low': 3},
    })
    return diagnostics.compute_motion_contrast(None, None, {}, tmp_path, 'example', **kwargs)


def test_region_raw_retains_eligible_cells_when_standardization_is_undefined():
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    rows = [
        _region_row('A', {'near': (_contrast_acc(200, 4), _contrast_acc(200, 8)),
                          'far': (_contrast_acc(800, 10), _contrast_acc(200, scored=0))}),
        _region_row('B', {'near': (_contrast_acc(200, 2), _contrast_acc(200, 6))}),
    ]
    result = standardized_region_comparison(rows, 'object')
    assert result['status'] == result['standardized_status'] == 'undefined_prediction_support'
    assert result['raw_status'] == 'ok'
    assert result['n_cells_eligible'] == 3
    assert result['missing_prediction_cells'] == [('A', 'far')]
    assert result['n_sites'] == 2
    assert result['missing_raw_sites'] == []
    # A's full operand still includes the unpredicted ROI cell: 8 - 8.8 = -0.8.
    # Dropping that cell would incorrectly produce a raw mean of 4 instead of 1.6.
    assert result['raw_retained_difference_cm'] == pytest.approx(1.6)
    assert result['raw_retained_ci95'] == pytest.approx([-.8, 4.])
    assert result['standardized_difference_cm'] is None
    assert result['standardized_ci95'] is None


@pytest.mark.parametrize('missing_operand', ['full', 'region', 'both'])
def test_region_raw_never_drops_an_unpredicted_eligible_site(missing_operand):
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    full = _contrast_acc(200, 4, scored=0 if missing_operand in ('full', 'both') else 200)
    region = _contrast_acc(200, 8, scored=0 if missing_operand in ('region', 'both') else 200)
    result = standardized_region_comparison([
        _region_row('A', {'near': (full, region)}),
        _region_row('B', {'near': (_contrast_acc(200, 2), _contrast_acc(200, 6))}),
    ], 'object')
    assert result['raw_status'] == 'undefined_prediction_support'
    assert result['n_sites'] == 2
    assert result['missing_raw_sites'] == ['A']
    assert set(result['per_site']) == {'A', 'B'}
    assert result['raw_retained_difference_cm'] is None
    assert result['raw_retained_ci95'] is None
    assert result['standardized_difference_cm'] is None


@pytest.mark.parametrize('empty', [True, False])
def test_region_without_eligible_reference_reports_no_estimate(empty):
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    rows = [] if empty else [_region_row('A', {'near': (_contrast_acc(200, 4), _contrast_acc(199, 8))})]
    result = standardized_region_comparison(rows, 'object')
    assert result['status'] == result['raw_status'] == result['standardized_status'] == 'insufficient_support'
    assert result['n_sites'] == result['n_cells_eligible'] == 0
    assert result['raw_retained_difference_cm'] is None
    assert result['standardized_difference_cm'] is None


def test_region_one_site_keeps_point_estimates_without_intervals():
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    result = standardized_region_comparison([
        _region_row('A', {'near': (_contrast_acc(200, 0), _contrast_acc(200, 0))}),
    ], 'object')
    assert result['raw_status'] == result['standardized_status'] == 'ok_without_interval'
    assert result['raw_retained_difference_cm'] == result['standardized_difference_cm'] == 0.
    assert result['raw_retained_ci95'] is result['standardized_ci95'] is None


def test_region_full_support_preserves_frozen_point_estimates_and_bootstrap():
    from egodepth_eval.report.paired_regions import standardized_region_comparison
    rows = [_region_row(site, {'near': (_contrast_acc(*fn), _contrast_acc(*rn)),
                              'far': (_contrast_acc(*ff), _contrast_acc(*rf))})
            for site, fn, ff, rn, rf in [
                ('A', (1000, 1), (9000, 10), (900, 1), (300, 10)),
                ('B', (2000, 2), (2000, 6), (400, 5), (800, 10)),
                ('C', (1000, 0), (500, 2), (300, 0), (300, 1)),
            ]]
    result = standardized_region_comparison(rows, 'object', n_boot=17, seed=31)
    # Values frozen from the pre-change implementation, including its RNG draws.
    assert result['raw_retained_difference_cm'] == -0.5611111111111108
    assert result['standardized_difference_cm'] == 1.0555555555555556
    assert result['raw_retained_ci95'] == [-5.0922222222222215, 2.233333333333333]
    assert result['standardized_ci95'] == [-0.43333333333333335, 2.444444444444444]
    assert result['raw_status'] == result['standardized_status'] == 'ok'


def test_motion_raw_retains_eligible_cells_when_standardization_is_undefined(monkeypatch, tmp_path):
    result = _motion_result(monkeypatch, tmp_path,
        {'A::near': _contrast_acc(200, 8), 'A::far': _contrast_acc(200, scored=0),
         'B::near': _contrast_acc(200, 6)},
        {'A::near': _contrast_acc(200, 4), 'A::far': _contrast_acc(800, 10),
         'B::near': _contrast_acc(200, 2)})
    assert result['status'] == result['standardized_status'] == 'undefined_prediction_support'
    assert result['raw_status'] == 'ok'
    assert result['n_strata_retained'] == 3
    assert result['n_strata_discarded'] == 0
    assert result['missing_prediction_strata'] == ['A::far']
    comparison = result['comparisons'][0]
    assert comparison['unweighted'] == pytest.approx(1.6)
    assert comparison['unweighted_ci95'] == pytest.approx([-.8, 4.])
    assert comparison['standardized'] is comparison['standardized_ci95'] is None
    assert comparison['n_sites_compared'] == comparison['n_sites_eligible'] == 2
    assert result['paired_bootstrap']['n_sites_dropped'] == 0


@pytest.mark.parametrize('missing_operand', ['active', 'low', 'both'])
def test_motion_raw_never_drops_an_unpredicted_eligible_site(monkeypatch, tmp_path, missing_operand):
    result = _motion_result(monkeypatch, tmp_path,
        {'A::near': _contrast_acc(200, 8, scored=0 if missing_operand in ('active', 'both') else 200),
         'B::near': _contrast_acc(200, 6)},
        {'A::near': _contrast_acc(200, 4, scored=0 if missing_operand in ('low', 'both') else 200),
         'B::near': _contrast_acc(200, 2)})
    assert result['raw_status'] == 'undefined_prediction_support'
    assert result['n_sites_eligible'] == 2
    assert result['missing_raw_sites'] == ['A']
    comparison = result['comparisons'][0]
    assert comparison['unweighted'] is comparison['unweighted_ci95'] is None
    assert comparison['standardized'] is comparison['standardized_ci95'] is None
    assert comparison['n_sites_compared'] == 0
    assert 'paired_bootstrap' not in result


@pytest.mark.parametrize('empty', [True, False])
def test_motion_without_eligible_reference_reports_no_estimate(monkeypatch, tmp_path, empty):
    result = _motion_result(monkeypatch, tmp_path,
        {} if empty else {'A::near': _contrast_acc(199, 2)},
        {} if empty else {'A::near': _contrast_acc(200, 4)})
    assert result['status'] == result['raw_status'] == result['standardized_status'] == 'insufficient_support'
    assert result['n_strata_retained'] == 0
    assert result['comparisons'] == []


@pytest.mark.parametrize('partial', [True, False])
def test_motion_one_site_keeps_raw_without_interval(monkeypatch, tmp_path, partial):
    active = {'A::near': _contrast_acc(200, 0)}
    low = {'A::near': _contrast_acc(200, 0)}
    if partial:
        active['A::far'] = _contrast_acc(200, scored=0)
        low['A::far'] = _contrast_acc(200, 0)
    result = _motion_result(monkeypatch, tmp_path, active, low)
    comparison = result['comparisons'][0]
    assert result['raw_status'] == 'ok_without_interval'
    assert comparison['unweighted'] == 0.
    assert comparison['unweighted_ci95'] is comparison['standardized_ci95'] is None
    assert comparison['active_mae_z_cm'] == comparison['low_motion_mae_z_cm'] == 0.
    if partial:
        assert result['standardized_status'] == 'undefined_prediction_support'
        assert comparison['standardized'] is None
    else:
        assert result['standardized_status'] == 'ok_without_interval'
        assert comparison['standardized'] == 0.


def test_motion_full_support_preserves_frozen_point_estimates_and_bootstrap(monkeypatch, tmp_path):
    active = {key: _contrast_acc(n, cm) for key, n, cm in [
        ('A::near', 200, 8), ('A::far', 800, 10), ('B::near', 400, 0),
        ('B::far', 200, 6), ('C::near', 300, 2), ('C::far', 900, 9)]}
    low = {key: _contrast_acc(n, cm) for key, n, cm in [
        ('A::near', 400, 4), ('A::far', 200, 5), ('B::near', 200, 0),
        ('B::far', 800, 2), ('C::near', 300, 5), ('C::far', 300, 3)]}
    result = _motion_result(monkeypatch, tmp_path, active, low, n_boot=17, seed=31)
    comparison = result['comparisons'][0]
    # Values frozen from the pre-change implementation, including its RNG draws.
    assert comparison['unweighted'] == 2.972222222222222
    assert comparison['standardized'] == 3.375
    assert comparison['unweighted_ci95'] == [1.6188888888888888, 4.997777777777777]
    assert comparison['standardized_ci95'] == [2.7333333333333334, 4.408333333333333]
    assert result['raw_status'] == result['standardized_status'] == 'ok'
