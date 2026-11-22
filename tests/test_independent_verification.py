"""Cross-check failure and subset semantics for the independent full-image audit."""

import numpy as np
import pytest

from scripts.verify_formal_scores_independent import compare, macro_metrics, verify


def test_missing_support_stays_in_denominator():
    counts = {
        "a": np.array([4, 4, 0, 0, 4, 4]),
        "b": np.array([3, 0, 0, 0, 0, 0]),
        "empty": np.zeros(6),
    }
    metrics = macro_metrics(counts)
    assert metrics == {
        "coverage": 0.5,
        "abs_rel": None,
        "mae_z_cm": None,
        "delta_1": None,
        "a_5cm": 0.5,
    }
    expected = dict(metrics, coverage=1.0, abs_rel=0.1)
    comparison = compare(metrics, expected)
    assert not comparison["coverage"]["ok"] and not comparison["abs_rel"]["ok"]
    assert comparison["mae_z_cm"]["ok"]


def test_probe_reports_only_calculated_values(tmp_path):
    depth = np.ones((8, 8), dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    np.savez(tmp_path / "anchor.npz", depth_z=depth, valid=valid)
    dataset = [
        {
            "sample_token": "anchor",
            "site_id": "site",
            "target_depth_z": depth,
            "target_valid": valid,
        }
    ]
    result = {"method": "example"}
    report = verify(dataset, tmp_path, "example", result, limit=1)
    assert report["status"] == "computed_only" and not report["comparison"]
    assert report["macro"]["abs_rel"] == 0.0
    with pytest.raises(ValueError, match="at least one frame"):
        verify(dataset, tmp_path, "example", result, limit=0)
