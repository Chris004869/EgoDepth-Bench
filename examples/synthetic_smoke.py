#!/usr/bin/env python3
"""Exercise real data loading and scoring using explicitly synthetic plane data.

No trained model is run. Three analytic reference planes and independently
specified constant predictions demonstrate bias, missing support, and equal-site
aggregation. These numbers are an executable check, never benchmark results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from egodepth_eval.data.contract import CanonicalSample, DSCalibration
from egodepth_eval.score.external import score_predictions
from egodepth_eval.score.prediction_io import save_prediction


def create_synthetic_fixture(root: Path) -> tuple[dict, Path]:
    root = Path(root).resolve()
    assets, predictions = root / "synthetic_assets", root / "synthetic_predictions"
    assets.mkdir(parents=True)
    predictions.mkdir()
    shape = (16, 16)
    calibrations = {i: DSCalibration(i, 12 + i * .1, 12, 7.5, 7.5, .01, .57, 16, 16) for i in range(6)}
    samples = []
    # Site A has twice as many frames; it must still have the same macro weight.
    for token, site, reference_z, prediction_z in (
        ("synthetic_a0", "synthetic_site_a", 1.0, 1.125),
        ("synthetic_a1", "synthetic_site_a", 1.0, 1.125),
        ("synthetic_b0", "synthetic_site_b", .5, .625),
    ):
        np.save(assets / f"{token}.depth.npy", np.full(shape, reference_z, np.float32))
        np.save(assets / f"{token}.valid.npy", np.ones(shape, bool))
        support = np.ones(shape, bool)
        if site == "synthetic_site_b":
            support[:, 8:] = False
        save_prediction(predictions / f"{token}.npz", np.full(shape, prediction_z, np.float32), support)
        samples.append(CanonicalSample(sample_token=token, recording_id=f"recording_{site}",
            timestamp_ns=1_000_000_000 + len(samples) * 1_000_000_000,
            rgb_rel_paths={i: f"unused_camera{i}.jpg" for i in range(6)},
            depth_rel_path=f"{token}.depth.npy", valid_mask_rel_path=f"{token}.valid.npy",
            depth_shape_hw=shape, calibrations=calibrations, physical_site_id=site, split="dev",
            data_version="synthetic-only", selection_policy="SYNTHETIC_DEMONSTRATION").to_dict())
    manifest = root / "synthetic_manifest.jsonl"
    manifest.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    config = {"run": {"status": "development", "population_name": "synthetic_demo_not_benchmark"},
              "data": {"samples_jsonl": str(manifest), "asset_roots": [str(assets)],
                       "data_version": "synthetic-only", "mask_profile": "release_declared_mask"}}
    (root / "synthetic_config.yaml").write_text(yaml.safe_dump(config))
    return config, predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for inspectable synthetic inputs and results")
    args = parser.parse_args(argv)
    if args.output_dir:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="egodepth-synthetic-"))
    config, predictions = create_synthetic_fixture(root)
    result = score_predictions(config, predictions, root / "synthetic_result.json", method="synthetic_constant_planes")
    metrics = result["strata"]["roi::full"]["metrics"]
    expected = {"abs_rel": .1875, "mae_z_cm": 12.5, "coverage": .75, "a_5cm": 0.0}
    for name, value in expected.items():
        if not np.isclose(metrics[name], value, rtol=0, atol=1e-12):
            raise AssertionError(f"{name}: expected {value}, got {metrics[name]}")
    print(json.dumps({"status": "passed", "synthetic": True, "trained_model_used": False,
                      "directory": str(root), "expected_metrics": expected,
                      "population_status": result["population"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
