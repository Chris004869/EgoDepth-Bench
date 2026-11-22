#!/usr/bin/env python3
"""Recompute five full-image metrics from release shards and saved predictions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sdk"), str(ROOT / "src")]
from egodepth_dataset import EgoDepthDataset

# The paper declares these five relative-output baselines. All other archived
# methods emit metric Z, including the arm with a fixed training-set scale.
ORACLE_MODES = {
    "depth_anything_v2": "affine_inverse",
    "marigold": "affine_depth",
    "moge": "positive_scale",
    "vggt": "positive_scale",
    "vggt_omega": "positive_scale",
}


def aligned_depth(payload, gt, reference, support, mode):
    depth = payload["depth_z"].astype(np.float32)
    if depth.shape != gt.shape:
        raise ValueError("depth_z and the reference must share the scoring grid")
    if mode == "none":
        return depth
    # Historical runner files may keep raw on the native model grid. In that
    # case depth_z contains the runner's resampled representation for fitting.
    raw = payload["raw"] if "raw" in payload else depth
    raw = raw.astype(np.float32) if raw.shape == gt.shape else depth
    valid = reference & support & np.isfinite(raw)
    if mode == "positive_scale":
        valid &= raw > 0
    elif mode == "affine_inverse":
        valid &= raw >= 0
    elif mode != "affine_depth":
        raise ValueError(f"Unknown alignment mode {mode}")
    if np.count_nonzero(valid) < 32:
        return np.full_like(raw, np.nan)
    x = raw[valid].astype(np.float64)
    y = gt[valid].astype(np.float64)
    if mode == "positive_scale":
        denominator = np.sum(x * x)
        if denominator <= 1e-12:
            return np.full_like(raw, np.nan)
        return (raw * float(np.sum(x * y) / denominator)).astype(np.float32)

    if mode == "affine_inverse":
        y = 1 / y
    centered = x - x.mean()
    denominator = np.sum(centered**2)
    if denominator == 0:
        return np.full_like(raw, np.nan)
    slope = float(np.sum(centered * (y - y.mean())) / denominator)
    offset = float(y.mean() - slope * x.mean())
    if mode == "affine_depth":
        return (raw * slope + offset).astype(np.float32)
    inverse = raw.astype(np.float64) * slope + offset
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inverse > 1e-8, 1 / inverse, np.nan).astype(np.float32)


def frame_counts(sample, path, mode):
    gt = sample["target_depth_z"]
    reference = sample["target_valid"] & np.isfinite(gt) & (gt >= 0.25) & (gt <= 2.0)
    with np.load(path, allow_pickle=False) as payload:
        support = payload["valid"]
        if support.shape != gt.shape or support.dtype != np.bool_:
            raise ValueError(f"{path.name}: valid must be a boolean mask on the scoring grid")
        pred = aligned_depth(payload, gt, reference, support, mode)
    scored = reference & support & np.isfinite(pred) & (pred > 0)
    predicted = pred[scored].astype(np.float64)
    target = gt[scored].astype(np.float64)
    error = np.abs(predicted - target)
    return np.array(
        [
            reference.sum(),
            scored.sum(),
            np.sum(error / target),
            error.sum(),
            np.sum(np.maximum(predicted / target, target / predicted) < 1.25),
            np.sum(error <= 0.05),
        ],
        dtype=np.float64,
    )


def site_metrics(counts):
    reference, scored, relative, absolute, delta, accurate = counts
    return {
        "coverage": scored / reference if reference else None,
        "abs_rel": relative / scored if scored else None,
        "mae_z_cm": 100 * absolute / scored if scored else None,
        "delta_1": delta / scored if scored else None,
        "a_5cm": accurate / reference if reference else None,
    }


def macro_metrics(per_site):
    applicable = [site_metrics(counts) for counts in per_site.values() if counts[0] > 0]
    macro = {}
    for key in ("coverage", "abs_rel", "mae_z_cm", "delta_1", "a_5cm"):
        values = [metrics[key] for metrics in applicable]
        macro[key] = float(np.mean(values)) if values and None not in values else None
    return macro


def compare(macro, expected):
    comparison = {}
    for key, value in macro.items():
        target = expected[key]
        difference = None if value is None or target is None else abs(value - target)
        agrees = (
            (value is target)
            if difference is None
            else bool(np.isclose(value, target, rtol=2e-6, atol=1e-7))
        )
        comparison[key] = {
            "independent": value,
            "pipeline": target,
            "absolute_difference": difference,
            "ok": agrees,
        }
    return comparison


def verify(dataset, predictions, method, result, limit=None):
    if not len(dataset) or (limit is not None and limit <= 0):
        raise ValueError("The protocol and any requested probe must contain at least one frame")
    if result["method"] != method:
        raise ValueError(f"Expected results for {method}, found {result['method']}")
    mode = ORACLE_MODES.get(method, "none")
    indices = (
        range(len(dataset))
        if limit is None
        else np.linspace(0, len(dataset) - 1, min(limit, len(dataset)), dtype=int)
    )
    counts_by_site = defaultdict(lambda: np.zeros(6))
    frames = []
    for n, index in enumerate(indices, start=1):
        sample = dataset[index]
        token, site = sample["sample_token"], sample["site_id"]
        counts = frame_counts(sample, predictions / f"{token}.npz", mode)
        counts_by_site[site] += counts
        frames.append(
            {
                "sample_token": token,
                "site_id": site,
                "n_reference": int(counts[0]),
                "n_scored": int(counts[1]),
            }
        )
        if n == 1 or n % 500 == 0:
            print(f"{method}: {n}/{len(indices)}", flush=True)
    macro = macro_metrics(counts_by_site)
    comparison = {}
    identity = {}
    status = "computed_only"
    if limit is None:
        from egodepth_eval.data import paper_protocol as paper

        digest = paper.token_sites_sha256((f["sample_token"], f["site_id"]) for f in frames)
        population = result["population"]
        identity = {
            "matches_frozen_token_sites": digest == paper.TOKEN_SITES_SHA256,
            "matches_result_token_set": population["token_set_id"] == paper.TOKEN_SET_ID,
            "matches_sites": set(counts_by_site)
            == set(population["sites"])
            == set(paper.TEST_SITES),
            "matches_count": len(frames)
            == result["n_samples"]
            == population["n_samples"]
            == paper.N_TEST,
        }
        comparison = compare(macro, result["strata"]["roi::full"]["metrics"])
        status = (
            "passed"
            if all(identity.values()) and all(v["ok"] for v in comparison.values())
            else "failed"
        )
    return {
        "method": method,
        "status": status,
        "scope": "full_test" if limit is None else "probe_only",
        "alignment": mode,
        "n_samples": len(frames),
        "n_sites": len(counts_by_site),
        "site_metrics": {site: site_metrics(counts) for site, counts in counts_by_site.items()},
        "macro": macro,
        "comparison": comparison,
        "identity_checks": identity,
        "per_frame_support": frames,
        "implementation": "release SDK and NumPy; independent metric and alignment calculations",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="compute a probe without certifying the table")
    parser.add_argument("--result", type=Path, help="reference JSON, defaults to the run's results")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run = Path(config["run"]["out_dir"])
    path = args.result or run / "results" / f"{args.method}.json"
    result = json.loads(path.read_text())
    data = EgoDepthDataset(
        config["data"]["asset_roots"][0],
        split="test",
        protocol="protocols/test_1hz.jsonl",
        cameras=(),
    )
    try:
        output = verify(data, run / "predictions" / args.method, args.method, result, args.limit)
    finally:
        data.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("per_frame_support", "site_metrics")},
            indent=2,
        )
    )
    return int(output["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
