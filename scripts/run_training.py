#!/usr/bin/env python3
"""Fit the train-only scale control for the MapAnything adaptation experiment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from egodepth_eval.data.mask_profile import get_profile
from egodepth_eval.data.population import FormalRunBlocked, PopulationRole, resolve_population
from egodepth_eval.data.reader import SampleReader
from egodepth_eval.train.dataset import SixViewTrainingDataset, TrainingBatchSpec, resize_reference
from egodepth_eval.train.loss import fit_calibration_scale


def load_config(path: Path) -> dict:
    from egodepth_eval.infer.assets import configure_paths

    config = yaml.safe_load(Path(path).read_text())
    configure_paths(config)
    return config


def build_reader(config: dict) -> SampleReader:
    return SampleReader(
        [Path(root) for root in config["data"]["asset_roots"]],
        get_profile(config["data"]["mask_profile"]),
    )


def training_spec(config: dict) -> TrainingBatchSpec:
    from egodepth_eval.train.contract import adaptation_inputs

    training = adaptation_inputs(config)
    return TrainingBatchSpec(
        height=training["height"],
        width=training["width"],
        view_ids=tuple(training["view_ids"]),
        pose_mode=training["pose_mode"],
    )


def command_calibrate(args: argparse.Namespace) -> int:
    """Fit the Scale-only scalar s_C on a train-only calibration subset C."""
    config = load_config(Path(args.config))
    from egodepth_eval.train.contract import adaptation_arms, adaptation_inputs

    public_name = adaptation_arms(config)[0]
    calibration = config.get("training", {}).get("calibration_subset", {})
    seed = int(calibration.get("seed", 20260912))
    n_samples = int(
        calibration.get("n_samples", 32) if args.n_calibration is None else args.n_calibration
    )

    # CALIBRATION is pinned to the train split by ROLE_SPLIT, so a config typo
    # cannot point C at val or test.
    population = resolve_population(config, PopulationRole.CALIBRATION, limit=args.limit)
    if population.status == "formal" and n_samples != 1024:
        raise FormalRunBlocked("The paper scale control fits 1,024 training anchors. Use development status for a different count.")
    spec = training_spec(config)
    reader = build_reader(config)
    dataset = SixViewTrainingDataset(population.samples, reader, spec, seed=seed)

    if not 0 < n_samples <= len(dataset):
        raise ValueError(
            "Calibration count must be positive and no larger than the training population"
        )

    sys.path.insert(0, str(Path(config["paths"]["third_party_dir"]) / "map-anything"))
    from mapanything.models import MapAnything

    from egodepth_eval.infer.mapanything_backend import MapAnythingRunner

    from egodepth_eval.infer.assets import hf_model_options

    checkpoint = config["methods"][public_name]["checkpoint"]
    options = {} if Path(checkpoint).exists() else hf_model_options("mapanything")
    model = MapAnything.from_pretrained(checkpoint, strict=True, **options).to(args.device).eval()
    runner = MapAnythingRunner(
        model,
        device=args.device,
        view_ids=spec.view_ids,
        height=spec.height,
        width=spec.width,
        pose_mode=spec.pose_mode,
        amp_dtype=config["training"].get("amp_dtype", "bf16"),
    )

    chosen, seen = [], set()
    for index in dataset.site_balanced_stream(max(n_samples * 100, len(dataset) * 20)):
        if index not in seen:
            chosen.append(index)
            seen.add(index)
        if len(chosen) == n_samples:
            break
    if len(chosen) != n_samples:
        raise RuntimeError("Could not draw the requested unique calibration set")
    predictions, targets, valids, tokens = [], [], [], []
    for index in chosen:
        sample = dataset.samples[index]
        loaded = reader.load(sample)
        result = runner.predict(loaded)
        predictions.append(result["raw"])
        target, target_valid = resize_reference(
            loaded.reference_depth, loaded.reference_valid, spec.height, spec.width
        )
        targets.append(target)
        # Reference validity only. Intersecting the model's support here would fit
        # s_C on a narrower domain than fine-tuning supervises.
        valids.append(target_valid)
        tokens.append(sample.sample_token)

    fit = fit_calibration_scale(
        predictions,
        targets,
        valids,
        huber_delta=float(config["training"].get("huber_delta", 0.1)),
    )
    if fit["status"] != "ok" or fit.get("n_frames_used") != len(tokens):
        raise RuntimeError(f"Incomplete calibration support: {fit}")
    from egodepth_eval.infer.assets import weight_identity

    payload = {
        "status": population.status,
        "arm": "scale_only",
        "s_C": fit["scale"],
        "weights_digest": weight_identity(
            "mapanything", config["methods"][public_name]["checkpoint"]
        )["digest"],
        "input_contract": adaptation_inputs(config),
        "training_grid_hw": [spec.height, spec.width],
        "fit": fit,
        "population": population.identity(),
        "calibration_subset": {
            "n_requested": n_samples,
            "n_used": len(tokens),
            "seed": seed,
            "split": population.split,
            "sample_tokens": tokens,
            "sites": sorted({dataset.samples[i].physical_site_id for i in chosen}),
        },
        "contract": (
            "One positive scalar fitted on the train split only, fixed across every "
            "test sample and site. The fine-tuned arm is NOT multiplied by it."
        ),
    }
    out_dir = Path(args.out_dir or config["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scale_only_calibration.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(
        json.dumps(
            {
                "s_C": fit["scale"],
                "n_pixels": fit["n_pixels"],
                "n_frames": fit["n_frames_used"],
                "reduction": fit.get("reduction"),
                "split": population.split,
                "status": fit["status"],
            },
            indent=2,
        )
    )
    print(f"[calibrate] -> {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True).add_parser("calibrate")
    sub.add_argument("--config", required=True)
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--out-dir", "--out_dir", dest="out_dir")
    sub.add_argument("--n-calibration", "--n_calibration", dest="n_calibration", type=int)
    sub.add_argument("--limit", type=int, help="development population limit")
    args = parser.parse_args()
    try:
        return command_calibrate(args)
    except FormalRunBlocked as exc:
        print(f"Formal run blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
