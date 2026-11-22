#!/usr/bin/env python3
"""Apply a frozen train-split scalar to the public arm's metric-Z predictions."""
import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egodepth_eval.data.population import PopulationRole, resolve_population
from egodepth_eval.score.prediction_io import (
    PredictionContract, check_resumable, file_content_digest, load_prediction,
    save_prediction, write_run_manifest,
)
from egodepth_eval.train.contract import adaptation_arms, adaptation_inputs, validate_fitted_scalar


def validate_calibration(record, factor, inputs, train_tokens):
    validate_fitted_scalar(record, factor, inputs)
    if not set(record["calibration_subset"]["sample_tokens"]) <= train_tokens:
        raise ValueError("The calibration contains samples outside the training population")


def validate_source(public, name, calibration, inputs, token_set_id):
    if public.method != name or public.output_space != "metric_depth_z":
        raise ValueError("The source must be the configured public metric-Z arm")
    if not public.checkpoint_content_id or public.checkpoint_content_id != calibration["weights_digest"]:
        raise ValueError("The source checkpoint differs from the calibration checkpoint")
    if public.preprocessing.get("pose_mode", "none") != inputs["pose_mode"]:
        raise ValueError("The source and calibration pose modes differ")
    if public.precision != inputs["amp_dtype"] or public.preprocessing.get("model_input_hw") != [
        inputs["height"], inputs["width"]
    ]:
        raise ValueError("The source and calibration precision or input grids differ")
    if public.population_identity.get("token_set_id") != token_set_id:
        raise ValueError("The source predictions cover a different test population")
    if tuple(public.input_views) != tuple(range(6)) or public.scale_contract != "none (raw metric)":
        raise ValueError("Derivation requires unscaled predictions from the six-view public arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config = yaml.safe_load(args.config.read_text())
    public_name, scale_name, _ = adaptation_arms(config)
    inputs = adaptation_inputs(config)
    method = config["methods"][scale_name]
    calibration_path = Path(method["calibration_record"])
    calibration = json.loads(calibration_path.read_text())
    factor = float(method["scale_factor"])
    train = resolve_population(config, PopulationRole.CALIBRATION)
    validate_calibration(calibration, factor, inputs, {sample.sample_token for sample in train.samples})

    population = resolve_population(config, PopulationRole.TEST)
    predictions = Path(config["run"]["out_dir"]) / "predictions"
    source = predictions / public_name
    destination = predictions / scale_name
    source_contract = source / "prediction_contract.json"
    public = PredictionContract.from_dict(json.loads(source_contract.read_text()))
    validate_source(public, public_name, calibration, inputs, population.token_set_id())
    for sample in population.samples:
        if not (source / f"{sample.sample_token}.npz").is_file():
            raise FileNotFoundError(f"Missing public prediction: {sample.sample_token}")

    # Preserve the source network identity. Only the post-resampling multiplier changes.
    contract = dataclasses.replace(
        public, method=scale_name, scale_contract=f"fixed positive scalar s_C={factor:.9f} from train",
        notes={
            **public.notes,
            "calibration_record": str(calibration_path),
            "calibration_record_digest": file_content_digest(calibration_path),
            "derivation": "same frozen network predictions; scalar applied after output resampling",
            "source_contract_digest": file_content_digest(source_contract),
            "derivation_script_digest": file_content_digest(Path(__file__)),
        },
    )
    resume = check_resumable(destination, contract)
    if not resume["resumable"]:
        raise RuntimeError(resume)
    start = time.time()
    write_run_manifest(destination, contract, {"started": start, "status": "deriving"})

    def derive(sample):
        token = sample.sample_token
        prediction = load_prediction(source / f"{token}.npz")
        scaled = (prediction.depth_z * factor).astype(np.float32)
        out = destination / f"{token}.npz"
        if out.exists():
            cached = load_prediction(out)
            if not np.array_equal(cached.depth_z, scaled, equal_nan=True) or not np.array_equal(
                cached.valid, prediction.valid
            ):
                raise RuntimeError(f"Bad derived cache: {token}")
            return "verified"
        with NamedTemporaryFile(dir=destination, prefix=f".{token}.", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            save_prediction(temporary, scaled, prediction.valid, raw_output=prediction.raw,
                            raw_space=prediction.raw_space, raw_valid=prediction.raw_valid,
                            raw_transform=prediction.raw_transform)
            temporary.replace(out)
        finally:
            temporary.unlink(missing_ok=True)
        return "written"

    counts = {"written": 0, "verified": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, status in enumerate(pool.map(derive, population.samples), start=1):
            counts[status] += 1
            if index == 1 or index % 500 == 0:
                print(f"derive: {index}/{len(population.samples)}", flush=True)
    record = {
        "status": "complete", "n_expected": len(population.samples),
        "n_written": counts["written"], "n_verified": counts["verified"],
        "seconds": time.time() - start, "source": public_name, "factor": factor,
        "workers": args.workers, "gpu_forward_calls": 0,
        "cost_note": "reuses the public forward pass; CPU scalar application only",
    }
    write_run_manifest(destination, contract, record)
    (destination / "derivation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
