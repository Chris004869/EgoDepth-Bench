#!/usr/bin/env python3
"""Bind a completed validation-selected training run and measured scale into E4."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egodepth_eval.data.population import resolve_train_val
from egodepth_eval.infer.assets import weight_identity
from egodepth_eval.infer.assets import upstream_identity, configure_paths
from egodepth_eval.train.distributed import optimizer_config_identity, training_source_identity
from egodepth_eval.train.contract import (
    adaptation_arms,
    adaptation_inputs,
    verify_calibration_inputs,
)

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--seed-dir", required=True)
p.add_argument("--calibration", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
c = yaml.safe_load(Path(a.config).read_text())
directory = Path(a.seed_dir).resolve()
record = json.loads((directory / "training_run.json").read_text())
configure_paths(c)
public_name, scale_name, ft_name = adaptation_arms(c)
inputs = adaptation_inputs(c)
if record["status"] != "complete" or record["best_step"] is None:
    raise ValueError("Training is not complete with a selected checkpoint")
identity = record["identity"]
observed = identity["optimization"]
expected = dict(c["training"])
seed = observed.get("seed")
if seed not in expected.get("seeds", [20260912]):
    raise ValueError("Training seed is not declared by this evaluation recipe")
expected["seed"] = seed
if observed != optimizer_config_identity(expected, record["world_size"]):
    raise ValueError("Training optimization recipe differs from the frozen evaluation recipe")
if record["step"] != expected["max_steps"] or not 0 < record["best_step"] <= record["step"]:
    raise ValueError("Training did not complete the declared step budget")
root = Path(__file__).resolve().parents[1]
if identity.get("training_source") != training_source_identity(root):
    raise ValueError("Training source differs from this checkout; use the matching archived bundle")
upstream = upstream_identity("mapanything", c["paths"]["third_party_dir"])
if identity["upstream"]["source_digest"] != upstream["source_digest"]:
    raise ValueError("Training and evaluation MapAnything source differ")
train, val = resolve_train_val(c)
identity = record["identity"]
if identity["train_tokens"] != train.token_set_id() or identity["val_tokens"] != val.token_set_id():
    raise ValueError("Training/validation populations differ")
weights = weight_identity("mapanything", c["methods"][public_name]["checkpoint"])["digest"]
if identity.get("input_contract") != inputs:
    raise ValueError("Training and evaluation geometry/input contracts differ")
if identity["weights"] != weights:
    raise ValueError("Public initialization differs from evaluation")
state = directory / "checkpoint_best.pt"
ckpt = torch.load(state, map_location="cpu", weights_only=False)
if ckpt["identity"] != identity or ckpt["step"] != record["best_step"]:
    raise ValueError("Selected checkpoint differs from training record")
del ckpt
calibration_path = Path(a.calibration).resolve()
cal = json.loads(calibration_path.read_text())
s = cal["s_C"]
verify_calibration_inputs(cal, inputs)
if cal["status"] != "formal" or cal["fit"]["status"] != "ok" or not np.isfinite(s) or s <= 0:
    raise ValueError("Calibration did not complete")
if (
    cal["calibration_subset"]["split"] != "train"
    or cal["population"]["token_set_id"] != train.token_set_id()
):
    raise ValueError("Calibration uses a different training population")
if cal.get("weights_digest") != weights:
    raise ValueError("Calibration public weights differ")
if set(cal["calibration_subset"]["sample_tokens"]) - set(train.sample_tokens):
    raise ValueError("Calibration contains non-training samples")
c["methods"][ft_name]["finetuned_state_dict"] = str(state)
c["methods"][ft_name]["training_record"] = str(directory / "training_run.json")
c["methods"][scale_name]["scale_factor"] = float(s)
c["methods"][scale_name]["calibration_record"] = str(calibration_path)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(yaml.safe_dump(c, sort_keys=False))
print(a.out)
