"""Depth Anything 3 inference runner -- monocular metric depth (Group A).

Uses DA3METRIC-LARGE, the monocular metric variant. Its raw output is *not* in
metres: per the model card,

    metric_depth = focal * net_output / 300

where `focal` is the focal length in pixels (mean of fx, fy). Skipping that
conversion yields depth wrong by a factor of ~focal/300, which for this rig is
roughly 1.7x -- large enough to dominate the score and small enough to look
like a plausible model failure rather than a units bug. Whether the API already
applies the conversion is detected at runtime rather than assumed, by checking
the returned magnitude against the benchmark's evaluation range.

Weights are read from the local model directory when present, so the run does
not depend on network access.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, pinhole_K_for_model, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("depth_anything_3")
LOCAL_WEIGHTS = local_weight_path("depth_anything_3")
MODEL_ID = "depth-anything/DA3METRIC-LARGE"

# Constant from the DA3 model card's metric conversion.
DA3_METRIC_DIVISOR = 300.0


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY / "src"))
    from depth_anything_3.api import DepthAnything3

    source = str(LOCAL_WEIGHTS) if (LOCAL_WEIGHTS / "model.safetensors").exists() else MODEL_ID
    model = DepthAnything3.from_pretrained(source, **({} if Path(source).is_dir() else hf_model_options("depth_anything_3")))
    return model.to(device=torch.device(device)), source


def main():
    args = build_parser(__doc__).parse_args()
    model, source = load_model(args.device)
    conversion_log: list = []

    def predict(entry, data_dir):
        path = rgb_path(data_dir, entry["frame_id"])
        pred = model.inference([path])
        depth = np.asarray(pred.depth, dtype=np.float32).squeeze()

        # The focal must be expressed on the grid the returned depth lives on.
        # DA3 resizes internally and the output resolution follows that resize,
        # so the intrinsics are rescaled to `depth.shape` rather than to any
        # assumed constant. Using the 800-wide or native 1600-wide focal here
        # instead misscales the result by that same ratio.
        K = pinhole_K_for_model(data_dir, entry["scene_id"], depth.shape)
        focal = float((K[0, 0] + K[1, 1]) / 2.0)

        # The documented conversion is applied unconditionally. An earlier
        # version chose between raw and converted output by whichever landed
        # closer to the benchmark's depth range -- that is fitting the units
        # convention to the test set, so the score would no longer measure the
        # model. The shapes and medians are logged instead, which is enough to
        # diagnose a units problem after the fact without biasing the number.
        converted = focal * depth / DA3_METRIC_DIVISOR

        finite = depth[np.isfinite(depth) & (depth > 0)]
        finite_conv = converted[np.isfinite(converted) & (converted > 0)]
        conversion_log.append({
            "frame_id": entry["frame_id"],
            "returned_shape": list(depth.shape),
            "focal_px_on_returned_grid": focal,
            "median_raw": float(np.median(finite)) if finite.size else 0.0,
            "median_converted": float(np.median(finite_conv)) if finite_conv.size else 0.0,
        })
        return converted

    run_loop("depth_anything_3", args, predict,
             notes={"output_space": "metric_depth", "expected_alignment": "none",
                    "weights": source,
                    "metric_conversion": "metric = focal * net_output / 300 (DA3 model card)",
                    "conversion_log": lambda: conversion_log})


if __name__ == "__main__":
    main()
