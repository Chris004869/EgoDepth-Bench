#!/usr/bin/env python3
"""Check a local starter with the portable reader and CPU scoring functions.

This checks asset decoding, geometry and scoring conventions; it does not
measure a model or certify the accuracy of the reference depth.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sdk"), str(ROOT / "src")]

import numpy as np
from egodepth_dataset import EgoDepthDataset, unit_rays
from egodepth_eval.score.accumulate import accumulate_frame, prediction_domain, reference_domain


def check_starter(root):
    data = EgoDepthDataset(root, split="train")
    sites = set()
    reference_pixels = 0
    try:
        if len(data) != 12:
            raise ValueError(f"Expected the 12-anchor starter; found {len(data)} train anchors")
        for index in range(len(data)):
            sample = data[index]
            z, valid = sample["target_depth_z"], sample["target_valid"]
            if sample["split"] != "train" or z.shape != (416, 512) or z.dtype != np.float32:
                raise ValueError("Unexpected split, depth shape or depth dtype")
            if valid.shape != z.shape or valid.dtype != np.bool_ or not valid.any():
                raise ValueError("Expected a nonempty boolean reference-valid mask")
            if set(sample["images"]) != set(range(6)):
                raise ValueError("Expected all six synchronized RGB views")
            for camera, rgb in sample["images"].items():
                if rgb.shape != (416, 512, 3) or rgb.dtype != np.uint8:
                    raise ValueError(f"Unexpected RGB array for camera {camera}")
                rays = unit_rays(sample["calibrations"][str(camera)])
                lengths = np.linalg.norm(rays, axis=-1)
                if rays.shape != (416, 512, 3) or not np.isfinite(rays).all():
                    raise ValueError("Invalid ray field")
                if not np.allclose(lengths[lengths > 0], 1.0, atol=2e-6):
                    raise ValueError("Valid rays must have unit length")
            domain = reference_domain(z, valid)
            metrics = accumulate_frame(z, z, domain, prediction_domain(z, valid)).finalize()
            if metrics["abs_rel"] != 0 or metrics["mae_z_m"] != 0:
                raise ValueError("The identity-prediction scoring check failed")
            if metrics["coverage"] != 1 or metrics["a_5cm"] != 1:
                raise ValueError("The identity-prediction support check failed")
            sites.add(sample["site_id"])
            reference_pixels += int(valid.sum())
        if len(sites) != 4:
            raise ValueError(f"Expected four starter sites; found {len(sites)}")
        return {
            "status": "passed", "anchors": len(data), "training_sites": len(sites),
            "cameras_per_anchor": 6, "grid_hw": [416, 512],
            "reference_valid_pixels": reference_pixels,
            "depth": "camera-2 optical-axis Z in metres; bool True means reference-valid",
            "geometry": "six independent calibrated unit-ray fields checked",
            "identity_prediction": {"abs_rel": 0.0, "mae_z_m": 0.0, "coverage": 1.0, "a_5cm": 1.0},
            "scope": "reader and scoring smoke check; no model inference or benchmark claim",
        }
    finally:
        data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted egodepth-starter directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report outside the read-only data")
    args = parser.parse_args()
    report = check_starter(args.data)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
