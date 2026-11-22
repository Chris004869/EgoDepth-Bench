"""MASt3R inference runner -- two-view metric 3D reconstruction (Group C).

MASt3R is a *pairwise* model, so the ego rig's six views cannot all go in at
once as they can for VGGT. Camera 2 is paired with camera 3 (the stereo pair,
59.4 mm baseline), and the depth for camera 2's own view is read from the
prediction in its own frame. Using the metric checkpoint
(`..._catmlpdpt_metric`) is what makes this comparable to the rest of Group C
without alignment.

Two conventions worth stating:

1. `load_images(..., size=512)` resizes and centre-crops to the model's
   expected input, which loses the fisheye periphery. The crop is undone by
   mapping the prediction back through the recorded `true_shape`, so what gets
   saved covers the frame that was actually fed in. Pixels the model never saw
   stay NaN rather than being extrapolated.

2. `pts3d` is a pointmap in camera 2's frame, so depth is its Z channel --
   the same convention as the GT. Taking the norm instead would give range
   along the ray, which reads long off-axis.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("mast3r")
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
LEFT_CAM, RIGHT_CAM = 2, 3
INPUT_SIZE = 512


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    sys.path.insert(0, str(THIRD_PARTY / "dust3r"))
    sys.path.insert(0, str(THIRD_PARTY / "dust3r" / "croco"))
    from mast3r.model import AsymmetricMASt3R

    return AsymmetricMASt3R.from_pretrained(MODEL_NAME, **hf_model_options("mast3r")).to(device).eval()


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)

    sys.path.insert(0, str(THIRD_PARTY))
    sys.path.insert(0, str(THIRD_PARTY / "dust3r"))
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    def predict(entry, data_dir):
        lp = rgb_path(data_dir, entry["frame_id"], LEFT_CAM)
        rp = rgb_path(data_dir, entry["frame_id"], RIGHT_CAM)
        for p in (lp, rp):
            if not Path(p).exists():
                raise FileNotFoundError(p)

        images = load_images([lp, rp], size=INPUT_SIZE, verbose=False)
        out = inference([tuple(images)], model, args.device, batch_size=1, verbose=False)

        # pred1 is the first view (camera 2) in its own camera frame.
        pts3d = out["pred1"]["pts3d"].squeeze(0).detach().float().cpu().numpy()
        # Z channel, not the norm: matches the GT's along-optical-axis depth.
        return pts3d[..., 2]

    run_loop("mast3r", args, predict,
             notes={"output_space": "metric_depth_z",
                    "expected_alignment": "none (metric checkpoint)",
                    "model": MODEL_NAME,
                    "view_pair": [LEFT_CAM, RIGHT_CAM],
                    "input_size": INPUT_SIZE,
                    "coverage_note": ("pairwise model: 2 of 6 views used; input is "
                                      "resized/cropped to 512 so periphery coverage "
                                      "is lower than the monocular methods")})


if __name__ == "__main__":
    main()
