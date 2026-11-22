"""Depth Anything v2 inference runner -- affine-invariant (Group D).

The head emits *inverse* depth (relative disparity), not depth: `infer_image`
returns larger values for nearer surfaces. It is saved as-is, and the evaluator
converts it with `alignment: affine_inverse` (least-squares scale and shift in
disparity space, the MiDaS/DA protocol). Inverting here instead would throw
away the shift degree of freedom the model was trained with and make the method
look far worse than it is.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("depth_anything_v2")


def load_model(device):
    import torch
    sys.path.insert(0, str(THIRD_PARTY))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
    )
    ckpt_path = local_weight_path("depth_anything_v2")
    if not ckpt_path.exists():
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            "depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth", **hf_model_options("depth_anything_v2")
        )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)

    def predict(entry, data_dir):
        img = cv2.imread(rgb_path(data_dir, entry["frame_id"]))
        if img is None:
            raise FileNotFoundError(rgb_path(data_dir, entry["frame_id"]))
        # infer_image resizes internally and interpolates back to the input
        # resolution, so the full frame is preserved -- no crop to undo.
        # Official infer_image takes BGR and converts to RGB internally.
        return model.infer_image(img)

    run_loop("depth_anything_v2", args, predict,
             notes={"output_space": "inverse_depth", "expected_alignment": "affine_inverse"})


if __name__ == "__main__":
    main()
