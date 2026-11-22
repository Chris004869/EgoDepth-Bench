"""Marigold inference runner -- diffusion-based affine-invariant depth (Group D).

The pipeline returns affine-invariant *inverse* depth normalized to [0, 1]. As
with DA v2 it is saved raw and converted by the evaluator's `affine_inverse`
alignment; see run_depth_anything_v2.py for why.

`ensemble_size` and `num_inference_steps` are the accuracy/time knobs. The
defaults here follow the Marigold-LCM card (4 steps, ensemble 5), which is the
configuration the authors report for fast inference.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop, hf_model_options  # noqa: E402


def load_model(device):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0",
        custom_pipeline="marigold_depth_estimation",
        torch_dtype=torch.float16,
        variant="fp16",
        **hf_model_options("marigold"),
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    args = build_parser(__doc__).parse_args()
    pipe = load_model(args.device)

    def predict(entry, data_dir):
        from PIL import Image

        img = Image.open(rgb_path(data_dir, entry["frame_id"])).convert("RGB")
        out = pipe(img, denoising_steps=4, ensemble_size=5,
                   processing_res=768, match_input_res=True,
                   batch_size=0, show_progress_bar=False)
        # Attribute name differs across diffusers/Marigold versions.
        for attr in ("depth_np", "prediction", "depth"):
            d = getattr(out, attr, None)
            if d is not None:
                return np.asarray(d, dtype=np.float32).squeeze()
        raise RuntimeError(f"no depth field on Marigold output: {dir(out)}")

    run_loop("marigold", args, predict,
             notes={"output_space": "inverse_depth_normalized",
                    "expected_alignment": "affine_inverse",
                    "denoising_steps": 4, "ensemble_size": 5})


if __name__ == "__main__":
    main()
