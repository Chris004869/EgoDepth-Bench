"""ZoeDepth inference runner -- monocular metric depth (Group A).

Included as the 2023 temporal baseline that quantifies how much the field moved
by 2025-26, so the indoor-trained variant is the right choice: ZoeD_N is the
NYU-finetuned model, matching the indoor tabletop domain. ZoeD_NK (joint
indoor/outdoor) would confound "two years of progress" with "wrong domain
head".

`infer_pil` returns metric depth in metres at the input resolution, so there is
no crop, pad, or canonical transform to undo -- unusual among these repos and
worth stating explicitly.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("zoedepth")


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    model = torch.hub.load(str(THIRD_PARTY), "ZoeD_N", source="local", pretrained=True)
    return model.to(device).eval()


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)

    def predict(entry, data_dir):
        from PIL import Image

        img = Image.open(rgb_path(data_dir, entry["frame_id"])).convert("RGB")
        with torch.no_grad():
            return np.asarray(model.infer_pil(img), dtype=np.float32)

    run_loop("zoedepth", args, predict,
             notes={"output_space": "metric_depth", "expected_alignment": "none",
                    "variant": "ZoeD_N (NYU indoor head)"})


if __name__ == "__main__":
    main()
