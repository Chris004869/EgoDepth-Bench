"""MoGe inference runner -- affine-invariant monocular geometry (Group D).

Uses MoGe-1 (`Ruicheng/moge-vitl`), which is the affine-invariant model the
paper positions in Group D. MoGe-2 exists and predicts metric scale, but
swapping it in would move the method into Group A and change what the
comparison means, so the version is pinned here deliberately.

The Z channel of the recovered point map is depth-like up to one unknown
global scale, hence `alignment: positive_scale` rather than the affine fit the
inverse-depth models need. Masked (unreliable) pixels come back as `inf`,
which is preserved: the evaluator drops non-finite predictions, so those
pixels are excluded rather than being scored against a fabricated value.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("moge")


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    from moge.model.v1 import MoGeModel

    return MoGeModel.from_pretrained("Ruicheng/moge-vitl", **hf_model_options("moge")).to(device).eval()


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)

    def predict(entry, data_dir):
        import cv2

        bgr = cv2.imread(rgb_path(data_dir, entry["frame_id"]))
        if bgr is None:
            raise FileNotFoundError(rgb_path(data_dir, entry["frame_id"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(args.device)

        with torch.no_grad():
            # Output resolution equals input resolution regardless of
            # resolution_level, so the full frame is preserved.
            out = model.infer(tensor, resolution_level=9, apply_mask=True)

        depth = out["depth"].squeeze().float().cpu().numpy()
        # inf marks masked pixels; keep it so the evaluator excludes them.
        return np.where(np.isfinite(depth), depth, np.inf).astype(np.float32)

    run_loop("moge", args, predict,
             notes={"model": "Ruicheng/moge-vitl (MoGe-1, affine-invariant)",
                    "output_space": "pointmap_z_up_to_scale",
                    "expected_alignment": "positive_scale"})


if __name__ == "__main__":
    main()
