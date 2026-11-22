"""Depth Pro inference runner -- monocular metric depth (Group A).

Depth Pro estimates focal length itself and needs no intrinsics, which is the
property the paper cites. `f_px` is therefore left unset: supplying a focal
length would override the model's own estimate, and no single focal length is
correct for a fisheye anyway. The focal length the model settles on is recorded
per frame in metadata.json, since a wildly wrong estimate is the most likely
explanation for a bad metric score here and that should be visible rather than
inferred.

`infer` resizes to 1536x1536 internally and resamples the result back to the
input resolution, so the full frame survives; there is no crop to undo.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("depth_pro")


CHECKPOINT = local_weight_path("depth_pro")


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY / "src"))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    # The packaged default is the relative path "./checkpoints/depth_pro.pt",
    # which only resolves if the process happens to be launched from the repo
    # root. Pin it absolutely so the runner works from any cwd.
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    if not CHECKPOINT.exists():
        from huggingface_hub import hf_hub_download

        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(hf_hub_download("apple/DepthPro", "depth_pro.pt", **hf_model_options("depth_pro")), CHECKPOINT)
    config.checkpoint_uri = str(CHECKPOINT)

    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=torch.device(device), precision=torch.half
    )
    return model.eval(), transform


def main():
    args = build_parser(__doc__).parse_args()
    model, transform = load_model(args.device)
    focals: list = []

    def predict(entry, data_dir):
        import depth_pro

        path = rgb_path(data_dir, entry["frame_id"])
        image, _, f_px = depth_pro.load_rgb(path)
        with torch.no_grad():
            # f_px from the file's EXIF is deliberately not forwarded: these
            # frames carry none, and the point is to use the model's estimate.
            out = model.infer(transform(image))

        focals.append({"frame_id": entry["frame_id"],
                       "focallength_px": float(out["focallength_px"])})
        return out["depth"].squeeze().float().cpu().numpy()

    run_loop("depth_pro", args, predict,
             notes={"output_space": "metric_depth", "expected_alignment": "none",
                    "intrinsics": "self_estimated (f_px not supplied)",
                    # Deferred: resolved after the loop, once populated.
                    "estimated_focals": lambda: focals})


if __name__ == "__main__":
    main()
