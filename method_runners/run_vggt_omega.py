"""Pose-free VGGT-Omega inference on six synchronized RGB views.

Obtain the checkpoint from https://huggingface.co/facebook/VGGT-Omega under
the upstream access terms. Pass its local file with `--checkpoint` or install
it under the configured weights directory. The evaluator fits one positive
scale to camera 2 depth per timestamp.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402
from run_vggt import autocast_dtype  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("vggt_omega")
NUM_CAMERAS = 6
ANCHOR_CAM = 2
IMAGE_RESOLUTION = 512

CHECKPOINT_CANDIDATES = [local_weight_path("vggt_omega")]


def find_checkpoint(explicit: str = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint not found: {p}")
        return p
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "VGGT-Omega checkpoint not found. The weights are gated on Hugging Face "
        "(facebook/VGGT-Omega returns 401 without approved access).\n"
        "Searched:\n  " + "\n  ".join(str(p) for p in CHECKPOINT_CANDIDATES) + "\n"
        "Request access, then pass --checkpoint /path/to/vggt_omega_1b_512.pt"
    )


def load_model(device, checkpoint: Path):
    sys.path.insert(0, str(THIRD_PARTY))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to vggt_omega_1b_512.pt (gated download)")
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.checkpoint)
    model = load_model(args.device, checkpoint)
    dtype = autocast_dtype(args.device)

    sys.path.insert(0, str(THIRD_PARTY))
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    def predict(entry, data_dir):
        import cv2

        paths = [rgb_path(data_dir, entry["frame_id"], cam) for cam in range(NUM_CAMERAS)]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} camera image(s) missing, e.g. {missing[0]}")

        native = cv2.imread(paths[ANCHOR_CAM])
        if native is None:
            raise FileNotFoundError(paths[ANCHOR_CAM])
        native_hw = native.shape[:2]

        # VGGT-Omega's loader takes mode="balanced"|"max_size", not VGGT's
        # "crop"|"pad". Either is safe for the fisheye periphery here: both
        # only centre-crop aspect ratios outside [0.5, 2.0], and ours is 1.23,
        # so no cropping happens. max_size (longest side -> image_resolution)
        # is the closer analogue of VGGT's pad mode, so the two Group C entries
        # see comparably framed input.
        images = load_and_preprocess_images(
            paths, mode="max_size", image_resolution=IMAGE_RESOLUTION
        ).to(args.device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if args.device != "cpu" else "cpu",
                                dtype=dtype, enabled=args.device != "cpu"):
                pred = model(images[None] if images.ndim == 4 else images)

        depth = pred["depth"]
        depth = depth[0, ANCHOR_CAM] if depth.ndim == 5 else depth[ANCHOR_CAM]
        out = depth.squeeze().float().cpu().numpy()

        # No unpadding here, unlike run_vggt.py. max_size resizes rather than
        # pads for our aspect ratio (1296x1600 -> 416x512, aspect 1.235 vs
        # 1.231), so the prediction already spans the whole frame and
        # save_depth's resize onto the GT grid is the only step needed.
        # Trimming would cut real content.
        expected = native_hw[1] / native_hw[0]
        got = out.shape[1] / out.shape[0]
        if abs(got - expected) > 0.05:
            raise RuntimeError(
                f"prediction aspect {got:.3f} does not match the frame's "
                f"{expected:.3f}; preprocessing cropped and the map would be "
                "misaligned with the GT"
            )
        return out

    run_loop("vggt_omega", args, predict,
             notes={"output_space": "depth_up_to_global_scale",
                    "expected_alignment": "positive_scale",
                    "checkpoint": str(checkpoint),
                    "image_resolution": IMAGE_RESOLUTION,
                    "n_input_views": NUM_CAMERAS, "anchor_camera": ANCHOR_CAM})


if __name__ == "__main__":
    main()
