"""VGGT inference runner -- feed-forward multi-view, pose-free (Group C).

All six ego cameras go in; camera 2's depth comes out. VGGT regresses camera
parameters itself, so no extrinsics are supplied (that is the point of the
pose-free half of Group C).

TWO THINGS THAT WOULD SILENTLY CORRUPT THIS

1. Preprocessing. `load_and_preprocess_images` defaults to mode="crop", which
   fixes width at 518 px and *centre-crops the height*. Our frames are
   1600x1296 (aspect 1.23), so cropping would discard the top and bottom of the
   fisheye circle -- precisely the periphery the paper's radial analysis is
   about -- and the surviving map would then be scored against GT pixels the
   model never saw. mode="pad" is used instead, and the padding is removed
   afterwards so the returned map covers the full frame.

2. Depth convention. VGGT's depth head predicts depth along the optical axis in
   its own normalized frame, and the whole prediction is metric only up to an
   unknown global scale (a pose-free method cannot recover absolute scale from
   images alone). The paper evaluates Group C without alignment, so this runner
   saves the raw prediction: whether VGGT's implied scale is right is a finding,
   not something to correct away.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GT_SHAPE, build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("vggt")
NUM_CAMERAS = 6
ANCHOR_CAM = 2


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    from vggt.models.vggt import VGGT

    return VGGT.from_pretrained("facebook/VGGT-1B", **hf_model_options("vggt")).to(device).eval()


def autocast_dtype(device: str):
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def unpad_to_frame(depth: np.ndarray, native_hw: tuple) -> np.ndarray:
    """Strip the square padding that mode="pad" added, restoring the frame.

    mode="pad" scales the longer side to 518 and pads the shorter side to a
    square. For a landscape frame the padding is vertical and symmetric, so the
    content occupies a centred horizontal band.
    """
    h, w = depth.shape[:2]
    nh, nw = native_hw
    if nw >= nh:
        content_h = int(round(h * (nh / nw) / 14) * 14)
        content_h = min(content_h, h)
        top = (h - content_h) // 2
        return depth[top: top + content_h, :]
    content_w = int(round(w * (nw / nh) / 14) * 14)
    content_w = min(content_w, w)
    left = (w - content_w) // 2
    return depth[:, left: left + content_w]


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)
    dtype = autocast_dtype(args.device)

    sys.path.insert(0, str(THIRD_PARTY))
    from vggt.utils.load_fn import load_and_preprocess_images

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

        # mode="pad" preserves every pixel; mode="crop" would not.
        images = load_and_preprocess_images(paths, mode="pad").to(args.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if args.device != "cpu" else "cpu",
                                dtype=dtype, enabled=args.device != "cpu"):
                pred = model(images[None])

        depth = pred["depth"][0, ANCHOR_CAM].squeeze().float().cpu().numpy()
        return unpad_to_frame(depth, native_hw)

    run_loop("vggt", args, predict,
             notes={"output_space": "depth_up_to_global_scale",
                    "expected_alignment": "none (Group C scored unaligned by design)",
                    "preprocess_mode": "pad (not crop) so the fisheye periphery survives",
                    "n_input_views": NUM_CAMERAS, "anchor_camera": ANCHOR_CAM})


if __name__ == "__main__":
    main()
