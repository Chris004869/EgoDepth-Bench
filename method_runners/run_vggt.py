"""Pose-free VGGT inference on six synchronized RGB views.

Pad each image for inference and remove the padding from the target prediction
to preserve the native field of view. The runner saves camera 2 optical-axis
depth. The evaluator fits one positive scale per timestamp.
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
                    "expected_alignment": "positive_scale",
                    "preprocess_mode": "pad (not crop) so the fisheye periphery survives",
                    "n_input_views": NUM_CAMERAS, "anchor_camera": ANCHOR_CAM})


if __name__ == "__main__":
    main()
