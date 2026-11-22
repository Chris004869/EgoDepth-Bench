"""Metric3D v2 inference runner -- monocular metric depth (Group A).

Metric3D predicts depth in a *canonical camera space* and requires an explicit
de-canonicalization to become metric:

    metric_depth = canonical_depth * (fx / 1000)

where 1000 px is the canonical camera's focal length. Omitting that step leaves
depth wrong by fx/1000 (~0.5x here), which looks like a plausible model failure
rather than a missing transform, so it is done explicitly and recorded.

The preprocessing (keep-ratio resize to fit 616x1064, then symmetric mean-value
padding, and the matching unpad after inference) follows the reference example
in the repo's hubconf.py. It matters for this benchmark specifically: the pad
must be removed before the map is compared against GT, or every pixel is offset
and the periphery is scored against the wrong ground truth.

Note the intrinsics scale with the resize but the depth does not -- the
canonical transform uses the *resized* fx, which is why the scale factor is
applied to the intrinsics before use.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, pinhole_K_for_model, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("metric3d_v2")

INPUT_SIZE = (616, 1064)      # (H, W) for the ViT models
PAD_MEAN = [123.675, 116.28, 103.53]
CANONICAL_FOCAL = 1000.0


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    # torch.hub is the interface the repo documents; the local checkout
    # supplies the model code via hubconf.
    model = torch.hub.load(str(THIRD_PARTY), "metric3d_vit_large",
                           pretrain=True, source="local")
    return model.to(device).eval()


def preprocess(rgb: np.ndarray, fx: float, fy: float):
    """Keep-ratio resize + pad to INPUT_SIZE. Returns tensor, pad info, scale."""
    h, w = rgb.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    rh, rw = resized.shape[:2]
    pad_h, pad_w = INPUT_SIZE[0] - rh, INPUT_SIZE[1] - rw
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=PAD_MEAN)
    pad_info = [top, pad_h - top, left, pad_w - left]

    mean = torch.tensor(PAD_MEAN).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    tensor = ((tensor - mean) / std)[None]
    return tensor, pad_info, scale


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)
    scale_log: list = []

    def predict(entry, data_dir):
        path = rgb_path(data_dir, entry["frame_id"])
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        native_hw = rgb.shape[:2]

        K = pinhole_K_for_model(data_dir, entry["scene_id"], native_hw)
        fx, fy = float(K[0, 0]), float(K[1, 1])

        tensor, pad_info, resize_scale = preprocess(rgb, fx, fy)
        with torch.no_grad():
            pred_depth, _conf, _out = model.inference({"input": tensor.to(args.device)})

        depth = pred_depth.squeeze()
        # Undo the padding before anything else touches the map.
        depth = depth[pad_info[0]: depth.shape[0] - pad_info[1],
                      pad_info[2]: depth.shape[1] - pad_info[3]]
        depth = torch.nn.functional.interpolate(
            depth[None, None], native_hw, mode="bilinear", align_corners=False
        ).squeeze()

        # De-canonicalize with the *resized* focal length, matching the resize
        # the intrinsics underwent.
        canonical_to_real = (fx * resize_scale) / CANONICAL_FOCAL
        depth = torch.clamp(depth * canonical_to_real, 0, 300)

        out = depth.float().cpu().numpy()
        scale_log.append({"frame_id": entry["frame_id"],
                          "canonical_to_real_scale": canonical_to_real,
                          "median_depth_m": float(np.median(out[out > 0])) if (out > 0).any() else 0.0})
        return out

    run_loop("metric3d_v2", args, predict,
             notes={"output_space": "metric_depth", "expected_alignment": "none",
                    "canonical_transform": "depth * (fx_resized / 1000)",
                    "scale_log": lambda: scale_log})


if __name__ == "__main__":
    main()
