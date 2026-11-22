"""UniDepth v2 inference runner -- monocular metric depth (Group A).

UniDepth v2 is the one evaluated model with native support for non-pinhole
camera models, so it is given the *real* fisheye camera rather than a pinhole
approximation -- that capability is precisely what the paper's Group A
comparison is testing, and handing it a 3x3 K would silently discard it.

Our camera is double-sphere with xi ~= -0.0005, and DS with xi = 0 is
algebraically identical to EUCM with beta = 1. Verified numerically on this
rig's intrinsics: field angles agree to 0.03 degrees worst-case (~0.02 mean),
which is orders of magnitude below the depth errors being measured. So the DS
intrinsics are passed through UniDepth's EUCM camera as
[fx, fy, cx, cy, alpha, beta=1].

Intrinsics come from depth_gt_info (the 800x648 GT grid). The image is fed at
native 1600x1296, so fx/fy/cx/cy are scaled up by 2 to match; alpha and beta
are dimensionless and must not be scaled.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, gt_grid_intrinsics, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("unidepth_v2")


def load_model(device):
    sys.path.insert(0, str(THIRD_PARTY))
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", **hf_model_options("unidepth_v2"))
    # infer() warns and falls back to default bounds if this is unset.
    model.resolution_level = 9
    return model.to(device).eval()


def build_camera(data_dir: Path, scene_id: str, shape_hw: tuple, device: str):
    """UniDepth EUCM camera on the image grid, or None to let it self-estimate."""
    intr = gt_grid_intrinsics(data_dir, scene_id)
    if intr is None:
        return None, "self_estimated"

    sys.path.insert(0, str(THIRD_PARTY))
    from unidepth.utils.camera import EUCM

    h, w = shape_hw
    sx, sy = w / intr["width"], h / intr["height"]
    params = torch.tensor(
        [[intr["fx"] * sx, intr["fy"] * sy, intr["cx"] * sx, intr["cy"] * sy,
          intr["alpha"], 1.0]],
        dtype=torch.float32, device=device,
    )
    return EUCM(params=params), "eucm_from_depth_gt_info"


def main():
    args = build_parser(__doc__).parse_args()
    model = load_model(args.device)

    def predict(entry, data_dir):
        path = rgb_path(data_dir, entry["frame_id"])
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device)

        # The camera object is rebuilt for every frame and deliberately NOT
        # cached. UniDepth's infer() mutates the Camera in place (it rescales
        # the intrinsics to the working resolution), so a reused instance is
        # progressively corrupted: the first frame of a cached run scores
        # AbsRel 0.11 and later frames collapse toward a median depth of
        # 0.006 m, which reads as catastrophic model failure rather than a
        # caching bug. Construction is cheap relative to a forward pass.
        camera, _ = build_camera(data_dir, entry["scene_id"], rgb.shape[:2], args.device)

        with torch.no_grad():
            pred = model.infer(tensor, camera)
        return pred["depth"].squeeze().float().cpu().numpy()

    run_loop("unidepth_v2", args, predict,
             notes={"output_space": "metric_depth", "expected_alignment": "none",
                    "camera_model": "EUCM(beta=1) from DS intrinsics; "
                                    "DS<->EUCM agree to 0.03 deg on this rig"})


if __name__ == "__main__":
    main()
