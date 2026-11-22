"""UniK3D inference runner -- any-camera monocular metric depth (Group A).

UniK3D is one of the few models that takes a real non-pinhole camera, which is why
it is in the pool: without it the monocular group would be pinhole models only and
every gap could be blamed on fisheye projection.

THREE THINGS THAT ARE EASY TO GET WRONG HERE

1. OUTPUT CHANNEL. `out["depth"]` is `points[:, -1:]`, the Z component of the
   camera-frame pointmap -- optical-axis depth, matching our GT. `out["distance"]`
   is `points.norm()`, range along the ray, which reads long off-axis by 1/cos(theta)
   -- about 8x at this lens's rim. The output name is not evidence of the
   convention; this was checked against unik3d/models/unik3d.py.

2. CAMERA MODEL. Our camera is double-sphere. UniK3D ships EUCM, and DS with
   xi ~= 0 is algebraically identical to EUCM with beta = 1 -- the same
   equivalence run_unidepth_v2.py already relies on, verified numerically on this
   rig to 0.03 degrees of field angle. So the DS intrinsics go in as
   [fx, fy, cx, cy, alpha, beta=1].

3. THE CAMERA IS REBUILT PER FRAME. `infer()` rescales the camera in place to the
   working resolution, so a cached Camera is progressively corrupted -- the failure
   that once made UniDepth look like it collapsed to 6 mm depth. Construction is
   cheap next to a forward pass.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, gt_grid_intrinsics, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("unik3d")
# Largest released variant, to match the other Group A entries' capacity.
DEFAULT_CHECKPOINT = "lpiccinelli/unik3d-vitl"


def load_model(device: str, checkpoint: str):
    sys.path.insert(0, str(THIRD_PARTY))
    from unik3d.models import UniK3D

    model = UniK3D.from_pretrained(checkpoint, **({} if Path(checkpoint).exists() else hf_model_options("unik3d")))
    # infer() warns and falls back to default shape bounds without this.
    model.resolution_level = 9
    return model.to(device).eval()


def build_camera(data_dir: Path, scene_id: str, shape_hw: tuple, device: str):
    """UniK3D EUCM camera on the image grid, or None to let the model self-estimate."""
    intrinsics = gt_grid_intrinsics(data_dir, scene_id)
    if intrinsics is None:
        return None, "self_estimated"

    sys.path.insert(0, str(THIRD_PARTY))
    from unik3d.utils.camera import EUCM

    height, width = shape_hw
    sx, sy = width / intrinsics["width"], height / intrinsics["height"]
    params = torch.tensor(
        [[intrinsics["fx"] * sx, intrinsics["fy"] * sy,
          intrinsics["cx"] * sx, intrinsics["cy"] * sy,
          intrinsics["alpha"], 1.0]],
        dtype=torch.float32, device=device,
    )
    return EUCM(params=params), "eucm_from_depth_gt_info"


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    model = load_model(args.device, args.checkpoint)
    camera_modes: set = set()

    def predict(entry, data_dir):
        path = rgb_path(data_dir, entry["frame_id"])
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device)

        camera, mode = build_camera(
            data_dir, entry["scene_id"], rgb.shape[:2], args.device
        )
        camera_modes.add(mode)

        with torch.no_grad():
            prediction = model.infer(tensor, camera)

        # depth, NOT distance: Z along the optical axis.
        return prediction["depth"].squeeze().float().cpu().numpy()

    run_loop(
        "unik3d", args, predict,
        notes={
            "output_space": "metric_depth_z",
            "expected_alignment": "none",
            "checkpoint": args.checkpoint,
            "output_channel": "out['depth'] = points[:, -1:] (optical Z), not out['distance'] (range)",
            "camera_model": lambda: (
                f"{sorted(camera_modes)}; EUCM(beta=1) from DS intrinsics, which "
                "agree to 0.03 deg on this rig"
            ),
        },
    )


if __name__ == "__main__":
    main()
