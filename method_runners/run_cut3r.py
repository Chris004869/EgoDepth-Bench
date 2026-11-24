"""Pose-free CUT3R inference on six synchronized RGB views.

Start a fresh recurrent sequence for each timestamp. Process camera 2 last so
its prediction incorporates every preceding view. Within a sequence, preserve
the state after each update with `reset=False`. The Z channel of the camera
2 point map in `pts3d_in_self_view` gives optical-axis depth in the target
frame. The evaluator scores this output directly.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("cut3r")
WEIGHTS_CANDIDATES = [local_weight_path("cut3r")]
NUM_CAMERAS = 6
ANCHOR_CAM = 2
INPUT_SIZE = 512
# The target camera is fed LAST so the causal state has seen all six views by the
# time its prediction is produced.
VIEW_ORDER = [c for c in range(NUM_CAMERAS) if c != ANCHOR_CAM] + [ANCHOR_CAM]


def find_weights(explicit: str = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--weights not found: {p}")
        return p
    for p in WEIGHTS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "CUT3R checkpoint not found. The official release is on Google Drive, "
        "linked from the official CUT3R repository. Searched:\n  "
        + "\n  ".join(str(p) for p in WEIGHTS_CANDIDATES)
    )


def load_model(device, weights: Path):
    sys.path.insert(0, str(THIRD_PARTY))
    from add_ckpt_path import add_path_to_dust3r

    # CUT3R resolves its vendored dust3r package relative to the checkpoint.
    add_path_to_dust3r(str(weights))
    from src.dust3r.model import ARCroco3DStereo

    return ARCroco3DStereo.from_pretrained(str(weights)).to(device).eval()


def build_views(img_paths: list, size: int) -> list:
    """One view dict per image, matching demo.py's images-only construction."""
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []
    for i, im in enumerate(images):
        c, h, w = im["img"].shape[-3:]
        views.append({
            "img": im["img"],
            # NaN ray_map with ray_mask=False marks "no ray prior supplied".
            "ray_map": torch.full((im["img"].shape[0], 6, h, w), torch.nan),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            # reset is applied AFTER the state update, so reset=True on view 0
            # would throw away view 0's own contribution -- measured: perturbing
            # view 0 then changed the target output by exactly nothing. Per-frame
            # isolation does not need it; `_forward_impl` initializes the state
            # from this call's own first view, verified to reproduce a frame
            # bit-exactly across an intervening different frame.
            "reset": torch.tensor(False).unsqueeze(0),
        })
    return views


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    weights = find_weights(args.weights)
    model = load_model(args.device, weights)

    from src.dust3r.inference import inference

    def predict(entry, data_dir):
        paths = [rgb_path(data_dir, entry["frame_id"], c) for c in VIEW_ORDER]
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} camera image(s) missing, e.g. {missing[0]}")

        views = build_views(paths, INPUT_SIZE)
        with torch.no_grad():
            outputs, _state = inference(views, model, args.device)

        preds = outputs["pred"] if isinstance(outputs, dict) else outputs
        # The target camera is LAST in VIEW_ORDER, so its prediction is the last
        # one -- indexing by camera id here would silently read another camera.
        pred = preds[VIEW_ORDER.index(ANCHOR_CAM)]
        # Own-camera-frame pointmap; Z is depth along the optical axis.
        pts = pred["pts3d_in_self_view"].squeeze(0).float().cpu().numpy()
        return pts[..., 2]

    run_loop("cut3r", args, predict,
             notes={"output_space": "metric_depth_z",
                    "expected_alignment": "none",
                    "weights": str(weights),
                    "input_size": INPUT_SIZE,
                    "state_handling": "fresh recurrent sequence for each timestamp",
                    "n_input_views": NUM_CAMERAS, "anchor_camera": ANCHOR_CAM,
                    "view_order": VIEW_ORDER,
                    "target_view_position": VIEW_ORDER.index(ANCHOR_CAM),
                    "six_views_reach_target_output": "all six views update state before target readout"})


if __name__ == "__main__":
    main()
