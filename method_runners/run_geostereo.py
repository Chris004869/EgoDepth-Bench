"""GeoStereo inference runner -- diffusion-enhanced stereo (Group B).

GeoStereo (ACM MM 2026) extends FoundationStereo with two diffusion-based
conditioning branches (warped-view ControlNet + normal-prior ControlNet) that
inject geometric priors into a Stable Diffusion UNet. It predicts both
disparity and surface normals jointly.

For the benchmark, only the disparity output is used (converted to depth via
the same rectified-pinhole geometry as FoundationStereo). This makes the
comparison apples-to-apples: same stereo pair, same rectification, same
depth conversion --- the only difference is whether the diffusion geometric
prior improves the disparity.

The pipeline is relatively heavy (~6.7 GB of weights, ~3.5s/frame on 4090)
due to the diffusion UNet, so it runs slower than pure FoundationStereo.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from _common import GT_SHAPE, build_parser, gt_grid_intrinsics, load_calibration, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("geostereo")
WEIGHTS_DIR = local_weight_path("geostereo")

LEFT_CAM, RIGHT_CAM = 2, 3


def load_pipeline(device: str, weights_dir: Path):
    sys.path.insert(0, str(THIRD_PARTY))
    from GeoStereo.GeoStereo_pipeline import GeoStereoPipeline

    import json
    config = json.loads((weights_dir / "GeoStereo_config.json").read_text())

    pipeline = GeoStereoPipeline.from_pretrained(
        str(weights_dir),
        device=torch.device(device),
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, config


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--weights_dir", default=str(WEIGHTS_DIR))
    parser.add_argument("--rectify_fov", type=float, default=90.0)
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    if not (weights_dir / "GeoStereo_config.json").exists():
        raise SystemExit(f"GeoStereo weights not found at {weights_dir}. "
                         "Download from https://huggingface.co/qz-wei/GeoStereo")

    pipeline, config = load_pipeline(args.device, weights_dir)
    generator = torch.Generator(device=args.device).manual_seed(42)

    # Reuse FoundationStereo's rectification infrastructure
    import run_foundation_stereo as FS

    rect_cache: dict = {}
    RECT_SIZE = FS.RECT_SIZE

    def predict(entry, data_dir):
        import cv2

        scene_id = entry["scene_id"]
        left = cv2.imread(rgb_path(data_dir, entry["frame_id"], LEFT_CAM))
        right = cv2.imread(rgb_path(data_dir, entry["frame_id"], RIGHT_CAM))
        if left is None or right is None:
            raise FileNotFoundError("missing stereo image")

        key = (scene_id, left.shape[:2])
        if key not in rect_cache:
            intr = gt_grid_intrinsics(data_dir, scene_id)
            if intr is None:
                raise FileNotFoundError(f"no depth-grid intrinsics for {scene_id}")
            rect_cache[key] = (FS.build_rectification(
                load_calibration(data_dir, scene_id), intr, left.shape[:2], args.rectify_fov
            ), intr)
        (map_l, map_r, fx_rect, baseline, R_rect), intr = rect_cache[key]

        lr = cv2.remap(left, map_l[0], map_l[1], cv2.INTER_LINEAR)
        rr = cv2.remap(right, map_r[0], map_r[1], cv2.INTER_LINEAR)

        # Convert to tensors as GeoStereo expects [0,1] float tensors
        def to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(args.device)

        t_l, t_r = to_tensor(lr), to_tensor(rr)

        with torch.inference_mode():
            output = pipeline(
                image1=t_l,
                image2=t_r,
                generator=generator,
                output_type="pt",
            )

        # GeoStereo returns disparity in output.disparity [B, 1, H, W]
        disp = output.disparity.squeeze().float().cpu().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_rect = np.where(disp > 1e-3, fx_rect * baseline / disp, np.nan)

        return FS.rect_depth_to_gt_grid(depth_rect, intr, fx_rect, R_rect)

    run_loop("geostereo", args, predict,
             notes={"output_space": "metric_depth_z", "expected_alignment": "none",
                    "weights_dir": str(weights_dir),
                    "stereo_pair": [LEFT_CAM, RIGHT_CAM],
                    "rectify_fov_deg": args.rectify_fov,
                    "rectified_size": list(RECT_SIZE),
                    "architecture": "FoundationStereo + diffusion normal/warp ControlNets",
                    "venue": "ACM MM 2026"})


if __name__ == "__main__":
    main()
