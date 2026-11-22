"""DAC (Depth Any Camera) inference runner -- ERP-based metric monocular (Group A).

DAC is a SEPARATE MODEL from UniDAC, not an earlier name for it: iDisc-ERP with a
Swin-L encoder (CVPR 2025) versus UniDAC's DINOv3+DPT with a scale-estimation head
(CVPR 2026). They share the ERP reprojection idea and the `erp_geometry` helpers,
which is exactly why this runner was NOT written as an alias -- aliasing would put
one model's predictions in two table rows. The two checkpoints are different files
from different repositories, so both rows are real.

The pipeline mirrors run_unidac.py because the ERP geometry is upstream's, but four
things are specific to DAC:

1. THE MODEL TAKES NO ATTENTION MASK OR LATITUDE GRID. `IDiscERP.forward` is
   (image, lat_range, long_range) -- UniDAC's extra `attn_mask` / `lat_grid`
   arguments do not exist here, and `resize_for_input` in this repo returns five
   values, not seven.

2. OUTPUT IS EUCLIDEAN DISTANCE. Upstream states it outright: "Our depth maps
   record the Euclidean distance to the camera center instead of the z-distance to
   the image plane, which is not appropriate for large-FoV camera models." Our GT
   is optical-axis Z, so `z = range * u_z` is applied with the same rays used to
   build the ERP patch. Reporting the raw output as Z reads long off-axis by
   1/cos(theta) -- about 8x at this lens's rim.

3. `erp_h` IS SCALED BY `pred_scale_factor` ON THE WAY BACK. The demo does
   `erp_h = cano_sz[0] * pred_scale_factor` before `erp_patch_to_cam_fast`, because
   the prediction was made on a resized patch. Passing the unscaled `cano_sz`
   samples the reverse grid at the wrong span.

4. DOUBLE-SPHERE NEEDS BOTH DIRECTIONS PATCHED. Upstream supports OPENCV_FISHEYE,
   MEI and ERP; DS is none of them (MEI has no `alpha`). The forward direction gets
   a DS branch, and the reverse direction is switched to use the supplied ray table
   instead of upstream's dataset-name whitelist -- without that second patch a new
   camera silently falls back to a PINHOLE unprojection. Both are recorded in
   docs/third_party_patches.md.

STATUS: implemented against the upstream demo path; NOT validated against a DAC
reference output on a double-sphere camera. The runner records that in metadata so
a score from it is not mistaken for a verified one.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_parser, gt_grid_intrinsics, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("dac")
DEFAULT_CHECKPOINT = local_weight_path("dac")
# Indoor model: this benchmark is indoor near-field manipulation. The config that
# ships beside the checkpoint carries the ERP canonical size it was trained on.
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_suffix(".json")
# Forward patch size. The repo's own fisheye test config (scannetpp) uses this.
FORWARD_SIZE = (500, 750)
CROP_WFOV_DEG = 180


def load_model(device: str, checkpoint: Path, config: dict):
    sys.path.insert(0, str(THIRD_PARTY))
    from dac.models.idisc_erp import IDiscERP

    name = config["model_name"]
    if name != "IDiscERP":
        raise SystemExit(
            f"this runner drives IDiscERP (the ERP variant); config asks for {name}. "
            "The non-ERP variants take no lat/long range and would be a different row."
        )
    model = IDiscERP.build(config)
    model.load_pretrained(str(checkpoint))
    return model.to(device).eval()


def ds_ray_grid(data_dir: Path, scene_id: str, shape_hw: tuple) -> np.ndarray:
    """(H, W, 4) ray lookup table: unit ray xyz plus an invalid flag.

    The 4th channel is load-bearing: `erp_patch_to_cam_fast` reads
    `fisheye_grid2ray[:, :, 3]` and treats 1 as "no valid ray here", using it to
    build `mask_active`. Zeros there would claim every corner of a 200-degree lens
    unprojects, including the pixels the DS model cannot invert.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from egodepth_eval.evaluate.geometry import DSCamera

    intrinsics = gt_grid_intrinsics(data_dir, scene_id)
    if intrinsics is None:
        raise FileNotFoundError(
            f"no depth-grid intrinsics for scene {scene_id}; DAC needs the real "
            "camera to build its ray table"
        )
    height, width = shape_hw
    sx, sy = width / intrinsics["width"], height / intrinsics["height"]
    camera = DSCamera(
        fx=intrinsics["fx"] * sx, fy=intrinsics["fy"] * sy,
        cx=intrinsics["cx"] * sx, cy=intrinsics["cy"] * sy,
        xi=intrinsics["xi"], alpha=intrinsics["alpha"],
        width=width, height=height,
    )
    vs, us = np.mgrid[:height, :width].astype(np.float64)
    x, y, z = camera.unproject(us, vs)
    invalid = (np.sqrt(x * x + y * y + z * z) <= 1e-6).astype(np.float64)
    return np.stack([x, y, z, invalid], axis=-1).astype(np.float32)


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model_config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    checkpoint, config_path = Path(args.checkpoint), Path(args.model_config)
    if not checkpoint.exists():
        raise SystemExit(
            f"DAC checkpoint not found: {checkpoint}. Download it with\n"
            "  huggingface-cli download yuliangguo/depth-any-camera "
            f"dac_swinl_indoor.pt --local-dir {DEFAULT_CHECKPOINT.parent}"
        )
    if not config_path.exists():
        raise SystemExit(f"DAC config not found: {config_path}")

    config = json.loads(config_path.read_text())
    model = load_model(args.device, checkpoint, config)
    cano_sz = config["cano_sz"]

    sys.path.insert(0, str(THIRD_PARTY))
    import torchvision.transforms.functional as TF
    from dac.dataloders.dataset import resize_for_input
    from dac.utils.erp_geometry import cam_to_erp_patch_fast, erp_patch_to_cam_fast

    ray_cache: dict = {}
    intrinsics_cache: dict = {}

    def predict(entry, data_dir):
        scene_id = entry["scene_id"]
        path = rgb_path(data_dir, entry["frame_id"])
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        key = (scene_id, height, width)
        if key not in ray_cache:
            ray_cache[key] = ds_ray_grid(data_dir, scene_id, (height, width))
        rays = ray_cache[key]
        if scene_id not in intrinsics_cache:
            resolved = gt_grid_intrinsics(data_dir, scene_id)
            if resolved is None:
                raise FileNotFoundError(f"no depth-grid intrinsics for {scene_id}")
            intrinsics_cache[scene_id] = resolved
        intrinsics = intrinsics_cache[scene_id]

        sx, sy = width / intrinsics["width"], height / intrinsics["height"]
        camera_params = {
            "dataset": "egodepth_336l",
            "camera_model": "DS",
            "fx": intrinsics["fx"] * sx, "fy": intrinsics["fy"] * sy,
            "cx": intrinsics["cx"] * sx, "cy": intrinsics["cy"] * sy,
            "xi": intrinsics["xi"], "alpha": intrinsics["alpha"],
            "fisheye_grid": rays,
        }

        # No GT is supplied: zero depth, zero valid mask. Nothing derived from the
        # reference may enter the input.
        depth_placeholder = np.zeros((height, width, 1), dtype=np.float32)
        mask_placeholder = np.zeros((height, width, 1), dtype=np.float32)

        crop_width = int(cano_sz[0] * CROP_WFOV_DEG / 180)
        crop_height = int(crop_width * FORWARD_SIZE[0] / FORWARD_SIZE[1])

        erp_image, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image.astype(np.float32) / 255.0, depth_placeholder, mask_placeholder,
            0.0, np.array(0.0, dtype=np.float32),
            crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
            camera_params, np.array(0.0, dtype=np.float32), scale_fac=None,
        )
        # This repo's resize_for_input returns FIVE values with a mask (UniDAC's
        # returns seven; it also hands back lat/long grids this model does not take).
        resized_image, _resized_depth, _pad, pred_scale_factor, attention_mask = (
            resize_for_input(
                (erp_image * 255.0).astype(np.uint8), erp_depth, FORWARD_SIZE, None,
                [erp_image.shape[0], erp_image.shape[1]], 1.0,
                padding_rgb=[0, 0, 0], mask=erp_mask,
            )
        )

        normalized = TF.normalize(
            TF.to_tensor(resized_image),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ).unsqueeze(0).to(args.device)
        latitude_range = torch.tensor(
            [[float(np.min(latitude)), float(np.max(latitude))]],
            dtype=torch.float32, device=args.device,
        )
        longitude_range = torch.tensor(
            [[float(np.min(longitude)), float(np.max(longitude))]],
            dtype=torch.float32, device=args.device,
        )

        with torch.no_grad():
            predictions, _, _ = model(normalized, latitude_range, longitude_range)
        # The resize factor the demo applies; omitting it scales depth by the resize
        # ratio while the map still looks plausible.
        predictions = predictions * pred_scale_factor

        # Back to the camera. erp_h carries the same factor -- see the demo.
        erp_prediction = predictions[0].detach().cpu()
        attention = torch.from_numpy(
            (attention_mask > 0).astype(np.float32)
        ).unsqueeze(0)
        erp_h = cano_sz[0] * pred_scale_factor
        _image_out, camera_distance, _valid, mask_active = erp_patch_to_cam_fast(
            torch.from_numpy(
                (erp_image * 255.0).astype(np.float32).transpose(2, 0, 1)
            ),
            erp_prediction, attention,
            0.0, 0.0, out_h=height, out_w=width,
            erp_h=erp_h, erp_w=erp_h * 2,
            cam_params=camera_params, fisheye_grid2ray=rays,
        )
        distance = np.asarray(
            camera_distance.squeeze().float().numpy(), dtype=np.float32
        )
        active = np.asarray(
            mask_active.squeeze().float().numpy(), dtype=np.float32
        ) > 0.5

        # z = range * u_z with the same rays. Pixels the ERP round trip could not
        # reach become NaN, not zero: zero would score as a 0 m prediction, while
        # NaN correctly leaves them out of the prediction support.
        u_z = rays[..., 2]
        forward = (rays[..., 3] < 0.5) & (u_z > 0) & active
        with np.errstate(invalid="ignore"):
            depth_z = np.where(forward, distance * u_z, np.nan)
        return depth_z.astype(np.float32)

    run_loop(
        "dac", args, predict,
        notes={
            "output_space": "metric_depth_z",
            "expected_alignment": "none",
            "checkpoint": str(checkpoint),
            "model_config": str(config_path),
            "model_name": config["model_name"],
            "distinct_from_unidac": (
                "separate model and checkpoint: iDisc-ERP + Swin-L (CVPR 2025) vs "
                "UniDAC's DINOv3+DPT with a scale head (CVPR 2026); shares only the "
                "upstream ERP helpers, so this is not an alias of that row"
            ),
            "camera_handling": (
                "forward (camera->ERP) uses a double-sphere branch added to "
                "dac/utils/erp_geometry.py; reverse (ERP->camera) uses the DS ray "
                "table, enabled by a second patch -- upstream gates that branch on a "
                "dataset-name whitelist, so a new camera would silently be "
                "unprojected as a pinhole"
            ),
            "third_party_patches": [
                "dac/utils/erp_geometry.py: added camera_model='DS' forward projection",
                "dac/utils/erp_geometry.py: reverse branch keys on fisheye_grid2ray",
            ],
            "range_to_z": (
                "upstream states the output is Euclidean distance to the camera "
                "centre, not z; z = range * u_z applied"
            ),
            "pred_scale_factor_applied": True,
            "erp_h_scaled_by_pred_scale_factor": True,
            "validation_status": (
                "implemented against the upstream demo path; NOT validated against a "
                "DAC reference output on a double-sphere camera"
            ),
        },
    )


if __name__ == "__main__":
    main()
