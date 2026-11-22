"""UniDAC inference runner -- any-camera monocular metric depth via ERP (Group A).

UniDAC does not predict in the input camera's frame. It converts the image to an
equirectangular (ERP) patch, predicts there, and the prediction must be projected
back to the camera. The upstream demo does this with `cam_to_erp_patch_fast` /
`erp_patch_to_cam_fast`, and this runner uses the same helpers so the geometry is
theirs rather than a reimplementation.

FOUR THINGS THAT DECIDE WHETHER THIS IS CORRECT

1. CAMERA MODEL. UniDAC's ERP conversion supports MEI, OPENCV_FISHEYE, ERP and a
   `fisheye_grid` ray lookup table. Our camera is double-sphere, which is none of
   those -- and DS is NOT the same as MEI (MEI has no `alpha`). So instead of
   claiming a model it does not have, the runner supplies the ray lookup table:
   the DS unit rays for this camera, computed by our own `DSCamera`, handed over as
   `fisheye_grid`. That is the sanctioned path for "a model with no closed-form
   inversion" and it carries the real geometry.

2. OUTPUT IS EUCLIDEAN DISTANCE, NOT Z. The upstream pipeline converts z-buffer
   depth to Euclidean distance on the way in (`depth / fisheye_grid_z`), so the
   prediction that comes back is range along the ray. Our GT is optical-axis Z, so
   the output is multiplied by `u_z` -- exactly the `z = range * u_z` relation, using
   the same rays. Reporting the raw output as Z would read long off-axis by up to
   8x at this lens's rim.

3. `pred_scale_factor`. `resize_for_input` returns a scale factor that the demo
   applies to the prediction (`preds *= pred_scale_factor`). Skipping it makes the
   depth wrong by the resize ratio while still looking plausible.

4. INDOOR CONFIG. The benchmark is indoor near-field, so the indoor config and
   `cano_sz` are used. The config choice is recorded, because it sets the ERP
   canonical size and the forward resolution.

STATUS: the geometry path is implemented against the upstream helpers, but this
runner has not been validated against a UniDAC reference output on a DS camera. It
reports its own status in metadata so a score from it is not mistaken for a
verified one.
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

THIRD_PARTY = source_path("unidac")
DEFAULT_CHECKPOINT = local_weight_path("unidac")
# Indoor config: this benchmark is indoor near-field manipulation.
DEFAULT_CONFIG = "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json"
# Field of view to carve out of the ERP, in degrees. The lens reaches ~200 degrees
# diagonally; the demo uses 180 plus buffer for a comparable fisheye.
CROP_WFOV_DEG = 180


def load_model(device: str, checkpoint: Path, config_path: Path):
    """Build UniDAC and load its released weights.

    The shipped configs point `pixel_encoder.weights` at a DINOv3 pretrain file on
    the authors' own filesystem, which is unreachable and gated upstream. That file
    is only used to initialize the backbone before the UniDAC checkpoint overwrites
    it: the released `unidac.pt` contains all 368 `pixel_encoder.*` tensors
    (verified by listing the state dict). So the pretrain path is cleared and the
    fact is recorded, rather than the run failing on a weight file whose contents are
    about to be replaced.

    If a future checkpoint did NOT carry the backbone, `load_pretrained` uses
    `strict=False` and would silently leave it random -- so the encoder tensor count
    is checked here instead of trusted.
    """
    sys.path.insert(0, str(THIRD_PARTY))
    import torch as _torch
    from unidac.models.unidac import UniDAC

    config = json.loads(config_path.read_text())
    encoder_config = config.get("model", {}).get("pixel_encoder", {})
    declared_weights = encoder_config.get("weights")

    payload = _torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    encoder_keys = [k for k in state if "pixel_encoder" in k]
    if not encoder_keys:
        raise SystemExit(
            f"{checkpoint} carries no pixel_encoder tensors, and the config's DINOv3 "
            f"pretrain file ({declared_weights}) is unreachable. The backbone would "
            "be left randomly initialized, which load_pretrained(strict=False) would "
            "not report. Obtain the DINOv3 weights before running UniDAC."
        )

    backbone_source = f"config path {declared_weights!r}"
    if not declared_weights or not Path(declared_weights).exists():
        # The builder requires a real file (it calls Path(...).as_uri()), and passing
        # None raises. So the backbone weights are written out FROM the UniDAC
        # checkpoint, which carries all of them -- the init load is then a no-op that
        # `load_pretrained` immediately confirms with the same tensors.
        extracted = checkpoint.parent / "dinov3_backbone_from_unidac.pth"
        if not extracted.exists():
            prefix = "module.pixel_encoder."
            backbone = {
                k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
            }
            if not backbone:
                # Fall back to whatever prefix this checkpoint uses.
                marker = "pixel_encoder."
                backbone = {
                    k[k.index(marker) + len(marker):]: v
                    for k, v in state.items() if marker in k
                }
            extracted.parent.mkdir(parents=True, exist_ok=True)
            _torch.save(backbone, extracted)
        encoder_config["weights"] = str(extracted)
        backbone_source = (
            f"config pointed at unreachable {declared_weights!r} (gated upstream); "
            f"backbone extracted from {checkpoint.name} to {extracted.name}"
        )

    model = UniDAC.build(config)
    model.load_pretrained(str(checkpoint))
    return model.to(device).eval(), config, {
        "backbone_source": backbone_source,
        "n_encoder_tensors_in_checkpoint": len(encoder_keys),
    }


def ds_ray_grid(data_dir: Path, scene_id: str, shape_hw: tuple) -> np.ndarray:
    """(H, W, 4) ray lookup table: unit ray xyz plus an invalid flag.

    UniDAC accepts a ray lookup table for cameras it has no closed-form inverse for.
    Double-sphere is such a camera -- not MEI (no alpha), not OPENCV_FISHEYE -- so
    handing over the real rays is the upstream-supported route.

    The FOURTH channel is required: `erp_patch_to_cam_fast` reads
    `fisheye_grid2ray[:, :, 3]` and treats 1 as "this pixel has no valid ray", using
    it to build `mask_active`. Supplying only xyz raises an IndexError; supplying
    zeros there would silently claim every pixel of a 200-degree lens has a valid
    ray, including the corners the DS model cannot unproject.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from egodepth_eval.evaluate.geometry import DSCamera

    intrinsics = gt_grid_intrinsics(data_dir, scene_id)
    if intrinsics is None:
        raise FileNotFoundError(
            f"no depth-grid intrinsics for scene {scene_id}; UniDAC needs the real "
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
    # DSCamera.unproject returns exact zeros where the model has no valid ray.
    invalid = (np.sqrt(x * x + y * y + z * z) <= 1e-6).astype(np.float64)
    return np.stack([x, y, z, invalid], axis=-1).astype(np.float32)


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model_config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"UniDAC checkpoint not found: {checkpoint}. Download it with\n"
            "  huggingface-cli download girish1511/UniDAC unidac.pt "
            f"--local-dir {DEFAULT_CHECKPOINT.parent}"
        )
    config_path = THIRD_PARTY / args.model_config
    if not config_path.exists():
        raise SystemExit(f"UniDAC config not found: {config_path}")

    model, config, load_notes = load_model(args.device, checkpoint, config_path)

    sys.path.insert(0, str(THIRD_PARTY))
    import torchvision.transforms.functional as TF
    from unidac.dataloaders.dataset import resize_for_input
    from unidac.utils.erp_geometry import cam_to_erp_patch_fast, erp_patch_to_cam_fast

    cano_sz = config["data"]["cano_sz"]
    forward_size = tuple(config["data"].get("fwd_sz", (512, 704)))
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
                raise FileNotFoundError(
                    f"no depth-grid intrinsics for scene {scene_id}"
                )
            intrinsics_cache[scene_id] = resolved

        # Forward direction (camera -> ERP) uses the DS projection added to
        # erp_geometry.py; the reverse (ERP -> camera) uses the DS ray table, which
        # upstream already supports for models without a closed-form inverse.
        intrinsics = intrinsics_cache[scene_id]
        sx, sy = width / intrinsics["width"], height / intrinsics["height"]
        camera_params = {
            "dataset": "egodepth_336l",
            "camera_model": "DS",
            "fx": intrinsics["fx"] * sx,
            "fy": intrinsics["fy"] * sy,
            "cx": intrinsics["cx"] * sx,
            "cy": intrinsics["cy"] * sy,
            "xi": intrinsics["xi"],
            "alpha": intrinsics["alpha"],
            "fisheye_grid": rays,
        }

        # No GT is supplied: depth is a zero placeholder, and the valid mask is
        # zero, so nothing derived from the reference can enter the input.
        depth_placeholder = np.zeros((height, width, 1), dtype=np.float32)
        mask_placeholder = np.zeros((height, width, 1), dtype=np.float32)

        crop_width = int(cano_sz[0] * CROP_WFOV_DEG / 180)
        crop_height = int(crop_width * forward_size[0] / forward_size[1])

        erp_image, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image.astype(np.float32) / 255.0, depth_placeholder, mask_placeholder,
            0.0, np.array(0.0, dtype=np.float32),
            crop_height, crop_width, cano_sz[0], cano_sz[0] * 2,
            camera_params, np.array(0.0, dtype=np.float32), scale_fac=None,
        )
        (
            resized_image, _resized_depth, _pad, pred_scale_factor,
            attention_mask, latitude_grid, _longitude_grid,
        ) = resize_for_input(
            (erp_image * 255.0).astype(np.uint8), erp_depth, forward_size, None,
            [erp_image.shape[0], erp_image.shape[1]], 1.0,
            padding_rgb=[0, 0, 0], mask=erp_mask,
            lat_grid=latitude, long_grid=longitude,
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
        attention = TF.to_tensor(
            (attention_mask > 0).astype(np.float32)
        ).unsqueeze(0).to(args.device)
        latitude_tensor = torch.tensor(
            latitude_grid, dtype=torch.float32, device=args.device
        ).unsqueeze(0)

        with torch.no_grad():
            predictions, _, _ = model(
                normalized, latitude_range, longitude_range,
                attn_mask=attention, lat_grid=latitude_tensor,
            )
        # The resize factor the demo applies; omitting it scales the depth wrong.
        predictions = predictions * pred_scale_factor

        # Back to the camera. The ERP prediction is Euclidean distance along the ray.
        erp_prediction = torch.nn.functional.interpolate(
            predictions, size=(erp_image.shape[0], erp_image.shape[1]),
            mode="bilinear", align_corners=False,
        )[0]
        # erp_patch_to_cam_fast indexes this as a numpy array
        # (`torch.from_numpy(fisheye_grid2ray[:, :, 3])`), so it stays numpy.
        ray_table = rays
        # Returns (img_out, depth_out, mask_valid_out, mask_active). `mask_active`
        # is where the inverse mapping landed inside the ERP patch AND the ray table
        # had a valid ray -- outside it the value is a border-padded artifact, so it
        # becomes NaN rather than a scored prediction.
        _image_out, camera_distance, _mask_valid, mask_active = erp_patch_to_cam_fast(
            torch.from_numpy(
                (erp_image * 255.0).astype(np.float32).transpose(2, 0, 1)
            ).to(erp_prediction.device),
            erp_prediction,
            torch.ones_like(erp_prediction),
            0.0, 0.0, height, width, cano_sz[0], cano_sz[0] * 2,
            camera_params, fisheye_grid2ray=ray_table,
        )
        distance = np.asarray(
            camera_distance.squeeze().float().cpu().numpy(), dtype=np.float32
        )
        active = np.asarray(
            mask_active.squeeze().float().cpu().numpy(), dtype=np.float32
        ) > 0.5

        # z = range * u_z, with the same rays used to build the ERP patch. Pixels the
        # ERP round trip could not reach are NaN, not zero: zero would read as a
        # 0 m prediction, while NaN leaves them out of the prediction support.
        u_z = rays[..., 2]
        forward = (rays[..., 3] < 0.5) & (u_z > 0) & active
        with np.errstate(invalid="ignore"):
            depth_z = np.where(forward, distance * u_z, np.nan)
        return depth_z.astype(np.float32)

    run_loop(
        "unidac", args, predict,
        notes={
            "output_space": "metric_depth_z",
            "expected_alignment": "none",
            "checkpoint": str(checkpoint),
            "model_config": args.model_config,
            "camera_handling": (
                "forward (camera->ERP) uses a double-sphere branch added to "
                "unidac/utils/erp_geometry.py, because upstream supports only "
                "OPENCV_FISHEYE/MEI/ERP and DS is none of them; reverse (ERP->camera) "
                "uses the DS ray table, which upstream already supports"
            ),
            "third_party_patch": "unidac/utils/erp_geometry.py: added camera_model='DS'",
            "range_to_z": "model output is Euclidean distance; z = range * u_z applied",
            "pred_scale_factor_applied": True,
            **load_notes,
            "validation_status": (
                "geometry implemented against upstream ERP helpers; NOT validated "
                "against a UniDAC reference output on a double-sphere camera"
            ),
        },
    )


if __name__ == "__main__":
    main()
