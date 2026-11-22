"""FoundationStereo inference runner -- RGB stereo (Group B).

Camera 2 (left) and camera 3 (right) form the evaluation stereo pair: 59.4 mm
baseline, 0.54 degrees relative rotation, translation essentially pure-x.
FoundationStereo needs *rectified* pinhole input, so the fisheye pair has to be
remapped onto a common virtual pinhole before matching, and the resulting
disparity converted back to depth on the fisheye GT grid.

NO CIRCULARITY WITH THE GT. The ground truth comes from five 336L active-IR
modules; this method sees only the ego RGB cameras. The GT pipeline also uses
FoundationStereo, but on different input pixels (IR, 95 mm baseline, separate
viewpoints) and gated against independent hardware depth. So this row measures
RGB stereo in this regime, not the GT reproducing itself.

THE GEOMETRY, IN THE ORDER IT MATTERS

1. Rectify. A virtual pinhole is chosen covering a central FoV (fisheye cannot
   be rectified globally -- a 200-degree lens has no pinhole equivalent, points
   at 90+ degrees would map to infinity). `--rectify_fov` sets it; the default
   90 degrees keeps the region where a pinhole approximation is defensible.
2. Match, giving disparity on the rectified grid.
3. depth_rect = fx_rect * baseline / disparity  (metric, from calibration).
4. Map back. Each fisheye GT pixel is unprojected to a ray, transformed into
   the rectified frame, and sampled -- so the saved map is on the GT grid and
   pixel-aligned with it. Pixels outside the rectified FoV come back as NaN and
   the evaluator excludes them, which is honest: this method genuinely cannot
   see the fisheye periphery.

Because of step 4, Group B's coverage is lower than the monocular methods' by
construction, and the reported score is over the pixels it can actually reach.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from _common import build_parser, gt_grid_intrinsics, load_calibration, rgb_path, run_loop  # noqa: E402

from _common import source_path, local_weight_path, hf_model_options

THIRD_PARTY = source_path("foundation_stereo")
WEIGHTS = local_weight_path("foundation_stereo")

LEFT_CAM, RIGHT_CAM = 2, 3
RECT_SIZE = (640, 800)  # (H, W) of the rectified virtual pinhole


def load_model(device, weights: Path):
    sys.path.insert(0, str(THIRD_PARTY))
    sys.path.insert(0, str(THIRD_PARTY / "core"))
    from omegaconf import OmegaConf, open_dict

    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(weights.parent / "cfg.yaml")
    if "args" in cfg:
        cfg = cfg["args"]
    # The released cfg.yaml predates the current FoundationStereo code, which
    # reads args.vit_size for the DepthAnything feature encoder. vitl matches
    # these released weights.
    if "vit_size" not in cfg:
        with open_dict(cfg):
            cfg["vit_size"] = "vitl"
    model = FoundationStereo(cfg)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def ds_camera(intr: dict, shape_hw: tuple):
    """DSCamera from a gt_grid_intrinsics-style dict, rescaled to shape_hw."""
    from egodepth_eval.evaluate.geometry import DSCamera

    h, w = shape_hw
    sx, sy = w / intr["width"], h / intr["height"]
    return DSCamera(fx=intr["fx"] * sx, fy=intr["fy"] * sy,
                    cx=intr["cx"] * sx, cy=intr["cy"] * sy,
                    xi=intr["xi"], alpha=intr["alpha"], width=w, height=h)


def ds_camera_from_calib(calib: dict, cam_idx: int, shape_hw: tuple):
    """DSCamera for one camera from its own camera_info entry.

    Each camera has its own intrinsics and they are not interchangeable: on
    this rig cam3's principal point sits 37 px from cam2's in x. Reusing cam2's
    model for cam3 injects that offset straight into the disparity as a
    constant, which inflates every depth by a plausible-looking factor instead
    of failing visibly. `D` for the DS model is [fx, fy, cx, cy, xi, alpha] on
    the native grid.
    """
    from egodepth_eval.evaluate.geometry import DSCamera

    entry = calib[f"cam{cam_idx}"]
    D = entry["D"]
    if len(D) < 6:
        raise ValueError(f"cam{cam_idx} has no DS distortion params: {D}")
    fx, fy, cx, cy, xi, alpha = D[:6]
    h, w = shape_hw
    sx = w / float(entry.get("width") or w)
    sy = h / float(entry.get("height") or h)
    return DSCamera(fx=fx * sx, fy=fy * sy, cx=cx * sx, cy=cy * sy,
                    xi=xi, alpha=alpha, width=w, height=h)


def ds_project(cam, X, Y, Z):
    """Double-sphere projection, matching the pipeline's camera_model."""
    r2 = X**2 + Y**2
    d1 = np.sqrt(r2 + Z**2)
    d2 = np.sqrt(r2 + (cam.xi * d1 + Z) ** 2)
    denom = cam.alpha * d2 + (1.0 - cam.alpha) * (cam.xi * d1 + Z)
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return cam.fx * X / denom + cam.cx, cam.fy * Y / denom + cam.cy


def build_rectification(calib: dict, intr: dict, native_hw: tuple, fov_deg: float):
    """Remap tables for both cameras onto a shared virtual pinhole.

    Returns (map_left, map_right, fx_rect, baseline_m, R_rect_from_left).
    """
    import cv2

    T2 = np.asarray(calib[f"cam{LEFT_CAM}"]["T_body_cam"], dtype=np.float64)
    T3 = np.asarray(calib[f"cam{RIGHT_CAM}"]["T_body_cam"], dtype=np.float64)
    # T_body_cam is cam->body, so right-from-left is inv(T3) @ T2.
    T_rl = np.linalg.inv(T3) @ T2
    R_rl, t_rl = T_rl[:3, :3], T_rl[:3, 3]
    baseline = float(np.linalg.norm(t_rl))

    # Rectified frame (Fusiello construction): x along the baseline, z as
    # close to the original forward direction as possible.
    #
    # The sign and handedness both matter. e1 must point from the left camera
    # toward the right one, so that a nearer surface shifts a pixel toward -x
    # in the right image and disparity comes out positive. And the basis has to
    # stay right-handed with e2 pointing the same way as the original +y --
    # building e2 as cross(z, e1) can yield a frame rolled 180 degrees about
    # the optical axis, which renders both rectified images upside-down. The
    # matcher then still finds confident correspondences (they are consistently
    # flipped), so disparity looks plausible while being several times too
    # large: here 160 px against a geometric maximum of 95 px at the closest GT
    # depth. Hence the explicit orientation checks below.
    e1 = -t_rl / baseline          # left -> right direction
    z_ref = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(z_ref, e1)
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:                  # baseline parallel to the optical axis
        e2 = np.cross(np.array([0.0, 1.0, 0.0]), e1)
        n2 = np.linalg.norm(e2)
    e2 /= n2
    if e2[1] < 0:                  # keep +y downward, as in the source images
        e2 = -e2
    e3 = np.cross(e1, e2)
    if e3[2] < 0:                  # keep +z forward
        e2, e3 = -e2, -e3
    R_rect = np.stack([e1, e2, e3])  # left-camera frame -> rectified frame

    h_r, w_r = RECT_SIZE
    fx_rect = (w_r / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx_r, cy_r = w_r / 2.0, h_r / 2.0

    # Rays of the rectified grid, expressed in each fisheye camera's frame.
    vs, us = np.mgrid[:h_r, :w_r].astype(np.float64)
    dirs = np.stack([(us - cx_r) / fx_rect, (vs - cy_r) / fx_rect, np.ones_like(us)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # Each camera is projected through its *own* intrinsics.
    cam_left = ds_camera_from_calib(calib, LEFT_CAM, native_hw)
    cam_right = ds_camera_from_calib(calib, RIGHT_CAM, native_hw)
    flat = dirs.reshape(-1, 3)

    # Both cameras must land in the *same* rectified orientation, so the pair's
    # 0.54-degree relative rotation is removed here rather than ignored: R_rect
    # is defined relative to the left camera, and the right camera's map
    # additionally passes through R_rl. Verified against theory on this rig --
    # recovered disparity matches fx*b/Z to 0.02 px and residual vertical
    # misalignment is under 0.04 px across 0.3-2.0 m.
    #
    # rect -> left camera frame
    d_left = flat @ R_rect
    ul, vl = ds_project(cam_left, d_left[:, 0], d_left[:, 1], d_left[:, 2])
    # rect -> right camera frame (through the relative rotation)
    d_right = d_left @ R_rl.T
    ur, vr = ds_project(cam_right, d_right[:, 0], d_right[:, 1], d_right[:, 2])

    def as_map(u, v):
        return (u.reshape(h_r, w_r).astype(np.float32), v.reshape(h_r, w_r).astype(np.float32))

    return as_map(ul, vl), as_map(ur, vr), fx_rect, baseline, R_rect


def rect_depth_to_gt_grid(depth_rect, intr, fx_rect, R_rect):
    """Resample rectified-frame depth onto the fisheye GT grid.

    Returns Z along the *fisheye* optical axis, matching the GT convention.
    """
    import cv2

    h_r, w_r = depth_rect.shape
    # ``intr`` comes from calibration/{scene}_depth_info.json and carries the
    # actual evaluation-grid dimensions. Ball-bar remains 648x800 while the
    # Gold-v0 336L export is 416x512; using a fixed intermediate grid here
    # would make the fisheye rays and GT pixels disagree before evaluation.
    target_shape = (int(intr["height"]), int(intr["width"]))
    if any(v <= 0 for v in target_shape):
        raise ValueError(f"invalid GT target shape from depth info: {target_shape}")
    cam = ds_camera(intr, target_shape)
    h_g, w_g = target_shape
    vs, us = np.mgrid[:h_g, :w_g].astype(np.float64)
    X, Y, Z = cam.unproject(us, vs)

    valid_ray = ~((X == 0) & (Y == 0) & (Z == 0))
    rays = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # left camera frame -> rectified frame
    r = rays @ R_rect.T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx_rect * r[:, 0] / r[:, 2] + w_r / 2.0
        v = fx_rect * r[:, 1] / r[:, 2] + h_r / 2.0
    in_front = r[:, 2] > 1e-6

    mx = np.where(in_front, u, -1).reshape(h_g, w_g).astype(np.float32)
    my = np.where(in_front, v, -1).reshape(h_g, w_g).astype(np.float32)
    sampled = cv2.remap(depth_rect.astype(np.float32), mx, my,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    # depth_rect is Z along the *rectified* axis; convert to range, then to Z
    # along the fisheye axis for this pixel's ray.
    rz = r[:, 2].reshape(h_g, w_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = sampled / np.clip(rz, 1e-6, None)      # range along the ray
        out = rng * Z                                # Z on the fisheye axis
    out[~valid_ray] = np.nan
    out[~in_front.reshape(h_g, w_g)] = np.nan
    return out.astype(np.float32)


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--rectify_fov", type=float, default=90.0,
                        help="Full FoV of the virtual pinhole, in degrees")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"FoundationStereo weights not found: {weights}")
    model = load_model(args.device, weights)
    rect_cache: dict = {}

    def predict(entry, data_dir):
        import cv2

        scene_id = entry["scene_id"]
        lp = rgb_path(data_dir, entry["frame_id"], LEFT_CAM)
        rp = rgb_path(data_dir, entry["frame_id"], RIGHT_CAM)
        left, right = cv2.imread(lp), cv2.imread(rp)
        if left is None or right is None:
            raise FileNotFoundError(f"missing stereo image: {lp if left is None else rp}")

        key = (scene_id, left.shape[:2])
        if key not in rect_cache:
            intr = gt_grid_intrinsics(data_dir, scene_id)
            if intr is None:
                raise FileNotFoundError(
                    f"no depth-grid intrinsics for {scene_id}; re-run extraction")
            rect_cache[key] = (build_rectification(
                load_calibration(data_dir, scene_id), intr, left.shape[:2], args.rectify_fov
            ), intr)
        (map_l, map_r, fx_rect, baseline, R_rect), intr = rect_cache[key]

        lr = cv2.remap(left, map_l[0], map_l[1], cv2.INTER_LINEAR)
        rr = cv2.remap(right, map_r[0], map_r[1], cv2.INTER_LINEAR)

        def to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(args.device)

        # InputPadder / unpad is not optional: the network requires dimensions
        # divisible by 32, and skipping the round-trip leaves the disparity on
        # a padded grid, misaligned with the rectified image it is supposed to
        # describe. That shows up as a uniform disparity inflation (~1.9x here)
        # that still correlates well with GT, so it reads as a plausible model
        # error rather than a preprocessing bug.
        from core.utils.utils import InputPadder

        t_l, t_r = to_tensor(lr), to_tensor(rr)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)

        with torch.no_grad(), torch.autocast(
            device_type="cuda" if args.device != "cpu" else "cpu",
            dtype=torch.float16, enabled=args.device != "cpu"
        ):
            disp = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True)
        disp = padder.unpad(disp.float()).squeeze().cpu().numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_rect = np.where(disp > 1e-3, fx_rect * baseline / disp, np.nan)
        return rect_depth_to_gt_grid(depth_rect, intr, fx_rect, R_rect)

    run_loop("foundation_stereo", args, predict,
             notes={"output_space": "metric_depth_z", "expected_alignment": "none",
                    "stereo_pair": [LEFT_CAM, RIGHT_CAM],
                    "rectify_fov_deg": args.rectify_fov,
                    "rectified_size": list(RECT_SIZE),
                    "valid_iters": args.valid_iters,
                    "weights": str(weights),
                    "coverage_note": ("limited to the rectified FoV; fisheye periphery "
                                      "is NaN because this method cannot observe it")})


if __name__ == "__main__":
    main()
