"""Shared plumbing for the method runners.

Every `run_*.py` follows the same contract, and the parts that are easy to get
subtly wrong live here rather than being re-derived fifteen times:

Input
  --data_dir           extracted/ directory
  --output_dir         predictions/{method}/
  --manifest           manifest.json
  --device             cuda or cpu
  --inject_extrinsics  (Group C extrinsics-conditioned only)
  --limit N            first N frames only, for smoke tests

Output, per frame
  {output_dir}/{frame_id}.npz   key "depth", float16, metres, GT grid

Completion
  {output_dir}/metadata.json    timings, per-frame errors, and the geometry
                                contract that was actually applied

Predictions cover the full native fisheye frame and are resampled to each
reference map's grid. Undo padding after upstream preprocessing so that the
predicted and reference image coordinates agree at the periphery.
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egodepth_eval.infer.assets import source_path, local_weight_path, hf_model_options

# Legacy Ball-bar grid. Kept as a fallback for old depth-info files that did
# not persist ``depth_shape``; new data must take its grid from depth_info/GT.
GT_SHAPE = (648, 800)  # (H, W)


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--inject_extrinsics", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N frames (smoke test)")
    return p


def load_manifest(path: str, limit: Optional[int] = None) -> list:
    entries = json.loads(Path(path).read_text())
    return entries[:limit] if limit else entries


def rgb_path(data_dir: Path, frame_id: str, cam: int = 2) -> str:
    root = data_dir / 'rgb' / f'cam{cam}'
    lossless = root / f'{frame_id}.png'
    return str(lossless if lossless.exists() else root / f'{frame_id}.jpg')


def load_calibration(data_dir: Path, scene_id: str) -> dict:
    return json.loads((data_dir / "calibration" / f"{scene_id}.json").read_text())


def load_depth_info(data_dir: Path, scene_id: str) -> Optional[dict]:
    """Intrinsics on the GT grid, if the extractor persisted them."""
    p = data_dir / "calibration" / f"{scene_id}_depth_info.json"
    return json.loads(p.read_text()) if p.exists() else None


def gt_grid_intrinsics(data_dir: Path, scene_id: str) -> Optional[dict]:
    """(fx, fy, cx, cy, xi, alpha) on the GT grid, from depth_gt_info.

    Models that accept intrinsics should be given *these*, not the camera_info
    values, which may describe a different native grid. Passing unscaled
    intrinsics is silent -- the depth still looks plausible. ``depth_shape``
    is authoritative for the grid size; the fixed 800x648 value is only a
    compatibility fallback for legacy Ball-bar metadata.
    """
    info = load_depth_info(data_dir, scene_id)
    if not info:
        return None
    cam = info.get("camera", info)
    K = cam.get("K")
    if not K or len(K) < 6:
        return None
    depth_shape = info.get("depth_shape", GT_SHAPE)
    if not isinstance(depth_shape, (list, tuple)) or len(depth_shape) != 2:
        return None
    height, width = int(depth_shape[0]), int(depth_shape[1])
    if height <= 0 or width <= 0:
        return None
    return {
        "fx": float(K[0]), "fy": float(K[4]),
        "cx": float(K[2]), "cy": float(K[5]),
        "xi": float(cam.get("xi", 0.0)),
        "alpha": float(cam.get("alpha", 0.5)),
        "width": width, "height": height,
    }


def pinhole_K_for_model(data_dir: Path, scene_id: str, shape_hw: tuple) -> np.ndarray:
    """A 3x3 pinhole K on `shape_hw`, for models that demand pinhole intrinsics.

    This is a deliberate approximation: the camera is a double-sphere fisheye
    and no 3x3 K describes it. It exists because several metric models require
    *some* K and refuse to run without one. The fisheye-versus-pinhole gap this
    introduces is part of what the benchmark measures, so it is recorded in
    metadata rather than hidden.
    """
    intr = gt_grid_intrinsics(data_dir, scene_id)
    h, w = shape_hw
    if intr is None:
        # Refuse rather than guess. A fabricated focal length propagates
        # straight into any metric conversion that uses it (DA3's
        # focal/300, Metric3D's fx/1000), producing depth that is wrong by
        # the ratio of guess to truth while looking entirely plausible --
        # which is exactly the class of silent error this benchmark exists
        # to catch. Re-run extraction to produce
        # calibration/{scene}_depth_info.json.
        raise FileNotFoundError(
            f"No depth-grid intrinsics for scene {scene_id}: expected "
            f"{data_dir / 'calibration' / f'{scene_id}_depth_info.json'}. "
            "Include the depth-grid intrinsics for this scene."
        )
    sx, sy = w / intr["width"], h / intr["height"]
    return np.array([
        [intr["fx"] * sx, 0, intr["cx"] * sx],
        [0, intr["fy"] * sy, intr["cy"] * sy],
        [0, 0, 1],
    ], dtype=np.float32)


def save_depth(out_path: Path, depth: np.ndarray, target_shape: tuple[int, int]) -> None:
    """Write one prediction, resampled onto the GT grid.

    The evaluator would resize anyway, but doing it here means the stored file
    is already the thing that gets scored, so a shape mismatch shows up now --
    while the runner that caused it is still identifiable -- rather than as a
    quiet interpolation weeks later.
    """
    import cv2

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {depth.shape}")

    if len(target_shape) != 2 or any(int(v) <= 0 for v in target_shape):
        raise ValueError(f"invalid GT target shape: {target_shape}")
    target_shape = (int(target_shape[0]), int(target_shape[1]))
    if depth.shape != target_shape:
        depth = cv2.resize(
            depth,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    np.savez_compressed(str(out_path), depth=depth.astype(np.float16))


def run_loop(
    method: str,
    args,
    predict: Callable,
    notes: Optional[dict] = None,
) -> dict:
    """Iterate the manifest, calling `predict(entry, data_dir) -> depth`.

    Resumes by skipping existing .npz files, records per-frame timing, and
    keeps going past a single frame's failure so one bad frame does not cost
    the whole method. Errors are written to metadata.json.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    manifest = load_manifest(args.manifest, args.limit)

    metadata = {
        "method": method,
        "device": args.device,
        "inject_extrinsics": bool(args.inject_extrinsics),
        "gt_shape": None,
        "geometry_contract": "full native frame, no centre crop, resampled to GT grid",
        "timings": [],
        "errors": [],
    }
    if notes:
        metadata.update(notes)

    # Resume only over predictions from the same method.
    existing_meta = output_dir / "metadata.json"
    if existing_meta.exists():
        try:
            prior = json.loads(existing_meta.read_text()).get("method")
        except (json.JSONDecodeError, OSError):
            prior = None
        if prior and prior != method:
            raise SystemExit(
                f"{output_dir} contains predictions from {prior!r}, not {method!r}. "
                "Delete the directory before running, or point --output_dir elsewhere."
            )

    n_done = 0
    gt_shapes: set[tuple[int, int]] = set()
    for entry in manifest:
        frame_id = entry["frame_id"]
        out_path = output_dir / f"{frame_id}.npz"

        # The final prediction grid follows the actual GT file, not a global
        # resolution constant. This both supports datasets with different
        # grids and catches a corrupt/mismatched frame before it is scored.
        try:
            gt_path = data_dir / "gt_depth" / f"{frame_id}.npz"
            with np.load(gt_path) as gt_data:
                target_shape = tuple(int(v) for v in gt_data["depth"].shape)
            if len(target_shape) != 2:
                raise ValueError(f"GT depth must be 2D, got shape {target_shape}")
            gt_shapes.add(target_shape)
        except Exception as e:  # noqa: BLE001 - keep processing other frames
            import traceback
            metadata["errors"].append({
                "frame_id": frame_id,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            })
            print(f"  [{method}] frame {frame_id} failed: {type(e).__name__}: {e}", flush=True)
            continue

        if out_path.exists():
            n_done += 1
            continue

        t0 = time.perf_counter()
        try:
            depth = predict(entry, data_dir)
            if depth is None:
                raise RuntimeError("predict returned None")
            save_depth(out_path, depth, target_shape)
            metadata["timings"].append(
                {"frame_id": frame_id, "ms": (time.perf_counter() - t0) * 1000}
            )
            n_done += 1
        except Exception as e:  # noqa: BLE001 - one frame must not kill the run
            import traceback
            metadata["errors"].append({
                "frame_id": frame_id,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            })
            print(f"  [{method}] frame {frame_id} failed: {type(e).__name__}: {e}", flush=True)

    if len(gt_shapes) == 1:
        metadata["gt_shape"] = list(next(iter(gt_shapes)))
    elif gt_shapes:
        metadata["gt_shapes"] = [list(shape) for shape in sorted(gt_shapes)]

    if metadata["timings"]:
        ms = [t["ms"] for t in metadata["timings"]]
        metadata["mean_ms"] = float(np.mean(ms))
        metadata["median_ms"] = float(np.median(ms))

    # Any note passed as a zero-arg callable is resolved now, so a runner can
    # record something it only learns while predicting (e.g. the focal length
    # a model estimated per frame) without threading it back out by hand.
    for key, value in list(metadata.items()):
        if callable(value):
            try:
                metadata[key] = value()
            except Exception as e:  # noqa: BLE001
                metadata[key] = f"<unavailable: {type(e).__name__}: {e}>"

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    n_err = len(metadata["errors"])
    print(f"  [{method}] {n_done}/{len(manifest)} frames written"
          + (f", {n_err} failed" if n_err else "")
          + (f", median {metadata['median_ms']:.0f} ms/frame" if metadata["timings"] else ""))

    if n_done == 0:
        raise SystemExit(f"{method}: no frames produced ({n_err} errors)")
    return metadata
