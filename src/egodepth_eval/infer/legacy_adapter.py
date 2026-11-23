"""Run third-party depth models in their own Python environments.

Each runner reads staged images and calibration from the selected population.
The adapter converts predictions to float32 with a separate support mask and
records the runner digest, input views, representation and scale contract.
A depth-only runner declares its finite-positive support rule. Methods with a
restricted field of view can supply their own masks.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.data.contract import NUM_CAMERAS, TARGET_CAMERA_ID
from egodepth_eval.score.prediction_io import (
    PredictionContract,
    code_identity,
    default_environment,
    file_content_digest,
    save_prediction,
)

from egodepth_eval.infer.assets import configure_paths, local_weight_path

def conda_prefix():
    configured = os.environ.get("EGODEPTH_CONDA_PREFIX")
    if configured:
        return Path(configured).expanduser()
    current = Path(sys.prefix)
    return current.parent.parent if current.parent.name == "envs" else current


@dataclass
class LegacyMethodSpec:
    """One method's declared contract. Nothing here is inferred at runtime."""

    name: str
    runner: str                       # method_runners/run_{runner}.py
    env: str                          # conda environment name
    input_views: tuple                # cameras the method actually consumes
    output_space: str                 # metric_depth_z | inverse_depth | depth_affine ...
    alignment: str                    # none | positive_scale | affine_inverse | affine_depth
    units: str = "metres"
    # True only when the runner writes its own validity mask. Left False otherwise,
    # and recorded in the contract, because a fabricated mask would be worse than an
    # honestly-derived one.
    support_from_runner: bool = False
    # Set for methods whose output is range along the ray rather than optical Z; the
    # conversion then uses the calibrated rays.
    output_is_range: bool = False
    checkpoint: Optional[str] = None
    upstream_revision: Optional[str] = None
    extra_args: tuple = ()
    status: str = "wired"             # wired | no_runner | weights_unavailable
    notes: str = ""

    @property
    def runner_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "method_runners" / f"run_{self.runner}.py"
        )

    @property
    def python_path(self) -> Path:
        return conda_prefix() / "envs" / self.env / "bin" / "python"

    def availability(self) -> dict:
        return {
            "runner_exists": self.runner_path.exists(),
            "env_exists": self.python_path.exists(),
            "runner_path": str(self.runner_path),
            "python_path": str(self.python_path),
            "status": self.status,
        }


# The E1 pool. `alignment` and `output_space` are the protocol's declared handling
# for each method, not whatever an older YAML happened to say -- note Marigold is
# `affine_depth` here, fitted in depth space, per the 2026-09-12 protocol.
E1_METHODS: dict = {
    # Metric monocular, camera 2 only.
    "unidepth_v2": LegacyMethodSpec(
        "unidepth_v2", "unidepth_v2", "egodepth_mono", (2,),
        "metric_depth_z", "none"),
    "depth_pro": LegacyMethodSpec(
        "depth_pro", "depth_pro", "egodepth_depthpro", (2,),
        "metric_depth_z", "none"),
    "metric3d_v2": LegacyMethodSpec(
        "metric3d_v2", "metric3d_v2", "egodepth_mono", (2,),
        "metric_depth_z", "none"),
    "depth_anything_3": LegacyMethodSpec(
        "depth_anything_3", "depth_anything_3", "egodepth_da3", (2,),
        "metric_depth_z", "none"),
    "zoedepth": LegacyMethodSpec(
        "zoedepth", "zoedepth", "egodepth_zoedepth", (2,),
        "metric_depth_z", "none"),
    "unik3d": LegacyMethodSpec(
        "unik3d", "unik3d", "egodepth_mono", (2,),
        # The runner reads out["depth"] = points[:, -1:], which is optical Z, NOT
        # out["distance"] (range). So no range->Z conversion is needed here.
        "metric_depth_z", "none",
        checkpoint="lpiccinelli/unik3d-vitl",
        notes=(
            "any-camera monocular; DS intrinsics supplied through UniK3D's EUCM "
            "camera with beta=1 (equivalent for xi~=0, 0.03 deg agreement on this "
            "rig). Reads `depth` (optical Z), not `distance` (range)."
        )),
    "unidac": LegacyMethodSpec(
        "unidac", "unidac", "egodepth_mono", (2,),
        # The runner converts range->Z internally using the same DS rays it built the
        # ERP patch with, so what reaches the adapter is already optical Z.
        "metric_depth_z", "none",
        checkpoint="girish1511/UniDAC:unidac.pt",
        notes=(
            "ERP inference then back-projection to the camera. Double-sphere is "
            "neither MEI nor OPENCV_FISHEYE, so the DS ray table is supplied as "
            "UniDAC's fisheye_grid. Model output is Euclidean distance; the runner "
            "applies z = range * u_z. NOT validated against a UniDAC reference "
            "output on a DS camera."
        )),
    # DAC is a SEPARATE model from UniDAC, not an earlier name for it: iDisc-ERP +
    # Swin-L (CVPR 2025, 205.9 M params, yuliangguo/depth-any-camera) vs UniDAC's
    # DINOv3+DPT with a scale head (CVPR 2026, girish1511/UniDAC). They share the
    # upstream ERP helpers, which is why run_dac.py is a real runner against DAC's
    # own checkpoint rather than an alias of the UniDAC row -- an alias would report
    # one model's predictions as two independent table rows.
    "dac": LegacyMethodSpec(
        "dac", "dac", "egodepth_mono", (2,),
        # The runner converts range->Z internally with the same DS rays it built the
        # ERP patch with, so what reaches the adapter is already optical Z.
        "metric_depth_z", "none",
        checkpoint="yuliangguo/depth-any-camera:dac_swinl_indoor.pt",
        notes=(
            "ERP inference then back-projection to the camera, like UniDAC but a "
            "different model and checkpoint. Upstream states the output is Euclidean "
            "distance to the camera centre, not z; the runner applies z = range * "
            "u_z. Needs two recorded patches to dac/utils/erp_geometry.py: a DS "
            "forward branch, and a reverse branch that keys on the supplied ray "
            "table instead of upstream's dataset-name whitelist (without which a new "
            "camera is silently unprojected as a pinhole). NOT validated against a "
            "DAC reference output on a DS camera."
        )),
    # Calibrated stereo. Support is genuinely the rectified overlap, so these must
    # declare it rather than have it derived.
    "foundation_stereo": LegacyMethodSpec(
        "foundation_stereo", "foundation_stereo", "egodepth_fstereo", (2, 3),
        "metric_depth_z", "none", support_from_runner=True,
        notes="support is the rectified overlap cone, not the full fisheye frame"),
    "geostereo": LegacyMethodSpec(
        "geostereo", "geostereo", "egodepth_geostereo", (2, 3),
        "metric_depth_z", "none", support_from_runner=True,
        notes="support is the rectified overlap cone"),
    # Metric reconstruction from RGB.
    "mast3r": LegacyMethodSpec(
        "mast3r", "mast3r", "egodepth_mast3r", (2, 3),
        "metric_depth_z", "none"),
    "cut3r": LegacyMethodSpec(
        "cut3r", "cut3r", "egodepth_cut3r", tuple(range(NUM_CAMERAS)),
        "metric_depth_z", "none",
        checkpoint=None,
        notes=(
            "Use the official CUT3R checkpoint. The six-view "
            "question IS now resolved by perturbation on the real checkpoint: state "
            "flows forward only, so the target camera is fed LAST and each of the "
            "other five then changes its depth by 0.12-1.57 m max. In the natural "
            "order the target sat third and was bit-identical when views 3-5 were "
            "replaced by noise -- a three-view row labelled six-view."
        )),
    # Oracle multi-view: one positive Z scale per synchronized inference call.
    "vggt": LegacyMethodSpec(
        "vggt", "vggt", "egodepth_vggt", tuple(range(NUM_CAMERAS)),
        "depth_up_to_scale", "positive_scale"),
    "vggt_omega": LegacyMethodSpec(
        "vggt_omega", "vggt_omega", "egodepth_vggt", tuple(range(NUM_CAMERAS)),
        "depth_up_to_scale", "positive_scale",
        checkpoint=None,
        notes=(
            "Weights require access under the upstream VGGT-Omega terms."
        )),
    # Oracle monocular, representation-specific alignment.
    "depth_anything_v2": LegacyMethodSpec(
        "depth_anything_v2", "depth_anything_v2", "egodepth_group_d", (2,),
        "inverse_depth", "affine_inverse"),
    "marigold": LegacyMethodSpec(
        "marigold", "marigold", "egodepth_group_d", (2,),
        "depth_affine", "affine_depth",
        notes=(
            "fitted in DEPTH space per the 2026-09-12 protocol; the older config "
            "used affine_inverse"
        )),
    "moge": LegacyMethodSpec(
        "moge", "moge", "egodepth_moge", (2,),
        "depth_up_to_scale", "positive_scale",
        notes="pointmap Z, depth-like up to one positive scale"),
}


def export_runner_inputs(
    population,
    reader,
    staging_dir: Path,
    view_ids: Sequence[int],
    overwrite: bool = False,
) -> dict:
    """Stage split-filtered inputs for the selected runner.

    Runners read `rgb/cam{i}/{frame_id}.jpg`, `gt_depth/{frame_id}.npz`,
    `calibration/{scene_id}.json` and a flat `manifest.json`.

    `scene_id` combines physical site and calibration identity. A site can have
    several calibration sessions; these must never share a geometry cache.
    """
    staging_dir = Path(staging_dir)
    manifest: list = []
    calibrations_written: set = set()

    for directory in [
        *(staging_dir / "rgb" / f"cam{i}" for i in view_ids),
        staging_dir / "gt_depth",
        staging_dir / "calibration",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    import cv2

    for sample in population.samples:
        frame_id = sample.sample_token
        # A site may contain recordings from several calibration sessions.
        # Runners cache geometry by scene_id, so the key must include calibration.
        import hashlib
        geometry = {str(k): v.to_dict() for k,v in sorted(sample.calibrations.items())}
        calibration_id = hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()[:12]
        scene_id = (sample.physical_site_id or f"unmapped_{sample.recording_id}") + '__' + calibration_id
        depth_path = staging_dir / "gt_depth" / f"{frame_id}.npz"
        image_paths = {c: staging_dir / 'rgb' / f'cam{c}' / f'{frame_id}.png' for c in view_ids}
        if overwrite or not depth_path.exists() or any(not p.exists() for p in image_paths.values()) or scene_id not in calibrations_written:
            loaded = reader.load(sample, cameras=list(view_ids))
            if overwrite or not depth_path.exists():
                np.savez_compressed(depth_path,depth=loaded.reference_depth.astype(np.float32),valid=loaded.reference_valid)
            for camera_id in view_ids:
                image_path = image_paths[camera_id]
                if overwrite or not image_path.exists():
                    if not cv2.imwrite(
                        str(image_path),
                        cv2.cvtColor(loaded.images[camera_id], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1],
                    ):raise OSError(f'Cannot stage RGB: {image_path}')
            if scene_id not in calibrations_written:
                _write_calibration(staging_dir, scene_id, sample, loaded)
                calibrations_written.add(scene_id)

        manifest.append({
            "frame_id": frame_id,
            "scene_id": scene_id,
            "physical_site_id": sample.physical_site_id,
            "timestamp_ns": int(sample.timestamp_ns),
            "split": sample.split,
        })

    (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "staging_dir": str(staging_dir),
        "n_frames": len(manifest),
        "n_scenes": len(calibrations_written),
        "view_ids": list(view_ids),
        "manifest_path": str(staging_dir / "manifest.json"),
    }


def _write_calibration(staging_dir: Path, scene_id: str, sample, loaded) -> None:
    """Per-camera DS calibration in the old `camera_info` shape, plus depth_gt_info.

    Each camera gets its OWN entry. `depth_shape` is persisted so
    `_common.gt_grid_intrinsics` reads the real grid rather than falling back to the
    legacy 800x648 Ball-bar constant.
    """
    payload: dict = {}
    for camera_id, calibration in sorted(sample.calibrations.items()):
        calibration = loaded.calibrations.get(camera_id, calibration)
        payload[f"cam{camera_id}"] = {
            "width": int(calibration.width),
            "height": int(calibration.height),
            "distortion_model": "ds",
            "D": [float(v) for v in calibration.ds_params],
            "K": [
                [calibration.fx, 0.0, calibration.cx],
                [0.0, calibration.fy, calibration.cy],
                [0.0, 0.0, 1.0],
            ],
            "T_body_cam": calibration.T_body_cam,
        }
    (staging_dir / "calibration" / f"{scene_id}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    reference = loaded.reference_calibration
    (staging_dir / "calibration" / f"{scene_id}_depth_info.json").write_text(
        json.dumps({
            "depth_shape": [int(reference.height), int(reference.width)],
            "camera": {
                "K": [reference.fx, 0.0, reference.cx,
                      0.0, reference.fy, reference.cy,
                      0.0, 0.0, 1.0],
                "xi": reference.xi,
                "alpha": reference.alpha,
            },
        }, indent=2) + "\n"
    )


def build_legacy_contract(
    spec: LegacyMethodSpec, config: dict, population
) -> PredictionContract:
    """The prediction contract for a legacy runner.

    The runner script's own content digest is included, so editing a runner
    invalidates its predictions -- the same rule the in-process path follows.
    """
    from egodepth_eval.infer.assets import weight_identity, upstream_identity
    configure_paths(config)
    weights=weight_identity(spec.name, spec.checkpoint)
    upstream=upstream_identity(spec.name, config.get("paths",{}).get("third_party_dir"))
    auxiliary = {}
    if spec.name in {"dac", "foundation_stereo"}:
        checkpoint = local_weight_path(spec.name)
        model_config = (checkpoint.with_suffix(".json") if spec.name == "dac"
                        else checkpoint.with_name("cfg.yaml"))
        digest = file_content_digest(model_config)
        if digest is None:
            raise FileNotFoundError(f"{spec.name}: missing model configuration {model_config}")
        auxiliary["model_config_content_id"] = digest
    return PredictionContract(
        method=spec.name,
        checkpoint=spec.checkpoint,
        checkpoint_content_id=weights["digest"],
        upstream_revision=json.dumps(upstream,sort_keys=True),
        input_views=tuple(spec.input_views),
        preprocessing={
            "runner": str(spec.runner_path.name),
            "runner_content_id": file_content_digest(spec.runner_path),
            **auxiliary,
            "conda_env": spec.env,
            "geometry_contract": "full native frame, resampled to the scoring grid",
            "extra_args": list(spec.extra_args),
        },
        precision="runner_default",
        output_space=spec.output_space,
        scale_contract=spec.alignment,
        support_rule=(
            "runner-declared validity mask" if spec.support_from_runner
            else f"derived_representation_valid ({spec.alignment}; positivity after alignment)"
            if spec.alignment in ("affine_inverse", "affine_depth")
            else "derived_finite_positive (runner writes no support mask)"
        ),
        scoring_grid_hw=tuple(config["data"].get("scoring_grid_hw", (416, 512))),
        code_identity=code_identity(),
        data_version=config["data"].get("data_version"),
        manifest_id=str(config["data"]["samples_jsonl"]),
        manifest_content_id=population.manifest_content_id,
        population_identity={
            k: v for k, v in population.identity().items() if k != "sites"
        },
        environment=default_environment(),
        notes={
            "units": spec.units,
            "output_is_range": spec.output_is_range,
            "declared_status": spec.status,
            "method_notes": spec.notes,
        },
    )


def convert_legacy_output(
    legacy_path: Path,
    spec: LegacyMethodSpec,
    reference_calibration,
    scoring_grid_hw: tuple,
) -> dict:
    """One legacy `.npz` -> the new artifact fields.

    Three conversions, each recorded rather than assumed:
      - float16 -> float32 (the legacy runners write float16);
      - range -> optical Z via the calibrated rays, for the methods that emit range;
      - support: the runner's own mask when it wrote one, otherwise derived from
        finiteness and positivity with `support_rule` saying so in the contract.
    """
    from egodepth_eval.score.rays import build_ray_field

    with np.load(str(legacy_path), allow_pickle=False) as payload:
        keys = set(payload.files)
        if "depth" in keys:
            raw = payload["depth"].astype(np.float32)
        elif "depth_z" in keys:
            raw = payload["depth_z"].astype(np.float32)
        else:
            raise KeyError(f"{legacy_path}: no depth array (keys: {sorted(keys)})")
        declared_support = payload["valid"].astype(bool) if "valid" in keys else None

    raw_space = spec.output_space
    depth = raw
    conversions: list = []

    if spec.output_is_range:
        # z = range * u_z, with non-forward rays becoming NaN rather than 0.
        field = build_ray_field(reference_calibration, grid_hw=raw.shape)
        depth = field.range_to_z(raw)
        conversions.append("range -> optical Z via calibrated unit rays")

    if declared_support is not None:
        support = declared_support
        conversions.append("support: runner-declared mask")
    else:
        from egodepth_eval.score.alignment import representation_valid
        support = representation_valid(depth, spec.alignment)
        conversions.append(
            f"support: representation-valid for {spec.alignment}; physical positivity after alignment"
            if spec.alignment in ("affine_inverse", "affine_depth")
            else "support: derived finite-and-positive (no runner mask)"
        )

    if tuple(depth.shape) != tuple(scoring_grid_hw):
        from egodepth_eval.infer.mapanything_backend import resample_to_scoring_grid

        depth, support = resample_to_scoring_grid(
            depth, support, scoring_grid_hw,
            require_positive=spec.alignment not in ("affine_inverse", "affine_depth"),
        )
        conversions.append(
            "support-weighted INTER_AREA resample to the scoring grid"
        )

    return {
        "depth_z": depth.astype(np.float32),
        "valid": support,
        "raw": raw,
        "raw_space": raw_space,
        "raw_valid": declared_support,
        "raw_transform": "; ".join(conversions),
    }


def run_legacy_method(
    spec: LegacyMethodSpec,
    config: dict,
    population,
    reader,
    staging_dir: Path,
    out_dir: Path,
    device: str = "cuda",
    timeout: int = 7200,
    limit: Optional[int] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """Export inputs, run the subprocess, convert its output to the new contract."""
    configure_paths(config)
    availability = spec.availability()
    if not availability["runner_exists"]:
        return {
            "method": spec.name, "status": "no_runner",
            "reason": (
                f"{spec.runner_path} does not exist. This is a missing "
                "implementation, not a missing run."
            ),
            "availability": availability, "declared": spec.notes,
        }
    if not availability["env_exists"]:
        return {
            "method": spec.name, "status": "env_unavailable",
            "reason": f"conda env {spec.env!r} not found at {spec.python_path}",
            "availability": availability,
        }

    exported = export_runner_inputs(
        population, reader, staging_dir, spec.input_views, overwrite=overwrite
    )
    legacy_out = Path(out_dir).parent / f".legacy_{spec.name}"
    # `run_loop` in every runner resumes by SKIPPING existing .npz files, which is
    # what makes a long sweep restartable. But it also means a fixed runner, or a
    # fixed third-party file the runner imports, re-reports the OLD predictions:
    # only metadata.json gets rewritten, so the run looks fresh and the metrics are
    # byte-identical to the buggy ones. That is exactly how a corrected UniDAC
    # reverse-projection reproduced its pre-fix AbsRel to 16 decimals here.
    # `--overwrite` must therefore reach the raw staging, not just the converted
    # predictions.
    if overwrite and legacy_out.exists():
        removed = 0
        for stale in sorted(legacy_out.glob("*.npz")):
            stale.unlink()
            removed += 1
        stale_metadata = legacy_out / "metadata.json"
        if stale_metadata.exists():
            stale_metadata.unlink()
        print(
            f"[legacy] {spec.name}: cleared {removed} cached raw output(s) from "
            f"{legacy_out} because --overwrite was given"
        )
    command = [
        str(spec.python_path), str(spec.runner_path),
        "--data_dir", str(staging_dir),
        "--output_dir", str(legacy_out),
        "--manifest", exported["manifest_path"],
        "--device", device,
        *spec.extra_args,
    ]
    if limit is not None:
        command += ["--limit", str(limit)]

    if dry_run:
        return {
            "method": spec.name, "status": "dry_run",
            "command": command, "exported": exported,
            "availability": availability,
        }

    started = time.time()
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=os.environ.copy(),
    )
    elapsed = time.time() - started
    if completed.returncode != 0:
        return {
            "method": spec.name, "status": "runner_failed",
            "returncode": completed.returncode,
            # Warning-free tail: a multi-line FutureWarning otherwise fills the
            # window and the actual traceback never appears in the report.
            "stderr_tail": "\n".join([
                line for line in completed.stderr.strip().splitlines()
                if line.strip() and not any(
                    marker in line for marker in
                    ("FutureWarning", "UserWarning", "DeprecationWarning",
                     "weights_only", "SECURITY.md", "add_safe_globals",
                     "We recommend you start setting", "Please open an issue",
                     "warnings.warn", "it/s]", "%|")
                )
            ][-25:]),
            "stderr_full_tail": "\n".join(completed.stderr.strip().splitlines()[-40:]),
            "command": command, "elapsed_s": round(elapsed, 1),
            "availability": availability,
        }

    # Convert every produced frame into the new artifact format.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scoring_grid = tuple(config["data"].get("scoring_grid_hw", (416, 512)))
    n_converted, failures = 0, []
    transforms: set = set()

    for sample in population.samples:
        legacy_path = legacy_out / f"{sample.sample_token}.npz"
        if not legacy_path.exists():
            continue
        try:
            loaded = reader.load(sample, load_images=False)
            converted = convert_legacy_output(
                legacy_path, spec, loaded.reference_calibration, scoring_grid
            )
            save_prediction(
                out_dir / f"{sample.sample_token}.npz",
                converted["depth_z"], converted["valid"],
                raw_output=converted["raw"], raw_space=converted["raw_space"],
                raw_valid=converted["raw_valid"],
                raw_transform=converted["raw_transform"],
            )
            transforms.add(converted["raw_transform"])
            n_converted += 1
        except Exception as exc:  # noqa: BLE001 - one frame must not kill the method
            failures.append({
                "sample_token": sample.sample_token,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })

    legacy_metadata = legacy_out / "metadata.json"
    timings = {}
    if legacy_metadata.exists():
        try:
            payload = json.loads(legacy_metadata.read_text())
            timings = {
                "median_ms": payload.get("median_ms"),
                "mean_ms": payload.get("mean_ms"),
                "n_runner_errors": len(payload.get("errors", [])),
            }
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "method": spec.name,
        "status": "converted" if n_converted else "no_output",
        "n_converted": n_converted,
        "n_expected": len(population.samples),
        "n_conversion_failures": len(failures),
        "conversion_failures": failures[:10],
        "conversions_applied": sorted(transforms),
        "elapsed_s": round(elapsed, 1),
        "runner_timings": timings,
        "legacy_dir": str(legacy_out),
        "availability": availability,
    }


def method_status_table() -> list:
    """Per-method wiring status, so 'not implemented' and 'not run' stay distinct."""
    rows = []
    for name, spec in sorted(E1_METHODS.items()):
        availability = spec.availability()
        rows.append({
            "method": name,
            "declared_status": spec.status,
            "runner_exists": availability["runner_exists"],
            "env_exists": availability["env_exists"],
            "input_views": list(spec.input_views),
            "output_space": spec.output_space,
            "alignment": spec.alignment,
            "output_is_range": spec.output_is_range,
            "support_from_runner": spec.support_from_runner,
            "wiring": (
                "wired" if availability["runner_exists"] and availability["env_exists"]
                else "no_runner" if not availability["runner_exists"]
                else "env_missing"
            ),
            "notes": spec.notes,
        })
    return rows
