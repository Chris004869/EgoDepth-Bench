"""Prediction storage and provenance.

Store scored Z as float32, with a separate validity mask for prediction
coverage. An optional raw representation supports fitting oracle alignments
without rerunning inference. A saved file is reusable only when its recorded
checkpoint, source, preprocessing, inputs and scoring-grid contract match.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

PREDICTION_SCHEMA = "egodepth-prediction-2"

# Fields that define whether an existing prediction can be reused. Anything that
# changes the numbers in the file belongs here; anything downstream of the file
# (aggregation, plotting, table formatting) deliberately does not.
CONTRACT_FIELDS = (
    "method", "checkpoint", "checkpoint_content_id", "upstream_revision",
    "input_views", "preprocessing", "precision", "output_space",
    "scale_contract", "support_rule", "target_camera_id", "scoring_grid_hw",
    "code_identity", "data_version", "manifest_id", "manifest_content_id",
    "population_identity",
)


def _git_output(args: list, cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# Hash implementation files by content, including local edits.
SOURCE_ROOTS_FOR_DIGEST = ("src/egodepth_eval", "method_runners", "scripts")
SOURCE_SUFFIXES_FOR_DIGEST = (".py", ".yaml", ".yml")


def file_content_digest(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """Digest of a file's contents, or None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def source_tree_digest(
    repo_root: Path,
    roots: tuple = SOURCE_ROOTS_FOR_DIGEST,
    suffixes: tuple = SOURCE_SUFFIXES_FOR_DIGEST,
) -> dict:
    """Content digest over the implementation source, tracked or not.

    Walks the source roots, hashes each file's bytes, and combines them in sorted
    path order. An edit to any untracked module changes this, so a prediction
    produced before the edit is no longer resumable.
    """
    repo_root = Path(repo_root)
    combined = hashlib.sha256()
    n_files = 0
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if rel.startswith(("src/egodepth_eval/report/", "src/egodepth_eval/train/", "src/egodepth_eval/annotate/")):
                continue
            if rel.startswith("src/egodepth_eval/evaluate/") and path.name != "geometry.py":
                continue
            if rel in {"src/egodepth_eval/score/aggregate.py", "src/egodepth_eval/score/scorer.py", "src/egodepth_eval/score/accumulate.py"}:
                continue
            if rel.startswith("scripts/") and path.name != "run_eval_pipeline.py":
                continue
            digest = file_content_digest(path)
            if digest is None:
                continue
            combined.update(str(path.relative_to(repo_root)).encode())
            combined.update(digest.encode())
            n_files += 1
    return {
        "roots": list(roots),
        "n_files": n_files,
        "digest": combined.hexdigest()[:16],
        "scheme": "sha256 over sorted (relative path, file content digest) pairs",
    }


def code_identity(repo_root: Optional[Path] = None) -> dict:
    """Content identity of the code that produces or converts predictions.

    This digest incorporates local edits under the prediction source roots.
    """
    repo_root = Path(repo_root or Path(__file__).resolve().parents[3])
    # Runtime outputs, commit IDs and descriptive dirty-file counts cannot change
    # a prediction. Content of relevant source is the reproducibility identity.
    return {"scheme": "prediction-source-v3", "source_tree": source_tree_digest(repo_root)}


@dataclass
class PredictionContract:
    """The identity of a prediction run. Two runs match only if these all match."""

    method: str
    checkpoint: Optional[str] = None
    # Digest of the actual weight FILE, so two fine-tuned checkpoints at different
    # paths -- or the same path with replaced contents -- get different identities.
    checkpoint_content_id: Optional[str] = None
    upstream_revision: Optional[str] = None
    input_views: tuple = ()
    preprocessing: dict = field(default_factory=dict)
    # Numeric precision changes the predictions, so it belongs in the identity.
    precision: Optional[str] = None
    output_space: str = "metric_depth_z"
    scale_contract: str = "none"
    support_rule: str = "finite_positive"
    target_camera_id: int = 2
    scoring_grid_hw: tuple = (416, 512)
    code_identity: dict = field(default_factory=dict)
    data_version: Optional[str] = None
    manifest_id: Optional[str] = None
    # Manifest CONTENT digest and the resolved population identity (split, sites,
    # token set), so a regenerated manifest at the same path is a different run.
    manifest_content_id: Optional[str] = None
    population_identity: dict = field(default_factory=dict)
    # Descriptive only; excluded from the identity hash so a hardware change does
    # not force re-inference.
    environment: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def contract_payload(self) -> dict:
        payload = {}
        for key in CONTRACT_FIELDS:
            value = getattr(self, key)
            if isinstance(value, (tuple, list)):
                value = list(value)
            payload[key] = value
        return payload

    def identity_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.contract_payload(), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["input_views"] = list(self.input_views)
        payload["scoring_grid_hw"] = list(self.scoring_grid_hw)
        payload["schema"] = PREDICTION_SCHEMA
        payload["contract_hash"] = self.identity_hash()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PredictionContract":
        payload = dict(payload)
        payload.pop("schema", None)
        payload.pop("contract_hash", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in payload.items() if k in known}
        if "input_views" in filtered:
            filtered["input_views"] = tuple(filtered["input_views"])
        if "scoring_grid_hw" in filtered:
            filtered["scoring_grid_hw"] = tuple(filtered["scoring_grid_hw"])
        return cls(**filtered)

    def describe_mismatch(self, other: "PredictionContract") -> list:
        differences = []
        mine, theirs = self.contract_payload(), other.contract_payload()
        for key in CONTRACT_FIELDS:
            if mine[key] != theirs[key]:
                differences.append(f"{key}: {theirs[key]!r} on disk vs {mine[key]!r} now")
        return differences


def default_environment() -> dict:
    environment = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    try:
        import torch

        environment["torch"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            environment["gpu_name"] = torch.cuda.get_device_name(0)
            environment["gpu_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - torch is optional for CPU-only tooling
        environment["torch"] = None
    return environment


def save_prediction(
    path: Path,
    depth_z: np.ndarray,
    valid: np.ndarray,
    raw_output: Optional[np.ndarray] = None,
    raw_space: Optional[str] = None,
    raw_valid: Optional[np.ndarray] = None,
    raw_transform: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write one prediction: metric Z float32, an independent support mask, raw output.

    `valid` is the model's declared support, not a re-derivation from the values.
    Saving it separately is what lets the scorer tell "no output here" from
    "output happened to be non-positive here".
    """
    depth_z = np.asarray(depth_z, dtype=np.float32)
    if depth_z.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {depth_z.shape}")
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != depth_z.shape:
        raise ValueError(f"valid shape {valid.shape} != depth shape {depth_z.shape}")

    payload = {"depth_z": depth_z, "valid": valid}
    if raw_output is not None:
        # `raw` may be on the model's own grid, which is generally NOT the scoring
        # grid. Its own support and the transform that produced `depth_z` are stored
        # alongside, so an oracle fit can be re-derived on the original support
        # rather than on an already-resampled array.
        payload["raw"] = np.asarray(raw_output, dtype=np.float32)
        payload["raw_space"] = np.array(str(raw_space or "unknown"))
        if raw_valid is not None:
            payload["raw_valid"] = np.asarray(raw_valid, dtype=bool)
        if raw_transform:
            payload["raw_transform"] = np.array(str(raw_transform))
    if extra:
        for key, value in extra.items():
            payload[f"extra_{key}"] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


@dataclass
class LoadedPrediction:
    depth_z: np.ndarray
    valid: np.ndarray
    raw: Optional[np.ndarray] = None
    raw_space: Optional[str] = None
    raw_valid: Optional[np.ndarray] = None
    raw_transform: Optional[str] = None

    @property
    def raw_on_scoring_grid(self) -> bool:
        """Whether `raw` shares the scored grid, so an alignment may use it directly."""
        return self.raw is not None and self.raw.shape == self.depth_z.shape


def load_prediction(path: Path) -> LoadedPrediction:
    """Read a prediction, accepting the legacy float16 `depth`-only layout.

    Legacy files have no support mask, so validity falls back to finite-and-
    positive and `raw` is absent. Kept readable so existing Ball-bar and Gold-v0
    predictions stay usable for aggregation work, but a legacy file cannot supply
    a real support mask and the scorer records that.
    """
    with np.load(str(path), allow_pickle=False) as payload:
        keys = set(payload.files)
        if "depth_z" in keys:
            depth = payload["depth_z"].astype(np.float32)
            valid = payload["valid"].astype(bool) if "valid" in keys else None
            raw = payload["raw"].astype(np.float32) if "raw" in keys else None
            raw_space = str(payload["raw_space"]) if "raw_space" in keys else None
            raw_valid = payload["raw_valid"].astype(bool) if "raw_valid" in keys else None
            raw_transform = (
                str(payload["raw_transform"]) if "raw_transform" in keys else None
            )
        elif "depth" in keys:
            depth = payload["depth"].astype(np.float32)
            valid = payload["valid"].astype(bool) if "valid" in keys else None
            raw, raw_space = None, "legacy_depth_only"
            raw_valid, raw_transform = None, None
        else:
            raise KeyError(f"{path}: no depth_z or depth array (keys: {sorted(keys)})")
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    return LoadedPrediction(
        depth_z=depth, valid=valid, raw=raw, raw_space=raw_space,
        raw_valid=raw_valid, raw_transform=raw_transform,
    )


def write_run_manifest(output_dir: Path, contract: PredictionContract, extra: Optional[dict] = None) -> Path:
    """Persist the contract that produced this prediction directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = contract.to_dict()
    if extra:
        payload["run"] = extra
    path = output_dir / "prediction_contract.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    return path


def check_resumable(output_dir: Path, contract: PredictionContract) -> dict:
    """Decide whether an existing prediction directory may be resumed.

    Returns {"resumable": bool, "reason": str, "differences": [...]}. A mismatched
    contract is refused rather than merged: mixing predictions from two
    preprocessing settings in one directory produces a score for a model that
    never existed.
    """
    output_dir = Path(output_dir)
    path = output_dir / "prediction_contract.json"
    existing_npz = sorted(output_dir.glob("*.npz"))

    if not path.exists():
        legacy = output_dir / "metadata.json"
        if legacy.exists():
            try:
                prior_method = json.loads(legacy.read_text()).get("method")
            except (OSError, json.JSONDecodeError):
                prior_method = None
            return {
                "resumable": False,
                "reason": (
                    f"directory holds legacy predictions from {prior_method!r}, "
                    f"not {contract.method!r}"
                    if prior_method and prior_method != contract.method
                    else "directory holds legacy predictions with no recorded "
                         "contract; they cannot be verified against the current one"
                ),
                "differences": [],
                "n_existing_npz": len(existing_npz),
            }
        if existing_npz:
            # Existing files without a prediction contract have unknown provenance.
            return {
                "resumable": False,
                "reason": (
                    f"{len(existing_npz)} prediction files exist with no "
                    "prediction_contract.json, so their provenance cannot be "
                    "verified. Delete them or point run.out_dir elsewhere."
                ),
                "differences": [],
                "n_existing_npz": len(existing_npz),
            }
        return {
            "resumable": True, "reason": "no existing predictions",
            "differences": [], "n_existing_npz": 0,
        }

    try:
        stored = PredictionContract.from_dict(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {
            "resumable": False,
            "reason": f"unreadable contract at {path}: {exc}",
            "differences": [],
        }

    differences = contract.describe_mismatch(stored)
    if differences:
        return {
            "resumable": False,
            "reason": "prediction contract changed since the existing run",
            "differences": differences,
            "n_existing_npz": len(existing_npz),
        }
    return {
        "resumable": True, "reason": "contract matches", "differences": [],
        "n_existing_npz": len(existing_npz),
    }


def verify_prediction_artifact(
    path: Path, expected_grid_hw: Optional[tuple] = None
) -> dict:
    """Check a kept prediction file before skipping its recomputation.

    A matching contract says the run *configuration* is the same; it says nothing
    about whether an individual file is intact. A truncated or half-written NPZ
    would otherwise be skipped as done and then scored.
    """
    try:
        loaded = load_prediction(Path(path))
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad: a half-written NPZ surfaces as zipfile.BadZipFile,
        # a truncated one as EOFError, a wrong-keyed one as KeyError. Any failure
        # to read means the file cannot be trusted as finished work, and the point
        # is to recompute it rather than to classify the corruption.
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

    problems = []
    if loaded.depth_z.ndim != 2:
        problems.append(f"depth is {loaded.depth_z.ndim}D, expected 2D")
    if loaded.valid.shape != loaded.depth_z.shape:
        problems.append(
            f"support shape {loaded.valid.shape} != depth shape {loaded.depth_z.shape}"
        )
    if expected_grid_hw and tuple(loaded.depth_z.shape) != tuple(expected_grid_hw):
        problems.append(
            f"grid {tuple(loaded.depth_z.shape)} != expected {tuple(expected_grid_hw)}"
        )
    if loaded.raw_space == "legacy_depth_only":
        problems.append(
            "legacy float16 layout with no independent support mask"
        )
    return {
        "ok": not problems,
        "reason": "; ".join(problems) if problems else "ok",
        "shape": list(loaded.depth_z.shape),
        "n_support": int(loaded.valid.sum()),
    }
