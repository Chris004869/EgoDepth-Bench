"""Score user-supplied predictions without installing any model or PyTorch.

Each <sample_token>.npz contains depth_z (float32 optical-axis metres) and
valid (bool) on the released camera-2 grid. Relative-depth evaluations also
provide raw on that same grid and explicitly select a GT-based alignment.
No predictions are resized, clipped, generated, or inferred by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from egodepth_eval.annotate.pipeline import load_records, region_masks_from_record
from egodepth_eval.data.mask_profile import get_profile
from egodepth_eval.data.population import FormalRunBlocked, PopulationRole, resolve_population
from egodepth_eval.data.reader import SampleReader
from egodepth_eval.score.prediction_io import PREDICTION_SCHEMA, LoadedPrediction, PredictionContract
from egodepth_eval.score.scorer import MethodScorer, build_result_document, write_result

PROGRESS_INTERVAL = 500
ALIGNMENTS = ("none", "positive_scale", "affine_depth", "affine_inverse")
OUTPUT_SPACES = {"none": "metric_depth_z", "positive_scale": "depth_up_to_scale",
                 "affine_depth": "depth_affine", "affine_inverse": "inverse_depth"}


def release_config(root: Path) -> dict:
    """Select the published official test protocol, never the dense training set."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    pointer = json.loads((root / "annotation_version.json").read_text())
    directory = Path(pointer["directory"])
    annotation_dir = (root / directory).resolve()
    if directory.is_absolute() or ".." in directory.parts or not annotation_dir.is_relative_to(root):
        raise ValueError("The annotation pointer must stay inside the release root")
    summary_path = annotation_dir / "annotation_summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "frozen" or
            summary.get("annotation_version") != pointer["annotation_version"]):
        raise ValueError("The annotation pointer and frozen summary versions disagree")
    return {
        "run": {"status": "formal", "population_name": "official_test_1hz"},
        "data": {
            "samples_jsonl": str(root / "protocols/test_1hz.jsonl"),
            "asset_roots": [str(root)],
            "site_map_path": str(root / "splits/site_map.json"),
            "data_version": manifest["version"],
            "mask_profile": "release_declared_mask",
            "target_camera_id": 2,
            "scoring_grid_hw": [416, 512],
        },
        "annotation": {"dir": str(annotation_dir), "asset_root": str(root),
                       "version": pointer["annotation_version"], "status": "frozen",
                       "summary": str(summary_path),
                       "diagnostics_manifest": str(root / "protocols/diagnostics.jsonl")},
    }


def scoring_source_identity() -> dict:
    """Hash the actual evaluator and data/annotation readers, including the scorer."""
    package = Path(__file__).resolve().parents[1]
    paths = [path for name in ("score", "data", "annotate")
             for path in (package / name).rglob("*.py")]
    paths.append(package / "evaluate/geometry.py")
    return {"scheme": "sha256-per-source-file", "files": {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)}}


def load_annotations(config: dict) -> dict:
    annotation = config.get("annotation", {})
    if not annotation.get("dir"):
        return {}
    directory = Path(annotation["dir"])
    if not directory.is_dir():
        raise FileNotFoundError(f"Configured annotation directory is missing: {directory}")
    records = load_records(directory, annotation.get("version"), annotation.get("asset_root"))
    if not records:
        raise ValueError("No annotation records match the configured version")
    if annotation.get("summary"):
        summary = json.loads(Path(annotation["summary"]).read_text())
        if summary.get("annotation_version") != annotation.get("version"):
            raise ValueError("Annotation summary version disagrees with the configuration")
        if len(records) != summary.get("n_records"):
            raise ValueError("Annotation record count disagrees with the frozen summary")
    for record in records.values():
        if not record.region_mask_path or not Path(record.region_mask_path).is_file():
            raise FileNotFoundError(f"Missing annotation mask for {record.sample_token}")
    return records


def read_external_prediction(
    path: Path, grid_hw: tuple, alignment: str, *, contents: bytes | None = None
) -> LoadedPrediction:
    """Decode the named file or an already captured snapshot of its bytes."""
    source = path if contents is None else io.BytesIO(contents)
    with np.load(source, allow_pickle=False) as payload:
        missing = {"depth_z", "valid"} - set(payload.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        depth = np.asarray(payload["depth_z"])
        valid = np.asarray(payload["valid"])
        if depth.shape != tuple(grid_hw) or valid.shape != depth.shape:
            raise ValueError(f"{path.name}: depth_z and valid must have shape {tuple(grid_hw)}")
        if depth.dtype.kind != "f" or valid.dtype != np.bool_:
            raise ValueError(f"{path.name}: depth_z must be floating point; valid must be bool")
        raw = None
        if alignment != "none":
            if "raw" not in payload:
                raise ValueError(f"{path.name}: alignment={alignment} requires an explicit raw array")
            raw = np.asarray(payload["raw"], dtype=np.float32)
            if raw.shape != depth.shape:
                raise ValueError(f"{path.name}: raw must share the scoring grid")
        return LoadedPrediction(depth.astype(np.float32), valid.copy(), raw=raw,
                                raw_space=alignment if raw is not None else None)


def score_predictions(config: dict, predictions_dir: Path, output: Path, *,
                      method: str, alignment: str = "none", allow_missing: bool = False,
                      overwrite: bool = False, contract_path: Path | None = None,
                      progress: bool = False) -> dict:
    """Use the same MethodScorer and population gate as the paper's model runners."""
    if alignment not in ALIGNMENTS:
        raise ValueError(f"Unsupported alignment: {alignment}")
    output, predictions_dir = Path(output), Path(predictions_dir)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; choose a new path or use --overwrite")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(predictions_dir)
    # Bind provenance when the evaluation starts. A long-running checkout can
    # change after Python has loaded the implementation used for these frames.
    code_identity = scoring_source_identity()
    role = PopulationRole.TEST if config.get("run", {}).get("status") == "formal" else PopulationRole.DEVELOPMENT
    population = resolve_population(config, role)
    if not len(population.samples):
        raise ValueError("The selected population is empty")
    identity = population.identity()
    tokens = population.sample_tokens
    if len(tokens) != len(set(tokens)):
        raise ValueError("The selected manifest contains duplicate sample tokens")
    if any(Path(token).name != token or token in (".", "..") for token in tokens):
        raise ValueError("Unsafe sample token in manifest")
    provided = {path.stem for path in predictions_dir.glob("*.npz")}
    extra, missing = provided - set(tokens), set(tokens) - provided
    if extra:
        raise ValueError(f"{len(extra)} unexpected prediction tokens; examples: {sorted(extra)[:5]}")
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} predictions missing; use --allow-missing to retain them as zero coverage")
    records = load_annotations(config)
    grid_hw = tuple(config["data"].get("scoring_grid_hw", population.samples[0].depth_shape_hw))
    if contract_path:
        payload = json.loads(Path(contract_path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("Prediction contract must be a JSON object")
        if "schema" in payload and payload["schema"] != PREDICTION_SCHEMA:
            raise ValueError(f"Unsupported prediction contract schema: {payload['schema']}")
        contract = PredictionContract.from_dict(payload)
        if "contract_hash" in payload and payload["contract_hash"] != contract.identity_hash():
            raise ValueError("Prediction contract hash disagrees with its declared fields")
        if contract.method != method:
            raise ValueError("The supplied prediction contract names a different method")
        if contract.target_camera_id != 2:
            raise ValueError("Only camera 2 is an evaluation target")
        if tuple(contract.scoring_grid_hw) != grid_hw:
            raise ValueError("Prediction contract scoring_grid_hw disagrees with the selected grid")
        scale_matches = contract.scale_contract == alignment
        if alignment == "none":
            scale_matches |= contract.scale_contract == "none (raw metric)"
            scale_matches |= contract.scale_contract.startswith("fixed positive scalar s_C=")
        if not scale_matches or contract.output_space != OUTPUT_SPACES[alignment]:
            raise ValueError("Prediction contract scale_contract/output_space disagrees with alignment")
        if contract.data_version is not None and contract.data_version != population.data_version:
            raise ValueError("Prediction contract data_version disagrees with the population")
        declared = contract.population_identity
        if not isinstance(declared, dict):
            raise ValueError("Prediction contract population_identity must be an object")
        for key in ("token_set_id", "n_samples", "n_sites", "role", "status", "data_version",
                    "mask_profile", "manifest_content_id", "annotation_version", "formal_ready"):
            if key in declared and (type(declared[key]) is not type(identity[key]) or declared[key] != identity[key]):
                raise ValueError(f"Prediction contract population_identity.{key} disagrees with the selected population")
        if "sites" in declared:
            sites = declared["sites"]
            if (not isinstance(sites, list) or not all(isinstance(site, str) for site in sites)
                    or sorted(sites) != sorted(identity["sites"])):
                raise ValueError("Prediction contract population_identity.sites disagrees with the selected population")
        if "split" in declared and declared["split"] != identity["split"]:
            # Development may score a single-split manifest without requesting a
            # formal split gate. Check its actual rows before accepting that label.
            if (role != PopulationRole.DEVELOPMENT or not isinstance(declared["split"], str)
                    or any(sample.split != declared["split"] for sample in population.samples)):
                raise ValueError("Prediction contract population_identity.split disagrees with the selected population")
        producer_provenance = "supplied_by_user; not independently verified"
    else:
        contract = PredictionContract(method=method, scale_contract=alignment,
            output_space=OUTPUT_SPACES[alignment], scoring_grid_hw=grid_hw,
            support_rule="declared_valid_and_finite_positive",
            notes={"origin": "external predictions", "producer_identity": "not supplied"})
        producer_provenance = "not supplied"
    reader = SampleReader([Path(p) for p in config["data"]["asset_roots"]],
                          get_profile(config["data"]["mask_profile"]))
    scorer = MethodScorer(alignment_mode=alignment)
    scores = []
    prediction_hashes = {}
    total = len(tokens)
    interactive = progress and sys.stderr.isatty() and total >= PROGRESS_INTERVAL
    if interactive:
        print(f"Scoring {method}: 0/{total} frames", file=sys.stderr, flush=True)
    for n, sample in enumerate(population.samples, start=1):
        loaded = reader.load(sample, load_images=False)
        if tuple(loaded.grid_hw) != grid_hw:
            raise ValueError(f"{sample.sample_token}: reference grid disagrees with the prediction contract")
        path = predictions_dir / f"{sample.sample_token}.npz"
        prediction = None
        if sample.sample_token in provided:
            # Score and identify the same bytes, even if a producer replaces the
            # file during a long run. A file lost after discovery is an I/O error.
            contents = path.read_bytes()
            prediction = read_external_prediction(path, loaded.grid_hw, alignment, contents=contents)
            prediction_hashes[sample.sample_token] = hashlib.sha256(contents).hexdigest()
        record = records.get(sample.sample_token)
        masks = region_masks_from_record(record, loaded.grid_hw) if record else None
        scores.append(scorer.score_sample(sample, sample.physical_site_id or f"unmapped::{sample.recording_id}",
            loaded.reference_depth, loaded.reference_valid, prediction, loaded.reference_calibration, roi_masks=masks))
        if interactive and (n % PROGRESS_INTERVAL == 0 or n == total):
            print(f"Scoring {method}: {n}/{total} frames", file=sys.stderr, flush=True)
    document = build_result_document(method, contract, scores, identity, extra={
        "external_prediction_evaluation": {
            "alignment": alignment,
            "uses_test_reference_for_alignment": alignment != "none",
            "producer_provenance": producer_provenance,
            "n_prediction_files": len(provided), "n_missing_predictions": len(missing),
            "prediction_files_sha256": prediction_hashes,
            "n_annotation_records": len(records),
            "scoring_code_identity": code_identity,
        },
    })
    write_result(output, document, overwrite=overwrite)
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-root", type=Path, help="Restored release root; uses its official test_1hz protocol")
    source.add_argument("--config", type=Path, help="Benchmark YAML (paths relative to the current working directory)")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--method", required=True, help="Name of the user's model")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON; existing files are refused by default")
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="none",
                        help="Other choices use per-frame reference GT; they are not unaligned metric results")
    parser.add_argument("--prediction-contract", type=Path, help="Optional producer provenance JSON; never rewritten")
    parser.add_argument("--allow-missing", action="store_true", help="Count missing predictions as zero coverage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = release_config(args.release_root) if args.release_root else yaml.safe_load(args.config.read_text())
        result = score_predictions(config, args.predictions_dir, args.output, method=args.method,
            alignment=args.alignment, allow_missing=args.allow_missing, overwrite=args.overwrite,
            contract_path=args.prediction_contract, progress=True)
    except (ValueError, OSError, KeyError, FormalRunBlocked) as error:
        print(f"Scoring failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "population_status": result["population"]["status"],
        "n_samples": result["n_samples"], "alignment": args.alignment,
        "whole_image": result["strata"]["roi::full"]["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
