"""External predictions obey the same support and site-macro scientific contract."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.synthetic_smoke import create_synthetic_fixture
from egodepth_eval.annotate.schema import AnnotationRecord
from egodepth_eval.score.external import load_annotations, release_config, read_external_prediction, score_predictions
from egodepth_eval.score.prediction_io import PredictionContract


def test_external_cli_scores_saved_predictions_without_model_dependencies(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    output = tmp_path / "scored.json"
    process = subprocess.run([sys.executable, "-m", "egodepth_eval.score.external",
        "--config", str(tmp_path / "synthetic_config.yaml"), "--predictions-dir", str(predictions),
        "--method", "synthetic", "--output", str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    metrics = result["strata"]["roi::full"]["metrics"]
    assert metrics["abs_rel"] == pytest.approx(.1875)
    assert metrics["mae_z_cm"] == pytest.approx(12.5)
    assert metrics["coverage"] == pytest.approx(.75)
    assert result["population"]["status"] == "development"
    assert len(result["external_prediction_evaluation"]["prediction_files_sha256"]) == 3


def test_missing_predictions_remain_in_site_and_coverage_denominators(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    (predictions / "synthetic_b0.npz").unlink()
    output = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="predictions missing"):
        score_predictions(config, predictions, output, method="synthetic")
    assert not output.exists()
    result = score_predictions(config, predictions, output, method="synthetic", allow_missing=True)
    full = result["strata"]["roi::full"]
    assert full["metrics"]["coverage"] == pytest.approx(.5)
    assert full["metrics"]["abs_rel"] is None
    assert full["support"]["n_sites_no_predictions"] == 1
    assert result["sample_status_counts"]["missing_prediction"] == 1


def test_external_input_rejects_missing_support_wrong_grid_and_implicit_relative_depth(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, depth_z=np.ones((8, 8), np.float32))
    with pytest.raises(ValueError, match="missing arrays"):
        read_external_prediction(path, (8, 8), "none")
    np.savez(path, depth_z=np.ones((8, 8), np.float32), valid=np.ones((8, 8), bool))
    with pytest.raises(ValueError, match="shape"):
        read_external_prediction(path, (16, 16), "none")
    with pytest.raises(ValueError, match="explicit raw"):
        read_external_prediction(path, (8, 8), "affine_depth")


def test_relative_alignment_is_explicit_and_cannot_be_reported_as_raw_metric(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    for path in predictions.glob("*.npz"):
        with np.load(path) as data:
            depth, valid = data["depth_z"].copy(), data["valid"].copy()
        np.savez(path, depth_z=depth, valid=valid, raw=depth)
    result = score_predictions(config, predictions, tmp_path / "aligned.json", method="synthetic",
                               alignment="positive_scale")
    assert result["strata"]["roi::full"]["metrics"]["abs_rel"] == pytest.approx(0, abs=1e-7)
    assert result["external_prediction_evaluation"]["uses_test_reference_for_alignment"] is True


def test_no_silent_overwrite_or_extra_prediction_population(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("existing result")
    with pytest.raises(FileExistsError):
        score_predictions(config, predictions, output, method="synthetic")
    assert output.read_text() == "existing result"
    np.savez(predictions / "wrong-token.npz", depth_z=np.ones((16, 16)), valid=np.ones((16, 16), bool))
    with pytest.raises(ValueError, match="unexpected prediction tokens"):
        score_predictions(config, predictions, output, method="synthetic", overwrite=True)


@pytest.mark.parametrize("field,value", [
    ("scale_contract", "affine_depth"), ("output_space", "inverse_depth"),
    ("scoring_grid_hw", [416, 512]), ("target_camera_id", 3),
    ("data_version", "different-release"),
])
def test_producer_contract_cannot_mislabel_alignment_or_coordinates(tmp_path, field, value):
    config, predictions = create_synthetic_fixture(tmp_path)
    contract = PredictionContract("synthetic", scoring_grid_hw=(16, 16)).to_dict()
    contract[field] = value
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        score_predictions(config, predictions, tmp_path / "result.json", method="synthetic", contract_path=path)
    assert not (tmp_path / "result.json").exists()


def test_release_annotation_pointer_and_portable_masks_reach_real_roi_scoring(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    (tmp_path / "release_manifest.json").write_text(json.dumps({"version": "synthetic-only"}))
    pointer = {"directory": "annotations_v6", "annotation_version": "synthetic-v6"}
    (tmp_path / "annotation_version.json").write_text(json.dumps(pointer))
    directory = tmp_path / "annotations_v6"
    (directory / "masks").mkdir(parents=True)
    summary = {"status": "frozen", "annotation_version": "synthetic-v6", "n_records": 1}
    (directory / "annotation_summary.json").write_text(json.dumps(summary))
    mask_path = directory / "masks/synthetic_a0.npz"
    np.savez(mask_path, region=np.zeros((16, 16), np.int8))
    record = AnnotationRecord("synthetic_a0", "synthetic_site_a", "synthetic-v6",
        region_mask_path="annotations_v6/masks/synthetic_a0.npz", hand_status="accepted_ai")
    record_path = directory / "synthetic_a0.json"
    record_path.write_text(json.dumps(record.to_dict()))
    original = record_path.read_bytes()
    config["annotation"] = release_config(tmp_path)["annotation"]
    result = score_predictions(config, predictions, tmp_path / "roi.json", method="synthetic")
    assert result["strata"]["roi::hand"]["metrics"]["abs_rel"] == pytest.approx(.125)
    assert result["external_prediction_evaluation"]["n_annotation_records"] == 1
    assert result["prediction_contract"]["scoring_grid_hw"] == [16, 16]
    assert record_path.read_bytes() == original
    mask_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_annotations(config)
    summary["annotation_version"] = "wrong-version"
    (directory / "annotation_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="versions disagree"):
        release_config(tmp_path)
    (tmp_path / "annotation_version.json").unlink()
    with pytest.raises(FileNotFoundError):
        release_config(tmp_path)
