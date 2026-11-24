"""Externally supplied contracts and concurrent scorers cannot mislabel results."""
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from examples.synthetic_smoke import create_synthetic_fixture
from egodepth_eval.data.population import PopulationRole, resolve_population
from egodepth_eval.score.external import score_predictions
from egodepth_eval.score.prediction_io import PredictionContract, default_environment
from egodepth_eval.score.scorer import write_result


def test_concurrent_writers_publish_one_complete_result(tmp_path):
    path = tmp_path / "result.json"

    def publish(index):
        try:
            write_result(path, {"writer": index}, overwrite=False)
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [index for index in pool.map(publish, range(16)) if index is not None]
    assert winners == [json.loads(path.read_text())["writer"]]
    assert list(tmp_path.iterdir()) == [path]
    write_result(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"replacement": True}


def test_external_score_preserves_racing_writer(tmp_path, monkeypatch):
    config, predictions = create_synthetic_fixture(tmp_path)
    output = tmp_path / "score.json"
    real_link = os.link

    def publish_competitor_then_link(source, target):
        Path(target).write_text("concurrent winner")
        real_link(source, target)

    monkeypatch.setattr(os, "link", publish_competitor_then_link)
    with pytest.raises(FileExistsError):
        score_predictions(config, predictions, output, method="synthetic")
    assert output.read_text() == "concurrent winner"
    assert not list(tmp_path.glob(".score.json.*.tmp"))


@pytest.mark.parametrize("key,invalid", [
    ("token_set_id", "wrong"), ("n_samples", 17), ("split", "test"),
    ("status", "formal"), ("role", "test"), ("sites", ["wrong-site"]),
])
def test_external_contract_population_conflicts_are_rejected(tmp_path, key, invalid):
    config, predictions = create_synthetic_fixture(tmp_path)
    actual = resolve_population(config, PopulationRole.DEVELOPMENT).identity()
    # The development population has no split filter; each manifest row is "dev".
    identity = {k: actual[k] for k in ("token_set_id", "n_samples", "sites", "role", "status")}
    identity["split"] = "dev"
    contract = PredictionContract("synthetic", scoring_grid_hw=(16, 16), population_identity=identity)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract.to_dict()))
    score_predictions(config, predictions, tmp_path / "valid.json", method="synthetic", contract_path=path)
    contract.population_identity[key] = invalid
    path.write_text(json.dumps(contract.to_dict()))
    with pytest.raises(ValueError, match=rf"population_identity\.{key}"):
        score_predictions(config, predictions, tmp_path / "bad.json", method="synthetic", contract_path=path)
    assert not (tmp_path / "bad.json").exists()


def test_scoring_legacy_contract_omits_machine_identifiers(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    contract = PredictionContract("synthetic", scoring_grid_hw=(16, 16))
    legacy = contract.to_dict()
    legacy["environment"] = {
        "hostname": "example-host", "pid": 42, "platform": "example-kernel",
        "python": "3.11.0", "gpu_name": "Example GPU",
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    result = score_predictions(config, predictions, tmp_path / "score.json",
                               method="synthetic", contract_path=path)
    published = result["prediction_contract"]
    assert published["contract_hash"] == legacy["contract_hash"]
    assert published["environment"] == {"python": "3.11.0", "gpu_name": "Example GPU"}
    assert result["population"]["manifest_path"] == "synthetic_manifest.jsonl"
    assert not {"platform", "pid", "hostname"}.intersection(default_environment())
