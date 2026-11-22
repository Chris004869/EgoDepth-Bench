"""Release commands distinguish partial paper tables from complete, matched runs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from egodepth_eval.data.population import FormalRunBlocked
from egodepth_eval.infer.benchmark_contract import read_formal_contract
from egodepth_eval.score.prediction_io import PredictionContract
from scripts.export_paper_artifacts import load_run_results


def run_fixture(root):
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/release.yaml").read_text()
    )
    config["run"]["out_dir"] = str(root)
    name = "mapanything_6view_rig"
    identity = dict(
        role="test",
        status="formal",
        split="test",
        token_set_id="fixture",
        n_samples=1,
        n_sites=1,
        mask_profile="release_declared_mask",
        data_version="fixture",
        annotation_version="v6",
    )
    population = SimpleNamespace(samples=["a"], sample_tokens=["a"], identity=lambda: identity)
    contract = PredictionContract(
        method=name,
        input_views=tuple(range(6)),
        checkpoint_content_id="weights",
        upstream_revision="source",
        data_version="fixture",
        scale_contract="none (raw metric)",
        preprocessing={"model_input_hw": [406, 518], "pose_mode": "rig"},
        population_identity=identity,
    )
    result = {
        "method": name,
        "n_samples": 1,
        "population": identity,
        "prediction_contract": contract.to_dict(),
        "sample_status_counts": {"scored": 1},
    }
    parent = root / "predictions" / name
    parent.mkdir(parents=True)
    manifest = parent / "prediction_contract.json"
    manifest.write_text(json.dumps(contract.to_dict()))
    (root / "results").mkdir()
    (root / "results" / f"{name}.json").write_text(json.dumps(result))
    return config, name, population, manifest, contract


def test_single_method_export_and_complete_gate(tmp_path):
    config, name, population, _, _ = run_fixture(tmp_path)
    assert set(load_run_results(config, False, population)) == {name}
    with pytest.raises(FormalRunBlocked, match="Missing result"):
        load_run_results(config, True, population)
    config["methods"] = {name: config["methods"][name]}
    with pytest.raises(FormalRunBlocked, match="24-configuration"):
        load_run_results(config, True, population)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("token_set_id", "population differs"),
        ("method", "producer method"),
        ("pose_mode", "pose mode"),
        ("input_views", "producer views"),
        ("digest", "content digest"),
        ("missing", "missing producer"),
    ],
)
def test_formal_scoring_rejects_conflicting_sources(tmp_path, change, reason):
    config, name, population, manifest, contract = run_fixture(tmp_path)
    if change == "token_set_id":
        contract.population_identity = dict(contract.population_identity, token_set_id="other")
    elif change == "pose_mode":
        contract.preprocessing["pose_mode"] = "none"
    elif change == "input_views":
        contract.input_views = (2,)
    elif change == "method":
        contract.method = "another_model"
    payload = contract.to_dict()
    if change == "digest":
        payload["contract_hash"] = "obsolete"
    manifest.write_text(json.dumps(payload))
    if change == "missing":
        manifest.unlink()
    with pytest.raises(FormalRunBlocked, match=reason):
        read_formal_contract(manifest, name, config["methods"][name], population.identity())


def test_export_refuses_result_from_a_different_checkpoint(tmp_path):
    config, name, population, manifest, contract = run_fixture(tmp_path)
    contract.checkpoint_content_id = "other-weights"
    manifest.write_text(json.dumps(contract.to_dict()))
    with pytest.raises(FormalRunBlocked, match="saved prediction contracts disagree"):
        load_run_results(config, False, population)


def test_complete_export_checks_the_named_view_conditions(tmp_path):
    config, name, population, _, _ = run_fixture(tmp_path)
    config["methods"][name]["view_ids"] = [2]
    with pytest.raises(FormalRunBlocked, match="inputs or alignment"):
        load_run_results(config, True, population)
