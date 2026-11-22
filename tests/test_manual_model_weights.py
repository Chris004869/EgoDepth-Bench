"""Manual assets participate in the same preparation and resume workflow as downloaded models."""

import pytest

from egodepth_eval.infer.assets import SOURCES, local_weight_path, weight_identity
from egodepth_eval.infer.legacy_adapter import E1_METHODS, build_legacy_contract
from scripts.prepare_models import prepare_weights


@pytest.mark.parametrize("name", ["cut3r", "foundation_stereo"])
def test_manually_provided_weights_are_verified(tmp_path, monkeypatch, name):
    monkeypatch.setenv("EGODEPTH_WEIGHTS_DIR", str(tmp_path))
    prepare_weights(name, dry_run=True)
    with pytest.raises(FileNotFoundError, match="official files"):
        prepare_weights(name)
    checkpoint = local_weight_path(name)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint")
    if name == "foundation_stereo":
        with pytest.raises(FileNotFoundError, match="cfg.yaml"):
            prepare_weights(name)
        checkpoint.with_name("cfg.yaml").write_text("fixture: true")
    monkeypatch.setitem(
        SOURCES[name], "paper_checkpoint_content_id", weight_identity(name)["digest"]
    )
    prepare_weights(name)
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="differs"):
        prepare_weights(name)


@pytest.mark.parametrize("name", ["dac", "foundation_stereo"])
def test_changing_model_config_invalidates_new_predictions(tmp_path, monkeypatch, name):
    from types import SimpleNamespace

    monkeypatch.setenv("EGODEPTH_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "egodepth_eval.infer.assets.upstream_identity", lambda *args: {"commit": "fixture"}
    )
    monkeypatch.setattr("egodepth_eval.infer.legacy_adapter.default_environment", lambda: {})
    checkpoint = local_weight_path(name)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint")
    config = {"data": {"samples_jsonl": "fixture"}, "paths": {}}
    population = SimpleNamespace(manifest_content_id="fixture", identity=lambda: {})
    with pytest.raises(FileNotFoundError, match="model configuration"):
        build_legacy_contract(E1_METHODS[name], config, population)
    auxiliary = (
        checkpoint.with_suffix(".json") if name == "dac" else checkpoint.with_name("cfg.yaml")
    )
    auxiliary.write_text("{}")
    first = build_legacy_contract(E1_METHODS[name], config, population)
    auxiliary.write_text('{"fixture": 2}')
    second = build_legacy_contract(E1_METHODS[name], config, population)
    assert first.checkpoint_content_id == second.checkpoint_content_id
    assert (
        first.preprocessing["model_config_content_id"]
        != second.preprocessing["model_config_content_id"]
    )
    assert first.identity_hash() != second.identity_hash()
