"""A selected model must not require unrelated adaptation arms."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from examples.synthetic_smoke import create_synthetic_fixture


def test_selected_method_limits_model_checks_but_keeps_data_checks(tmp_path):
    config, _ = create_synthetic_fixture(tmp_path)
    config["run"]["out_dir"] = str(tmp_path / "preflight")
    config["methods"] = {"public": {"kind": "mapanything"}, "unavailable_ft": {
        "kind": "mapanything", "finetuned_state_dict": str(tmp_path / "absent.pt")}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    script = Path(__file__).resolve().parents[1] / "scripts/run_eval_pipeline.py"
    command = [sys.executable, str(script), "preflight", "--config", str(path), "--method"]
    process = subprocess.run([*command, "public"], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    names = {check["name"] for check in report["checks"]}
    assert "weights::public" in names and "e4_arm_filled::unavailable_ft" not in names
    assert {"samples_manifest", "mask_profile", "site_map_loaded"} <= names
    unavailable = subprocess.run([*command, "unavailable_ft"], capture_output=True, text=True)
    assert unavailable.returncode == 0, unavailable.stderr  # development report still records formal blockers
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    check = next(c for c in report["checks"] if c["name"] == "e4_arm_filled::unavailable_ft")
    assert check["ok"] is False and "missing fine-tuned checkpoint" in check["detail"]
    bad = subprocess.run([*command, "unknown"], capture_output=True, text=True)
    assert bad.returncode != 0 and "Unknown method" in bad.stderr


def test_subset_preflight_and_sweep_pass_same_selection(tmp_path, monkeypatch):
    from scripts import run_formal_sweep

    config, _ = create_synthetic_fixture(tmp_path)
    config["run"]["out_dir"] = str(tmp_path / "preflight")
    config["methods"] = {"left": {}, "right": {}, "missing_ft": {"finetuned_state_dict": None}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    script = Path(__file__).resolve().parents[1] / "scripts/run_eval_pipeline.py"
    command = [sys.executable, str(script), "preflight", "--config", str(path), "--methods", "left", "right"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert report["selected_methods"] == ["left", "right"]
    assert {c["name"] for c in report["checks"] if c["name"].startswith("weights::")} == {
        "weights::left", "weights::right"
    }
    assert "e4_arm_filled::missing_ft" not in {c["name"] for c in report["checks"]}

    # Stop at the subprocess boundary to verify the actual sweep's preflight arguments.
    config["run"]["status"] = "formal"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["sweep", "--config", str(path), "--methods", "left", "right"])
    calls = []

    def stop_after_preflight(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(run_formal_sweep.subprocess, "run", stop_after_preflight)
    with pytest.raises(subprocess.CalledProcessError):
        run_formal_sweep.main()
    assert calls == [[sys.executable, "scripts/run_eval_pipeline.py", "preflight",
                      "--config", str(path), "--methods", "left", "right"]]


def test_explicit_score_without_prediction_directory_fails(tmp_path):
    config, _ = create_synthetic_fixture(tmp_path)
    config["run"]["out_dir"] = str(tmp_path / "out")
    config["methods"] = {"requested": {}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    script = Path(__file__).resolve().parents[1] / "scripts/run_eval_pipeline.py"
    result = subprocess.run([sys.executable, str(script), "score", "--method", "requested",
                             "--config", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing prediction directory" in result.stderr
