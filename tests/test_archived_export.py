import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.export_archived_results import verify_provenance


def test_public_results_omit_machine_identifiers_and_raw_training_logs():
    results = Path(__file__).resolve().parents[1] / "results"
    for directory in [results / "official", results / "paper_artifacts/diagnostic_subset"]:
        for path in directory.glob("*.json"):
            contract = json.loads(path.read_text())["prediction_contract"]
            assert {"platform", "hostname", "pid"}.isdisjoint(contract["environment"]), path.name

    training = json.loads((results / "training_run.json").read_text())
    assert {"hostname", "pid", "step_log", "validation_seconds"}.isdisjoint(training)


def test_roi_coverage_and_common_support_values_reach_the_archive_tables(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, root / "scripts/export_archived_results.py",
                    "--output", tmp_path], check=True)
    with (tmp_path / "adaptation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [round(float(row["common_support_aligned_abs_rel"]), 3) for row in rows] == [
        0.327, 0.327, 0.072,
    ]
    assert [round(float(row["common_support_aligned_object_mae_z_cm"]), 2) for row in rows] == [
        12.44, 12.44, 4.78,
    ]
    stereo = next(line for line in (tmp_path / "all_configurations.md").read_text().splitlines()
                  if line.startswith("| foundation_stereo |"))
    assert "| 35.4 | 2.15 / 72.8 | 2.52 / 81.0 |" in stereo
    assert "Per-frame scale alignment on common support" in (tmp_path / "adaptation.md").read_text()


def test_unlisted_archive_artifact_is_rejected(tmp_path):
    run = tmp_path / "training_run.json"
    run.write_text("{}")
    file_record = {"path": "results/training_run.json", "public_sha256": hashlib.sha256(b"{}").hexdigest()}
    (tmp_path / "provenance.json").write_text(json.dumps({"files": [file_record]}))
    assert verify_provenance(tmp_path) == 1
    official = tmp_path / "official"
    official.mkdir()
    (official / "accidentally_unverified.json").write_text("{}")
    with pytest.raises(ValueError, match="missing from provenance"):
        verify_provenance(tmp_path)
