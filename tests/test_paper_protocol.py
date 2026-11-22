"""Formal status must identify the v1 population and the annotations actually scored."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.synthetic_smoke import create_synthetic_fixture
from egodepth_eval.annotate.schema import AnnotationRecord
from egodepth_eval.data import paper_protocol as v1
from egodepth_eval.data.contract import CanonicalSample
from egodepth_eval.data.population import FormalRunBlocked, PopulationRole, resolve_population
from egodepth_eval.score.external import score_predictions


def test_formal_and_development_entry_points_on_the_same_subset(tmp_path):
    config, predictions = create_synthetic_fixture(tmp_path)
    sites = ["synthetic_site_a", "synthetic_site_b"] + [f"other_{i}" for i in range(35)]
    site_map = tmp_path / "sites.json"
    site_map.write_text(json.dumps({
        "source": "synthetic test", "status": "frozen",
        "recording_to_site": {f"recording_{site}": site for site in sites},
        "site_to_split": {site: "test" if i < 2 else "train" for i, site in enumerate(sites)},
    }))
    config["data"]["site_map_path"] = str(site_map)
    config["run"]["status"] = "formal"
    with pytest.raises(FormalRunBlocked, match="6574"):
        score_predictions(config, predictions, tmp_path / "formal.json", method="synthetic")
    assert not (tmp_path / "formal.json").exists()
    config["run"]["status"] = "development"
    population = resolve_population(config, PopulationRole.DEVELOPMENT)
    assert population.identity()["paper_protocol"] is None
    assert not population.formal_ready
    original = population.token_set_id()
    other = replace(population, samples=list(reversed(population.samples)))
    assert other.token_set_id() == original
    other.samples[0] = replace(other.samples[0], sample_token="different")
    assert other.token_set_id() != original


@pytest.fixture
def small_protocol(tmp_path, monkeypatch):
    config, _ = create_synthetic_fixture(tmp_path)
    rows = [json.loads(line) for line in Path(config["data"]["samples_jsonl"]).read_text().splitlines()]
    for row in rows:
        row.update(split="test", data_version=v1.DATA_VERSION, depth_shape_hw=v1.GRID_HW)
    config["data"].update(data_version=v1.DATA_VERSION, target_camera_id=2, scoring_grid_hw=v1.GRID_HW)
    samples = [CanonicalSample.from_dict(row) for row in rows]
    folder = tmp_path / "annotations"
    folder.mkdir()
    region = np.full(v1.GRID_HW, -1, dtype=np.int8)
    record = AnnotationRecord(samples[0].sample_token, samples[0].physical_site_id, v1.ANNOTATION_VERSION,
                              region_mask_path="annotations/one.npz")
    np.savez(folder / "one.npz", region=region)
    (folder / "one.json").write_text(json.dumps(record.to_dict()))
    config["annotation"] = {"status": "frozen", "version": v1.ANNOTATION_VERSION,
                            "dir": str(folder), "asset_root": str(tmp_path)}

    def encode(row):
        return json.dumps(row, separators=(",", ":")).encode()

    def digest(records):
        return hashlib.sha256(b"".join(encode(r) + b"\n" for r in sorted(records))).hexdigest()

    rig = [[v[k] for k in ("camera_id", "width", "height", "fx", "fy", "cx", "cy", "xi", "alpha", "T_body_cam")]
           for v in rows[0]["calibrations"].values()]
    rig_digest = hashlib.sha256(encode(rig)).hexdigest()
    monkeypatch.setattr(v1, "N_TEST", len(rows))
    monkeypatch.setattr(v1, "N_DIAGNOSTIC", 1)
    monkeypatch.setattr(v1, "TEST_ANCHORS_SHA256", digest([
        (s.sample_token, s.recording_id, s.timestamp_ns, s.physical_site_id) for s in samples]))
    monkeypatch.setattr(v1, "CALIBRATIONS_SHA256", digest([(s.sample_token, rig_digest) for s in samples]))
    monkeypatch.setattr(v1, "ANNOTATION_SHA256", digest([(
        record.sample_token, record.site_id, record.annotation_version, record.roi_status, record.hand_status,
        record.object_status, record.motion_status, record.motion_label, hashlib.sha256(region.tobytes()).hexdigest())]))
    return config, samples, folder


def test_protocol_identity_is_independent_of_manifest_order_and_npz_compression(small_protocol):
    config, samples, folder = small_protocol
    expected = v1.verify_paper_population(config, samples)
    np.savez_compressed(folder / "one.npz", region=np.full(v1.GRID_HW, -1, dtype=np.int8))
    assert v1.verify_paper_population(config, list(reversed(samples))) == expected


@pytest.mark.parametrize(("change", "message"), [
    ("repeat_token", "tokens"), ("swap_sites", "site"), ("timestamp", "timestamp"),
    ("geometry", "calibration"), ("mask", "records and masks"),
    ("annotation_status", "records and masks"), ("annotation_version", "annotations"),
])
def test_same_length_population_or_annotation_substitution_fails(small_protocol, change, message):
    config, samples, folder = small_protocol
    first, _, last = samples
    if change == "repeat_token":
        first.sample_token = last.sample_token
    elif change == "swap_sites":
        first.physical_site_id, last.physical_site_id = last.physical_site_id, first.physical_site_id
    elif change == "timestamp":
        first.timestamp_ns += 1
    elif change == "geometry":
        first.calibrations[2] = replace(first.calibrations[2], fx=8)
    elif change == "mask":
        np.savez(folder / "one.npz", region=np.zeros(v1.GRID_HW, dtype=np.int8))
    elif change == "annotation_status":
        data = json.loads((folder / "one.json").read_text())
        data["hand_status"] = "accepted_ai"
        (folder / "one.json").write_text(json.dumps(data))
    elif change == "annotation_version":
        config["annotation"]["version"] = "custom"
    with pytest.raises(ValueError, match=message):
        v1.verify_paper_population(config, samples)
