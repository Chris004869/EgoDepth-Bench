"""Development sample limits apply to every reported part of the population."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from examples.synthetic_smoke import create_synthetic_fixture
from egodepth_eval.data.population import FormalRunBlocked, PopulationRole, resolve_population


@pytest.mark.parametrize("indexed", [False, True], ids=["protocol-jsonl", "indexed-manifest"])
@pytest.mark.parametrize("role", [PopulationRole.TEST, PopulationRole.DEVELOPMENT])
def test_limit_reports_only_retained_tokens_recordings_and_sites(tmp_path, indexed, role):
    config, _ = create_synthetic_fixture(tmp_path)
    source = Path(config["data"]["samples_jsonl"])
    site_map = tmp_path / "site_map.json"
    sites = ["synthetic_site_a", "synthetic_site_b"]
    site_map.write_text(json.dumps({
        "status": "frozen", "source": "synthetic",
        "recording_to_site": {f"recording_{s}": s for s in sites},
        "site_to_split": {s: "test" for s in sites},
    }))
    config["data"]["site_map_path"] = str(site_map)
    if indexed:
        folder = tmp_path / "metadata"
        folder.mkdir()
        manifest = folder / "samples.jsonl"
        manifest.write_bytes(source.read_bytes())
        (folder / "index_meta.json").write_text("{}")
        with sqlite3.connect(folder / "index.sqlite") as db:
            db.execute("CREATE TABLE samples(sample_token, recording_id, timestamp_ns, byte_offset, byte_length)")
            offset = 0
            for line in manifest.read_bytes().splitlines(keepends=True):
                row = json.loads(line)
                db.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                           (row["sample_token"], row["recording_id"], row["timestamp_ns"], offset, len(line)))
                offset += len(line)
        config["data"]["samples_jsonl"] = str(manifest)

    full = resolve_population(config, role)
    limited = resolve_population(config, role, limit=2)
    identity = limited.identity()
    assert full.sites == sites and identity["sites"] == [sites[0]]
    assert identity["n_samples"] == 2 and identity["n_sites"] == 1
    assert identity["n_samples_before_limit"] == 3
    assert identity["site_mapping_summary"] == {
        "sites": [sites[0]], "splits": {"test": 2}, "n_mapped": 2,
        "n_unmapped": 0, "unmapped_examples": [], "n_recordings": 1,
    }
    assert limited.sample_tokens == ["synthetic_a0", "synthetic_a1"]
    assert identity["token_set_id"] == hashlib.sha256(b"synthetic_a0\nsynthetic_a1").hexdigest()[:16]
    assert full.token_set_id() != identity["token_set_id"]
    if role == PopulationRole.TEST:
        assert identity["split_assertion"]["n_probed"] == identity["split_assertion"]["n_retained"] == 2


def test_limit_must_be_positive_and_development_only(tmp_path):
    config, _ = create_synthetic_fixture(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        resolve_population(config, PopulationRole.DEVELOPMENT, limit=-1)
    config["run"]["status"] = "formal"
    with pytest.raises(FormalRunBlocked, match="limit"):
        resolve_population(config, PopulationRole.TEST, limit=2)
