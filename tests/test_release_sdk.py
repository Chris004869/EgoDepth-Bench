"""Portable label consumers must preserve uncertainty when paths are relocated."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk'))
from annotations import read_annotation
from local_annotation_records import materialize
from verify_checksums import inside


def test_partial_labels_survive_relocation_without_accepting_other_regions(tmp_path):
    root = tmp_path / 'relocated'
    (root / 'annotations/masks').mkdir(parents=True)
    region = np.full((416,512), -1, dtype=np.int8)
    region[:10,:10] = 0
    region[10:20,:10] = 1
    region[20:30,:10] = 2
    np.savez_compressed(root/'annotations/masks/s_example.npz', region=region)
    record = {'sample_token':'s_example', 'region_mask_path':'annotations/masks/s_example.npz',
              'hand_status':'accepted_ai', 'object_status':'uncertain', 'roi_status':'uncertain',
              'automatic_checks':{'background_confirmation':{'unpainted_is_background':True}}}
    (root/'annotations/s_example.json').write_text(json.dumps(record))
    _, usable = read_annotation(root,'s_example')
    assert int((usable == 0).sum()) == 100
    assert not (usable == 1).any() and not (usable == 2).any()
    local = tmp_path/'local-consumer'
    materialize(root,local)
    derived = json.loads((local/'s_example.json').read_text())
    assert Path(derived['region_mask_path']).is_absolute()
    assert Path(derived['region_mask_path']).exists()
    assert json.loads((root/'annotations/s_example.json').read_text()) == record


def test_public_paths_reject_traversal_and_absolute_names(tmp_path):
    for unsafe in ['../outside', '/tmp/outside']:
        with pytest.raises(ValueError):
            inside(tmp_path,unsafe)
    with pytest.raises(ValueError):
        read_annotation(tmp_path,'../s_example')


def test_restoration_checks_both_hashes_and_preserves_existing_file_on_failure(tmp_path):
    import hashlib
    zstd = pytest.importorskip("zstandard", reason="install the download extra for restoration tests")
    from restore_downloads import restore_shard

    payload = b"small synthetic shard payload" * 23
    compressed = zstd.ZstdCompressor().compress(payload)
    root = tmp_path / "release"
    (root / "downloads").mkdir(parents=True)
    (root / "downloads/shard.tar.zst").write_bytes(compressed)
    row = {"path": "downloads/shard.tar.zst", "restored_path": "shards/test/shard.tar",
           "sha256": hashlib.sha256(compressed).hexdigest(),
           "restored_sha256": hashlib.sha256(payload).hexdigest()}
    restored = restore_shard(root, row)
    assert restored.read_bytes() == payload
    restored.write_bytes(b"existing file must survive")
    with pytest.raises(ValueError, match="Restored checksum mismatch"):
        restore_shard(root, {**row, "restored_sha256": "0" * 64})
    assert restored.read_bytes() == b"existing file must survive"
    assert not list(restored.parent.glob("*.partial"))
    with pytest.raises(ValueError, match="Compressed checksum mismatch"):
        restore_shard(root, {**row, "sha256": "0" * 64})
    assert restored.read_bytes() == b"existing file must survive"


def test_restoration_rejects_symlink_escape(tmp_path):
    pytest.importorskip("zstandard")
    from restore_downloads import inside as restore_inside

    root = tmp_path / "release"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    for relative in ("../outside/file", "escape/file"):
        with pytest.raises(ValueError):
            restore_inside(root, relative)


@pytest.fixture
def indexed_protocol(tmp_path):
    """Two independently indexed manifest rows, no image assets needed."""
    import sqlite3

    root = tmp_path / 'test #1'
    metadata = root / 'metadata'
    metadata.mkdir(parents=True)
    (metadata / 'calibrations.json').write_text('{"rig": {}}')
    tokens = ('s_first', 's_other')
    rows = [
        dict(sample_token=t, recording_id='rec', timestamp_ns=i + 1,
             physical_site_id='site', split='test', calibration_id='rig')
        for i, t in enumerate(tokens)
    ]
    with (metadata / 'samples.jsonl').open('wb') as stream, sqlite3.connect(metadata / 'index.sqlite') as db:
        db.execute('CREATE TABLE samples (sample_token TEXT PRIMARY KEY, recording_id, timestamp_ns, '
                   'physical_site_id, split, byte_offset, byte_length)')
        for row in rows:
            offset = stream.tell()
            length = stream.write(json.dumps(row).encode() + b'\n')
            db.execute('INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)', (
                row['sample_token'], row['recording_id'], row['timestamp_ns'], row['physical_site_id'],
                row['split'], offset, length,
            ))
    return root, rows


def test_sdk_maps_protocol_order_and_checks_record_identity(indexed_protocol):
    import sqlite3
    from egodepth_dataset import EgoDepthDataset

    root, rows = indexed_protocol
    path = root / 'protocol.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in reversed(rows)))
    with EgoDepthDataset(root, split='test', protocol=path) as dataset:
        assert [dataset.metadata(i)['sample_token'] for i in range(2)] == ['s_other', 's_first']
    with sqlite3.connect(root / 'metadata/index.sqlite') as db:
        db.execute("UPDATE samples SET byte_offset=0 WHERE sample_token='s_other'")
    with EgoDepthDataset(root, split='test', protocol=path) as dataset:
        with pytest.raises(ValueError, match='manifest sample_token disagrees'):
            dataset.metadata(0)


@pytest.mark.parametrize('bad, field', [('repeat', 'Duplicate'), ('site', 'physical_site_id')])
def test_sdk_rejects_conflicting_protocol(indexed_protocol, bad, field):
    from egodepth_dataset import EgoDepthDataset

    root, rows = indexed_protocol
    entries = [rows[0], rows[0]] if bad == 'repeat' else [dict(rows[0], physical_site_id='different')]
    path = root / 'bad.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in entries))
    with pytest.raises(ValueError, match=field):
        EgoDepthDataset(root, split='test', protocol=path)


def test_legacy_materializer_requires_records_and_output_outside_release(tmp_path):
    release = tmp_path / 'release'
    (release / 'annotations').mkdir(parents=True)
    with pytest.raises(ValueError, match='outside'):
        materialize(release, release / 'annotations_v6')
    with pytest.raises(FileNotFoundError, match='No annotation records'):
        materialize(release, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
