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
