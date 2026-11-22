"""Restore downloaded .tar.zst files to the exact indexed .tar shards."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import zstandard as zstd

if __package__:
    from .release_paths import inside
else:
    from release_paths import inside


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_shard(root, row):
    """Verify both representations and replace a destination only after success."""
    source = inside(root, row["path"])
    destination = inside(root, row["restored_path"])
    if source == destination:
        raise ValueError("Compressed and restored paths must differ")
    if destination.exists() and sha(destination) == row["restored_sha256"]:
        return destination
    if sha(source) != row["sha256"]:
        raise ValueError(f"Compressed checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=destination.name + ".", suffix=".partial",
            dir=destination.parent, delete=False,
        ) as writer:
            temporary = Path(writer.name)
            with source.open("rb") as stream, zstd.ZstdDecompressor().stream_reader(stream) as reader:
                for block in iter(lambda: reader.read(8 << 20), b""):
                    digest.update(block)
                    writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != row["restored_sha256"]:
            raise ValueError(f"Restored checksum mismatch: {destination}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"])
    args = parser.parse_args()
    manifest = inside(args.root, "downloads/download_manifest.json")
    document = json.loads(manifest.read_text())
    for row in document["shards"]:
        inside(args.root, row["path"])
        inside(args.root, row["restored_path"])
        parts = Path(row["restored_path"]).parts
        if len(parts) < 3 or parts[0] != "shards" or parts[1] not in {"train", "val", "test"}:
            raise ValueError("Expected restored_path under shards/{train,val,test}/")
        if args.split and parts[1] != args.split:
            continue
        print(restore_shard(args.root, row), flush=True)


if __name__ == "__main__":
    main()
