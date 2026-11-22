"""Portable EgoDepth-Bench reader: Python, NumPy and Pillow only.

Works as a map-style PyTorch Dataset without importing torch. Large arrays and
open handles are never serialized to worker processes. Paths stay inside root.
"""

from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

if __package__:
    from .release_paths import inside
else:
    from release_paths import inside


def unit_rays(calibration):
    """Double-sphere unit rays on the declared pixel grid (H,W,3).

    Optical-axis depth Z converts to ray distance as Z/u_z for u_z>0.
    Intrinsics belong to this exact grid; no implicit image resizing is applied.
    """
    c = calibration
    v, u = np.mgrid[: c["height"], : c["width"]].astype(np.float64)
    x = (u - c["cx"]) / c["fx"]
    y = (v - c["cy"]) / c["fy"]
    r2 = x * x + y * y
    a = c["alpha"]
    xi = c["xi"]
    valid = r2 <= 1 / (2 * a - 1) if a > 0.5 else np.ones_like(r2, dtype=bool)
    mz = (1 - a * a * r2) / (a * np.sqrt(np.maximum(1 - (2 * a - 1) * r2, 1e-10)) + 1 - a)
    k = (mz * xi + np.sqrt(np.maximum(mz * mz + (1 - xi * xi) * r2, 1e-10))) / (
        mz * mz + r2 + 1e-12
    )
    rays = np.stack([k * x, k * y, k * mz - xi], axis=-1)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-10)
    rays[~valid] = 0
    return rays.astype(np.float32)


class EgoDepthDataset:
    """Map each protocol token, or each dense split row, to a released RGB-D sample."""

    def __init__(self, root, split=None, protocol=None, cameras=(0, 1, 2, 3, 4, 5)):
        self.root = Path(root).resolve()
        self.cameras = tuple(cameras)
        if split not in (None, "train", "val", "test"):
            raise ValueError("invalid split")
        self.calibrations = json.loads(inside(self.root, "metadata/calibrations.json").read_text())
        self._pid = None
        self._db = None
        self._manifest = None
        self._shards = OrderedDict()
        try:
            self._open()
            if protocol is not None:
                self.rowids = self._protocol_rows(self.root / Path(protocol), split)
            else:
                query = (
                    "SELECT rowid FROM samples"
                    + (" WHERE split=?" if split else "")
                    + " ORDER BY rowid"
                )
                self.rowids = np.fromiter(
                    (r[0] for r in self._db.execute(query, (split,) if split else ())),
                    dtype=np.int64,
                )
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _protocol_rows(self, path, split):
        ids = []
        seen = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                token = record["sample_token"]
                if token in seen:
                    raise ValueError(f"Duplicate protocol token: {token}")
                row = self._db.execute(
                    "SELECT rowid, recording_id, timestamp_ns, physical_site_id, split "
                    "FROM samples WHERE sample_token = ?",
                    (token,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown protocol token: {token}")
                if split is not None and row["split"] != split:
                    raise ValueError(f"{token}: protocol crosses the requested split {split}")
                for key in ("recording_id", "timestamp_ns", "physical_site_id", "split"):
                    if key in record and record[key] != row[key]:
                        raise ValueError(f"{token}: protocol {key} disagrees with the index")
                seen.add(token)
                ids.append(row["rowid"])
        return np.asarray(ids, dtype=np.int64)

    def _open(self):
        if self._pid == os.getpid() and self._db is not None:
            return
        self.close()
        uri = inside(self.root, "metadata/index.sqlite").as_uri() + "?mode=ro"
        self._db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        try:
            self._db.row_factory = sqlite3.Row
            self._manifest = os.open(inside(self.root, "metadata/samples.jsonl"), os.O_RDONLY)
        except (OSError, ValueError):
            self.close()
            raise
        self._pid = os.getpid()

    def close(self):
        if self._db is not None:
            self._db.close()
        if self._manifest is not None:
            os.close(self._manifest)
        for fd in self._shards.values():
            os.close(fd)
        self._db = None
        self._manifest = None
        self._shards = OrderedDict()
        self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_db=None, _manifest=None, _shards=OrderedDict(), _pid=None)
        return state

    def __len__(self):
        return len(self.rowids)

    def metadata(self, index):
        self._open()
        row = self._db.execute(
            "SELECT sample_token, recording_id, timestamp_ns, physical_site_id, split, byte_offset, "
            "byte_length FROM samples WHERE rowid = ?",
            (int(self.rowids[index]),),
        ).fetchone()
        if row is None:
            raise ValueError("Selected sample disappeared from the release index")
        offset, length = row["byte_offset"], row["byte_length"]
        if offset < 0 or length <= 0:
            raise ValueError(f"{row['sample_token']}: invalid manifest byte range")
        payload = os.pread(self._manifest, length, offset)
        if len(payload) != length:
            raise IOError("Truncated manifest")
        result = json.loads(payload)
        for key in ("sample_token", "recording_id", "timestamp_ns", "physical_site_id", "split"):
            if result[key] != row[key]:
                raise ValueError(f"{row['sample_token']}: manifest {key} disagrees with the index")
        result["calibrations"] = self.calibrations[result["calibration_id"]]
        return result

    def _bytes(self, locator):
        self._open()
        rel, span = locator.rsplit("::", 1)
        off, length = map(int, span.split(":"))
        if off < 0 or length <= 0:
            raise ValueError("Invalid shard byte range")
        if rel not in self._shards:
            shard = inside(self.root, rel)
            self._shards[rel] = os.open(shard, os.O_RDONLY)
            if len(self._shards) > 16:
                os.close(self._shards.popitem(last=False)[1])
        self._shards.move_to_end(rel)
        b = os.pread(self._shards[rel], length, off)
        if len(b) != length:
            raise IOError("truncated shard")
        return b

    def __getitem__(self, index):
        r = self.metadata(index)
        shape = tuple(r["depth_shape_hw"])
        if r["target_camera_id"] != 2 or len(shape) != 2:
            raise ValueError(f"{r['sample_token']}: expected a camera-2 depth map")
        images = {}
        for camera in self.cameras:
            with Image.open(io.BytesIO(self._bytes(r["rgb_rel_paths"][str(camera)]))) as image:
                if image.size != (shape[1], shape[0]):
                    raise ValueError(
                        f"{r['sample_token']}: camera {camera} RGB grid differs from {shape}"
                    )
                images[camera] = np.array(image.convert("RGB"))
        stored = np.load(io.BytesIO(self._bytes(r["depth_rel_path"])), allow_pickle=False)
        if stored.dtype.kind != "f" or stored.shape != shape:
            raise ValueError(f"{r['sample_token']}: expected floating-point depth on grid {shape}")
        depth = stored.astype(np.float32, copy=False)
        valid = np.load(io.BytesIO(self._bytes(r["valid_mask_rel_path"])), allow_pickle=False)
        if valid.dtype != np.bool_ or valid.shape != shape:
            raise ValueError(f"{r['sample_token']}: invalid reference mask grid or dtype")
        if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError(f"{r['sample_token']}: invalid depth within reference support")
        return {
            "sample_token": r["sample_token"],
            "recording_id": r["recording_id"],
            "site_id": r["physical_site_id"],
            "split": r["split"],
            "timestamp_ns": r["timestamp_ns"],
            "images": images,
            "calibrations": r["calibrations"],
            "target_depth_z": depth,
            "target_valid": valid,
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--split", default="train")
    p.add_argument("--protocol")
    args = p.parse_args()
    data = EgoDepthDataset(args.root, args.split, args.protocol)
    s = data[0]
    print(
        json.dumps(
            {
                "samples": len(data),
                "first_token": s["sample_token"],
                "split": s["split"],
                "rgb_shapes": {k: list(v.shape) for k, v in s["images"].items()},
                "depth_shape": list(s["target_depth_z"].shape),
                "valid_pixels": int(s["target_valid"].sum()),
            }
        )
    )
    data.close()
