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
    def __init__(self, root, split=None, protocol=None, cameras=(0, 1, 2, 3, 4, 5)):
        self.root = Path(root)
        self.cameras = tuple(cameras)
        if split not in (None, "train", "val", "test"):
            raise ValueError("invalid split")
        self.calibrations = json.loads((self.root / "metadata/calibrations.json").read_text())
        self._pid = None
        self._db = None
        self._manifest = None
        self._shards = OrderedDict()
        self._open()
        if protocol is not None:
            pp = Path(protocol)
            if not pp.is_absolute():
                pp = self.root / pp
            ids = []
            with pp.open() as f:
                for line in f:
                    r = json.loads(line)
                    row = self._db.execute(
                        "SELECT rowid,split FROM samples WHERE sample_token=?", (r["sample_token"],)
                    ).fetchone()
                    if row is None:
                        raise KeyError(r["sample_token"])
                    if split is not None and row[1] != split:
                        raise ValueError("protocol crosses requested split")
                    ids.append(row[0])
            self.rowids = np.asarray(ids, dtype=np.int64)
        else:
            query = (
                "SELECT rowid FROM samples"
                + (" WHERE split=?" if split else "")
                + " ORDER BY rowid"
            )
            self.rowids = np.fromiter(
                (r[0] for r in self._db.execute(query, (split,) if split else ())), dtype=np.int64
            )
        self.close()

    def _open(self):
        if self._pid == os.getpid() and self._db is not None:
            return
        self.close()
        self._pid = os.getpid()
        self._db = sqlite3.connect(
            f"file:{self.root / 'metadata/index.sqlite'}?mode=ro", uri=True, check_same_thread=False
        )
        self._manifest = os.open(self.root / "metadata/samples.jsonl", os.O_RDONLY)

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
            "SELECT byte_offset,byte_length FROM samples WHERE rowid=?", (int(self.rowids[index]),)
        ).fetchone()
        b = os.pread(self._manifest, row[1], row[0])
        if len(b) != row[1]:
            raise IOError("truncated manifest")
        result = json.loads(b)
        result["calibrations"] = self.calibrations[result["calibration_id"]]
        return result

    def _bytes(self, locator):
        self._open()
        rel, span = locator.rsplit("::", 1)
        off, length = map(int, span.split(":"))
        if Path(rel).is_absolute() or ".." in Path(rel).parts or off < 0 or length <= 0:
            raise ValueError("unsafe asset locator")
        if rel not in self._shards:
            self._shards[rel] = os.open(self.root / rel, os.O_RDONLY)
            if len(self._shards) > 16:
                os.close(self._shards.popitem(last=False)[1])
        self._shards.move_to_end(rel)
        b = os.pread(self._shards[rel], length, off)
        if len(b) != length:
            raise IOError("truncated shard")
        return b

    def __getitem__(self, index):
        r = self.metadata(index)
        images = {
            i: np.array(
                Image.open(io.BytesIO(self._bytes(r["rgb_rel_paths"][str(i)]))).convert("RGB")
            )
            for i in self.cameras
        }
        depth = np.load(io.BytesIO(self._bytes(r["depth_rel_path"])), allow_pickle=False).astype(
            np.float32
        )
        valid = np.load(io.BytesIO(self._bytes(r["valid_mask_rel_path"])), allow_pickle=False)
        if valid.dtype != np.bool_ or valid.shape != depth.shape:
            raise ValueError("bad released reference mask")
        if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError("invalid reference support")
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
