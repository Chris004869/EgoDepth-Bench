"""Read canonical samples from the release JSONL and SQLite index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

from egodepth_eval.data.contract import CanonicalSample


class ManifestStore:
    """Random access to an indexed manifest without loading it all.

    Use one instance per data-loading worker. Each lookup reads one JSONL record.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.jsonl_path = self.index_dir / "samples.jsonl"
        self.sqlite_path = self.index_dir / "index.sqlite"
        meta_path = self.index_dir / "index_meta.json"
        for path in (self.jsonl_path, self.sqlite_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing release metadata: {path}. Restore the metadata directory from the release."
                )
        self.meta = json.loads(meta_path.read_text())
        self._connection: Optional[sqlite3.Connection] = None
        self._stream = None
        self._calibrations = None

    def _sample(self, payload: dict) -> CanonicalSample:
        if "calibrations" not in payload and "calibration_id" in payload:
            if self._calibrations is None:
                self._calibrations = json.loads((self.index_dir / "calibrations.json").read_text())
            payload = dict(payload)
            payload["calibrations"] = self._calibrations[payload["calibration_id"]]
        return CanonicalSample.from_dict(payload)

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            # check_same_thread=False so a dataloader worker thread can read the
            # handle it inherited; all access here is read-only.
            self._connection = sqlite3.connect(
                f"file:{self.sqlite_path}?mode=ro", uri=True, check_same_thread=False
            )
        return self._connection

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])

    def tokens(self) -> list:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT sample_token FROM samples ORDER BY recording_id, timestamp_ns"
            )
        ]

    def get(self, sample_token: str) -> CanonicalSample:
        row = self.connection.execute(
            "SELECT byte_offset, byte_length FROM samples WHERE sample_token = ?",
            (sample_token,),
        ).fetchone()
        if row is None:
            raise KeyError(f"sample_token {sample_token!r} is not in {self.index_dir}")
        if self._stream is None:
            self._stream = self.jsonl_path.open("rb")
        self._stream.seek(int(row[0]))
        payload = self._stream.read(int(row[1]))
        return self._sample(json.loads(payload.decode("utf-8")))

    def iter_samples(self) -> Iterator[CanonicalSample]:
        with self.jsonl_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield self._sample(json.loads(line))

    def recordings(self) -> list:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT recording_id FROM samples ORDER BY recording_id"
            )
        ]

    def recording_samples(self, recording_id: str) -> list:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT sample_token FROM samples WHERE recording_id = ? ORDER BY timestamp_ns",
                (recording_id,),
            )
        ]

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
