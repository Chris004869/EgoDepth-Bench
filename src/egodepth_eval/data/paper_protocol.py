"""Content identities for the population and annotations used in the v1 paper.

Rows are sorted by sample token, encoded as compact JSON arrays in UTF-8,
and terminated by LF. Paths, manifest order, shard offsets and NPZ compression
are storage details. Dataset asset checksums are verified by the release SDK.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
from pathlib import Path

import numpy as np

from egodepth_eval.annotate.pipeline import load_records
from egodepth_eval.annotate.schema import AnnotationRecord
from egodepth_eval.data.contract import CanonicalSample


DATA_VERSION = "egodepth_v1_20260912"
ANNOTATION_VERSION = "ai-rgb-reviewed-v6"
N_TEST = 6574
N_DIAGNOSTIC = 800
TEST_SITES = ("site_010", "site_013", "site_014", "site_015", "site_016",
              "site_027", "site_032", "site_034")
GRID_HW = (416, 512)
TOKEN_SET_ID = "4f36b7e1100a8c2a"  # Identifier recorded in the original result JSONs.
TEST_ANCHORS_SHA256 = "dd24185eb1291ccd59bc447a57240bf3194dbf3a0e93a03f672cecd83e08167c"
TOKEN_SITES_SHA256 = "ed09fcb4ce70a36fe234a94a1f06b2a4c51051d96b327d2344f92d5569b7f5ce"
CALIBRATIONS_SHA256 = "22ad5e68b51919b6ae52a0fceab2031c341a49127cfceb5f2dfb453c25416c5a"
DIAGNOSTIC_ANCHORS_SHA256 = "1e2f2d72befaf912323eba8ae54644011a2c3ea4b4b70462180e34cf370bea63"
DIAGNOSTIC_CALIBRATIONS_SHA256 = "ee40afdcb65e149470934a8fba509c69618676855ae43eeb296e4af9e9565426"
DIAGNOSTIC_TOKEN_SITES_SHA256 = "2e9b7a649a0f9771ed036bf3e7a9f2c64da6809d493f21a9720478f425cd3c5f"
ANNOTATION_SHA256 = "0d3183eb711b563776693b0c15f619348dc2507bb641147b5d1c30a08f804ea8"


def _json_bytes(row):
    return json.dumps(row, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _rows_sha256(rows):
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def token_sites_sha256(pairs: Iterable[tuple[str, str]]) -> str:
    """Fingerprint an iterable of (sample_token, physical_site_id) pairs."""
    return _rows_sha256(pairs)


def annotation_sha256(records: dict[str, AnnotationRecord]) -> str:
    """Hash the actual v6 scoring fields and decoded native label pixels."""
    rows = []
    for token, record in records.items():
        if not record.region_mask_path:
            raise ValueError(f"{token}: missing diagnostic mask path")
        with np.load(record.region_mask_path, allow_pickle=False) as mask:
            region = mask["region"]
        if region.dtype != np.int8 or region.shape != GRID_HW:
            raise ValueError(f"{token}: expected a native {GRID_HW} int8 diagnostic mask")
        rows.append((token, record.site_id, record.annotation_version, record.roi_status,
                     record.hand_status, record.object_status, record.motion_status,
                     record.motion_label, hashlib.sha256(region.tobytes(order="C")).hexdigest()))
    return _rows_sha256(rows)


def protocol_identity(diagnostic: bool = False) -> dict:
    """Reference fingerprints for the main test or its diagnostic population."""
    return {
        "id": "egodepth-v1-diagnostics" if diagnostic else "egodepth-v1-main-test",
        "test_anchors_sha256": DIAGNOSTIC_ANCHORS_SHA256 if diagnostic else TEST_ANCHORS_SHA256,
        "token_sites_sha256": DIAGNOSTIC_TOKEN_SITES_SHA256 if diagnostic else TOKEN_SITES_SHA256,
        "calibrations_sha256": DIAGNOSTIC_CALIBRATIONS_SHA256 if diagnostic else CALIBRATIONS_SHA256,
        "annotation_sha256": ANNOTATION_SHA256,
    }


def verify_paper_population(config: dict, samples: Sequence[CanonicalSample], *, diagnostic: bool = False) -> dict:
    """Validate the inputs actually selected for scoring against the released v1."""
    data = config["data"]
    annotation = config.get("annotation", {})
    identity = protocol_identity(diagnostic)
    count = N_DIAGNOSTIC if diagnostic else N_TEST
    if len(samples) != count:
        raise ValueError(f"{identity['id']} requires all {count} anchors; found {len(samples)}. "
                         "Use run.status=development for a custom population.")
    if data.get("data_version") != DATA_VERSION or data.get("mask_profile") != "release_declared_mask":
        raise ValueError(f"v1 requires {DATA_VERSION} with the release_declared_mask profile")
    if data.get("target_camera_id") != 2 or tuple(data.get("scoring_grid_hw") or ()) != GRID_HW:
        raise ValueError(f"v1 scores camera 2 on its native {GRID_HW} grid")
    if (annotation.get("version") != ANNOTATION_VERSION or annotation.get("status") != "frozen"
            or not annotation.get("dir")):
        raise ValueError(f"v1 requires the frozen {ANNOTATION_VERSION} diagnostic annotations")

    anchors, geometries = [], []
    for sample in samples:
        sample.validate()
        if (sample.split != "test" or sample.data_version != DATA_VERSION
                or sample.target_camera_id != 2 or tuple(sample.depth_shape_hw) != GRID_HW):
            raise ValueError(f"{sample.sample_token}: split, version, camera or grid differs from v1")
        anchors.append((sample.sample_token, sample.recording_id, sample.timestamp_ns, sample.physical_site_id))
        views = []
        for i in range(6):
            c = sample.calibrations[i]
            views.append([c.camera_id, c.width, c.height, c.fx, c.fy, c.cx, c.cy,
                          c.xi, c.alpha, c.T_body_cam])
        geometries.append((sample.sample_token, hashlib.sha256(_json_bytes(views)).hexdigest()))
    if _rows_sha256(anchors) != identity["test_anchors_sha256"]:
        raise ValueError("Test tokens, recordings, timestamps or physical-site assignments differ from v1")
    if _rows_sha256(geometries) != identity["calibrations_sha256"]:
        raise ValueError("The test camera intrinsics or rig calibration differ from v1")

    records = load_records(Path(annotation["dir"]), ANNOTATION_VERSION, annotation.get("asset_root"))
    if len(records) != N_DIAGNOSTIC or annotation_sha256(records) != ANNOTATION_SHA256:
        raise ValueError(f"v1 requires the {N_DIAGNOSTIC} frozen diagnostic records and masks")
    return identity
