"""Read frozen diagnostic annotations; no annotation models are required.

The release preserves accepted/unknown states and per-region eligibility. AI
annotation production is outside this public benchmark consumer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
from egodepth_eval.annotate.schema import (AnnotationRecord, FieldStatus, MotionLabel,
    LABEL_BACKGROUND, LABEL_HAND, LABEL_OBJECT, LABEL_UNKNOWN)

VALID_REGION_LABELS = (LABEL_UNKNOWN, LABEL_HAND, LABEL_OBJECT, LABEL_BACKGROUND)


def resample_label_map(region: np.ndarray, grid_hw: tuple) -> np.ndarray:
    """Nearest-neighbour resample of an integer LABEL map, preserving every class.

    Resampling a boolean slice and re-decoding it as labels silently permutes the
    classes (see the caller). Nearest neighbour on the integer array cannot
    invent a label, and the post-condition below refuses any output holding a
    label the input did not have -- which is what an interpolating resize would
    produce at a class boundary.
    """
    import cv2

    region = np.asarray(region)
    grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
    if tuple(region.shape) == grid_hw:
        return region.astype(np.int8, copy=False)

    source_labels = set(np.unique(region).tolist())
    unexpected = source_labels - set(VALID_REGION_LABELS)
    if unexpected:
        raise ValueError(
            f"label map holds unknown values {sorted(unexpected)}; expected a "
            f"subset of {sorted(VALID_REGION_LABELS)}"
        )

    # int32 for the resize: OpenCV does not accept int8 here, and float would
    # allow interpolation to produce fractional labels.
    resampled = cv2.resize(
        region.astype(np.int32), (grid_hw[1], grid_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    produced = set(np.unique(resampled).tolist())
    invented = produced - source_labels
    if invented:
        raise ValueError(
            f"resampling invented labels {sorted(invented)} not present in the "
            f"source {sorted(source_labels)}: the interpolation is not nearest"
        )
    return resampled.astype(np.int8)


def region_masks_from_record(
    record: AnnotationRecord, grid_hw: tuple
) -> dict:
    """ROI masks for the scorer, from a frozen annotation.

    `unknown` is returned as its own ROI. `active_object` is the object region on
    frames whose motion label is an accepted active manipulation -- so an
    unresolved motion label yields an empty active-object ROI rather than
    quietly counting as inactive.
    """
    grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
    if not record.region_mask_path or not Path(record.region_mask_path).exists():
        empty = np.zeros(grid_hw, dtype=bool)
        return {
            "full": np.ones(grid_hw, dtype=bool),
            "hand": empty, "object": empty, "background": empty,
            "unknown": np.ones(grid_hw, dtype=bool),
            "active_object": empty,
        }

    with np.load(record.region_mask_path) as payload:
        region = payload["region"]
    if tuple(region.shape) != grid_hw:
        # The whole integer label map is resampled, preserving every class.
        # The previous code resampled only `region == LABEL_HAND` and then decoded
        # the resulting 0/1 array with HAND=0 and OBJECT=1, which inverted the
        # labels: hand became object, everything else became hand, and
        # unknown/background were lost entirely.
        region = resample_label_map(region, grid_hw)

    hand = region == LABEL_HAND
    obj = region == LABEL_OBJECT
    background = region == LABEL_BACKGROUND
    unknown = region == LABEL_UNKNOWN

    # Gate each region on its own status. An unaccepted region's pixels return to
    # `unknown` -- the un-annotated domain -- rather than being scored or being
    # relabelled background.
    frame_accepted = record.roi_status == FieldStatus.ACCEPTED_AI.value
    if not (frame_accepted or record.hand_usable):
        unknown = unknown | hand
        hand = np.zeros(grid_hw, dtype=bool)
    if not (frame_accepted or record.object_usable):
        unknown = unknown | obj
        obj = np.zeros(grid_hw, dtype=bool)
    if not frame_accepted:
        # Background is only trustworthy when the whole frame was accepted: it is
        # defined as "nothing was missed here", which a partial acceptance does
        # not establish.
        unknown = unknown | background
        background = np.zeros(grid_hw, dtype=bool)

    active = (
        obj
        if record.motion_usable
        and record.motion_label == MotionLabel.ACTIVE_MANIPULATION.value
        else np.zeros(grid_hw, dtype=bool)
    )
    return {
        "full": np.ones(grid_hw, dtype=bool),
        "hand": hand, "object": obj, "background": background,
        "unknown": unknown, "active_object": active,
    }


def save_record(record: AnnotationRecord, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.sample_token}.json"
    path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return path


def load_records(out_dir: Path, annotation_version: Optional[str] = None,
                 asset_root: Optional[Path] = None) -> dict:
    """Load frozen records, optionally only those matching an annotation version.

    A version mismatch is skipped rather than reused: a prompt change means the
    labels came from a different process, and mixing versions in one diagnostic
    would make the acceptance statistics meaningless.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return {}
    records: dict = {}
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # The directory also holds annotation_summary.json and any other
        # bookkeeping file. Identify records by their required fields rather than
        # by filename, so a new sidecar file cannot break the loader.
        if not isinstance(payload, dict) or "sample_token" not in payload:
            continue
        payload.setdefault("site_id", None)
        payload.setdefault("annotation_version", "unknown")
        record = AnnotationRecord.from_dict(payload)
        if annotation_version and record.annotation_version != annotation_version:
            continue
        if asset_root is not None and record.region_mask_path:
            root = Path(asset_root).resolve()
            relative = Path(record.region_mask_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Portable annotation mask must be release-relative")
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Annotation mask escapes the release root")
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            record.region_mask_path = str(resolved)
        records[record.sample_token] = record
    return records
