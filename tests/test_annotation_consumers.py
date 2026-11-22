"""Frozen partial AI annotations keep uncertainty; no annotation models run."""
import json

import numpy as np

from egodepth_eval.annotate.pipeline import load_records, region_masks_from_record, resample_label_map
from egodepth_eval.annotate.schema import AnnotationRecord


def test_unaccepted_objects_and_background_return_to_unknown(tmp_path):
    labels = np.array([[0, 1], [2, -1]], dtype=np.int8)
    path = tmp_path / "regions.npz"
    np.savez(path, region=labels)
    record = AnnotationRecord("sample", "site", "v6", region_mask_path=str(path),
                              hand_status="accepted_ai", object_status="uncertain")
    masks = region_masks_from_record(record, (2, 2))
    assert masks["hand"].sum() == 1
    assert not masks["object"].any() and not masks["background"].any()
    assert masks["unknown"].sum() == 3
    assert not masks["active_object"].any()


def test_resize_preserves_integer_semantic_classes():
    labels = np.array([[0, 1], [2, -1]], dtype=np.int8)
    resized = resample_label_map(labels, (4, 4))
    np.testing.assert_array_equal(resized, labels.repeat(2, axis=0).repeat(2, axis=1))


def test_version_and_record_identity_are_checked_before_consumption(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"n_samples": 2}))
    for token, version in (("old", "v4"), ("current", "v6")):
        (tmp_path / f"{token}.json").write_text(json.dumps(AnnotationRecord(token, "site", version).to_dict()))
    assert list(load_records(tmp_path, "v6")) == ["current"]
