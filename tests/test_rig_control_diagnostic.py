import json

import numpy as np
import pytest

from egodepth_eval.report import rig_control as rig
from egodepth_eval.score.prediction_io import PredictionContract


def test_all_arms_share_fit_support_and_reuse_full_scale_for_hand():
    gt = np.ones((6, 8), dtype=np.float32)
    valid = np.ones(gt.shape, dtype=bool)
    hand = valid.copy()
    hand[3:] = False
    gt[0, 0] = 2.5
    gt[0, 1] = np.nan
    invalid = valid.copy()
    invalid[0, 2] = False
    no_pose = np.where(hand, 2, 1).astype(np.float32)
    omega = gt.copy()
    omega[0, 3] = 0
    predictions = dict(zip(rig.METHODS, ((no_pose, valid), (gt, invalid), (omega, valid))))
    scores = rig.frame_statistics(gt, valid, predictions, {"hand": hand})
    n_hand, n_other = 20, 24
    scale = (n_hand * 2 + n_other) / (n_hand * 4 + n_other)
    full = scores[rig.METHODS[0], "full"]
    actual_hand = scores[rig.METHODS[0], "hand"]
    assert tuple(full[:2]) == (46, 44)
    assert tuple(actual_hand[:2]) == (22, 20)
    assert actual_hand[4] == pytest.approx(n_hand * abs(2 * scale - 1))
    assert full[4] == pytest.approx(actual_hand[4] + n_other * abs(scale - 1))
    assert scores[rig.METHODS[2], "full"][4] == 0
    invalid[2:] = False
    with pytest.raises(ValueError, match="fit failed"):
        rig.frame_statistics(gt, valid, predictions, {})


def test_site_weighting_and_paired_site_bootstrap():
    sums = {}
    for name in rig.METHODS:
        for region in rig.REGIONS:
            # Unequal counts and opposing site effects expose pixel pooling.
            for site, count, delta in (("a", 10, 0.2), ("b", 100, -0.1)):
                rel = 0.3 + (delta if name == rig.METHODS[1] else 0)
                sums[name, region, site] = [count, count, 0, 0, count * rel, 0]
    methods = rig._report(sums, ["b", "a"])
    assert methods[rig.METHODS[1]]["full"]["macro"]["aligned_abs_rel"] == pytest.approx(0.35)
    diff = rig._contrasts(methods, ["b", "a"])[0]
    assert diff["difference"] == "mapanything_6view_rig minus mapanything_6view"
    assert diff["estimate"] == pytest.approx(0.05)
    assert diff["ci95"] == pytest.approx([-0.1, 0.2])
    del sums[rig.METHODS[0], "object", "b"]
    with pytest.raises(ValueError, match="no common-support pixels"):
        rig._report(sums, ["b", "a"])


def test_rig_contracts_bind_pose_population_and_weights(tmp_path):
    identity = dict(token_set_id="test-tokens", split="test", data_version="v1", n_samples=2,
                    n_sites=2, mask_profile="release_declared_mask")
    for i, name in enumerate(rig.METHODS):
        contract = PredictionContract(
            method=name, checkpoint_content_id="weights", upstream_revision="revision",
            precision="bf16", input_views=tuple(range(6)), population_identity=identity,
            scale_contract="positive_scale" if i == 2 else "none (raw metric)",
            output_space="depth_up_to_scale" if i == 2 else "metric_depth_z",
            preprocessing={"pose_mode": "rig"} if i == 1 else {},
        )
        root = tmp_path / name
        root.mkdir()
        (root / "prediction_contract.json").write_text(json.dumps(contract.to_dict()))
    assert set(rig._read_contracts(tmp_path, identity)) == set(rig.METHODS)
    with pytest.raises(ValueError, match="token_set_id"):
        rig._read_contracts(tmp_path, {**identity, "token_set_id": "different"})
    path = tmp_path / rig.METHODS[1] / "prediction_contract.json"
    contract = PredictionContract.from_dict(json.loads(path.read_text()))
    contract.preprocessing["pose_mode"] = "none"
    path.write_text(json.dumps(contract.to_dict()))
    with pytest.raises(ValueError, match="expected pose_mode=rig"):
        rig._read_contracts(tmp_path, identity)
    contract.preprocessing["pose_mode"] = "rig"
    contract.checkpoint_content_id = "other-weights"
    path.write_text(json.dumps(contract.to_dict()))
    with pytest.raises(ValueError, match="checkpoint_content_id"):
        rig._read_contracts(tmp_path, identity)


def test_native_grid_requires_explicit_boolean_mask(tmp_path):
    path = tmp_path / "sample.npz"
    depth = np.ones((8, 8), np.float32)
    np.savez(path, depth_z=depth)
    with pytest.raises(ValueError, match="depth_z and valid"):
        rig._load_frame(path, depth.shape)
    np.savez(path, depth_z=depth, valid=depth.astype(np.uint8))
    with pytest.raises(ValueError, match="bool valid"):
        rig._load_frame(path, depth.shape)


def test_replacing_scored_pixels_changes_digest_under_same_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace

    grid = (6, 8)
    gt = np.ones(grid, np.float32)
    valid = np.ones(grid, bool)
    sites = ["a", "b"]
    samples = [SimpleNamespace(sample_token=s, physical_site_id=s) for s in sites]
    identity = dict(status="formal", split="test", formal_ready=True, token_set_id="fixture",
                    annotation_version="fixture", data_version="fixture", sites=sites)
    population = SimpleNamespace(samples=samples, identity=lambda: identity)
    frame = SimpleNamespace(reference_depth=gt, reference_valid=valid, grid_hw=grid)
    reader = SimpleNamespace(load=lambda sample, **kwargs: frame)
    records = {s: object() for s in sites}
    contracts = {method: "same-contract" for method in rig.METHODS}
    monkeypatch.setattr(rig, "GRID", grid)
    monkeypatch.setattr(rig, "_read_contracts", lambda root, identity: contracts)
    monkeypatch.setattr(rig, "region_masks_from_record", lambda record, shape: {"hand": valid, "object": valid})
    for method in rig.METHODS:
        (tmp_path / method).mkdir()
        for token in sites:
            np.savez(tmp_path / method / f"{token}.npz", depth_z=gt, valid=valid)
    before = rig.compare_rig_control(population, reader, records, tmp_path)
    edited = gt.copy()
    edited[0, 0] = 2
    np.savez(tmp_path / rig.METHODS[0] / "a.npz", depth_z=edited, valid=valid)
    after = rig.compare_rig_control(population, reader, records, tmp_path)
    assert before["input_identity"]["prediction_contracts"] == after["input_identity"]["prediction_contracts"]
    old = before["input_identity"]["scored_prediction_arrays_sha256"]
    new = after["input_identity"]["scored_prediction_arrays_sha256"]
    assert [name for name in old if old[name] != new[name]] == [rig.METHODS[0]]
    assert after["methods"][rig.METHODS[0]]["full"]["macro"]["aligned_abs_rel"] > 0
    assert before["support_pixels"]["full"]["a"] == {"common": 48, "reference": 48}
