"""Paired shape diagnostic for the six-view no-pose, rig and relative-depth arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from egodepth_eval.annotate.pipeline import region_masks_from_record
from egodepth_eval.score.accumulate import prediction_domain, reference_domain
from egodepth_eval.score.external import scoring_source_identity
from egodepth_eval.score.alignment import fit_alignment
from egodepth_eval.score.prediction_io import PREDICTION_SCHEMA, PredictionContract, file_content_digest

METHODS = ("mapanything_6view", "mapanything_6view_rig", "vggt_omega")
REGIONS = ("full", "hand", "object")
GRID = (416, 512)
N_BOOT, SEED = 2000, 20260912


def _read_contracts(root, identity):
    contracts, digests = {}, {}
    for name, output_space in zip(METHODS, ("metric_depth_z", "metric_depth_z", "depth_up_to_scale")):
        path = root / name / "prediction_contract.json"
        payload = json.loads(path.read_text())
        contract = PredictionContract.from_dict(payload)
        if payload.get("schema") != PREDICTION_SCHEMA or payload.get("contract_hash") != contract.identity_hash():
            raise ValueError(f"{name}: invalid prediction contract schema or digest")
        if (contract.method != name or contract.output_space != output_space
                or contract.input_views != tuple(range(6)) or contract.target_camera_id != 2
                or contract.scoring_grid_hw != GRID):
            raise ValueError(f"{name}: expected the named six-view arm on the native camera-2 grid")
        for key in ("token_set_id", "split", "data_version", "n_samples", "n_sites", "mask_profile"):
            if contract.population_identity.get(key) != identity[key]:
                raise ValueError(f"{name}: prediction contract disagrees with test population on {key}")
        contracts[name] = contract
        digests[name] = file_content_digest(path)
    no_pose, rig = (contracts[name] for name in METHODS[:2])
    for contract, expected in ((no_pose, "none"), (rig, "rig")):
        if contract.preprocessing.get("pose_mode", "none") != expected:
            raise ValueError(f"{contract.method}: expected pose_mode={expected}")
    if any(c.scale_contract != "none (raw metric)" for c in (no_pose, rig)):
        raise ValueError("MapAnything rig control requires two uncalibrated metric arms")
    if contracts[METHODS[2]].scale_contract != "positive_scale":
        raise ValueError("VGGT-Omega requires its relative positive-scale contract")
    for key in ("checkpoint_content_id", "upstream_revision", "precision", "support_rule"):
        if not getattr(no_pose, key) or getattr(no_pose, key) != getattr(rig, key):
            raise ValueError(f"MapAnything arms must share {key}")
    preprocessing = [{k: v for k, v in c.preprocessing.items() if k not in ("pose_mode", "pose_convention")}
                     for c in (no_pose, rig)]
    if preprocessing[0] != preprocessing[1]:
        raise ValueError("MapAnything arms must share preprocessing outside pose conditioning")
    return digests


def _load_frame(path, grid):
    with np.load(path, allow_pickle=False) as data:
        if not {"depth_z", "valid"}.issubset(data.files):
            raise ValueError(f"{path}: depth_z and valid are required")
        z, valid = data["depth_z"], data["valid"]
    if z.shape != grid or valid.shape != grid or z.dtype != np.float32 or valid.dtype != np.bool_:
        raise ValueError(f"{path}: expected float32 depth_z and bool valid on grid {grid}")
    return z, valid


def frame_statistics(gt, valid, predictions, regions):
    """Return error sums on the same pixels for every arm, with one scale per arm."""
    reference = reference_domain(gt, valid)
    shared = reference.copy()
    for depth, support in predictions.values():
        shared &= prediction_domain(depth, support)

    metrics = {}
    for name, (depth, _) in predictions.items():
        fit = fit_alignment(depth, gt, shared, mode="positive_scale")
        if not fit.ok:
            raise ValueError(f"{name}: common-support scale fit failed ({fit.reason})")
        scale = fit.coefficients["scale"]
        for region in REGIONS:
            if region == "full":
                eligible, selected = reference, shared
            elif region in regions:
                eligible = reference & regions[region]
                selected = shared & regions[region]
            else:
                continue
            n_reference = int(eligible.sum())
            if not n_reference:
                continue
            y, p = gt[selected].astype(np.float64), depth[selected].astype(np.float64)
            raw, aligned = np.abs(p - y), np.abs(scale * p - y)
            metrics[name, region] = np.array([n_reference, y.size, (raw / y).sum(), raw.sum(),
                                              (aligned / y).sum(), aligned.sum()])
    return metrics


def _report(sums, sites):
    report = {name: {} for name in METHODS}
    for name in METHODS:
        for region in REGIONS:
            per_site = {}
            for site in sorted(sites):
                values = sums.get((name, region, site))
                if values is None or values[1] == 0:
                    raise ValueError(f"{name}/{region}/{site}: no common-support pixels")
                n_ref, n, raw_rel, raw_abs, aligned_rel, aligned_abs = values
                per_site[site] = {
                    "coverage": float(n / n_ref), "raw_abs_rel": float(raw_rel / n),
                    "raw_mae_cm": float(100 * raw_abs / n), "aligned_abs_rel": float(aligned_rel / n),
                    "aligned_mae_cm": float(100 * aligned_abs / n),
                }
            report[name][region] = {
                "n_sites": len(per_site), "per_site": per_site,
                "macro": {k: float(np.mean([s[k] for s in per_site.values()]))
                          for k in next(iter(per_site.values()))},
            }
    return report


def _contrasts(methods, sites):
    sites = sorted(sites)
    draws = np.random.default_rng(SEED).integers(len(sites), size=(N_BOOT, len(sites)))
    contrasts = []
    for a, b in ((METHODS[1], METHODS[0]), (METHODS[2], METHODS[1])):
        for region in REGIONS:
            for key in ("aligned_abs_rel", "aligned_mae_cm"):
                differences = np.array([methods[a][region]["per_site"][s][key]
                                        - methods[b][region]["per_site"][s][key] for s in sites])
                contrasts.append({
                    "difference": f"{a} minus {b}", "region": region, "metric": key,
                    "estimate": float(differences.mean()), "n_sites": len(sites),
                    "ci95": np.quantile(differences[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                })
    return contrasts


def compare_rig_control(population, reader, records, predictions_dir):
    """Run the fixed E3 comparison and return archive-compatible statistics."""
    identity = population.identity()
    if identity["status"] != "formal" or identity["split"] != "test" or not identity["formal_ready"]:
        raise ValueError("Rig control requires the formal held-out test population")
    root = Path(predictions_dir)
    contracts = _read_contracts(root, identity)
    digests = {name: hashlib.sha256() for name in METHODS}
    reference_digest = hashlib.sha256()
    sums, seen = {}, set()
    for sample in population.samples:
        token, site = sample.sample_token, sample.physical_site_id
        if not token or Path(token).name != token or token in (".", "..") or token in seen:
            raise ValueError(f"Duplicate or invalid sample token {token!r}")
        seen.add(token)
        loaded = reader.load(sample, load_images=False, cameras=())
        if loaded.grid_hw != GRID:
            raise ValueError(f"{token}: expected the native camera-2 grid {GRID}")
        reference_digest.update(token.encode() + b"\0")
        reference_digest.update(np.ascontiguousarray(loaded.reference_depth, dtype="<f4"))
        reference_digest.update(np.ascontiguousarray(loaded.reference_valid, dtype=np.bool_))
        predictions = {}
        for name in METHODS:
            depth, support = _load_frame(root / name / f"{token}.npz", GRID)
            # Bind the scored arrays, including changes that preserve a contract file.
            digests[name].update(token.encode() + b"\0")
            digests[name].update(np.ascontiguousarray(depth, dtype="<f4"))
            digests[name].update(np.ascontiguousarray(support))
            predictions[name] = depth, support
        rois = region_masks_from_record(records[token], GRID) if token in records else {}
        try:
            stats = frame_statistics(loaded.reference_depth, loaded.reference_valid, predictions, rois)
        except ValueError as exc:
            raise ValueError(f"{token}: {exc}") from exc
        for (name, region), values in stats.items():
            key = (name, region, site)
            if key not in sums:
                sums[key] = np.zeros(6, dtype=np.float64)
            sums[key] += values

    methods = _report(sums, identity["sites"])
    support = {region: {} for region in REGIONS}
    for region in REGIONS:
        for site in identity["sites"]:
            eligible, shared = sums[METHODS[0], region, site][:2]
            support[region][site] = {"reference": int(eligible), "common": int(shared)}
    source = scoring_source_identity()
    source["files"]["report/rig_control.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "status": "passed", "n_samples": len(seen), "reference": "native Z in [0.25,2.0] m",
        "support": "G intersect all three prediction supports",
        "alignment": "Per-frame positive L2 Z scale fitted on full common support and reused on ROIs.",
        "omega_raw_is_not_a_metric_benchmark": True,
        "methods": methods, "support_pixels": support, "contrasts": _contrasts(methods, identity["sites"]),
        "bootstrap": {"unit": "paired site", "n_boot": N_BOOT, "seed": SEED},
        "scoring_source_identity": source,
        "input_identity": {
            **{k: identity[k] for k in ("data_version", "token_set_id", "sites", "annotation_version")},
            **({"paper_protocol": identity["paper_protocol"]} if "paper_protocol" in identity else {}),
            "prediction_contracts": contracts,
            "reference_arrays_sha256": reference_digest.hexdigest(),
            "scored_prediction_arrays_sha256": {name: h.hexdigest() for name, h in digests.items()},
        },
    }
