#!/usr/bin/env python3
"""Verify and tabulate the 24 archived configurations without data or a GPU.

Recompute equal-site averages from archived per-site metrics. This does not
re-score pixels, rerun inference, or regenerate paired bootstrap intervals.
Only the Python standard library is required.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ERRORS = ("abs_rel", "mae_z_m", "mae_z_cm", "signed_err_m", "rmse_m", "delta_1")
COVERAGE = ("coverage", "delta_1_all")
COUNTS = ("n_reference", "n_scored", "n_frames")


def check_equal(expected, actual, label):
    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(f"Undefined-value mismatch: {label}")
    elif not math.isfinite(float(expected)) or not math.isfinite(float(actual)):
        raise ValueError(f"Nonfinite metric: {label}")
    elif not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Archive aggregate mismatch: {label}: {expected} != {actual}")


def verify_macro(stratum, label):
    """Recompute the published aggregation from stored per-site summaries."""
    if stratum.get("aggregation") != "site_macro_equal_weight":
        raise ValueError(f"Unexpected aggregation: {label}")
    sites = stratum["per_site"]
    applicable = [s for s in sites.values() if s["n_reference"] > 0]
    predicted = [s for s in applicable if s["n_scored"] > 0]
    metrics = stratum["metrics"]
    available = stratum["available_site_metrics"]
    expected_support = {
        "n_sites_total": len(applicable), "n_sites_with_predictions": len(predicted),
        "n_sites_no_predictions": len(applicable) - len(predicted),
        "n_sites_not_applicable": len(sites) - len(applicable),
    }
    if stratum["support"] != expected_support:
        raise ValueError(f"Site support mismatch: {label}")
    complete = bool(applicable) and len(predicted) == len(applicable)
    if stratum["complete_macro_defined"] != complete:
        raise ValueError(f"Complete-macro flag mismatch: {label}")
    checked = 0
    for metric in ERRORS:
        values = [s[metric] for s in predicted if s[metric] is not None]
        mean = statistics.fmean(values) if values else None
        expected = mean if complete and len(values) == len(applicable) else None
        check_equal(expected, metrics[metric], f"{label}/{metric}")
        check_equal(mean, available[metric], f"{label}/available/{metric}")
        checked += 2
    for metric in COVERAGE:
        values = [s[metric] if s[metric] is not None else 0.0 for s in applicable]
        expected = statistics.fmean(values) if values else None
        check_equal(expected, metrics[metric], f"{label}/{metric}")
        checked += 1
    for key, value in metrics["a_tau"].items():
        values = [s["a_tau"].get(key) or 0.0 for s in applicable]
        check_equal(statistics.fmean(values) if values else None, value, f"{label}/a_tau/{key}")
        checked += 1
    check_equal(metrics["a_tau"]["0.050"], metrics["a_5cm"], f"{label}/a_5cm")
    for key in COUNTS:
        group = sites.values() if key == "n_frames" else applicable
        if sum(s[key] for s in group) != metrics[key]:
            raise ValueError(f"Summed count mismatch: {label}/{key}")
        checked += 1
    return checked + 1


def verify_provenance(results):
    manifest = json.loads((results / "provenance.json").read_text())
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "results":
            raise ValueError("Unsafe archived artifact path")
        path = results.joinpath(*relative.parts[1:])
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != entry["public_sha256"]:
            raise ValueError(f"Public archive checksum mismatch: {relative}")
    return len(manifest["files"])


def row_from_result(result):
    contract = result["prediction_contract"]
    row = {
        "method": result["method"], "samples": result["n_samples"],
        "sites": result["population"]["n_sites"],
        "views": len(contract["input_views"]),
        "scale_contract": contract["scale_contract"],
    }
    for region in ("full", "hand", "object"):
        metrics = result["strata"][f"roi::{region}"]["metrics"]
        for metric in ("abs_rel", "mae_z_cm", "coverage", "a_5cm"):
            row[f"{region}_{metric}"] = metrics[metric]
    return row


def markdown(rows):
    lines = [
        "# Archived results", "",
        "Equal-weight site averages over eight held-out-from-fine-tuning test sites. "
        "Full-image metrics use 6,574 anchors. Hand/object metrics use accepted "
        "v6 AI diagnostic regions, whose support is recorded in the JSON archives.", "",
        "Rows have different input and scale contracts. Coverage accompanies errors; "
        "relative-depth rows aligned using reference depth are diagnostic upper bounds.", "",
        "| Configuration | Views | Scale contract | AbsRel | MAE (cm) | Coverage (%) | Hand MAE (cm) | Object MAE (cm) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    def fmt(value, scale=1):
        return "—" if value is None else f"{value * scale:.4f}"
    for row in rows:
        scale = row["scale_contract"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['method']} | {row['views']} | {scale} | "
            f"{fmt(row['full_abs_rel'])} | {fmt(row['full_mae_z_cm'])} | "
            f"{fmt(row['full_coverage'], 100)} | {fmt(row['hand_mae_z_cm'])} | "
            f"{fmt(row['object_mae_z_cm'])} |"
        )
    return "\n".join(lines) + "\n"


def write_table(output, name, rows):
    with (output / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"{name}.md").write_text(markdown(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/archived_results")
    args = parser.parse_args()
    files_checked = verify_provenance(args.results)
    files = sorted((args.results / "official").glob("*.json"))
    if len(files) != 24:
        raise ValueError(f"Expected 24 configurations, found {len(files)}")
    rows, checks, strata_count = [], 0, 0
    for path in files:
        result = json.loads(path.read_text())
        population = result["population"]
        if result["n_samples"] != 6574 or population["n_sites"] != 8:
            raise ValueError(f"Unexpected formal population: {path.name}")
        if population["annotation_version"] != "ai-rgb-reviewed-v6":
            raise ValueError(f"Unexpected annotation version: {path.name}")
        for key, stratum in result["strata"].items():
            if "per_site" in stratum:
                checks += verify_macro(stratum, f"{result['method']}/{key}")
                strata_count += 1
        rows.append(row_from_result(result))
    if len({row["method"] for row in rows}) != 24:
        raise ValueError("Duplicate archived method names")
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output, "all_configurations", rows)
    lookup = {row["method"]: row for row in rows}
    write_table(args.output, "views", [lookup[name] for name in (
        "mapanything_1view", "mapanything_2view", "mapanything_6view", "mapanything_6view_rig")])
    write_table(args.output, "adaptation", [lookup[name] for name in (
        "mapanything_6view_rig", "mapanything_rig_scale_only", "mapanything_rig_ft")])
    report = {
        "status": "passed", "configurations": len(rows), "artifact_checksums_verified": files_checked,
        "strata_recomputed_from_site_summaries": strata_count, "scalar_checks": checks,
        "scope": "archived equal-site aggregates; no pixel rescoring, model inference, or CI recomputation",
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
