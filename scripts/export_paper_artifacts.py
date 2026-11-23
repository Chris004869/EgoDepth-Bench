#!/usr/bin/env python3
"""Export Tables II/V/VI, Fig. 4's five panels, the support CSV and paired CIs.

    python scripts/export_paper_artifacts.py --config configs/local.yaml \
        --out_dir outputs/benchmark/paper_artifacts

Tables read result documents; paired-frame, common-support, motion, and aligned
diagnostics are computed from archived predictions and reference arrays. No
value is retyped. Missing results stay visibly missing: tables render `\\pending`,
panels render an explicit notice, and the support CSV records which sites had no
predictions. Development exports are stamped so they cannot be mistaken for a
formal result, and they are written to the run directory rather than over the
paper's own table files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from egodepth_eval.train.contract import adaptation_arms, method_inputs

from egodepth_eval.annotate.pipeline import load_records, region_masks_from_record  # noqa: E402
from egodepth_eval.annotate.schema import MotionLabel, summarize_statuses  # noqa: E402
from egodepth_eval.data.contract import CanonicalSample  # noqa: E402
from egodepth_eval.data.mask_profile import get_profile  # noqa: E402
from egodepth_eval.data.population import (  # noqa: E402
    FormalRunBlocked,
    PopulationRole,
    resolve_population,
)
from egodepth_eval.data.reader import SampleReader  # noqa: E402
from egodepth_eval.infer.benchmark_contract import read_formal_contract
from egodepth_eval.infer.legacy_adapter import E1_METHODS  # noqa: E402
from egodepth_eval.report.diagnostics import (  # noqa: E402
    compute_aligned_three_arm,
    compute_common_support_pair,
    compute_motion_contrast,
)
from egodepth_eval.report.rig_control import METHODS as RIG_METHODS, compare_rig_control
from egodepth_eval.report.figure4 import build_figure4  # noqa: E402
from egodepth_eval.report.paired_regions import paired_region_comparisons
from egodepth_eval.report.tables_e14 import (  # noqa: E402
    export_support_csv,
    generate_adaptation_table,
    generate_main_table,
    generate_views_table,
)
from egodepth_eval.score.accumulate import (  # noqa: E402
    ScoreAccumulator,
    accumulate_frame,
    prediction_domain,
    reference_domain,
)
from egodepth_eval.score.aggregate import (  # noqa: E402
    paired_site_bootstrap,
    standardize_by_strata,
)
from egodepth_eval.score.alignment import fit_shared_positive_scale  # noqa: E402
from egodepth_eval.score.prediction_io import (  # noqa: E402
    PredictionContract,
    file_content_digest,
    load_prediction,
)
from egodepth_eval.score.rays import angle_bin_labels, z_bin_labels  # noqa: E402
from egodepth_eval.score.scorer import (  # noqa: E402
    ROI_NAMES,
    MethodScorer,
    build_result_document,
    write_result,
)


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def collect_cost(config: dict) -> dict:
    """Per-method runtime and peak memory, from the prediction contracts."""
    cost: dict = {}
    predictions_root = Path(config["run"]["out_dir"]) / "predictions"
    for method_dir in sorted(predictions_root.glob("*")):
        path = method_dir / "prediction_contract.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run = payload.get("run", {}).get("cost", {})
        cost[method_dir.name] = {
            "median_ms": run.get("median_ms"),
            "mean_ms": run.get("mean_ms"),
            "peak_gpu_gb": run.get("peak_gpu_gb"),
            "n_timed": run.get("n_timed"),
            "device": payload.get("environment", {}).get("gpu_name"),
            "amp": payload.get("preprocessing", {}).get("model_input_hw"),
            "timing_scope": run.get("timing_scope"),
        }
    return cost


def build_population_and_reader(config: dict):
    """The scored population plus a reader, through the gate."""
    role = (
        PopulationRole.TEST
        if str(config.get("run", {}).get("status", "development")) == "formal"
        else PopulationRole.DEVELOPMENT
    )
    population = resolve_population(config, role)
    reader = SampleReader(
        [Path(root) for root in config["data"]["asset_roots"]],
        get_profile(config["data"]["mask_profile"]),
    )
    return population, reader


def load_annotation_records(config: dict) -> dict:
    annotation = config.get("annotation", {})
    if annotation.get("dir") and Path(annotation["dir"]).exists():
        return load_records(Path(annotation["dir"]), annotation.get("version"), annotation.get("asset_root"))
    return {}


def diagnostic_subset(population, records: dict):
    """Select diagnostic anchors for the same-frame whole-image and ROI comparisons."""
    from dataclasses import replace

    from egodepth_eval.data.paper_protocol import DIAGNOSTIC_TOKEN_SITES_SHA256, protocol_identity, token_sites_sha256

    samples = [sample for sample in population.samples if sample.sample_token in records]
    subset = replace(population, role=PopulationRole.ANNOTATION, samples=samples, notes={
        "derived_from_token_set_id": population.token_set_id(),
        "selection": "fixed diagnostic anchors, including unresolved labels",
        "n_parent_samples": len(population.samples),
    })
    if population.paper_protocol:
        digest = token_sites_sha256((s.sample_token, s.physical_site_id) for s in samples)
        if digest != DIAGNOSTIC_TOKEN_SITES_SHA256:
            raise FormalRunBlocked("Diagnostic membership differs from the frozen v1 selection")
        subset.paper_protocol = protocol_identity(diagnostic=True)
    return subset


def score_diagnostic_subset(
    config: dict, population, reader, records: dict, methods: list
) -> dict:
    """Re-score the annotated subset for every method, whole image and ROIs together.

    This is what makes Fig. 4(a) a same-frame comparison: one scoring pass over one
    token set produces the full-frame and the ROI numbers, so they cannot come from
    different populations.
    """
    predictions_root = Path(config["run"]["out_dir"]) / "predictions"
    output: dict = {}
    for method in methods:
        pred_dir = predictions_root / method
        if not pred_dir.exists():
            continue
        method_config = config.get("methods", {}).get(method, {})
        spec = (
            E1_METHODS.get(method_config.get("spec", method))
            if method_config.get("kind") == "legacy" else None
        )
        alignment = spec.alignment if spec else method_config.get("alignment", "none")
        scorer = MethodScorer(alignment_mode=alignment)

        sample_scores = []
        for sample in population.samples:
            loaded = reader.load(sample, load_images=False)
            path = pred_dir / f"{sample.sample_token}.npz"
            prediction = load_prediction(path) if path.exists() else None
            record = records.get(sample.sample_token)
            roi_masks = (
                region_masks_from_record(record, loaded.grid_hw) if record else None
            )
            sample_scores.append(scorer.score_sample(
                sample=sample,
                site_id=sample.physical_site_id or f"unmapped::{sample.recording_id}",
                reference_depth=loaded.reference_depth,
                reference_valid=loaded.reference_valid,
                prediction=prediction,
                reference_calibration=loaded.reference_calibration,
                roi_masks=roi_masks,
            ))
        contract_path = pred_dir / "prediction_contract.json"
        contract = (
            PredictionContract.from_dict(json.loads(contract_path.read_text()))
            if contract_path.exists() else PredictionContract(method=method)
        )
        payload = {
            "name": "diagnostic_subset_same_frames",
            **population.identity(),
        }
        output[method] = build_result_document(
            method, contract, sample_scores, payload,
            extra={'paired_regions': paired_region_comparisons(sample_scores)},
        )
    return output


def collect_examples(config: dict, methods: list, max_examples: int = 2) -> list:
    """Panel (e): the same frames across RGB+ROI, reference, and model columns."""
    # Qualitative examples obey the same split gate as all scored samples.
    samples = build_population_and_reader(config)[0].samples
    reader = SampleReader(
        [Path(root) for root in config["data"]["asset_roots"]],
        get_profile(config["data"]["mask_profile"]),
    )
    annotation = config.get("annotation", {})
    records = (
        load_records(Path(annotation["dir"]), annotation.get("version"), annotation.get("asset_root"))
        if annotation.get("dir") and Path(annotation["dir"]).exists() else {}
    )
    predictions_root = Path(config["run"]["out_dir"]) / "predictions"

    examples: list = []
    for sample in samples:
        if len(examples) >= max_examples:
            break
        record = records.get(sample.sample_token)
        if record is None or not record.roi_usable:
            continue
        loaded = reader.load(sample, cameras=[sample.target_camera_id])
        image = loaded.images[sample.target_camera_id]
        roi_masks = region_masks_from_record(record, loaded.grid_hw)

        overlay = image.copy().astype(np.float32)
        if roi_masks["object"].any():
            overlay[roi_masks["object"]] = (
                0.5 * overlay[roi_masks["object"]] + 0.5 * np.array([0, 220, 0])
            )
        if roi_masks["hand"].any():
            overlay[roi_masks["hand"]] = (
                0.5 * overlay[roi_masks["hand"]] + 0.5 * np.array([230, 0, 0])
            )

        # Reference with invalid pixels as NaN, so panel (e) draws them grey
        # rather than as a plausible depth.
        reference = np.where(
            reference_domain(loaded.reference_depth, loaded.reference_valid),
            loaded.reference_depth, np.nan,
        )
        columns = [
            {"kind": "rgb", "array": np.clip(overlay, 0, 255), "title": "RGB + ROI"},
            {"kind": "depth", "array": reference, "title": "reference Z"},
        ]
        for method in methods[:3]:
            path = predictions_root / method / f"{sample.sample_token}.npz"
            if not path.exists():
                columns.append({"array": None, "missing": f"no {method}\nprediction",
                                "title": method})
                continue
            prediction = load_prediction(path)
            depth = np.where(
                prediction_domain(prediction.depth_z, prediction.valid),
                prediction.depth_z, np.nan,
            )
            columns.append({"kind": "depth", "array": depth, "title": method})

        examples.append({
            "sample_token": sample.sample_token,
            "site_id": sample.physical_site_id,
            "columns": columns,
        })
    return examples


def load_run_results(config: dict, require_complete: bool, population) -> dict:
    """Read only the selected run files and verify results against their producer."""
    run_dir = Path(config["run"]["out_dir"])
    methods = config["methods"]
    if require_complete:
        paper_config = Path(__file__).resolve().parents[1] / "configs/release.yaml"
        paper = yaml.safe_load(paper_config.read_text())
        if config["run"]["status"] != "formal" or set(methods) != set(paper["methods"]):
            raise FormalRunBlocked("--require-complete needs the formal 24-configuration paper protocol")
        if config.get("scoring") != paper["scoring"] or adaptation_arms(config) != adaptation_arms(paper):
            raise FormalRunBlocked("Scoring settings or adaptation arms differ from the paper protocol")
        for name, expected in paper["methods"].items():
            actual = methods[name]
            if (actual["kind"] != expected["kind"] or actual.get("spec", name) != expected.get("spec", name)
                    or method_inputs(actual) != method_inputs(expected)
                    or actual.get("alignment", "none") != expected.get("alignment", "none")):
                raise FormalRunBlocked(f"{name}: configured model inputs or alignment differ from the paper")

    results = {}
    identity = population.identity()
    expected_tokens = set(population.sample_tokens) if require_complete else set()
    for name, method in methods.items():
        path = run_dir / "results" / f"{name}.json"
        if not path.exists():
            if require_complete:
                raise FormalRunBlocked(f"Missing result document: {path}")
            continue
        result = json.loads(path.read_text())
        if result.get("method") != name:
            raise ValueError(f"{path}: result names a different method")
        if config["run"]["status"] == "formal":
            produced = read_formal_contract(
                run_dir / "predictions" / name / "prediction_contract.json", name, method, identity,
            )
            recorded = result["prediction_contract"]
            if (recorded.get("contract_hash") != produced.identity_hash()
                    or PredictionContract.from_dict(recorded).identity_hash() != produced.identity_hash()):
                raise FormalRunBlocked(f"{name}: scored result and saved prediction contracts disagree")
            population_fields = ("token_set_id", "status", "data_version", "n_sites", "split", "annotation_version")
            if (result.get("n_samples") != len(population.samples)
                    or any(result["population"].get(key) != identity[key] for key in population_fields)):
                raise FormalRunBlocked(f"{name}: wrong result population or annotation version")
        if require_complete:
            if result.get("sample_status_counts") != {"scored": len(population.samples)}:
                raise FormalRunBlocked(f"{name}: incomplete scores on the formal population")
            predicted = {file.stem for file in (run_dir / "predictions" / name).glob("*.npz")}
            if predicted != expected_tokens:
                raise FormalRunBlocked(
                    f"{name}: saved predictions differ from the test population "
                    f"({len(expected_tokens - predicted)} missing, {len(predicted - expected_tokens)} unexpected)"
                )
            if method.get("kind") == "legacy":
                spec = E1_METHODS[method.get("spec", name)]
                if recorded["preprocessing"].get("runner_content_id") != file_content_digest(spec.runner_path):
                    raise FormalRunBlocked(f"{name}: result predates the current model wrapper; rerun before export")
        results[name] = result
    if not results:
        raise ValueError(f"No scored result documents in {run_dir / 'results'}. Run inference and scoring first.")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--figure_methods", nargs="*", default=None)
    parser.add_argument("--skip_aligned", action="store_true")
    parser.add_argument("--skip_examples", action="store_true")
    parser.add_argument('--no_plots', action='store_true', help='Export verified numerical artifacts for the manuscript plotting script')
    parser.add_argument('--require-complete',action='store_true',help='Reject absent/stale/partial formal result sets before export')
    parser.add_argument('--workers', type=int, default=4, help='Independent CPU diagnostic jobs; each owns a reader')
    parser.add_argument('--refresh-tables', action='store_true', help='Refresh complete tables and their diagnostics from saved predictions (requires --require-complete)')
    args = parser.parse_args()

    config = load_config(Path(args.config))
    status = str(config.get("run", {}).get("status", "development"))
    if args.refresh_tables and not args.require_complete:
        parser.error("--refresh-tables requires --require-complete")
    population, _ = build_population_and_reader(config)
    results = load_run_results(config, args.require_complete, population)
    if args.figure_methods and (set(args.figure_methods) - set(results)):
        parser.error("--figure_methods must name methods with scored results in this run")
    out_dir = Path(args.out_dir or Path(config["run"]["out_dir"]) / "paper_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] {len(results)} result documents: {sorted(results)}")

    cost = collect_cost(config)
    manifest: dict = {"status": status, "results_found": sorted(results), "artifacts": {}}

    manifest["artifacts"]["main_table"] = str(
        generate_main_table(results, out_dir / "main_table.tex", status)
    )
    manifest["artifacts"]["views_table"] = str(
        generate_views_table(results, out_dir / "views_table.tex", cost, status)
    )

    records = load_annotation_records(config)
    manifest["complete_final_result_set_verified"] = args.require_complete
    manifest["scored_population"] = population.identity()
    manifest["n_annotation_records"] = len(records)

    # Fit each arm using its own predictions on the shared domain.
    aligned_arms = adaptation_arms(config)
    bootstrap_config = config.get("scoring", {}).get("bootstrap", {})
    public_arm, _, ft_arm = adaptation_arms(config)
    figure_primary = [name for name in ('unidepth_v2', 'foundation_stereo', public_arm)
                      if name in results]
    figure_methods = args.figure_methods or list(dict.fromkeys(
        name for name in (*figure_primary, 'mapanything_6view', ft_arm) if name in results
    )) or sorted(results)[:2]
    manifest['figure_primary_methods'] = figure_primary
    subset = diagnostic_subset(population, records)
    pairs = [pair for pair in config.get("scoring", {}).get("common_support_pairs", [])
             if all(name in results for name in pair)]
    pred_root = Path(config['run']['out_dir']) / 'predictions'
    diagnostic_methods = set(figure_methods) | {name for pair in pairs for name in pair}
    if not args.skip_aligned and set(aligned_arms).issubset(results):
        diagnostic_methods.update(aligned_arms)
    manifest['diagnostic_input_identity'] = {
        'annotation_version': config['annotation']['version'],
        'prediction_contracts': {name: file_content_digest(pred_root / name / 'prediction_contract.json')
                                 for name in sorted(diagnostic_methods)},
    }
    boot = dict(n_boot=int(bootstrap_config.get('n_boot', 2000)),
                seed=int(bootstrap_config.get('seed', 20260912)))
    def fresh_reader():
        return SampleReader([Path(x) for x in config['data']['asset_roots']],
                            get_profile(config['data']['mask_profile']))
    jobs = {}
    if status == "formal" and set(RIG_METHODS).issubset(results):
        jobs["rig"] = lambda: compare_rig_control(population, fresh_reader(), records, pred_root)
    if not args.skip_aligned and set(aligned_arms).issubset(results):
        jobs['aligned'] = lambda: compute_aligned_three_arm(
            population, fresh_reader(), records, pred_root,
            arms=aligned_arms, shared_arms=aligned_arms, scale_pair=adaptation_arms(config)[:2])
    for name in figure_methods:
        jobs['subset:' + name] = lambda name=name: score_diagnostic_subset(
            config, subset, fresh_reader(), records, [name])[name]
        jobs['motion:' + name] = lambda name=name: compute_motion_contrast(
            subset, fresh_reader(), records, pred_root / name, method=name, **boot)
    for i, pair in enumerate(pairs):
        jobs['pair:' + str(i)] = lambda pair=pair: compute_common_support_pair(
            population, fresh_reader(), records, pred_root, pair[0], pair[1], **boot)
    def compute(item):
        key, fn = item
        result = fn()
        print('[export] diagnostic complete: ' + key, flush=True)
        return key, result
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        computed = dict(pool.map(compute, jobs.items()))
    if "rig" in computed:
        manifest["rig_common_support"] = computed["rig"]
        manifest["rig_common_support_prediction_contracts"] = computed["rig"]["input_identity"]["prediction_contracts"]
    aligned = computed.get('aligned', {})
    manifest["aligned_diagnostic"] = aligned
    aligned_rows = {}
    if aligned.get("status") == "ok":
        for arm in adaptation_arms(config):
            if arm in aligned:
                aligned_rows[arm] = aligned[arm]
        manifest["scalar_invariance_check"] = aligned.get("scalar_invariance_check")
    manifest["artifacts"]["adaptation_table"] = str(
        generate_adaptation_table(
            results, out_dir / "adaptation_table.tex", aligned_rows, status, methods=list(adaptation_arms(config)),
        )
    )
    manifest["artifacts"]["support_csv"] = str(
        export_support_csv(results, out_dir / "support_summary.csv")
    )

    # Panel (a) uses the same annotated frames for whole-image and ROI scores.
    subset_results = {name: computed['subset:' + name] for name in figure_methods}
    for method, document in subset_results.items():
        write_result(out_dir / "diagnostic_subset" / f"{method}.json", document)
    manifest["diagnostic_subset"] = {
        "n_samples": len(subset.samples),
        "token_set_id": subset.token_set_id(),
        "main_token_set_id": population.token_set_id(),
        "same_token_set": subset.token_set_id() == population.token_set_id(),
        "methods_scored": sorted(subset_results),
        "note": (
            "Fig. 4(a) reads these results only, so its whole-image and ROI numbers "
            "share one frame set. The main test-set whole-image numbers stay in the "
            "tables."
        ),
    }

    # The same predeclared representative methods get the same diagnostic analyses.
    motion_by_method = {name: computed['motion:' + name] for name in figure_methods}
    motion = {'status':'ok', 'comparisons':[comparison for payload in motion_by_method.values() for comparison in payload.get('comparisons',[])]}
    manifest["motion_contrast"] = motion
    manifest['motion_by_method'] = motion_by_method

    examples = [] if args.skip_examples else collect_examples(config, figure_methods)
    figure_record = {} if args.no_plots else build_figure4(
        subset_results or results, out_dir / "interaction_diagnostics.pdf",
        methods=figure_methods,
        z_labels=z_bin_labels(), angle_labels=angle_bin_labels(),
        heatmap_method=figure_methods[0] if figure_methods else None,
        motion_payload=motion if motion.get("status") == "ok" else None,
        examples=examples, status=status,
    )
    manifest["artifacts"]["figure4"] = figure_record.get("output_pdf")
    manifest["figure4_panels"] = {
        key: value.get("status") for key, value in figure_record.get("panels", {}).items()
    }
    manifest["figure4_missing_panels"] = figure_record.get("missing_panels", [])

    # Compare each prespecified pair on its common prediction support.
    paired = [computed['pair:' + str(i)] for i in range(len(pairs))]
    manifest["paired_bootstrap"] = paired

    annotation = config.get("annotation", {})
    if annotation.get("dir") and Path(annotation["dir"]).exists():
        records = load_records(Path(annotation["dir"]), annotation.get("version"), annotation.get("asset_root"))
        manifest["annotation_summary"] = summarize_statuses(list(records.values()))

    manifest["note"] = (
        "Tables use result documents; paired, motion, and aligned diagnostics "
        "recompute the stated comparisons from archived predictions and reference. "
        "Missing results appear as \\pending "
        "in the tables and as explicit notices in the figure panels; no value is "
        "simulated. Development artifacts are not written into the paper tree."
    )
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    print(json.dumps({
        "artifacts": manifest["artifacts"],
        "figure4_panels": manifest["figure4_panels"],
        "paired_bootstrap": [
            {
                "pair": p["pair"], "status": p.get("status"),
                "own_support_difference": (p.get("own_support_difference") or {}).get("difference"),
                "common_support_difference": (p.get("common_support_difference") or {}).get("difference"),
                "shared_share_of_a": (p.get("support_pixels") or {}).get("shared_share_of_a"),
            }
            for p in paired
        ],
        "motion_contrast": {
            "status": motion.get("status"),
            "n_comparisons": len(motion.get("comparisons", [])),
        },
        "scalar_invariance_check": manifest.get("scalar_invariance_check"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormalRunBlocked as exc:
        print(f"\nFORMAL RUN BLOCKED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
