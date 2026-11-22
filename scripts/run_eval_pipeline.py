#!/usr/bin/env python3
"""Single entry point: preflight, inference, scoring, and E1-E4 exports.

    # what is missing before a formal run can start
    python scripts/run_eval_pipeline.py preflight --config configs/local.yaml

    # real inference on the official held-out test protocol
    python scripts/run_eval_pipeline.py infer --config configs/local.yaml \
        --method mapanything_6view

    # score whatever predictions exist
    python scripts/run_eval_pipeline.py score --config configs/local.yaml

    # E3 view sweep (1/2/6) in one command
    python scripts/run_eval_pipeline.py views --config configs/local.yaml

The generated public configuration writes under `outputs/benchmark/`.
`preflight` lists exactly which inputs a formal run still needs and exits
non-zero when the config claims formal status without them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from egodepth_eval.annotate.pipeline import load_records, region_masks_from_record  # noqa: E402
from egodepth_eval.data.contract import CanonicalSample  # noqa: E402
from egodepth_eval.data.mask_profile import (  # noqa: E402
    PROFILES,
    get_profile,
    is_formal_profile,
)
from egodepth_eval.data.reader import SampleReader  # noqa: E402
from egodepth_eval.infer.benchmark_contract import read_formal_contract
from egodepth_eval.data.manifest_store import ManifestStore  # noqa: E402
from egodepth_eval.data.population import (  # noqa: E402
    FormalRunBlocked,
    PopulationRole,
    resolve_population,
)
from egodepth_eval.data.site_map import SiteMapping, require_formal_sites  # noqa: E402
from egodepth_eval.score.prediction_io import (  # noqa: E402
    PredictionContract,
    check_resumable,
    code_identity,
    default_environment,
    file_content_digest,
    load_prediction,
    save_prediction,
    verify_prediction_artifact,
    write_run_manifest,
)
from egodepth_eval.infer.legacy_adapter import (  # noqa: E402
    E1_METHODS,
    build_legacy_contract,
    method_status_table,
    run_legacy_method,
)
from egodepth_eval.score.scorer import (  # noqa: E402
    MethodScorer,
    build_result_document,
    write_result,
)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: config must be a mapping")
    from egodepth_eval.infer.assets import configure_paths
    configure_paths(config)
    return config


def _population_line(population) -> str:
    """A one-line population summary for the console.

    The full identity -- including the site list and the mapping summary -- goes into
    the result document and the prediction contract; printing all of it per method
    buried the actual scores.
    """
    identity = population.identity()
    line = (
        f"role={identity['role']} split={identity['split']} "
        f"status={identity['status']} n={identity['n_samples']} "
        f"sites={identity['n_sites']} tokens={identity['token_set_id']} "
        f"manifest={identity['manifest_content_id']}"
    )
    if identity["blockers"]:
        line += f" | {len(identity['blockers'])} formal blocker(s): "
        line += "; ".join(b.split(":")[0] for b in identity["blockers"])
    return line


def resolve_eval_population(config: dict, limit=None, require_annotations: bool = False):
    """Samples for scoring, through the gate.

    Formal scoring reads the TEST split only. The previous `load_samples()` read the
    whole manifest and no caller filtered, so a manifest holding train/val/test was
    scored in full while the result claimed to be a test-set number.
    """
    role = (
        PopulationRole.TEST
        if str(config.get("run", {}).get("status", "development")) == "formal"
        else PopulationRole.DEVELOPMENT
    )
    return resolve_population(
        config, role, limit=limit, require_annotations=require_annotations
    )


def build_reader(config: dict) -> SampleReader:
    roots = [Path(root) for root in config["data"]["asset_roots"]]
    return SampleReader(roots, get_profile(config["data"]["mask_profile"]))


# ------------------------------------------------------------------ preflight

def scale_is_measured(method):
    """Require a recorded train fit for both rays-only and rig adaptation."""
    from egodepth_eval.train.contract import method_inputs, validate_fitted_scalar

    try:
        path = Path(method["calibration_record"])
        record = json.loads(path.read_text())
        validate_fitted_scalar(record, float(method["scale_factor"]), method_inputs(method))
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def command_preflight(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    selected_method = getattr(args, "method", None)
    methods = config.get("methods", {})
    selection = [selected_method] if selected_method else getattr(args, "methods", None)
    selected_names = list(methods) if selection is None else selection
    unknown = sorted(set(selected_names) - set(methods))
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
    intended = str(config.get("run", {}).get("status", "development"))
    report: dict = {
        "config": str(args.config),
        "intended_status": intended,
        "selected_method": selected_method,
        "selected_methods": sorted(selected_names),
        "checks": [],
        "missing_for_formal_run": [],
    }

    def check(name: str, ok: bool, detail: str, formal_blocker: bool = False):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok and formal_blocker:
            report["missing_for_formal_run"].append({"name": name, "detail": detail})

    data = config.get("data", {})
    samples_path = Path(data.get("samples_jsonl", ""))
    have_samples = samples_path.exists()
    n_samples = 0
    samples: list = []
    if have_samples:
        # Count and sample WITHOUT decoding the whole manifest: the 2026-09-12
        # release is 5.4 GB / 1,002,000 rows, and preflight is supposed to be the
        # cheap check that runs before anything else.
        index_dir = samples_path.parent
        if samples_path.name == 'samples.jsonl' and (index_dir / "index.sqlite").exists():
            store = ManifestStore(index_dir)
            n_samples = len(store)
            probe_tokens = store.tokens()[:200]
            samples = [store.get(token) for token in probe_tokens]
            notes_source = f"sqlite index ({len(samples)} of {n_samples} probed)"
        else:
            with samples_path.open() as stream:
                n_samples = sum(1 for line in stream if line.strip())
            with samples_path.open() as stream:
                samples = [
                    CanonicalSample.from_dict(json.loads(line))
                    for line in itertools.islice(
                        (l for l in stream if l.strip()), 200
                    )
                ]
            notes_source = f"jsonl scan ({len(samples)} of {n_samples} probed)"
        report["sample_probe"] = notes_source
    check("samples_manifest", have_samples,
          f"{samples_path} ({n_samples} samples)" if have_samples else f"missing: {samples_path}",
          formal_blocker=True)

    roots = [Path(root) for root in data.get("asset_roots", [])]
    # A privileged source root raises PermissionError rather than returning False.
    # That is a legitimate configuration -- a readable cache shadows it -- so it is
    # reported as unreadable instead of crashing preflight.
    readable, unreadable = [], []
    for root in roots:
        try:
            (readable if root.exists() else unreadable).append(root)
        except PermissionError:
            unreadable.append(root)
    check("asset_roots", bool(readable),
          f"{len(readable)}/{len(roots)} readable: {[str(r) for r in readable]}"
          + (f"; not readable by this user: {[str(r) for r in unreadable]}"
             if unreadable else ""),
          formal_blocker=True)

    profile_name = data.get("mask_profile")
    check("mask_profile", profile_name in PROFILES,
          f"{profile_name!r} (known: {sorted(PROFILES)})", formal_blocker=True)
    if profile_name in PROFILES:
        profile = get_profile(profile_name)
        # The test is validity_source, NOT one_is_valid: a batch whose stored mask
        # is inverted still has a real mask, and the direction is pinned by the
        # profile. What disqualifies a profile is deriving validity from the depth.
        check("mask_profile_reads_a_real_mask", is_formal_profile(profile_name),
              f"validity_source={profile.validity_source}, "
              f"one_is_valid={profile.one_is_valid}: {profile.notes[:120]}",
              formal_blocker=True)

    # Site mapping is the hard gate on any unseen-site claim.
    site_map_path = data.get("site_map_path")
    mapping = None
    if site_map_path and Path(site_map_path).exists():
        mapping = SiteMapping.load(Path(site_map_path))
        problems = mapping.validate(strict_site_count=(intended == "formal"))
        check("site_map_loaded", True,
              f"{len(mapping.sites)} sites, status={mapping.status}")
        check("site_map_usable_for_formal", not problems,
              "; ".join(problems)[:400] if problems else "no structural problems",
              formal_blocker=True)
        check("site_map_frozen", mapping.status == "frozen",
              f"status={mapping.status}", formal_blocker=True)
    else:
        check("site_map_loaded", False,
              f"no site map at {site_map_path!r}: recording IDs are not physical "
              "sites and cannot be substituted", formal_blocker=True)

    if samples and mapping is not None:
        # Coverage is checked exhaustively against the MAPPING (1,218 recordings),
        # which is where site and split actually come from; the sample probe only
        # confirms the manifest's recordings appear in it.
        recordings = {s.recording_id for s in samples}
        unmapped = sorted(recordings - set(mapping.recording_to_site))
        check("probed_recordings_are_mapped", not unmapped,
              f"{len(unmapped)} of {len(recordings)} probed recordings are not in the "
              f"site map, e.g. {unmapped[:3]}", formal_blocker=True)
        splits = sorted({
            str(mapping.site_to_split.get(mapping.recording_to_site.get(r)))
            for r in recordings
        })
        check("mapping_assigns_splits", "None" not in splits,
              f"splits reachable from the probed recordings: {splits}",
              formal_blocker=True)
    elif samples:
        check("probed_recordings_are_mapped", False,
              "no site map, so no sample can carry a site or a split",
              formal_blocker=True)

    annotation_dir = config.get("annotation", {}).get("dir")
    if annotation_dir and Path(annotation_dir).exists():
        version = config.get("annotation", {}).get("version")
        records = load_records(Path(annotation_dir), version, config["annotation"].get("asset_root"))
        accepted = sum(1 for r in records.values() if r.roi_usable)
        motion = sum(1 for r in records.values() if r.motion_usable)
        check("annotations_present", bool(records),
              f"{len(records)} records at version {version!r}; "
              f"{accepted} with usable ROI, {motion} with usable motion",
              formal_blocker=True)
        check("annotation_frozen", str(config["annotation"].get("status")) == "frozen",
              f"status={config['annotation'].get('status')!r}", formal_blocker=True)
    else:
        check("annotations_present", False,
              f"no annotation directory at {annotation_dir!r}", formal_blocker=True)

    check("methods_configured", bool(methods), f"{len(methods)} methods: {sorted(methods)}")

    # Every E1 method that HAS a runner should be in a formal config; a silently
    # short method list produces a main table missing rows, with nothing to signal it.
    declared_runnable = {
        n for n, spec in E1_METHODS.items() if spec.availability()["runner_exists"]
    }
    absent = sorted(declared_runnable - set(methods))
    check("all_runnable_methods_configured", not absent,
          f"{len(declared_runnable)} methods have runners; missing from this config: "
          f"{absent}" if absent else
          f"all {len(declared_runnable)} methods with runners are configured")

    # E4 arms whose defining parameter is still a placeholder. Each would RUN and
    # score, producing a row numerically identical to the public arm.
    checked_methods = {name: methods[name] for name in selected_names}
    for name, method in sorted(checked_methods.items()):
        if name.endswith("scale_only"):
            ok = scale_is_measured(method)
            detail = (f"scale_factor={method.get('scale_factor')}, "
                      f"calibration_record={method.get('calibration_record')}")
            if not ok:
                detail += " (missing or inconsistent train fit)"
            check(f"e4_arm_filled::{name}", ok, detail, formal_blocker=True)
        if name.endswith("_ft"):
            state = method.get("finetuned_state_dict")
            state_exists = bool(state) and Path(state).is_file()
            check(f"e4_arm_filled::{name}", state_exists,
                  (str(state) if state_exists else f"missing fine-tuned checkpoint: {state}") if state else
                  "finetuned_state_dict is null -- would load the PUBLIC weights and "
                  "report them as fine-tuned; point it at checkpoint_best.pt",
                  formal_blocker=True)

    # Weight availability, per method, without loading anything.
    for name, method in sorted(checked_methods.items()):
        checkpoint = method.get("checkpoint")
        local = method.get("weights_path")
        if local:
            check(f"weights::{name}", Path(local).exists(), f"{local}")
        else:
            check(f"weights::{name}", True,
                  f"{checkpoint!r} (resolved by the backend at load time)")

    # Exercise the exact role-specific gates; header/asset probes can pass while
    # a whitespace-sensitive manifest prefilter silently removes every row.
    if have_samples and intended == 'formal':
        for role in (PopulationRole.TRAIN, PopulationRole.VAL, PopulationRole.TEST):
            try:
                population = resolve_population(config, role)
                check(f'population::{role.value}', len(population.samples)>0,
                      f'{len(population.samples)} samples / {len(population.sites)} sites', formal_blocker=True)
            except Exception as exc:
                check(f'population::{role.value}', False, f'{type(exc).__name__}: {exc}', formal_blocker=True)
    report["formal_run_ready"] = not report["missing_for_formal_run"]
    report["summary"] = (
        f"{sum(1 for c in report['checks'] if c['ok'])}/{len(report['checks'])} "
        f"checks pass; {len(report['missing_for_formal_run'])} formal blockers"
    )

    out_dir = Path(config.get("run", {}).get("out_dir", "runs/preflight"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if intended == "formal" and report["missing_for_formal_run"]:
        print("\nFORMAL RUN BLOCKED: the inputs above are missing.", file=sys.stderr)
        return 2
    return 0


# ------------------------------------------------------------------ inference

def build_method_runner(name: str, method: dict, device: str, config: dict, population=None):
    """Instantiate one method's runner. Only MapAnything is wired in-process.

    The other seventeen methods keep their existing subprocess runners under
    `method_runners/`, because each needs its own conda environment.
    """
    kind = method.get("kind")
    if kind != "mapanything":
        raise SystemExit(
            f"method {name!r} has kind={kind!r}; build_method_runner handles only "
            "'mapanything'. Legacy methods go through command_infer's adapter branch."
        )
    # A FORMAL run must not silently report an unfilled E4 arm. Both placeholders
    # produce a working run that is numerically just the public arm:
    #   scale_only with scale_factor 1.0  -> multiplies by one
    #   fine-tuned with state_dict null   -> loads the public weights
    # Either way the table gains a row claiming to be a different arm while the
    # predictions are identical to `mapanything_6view`, which is the kind of result
    # that survives review precisely because nothing errors. The values are filled in
    # by `run_training.py calibrate` and by val-split selection respectively, so a
    # formal request for these arms before those steps is a mistake, not a choice.
    if str(config.get("run", {}).get("status", "")).lower() == "formal":
        if name.endswith("scale_only") and not scale_is_measured(method):
            raise SystemExit(
                f"{name}: the formal scale-only arm requires the fitted train scalar and "
                "its matching calibration record. See run_training.py calibrate."
            )
        if name.endswith("_ft") and not method.get("finetuned_state_dict"):
            raise SystemExit(
                f"{name}: run.status is formal but finetuned_state_dict is null, so "
                "this arm would load the PUBLIC weights and report them as "
                "fine-tuned. Point it at the checkpoint the val split selected "
                "(checkpoint_best.pt) first."
            )

    sys.path.insert(0, str(Path(config["paths"]["third_party_dir"]) / "map-anything"))
    from mapanything.models import MapAnything

    from egodepth_eval.infer.mapanything_backend import MapAnythingRunner

    checkpoint = method.get("checkpoint", "facebook/map-anything-apache")

    from egodepth_eval.infer.assets import hf_model_options
    model_options = {} if Path(checkpoint).exists() else hf_model_options("mapanything")
    model = MapAnything.from_pretrained(checkpoint, strict=True, **model_options).to(device).eval()

    state_path = method.get("finetuned_state_dict")
    if state_path:
        import torch

        payload = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"] if "model" in payload else payload)
        print(f"[infer] loaded fine-tuned weights from {state_path}")

    runner = MapAnythingRunner(
        model, device=device,
        view_ids=tuple(method.get("view_ids", range(6))),
        height=int(method.get("height", 406)),
        width=int(method.get("width", 518)),
        amp_dtype=method.get("amp_dtype", "bf16"),
        scale_factor=float(method.get("scale_factor", 1.0)),
        pose_mode=method.get("pose_mode", "none"),
    )
    from egodepth_eval.infer.assets import weight_identity, upstream_identity
    weights=weight_identity(name, state_path or checkpoint)
    upstream=upstream_identity(name, config["paths"]["third_party_dir"])
    contract = PredictionContract(
        method=name,
        checkpoint=checkpoint,
        # The weight FILE's content digest, so two fine-tuned checkpoints -- or the
        # same path with replaced bytes -- are different runs. Previously the path
        # lived in `notes`, which is excluded from the identity hash.
        checkpoint_content_id=(
            weights["digest"]
        ),
        upstream_revision=json.dumps(upstream,sort_keys=True),
        input_views=tuple(runner.view_ids),
        preprocessing={
            "resize": "INTER_AREA to model grid; rays rebuilt at target resolution",
            "normalization": "dinov2",
            "model_input_hw": [runner.height, runner.width],
            "rays": "per-camera double-sphere, verified to reach forward()",
            "upstream_preprocess_inputs_bypassed": True,
            "pose_mode": runner.pose_mode,
            "pose_convention": (
                "T_body_cam: camera-to-rig, translation in metres; official quaternion conversion; metric scale enabled"
                if runner.pose_mode == "rig" else "no rig poses supplied"
            ),
            "output_resample": (
                "support-weighted INTER_AREA to the scoring grid; one kernel for "
                "value and support"
            ),
        },
        precision=runner.amp_dtype,
        output_space="metric_depth_z",
        scale_contract=(
            "none (raw metric)" if runner.scale_factor == 1.0
            else f"fixed positive scalar s_C={runner.scale_factor:.9f} from train"
        ),
        support_rule="model non_ambiguous_mask, no confidence filter",
        scoring_grid_hw=tuple(config["data"].get("scoring_grid_hw", (416, 512))),
        code_identity=code_identity(),
        data_version=config["data"].get("data_version"),
        manifest_id=str(config["data"]["samples_jsonl"]),
        manifest_content_id=population.manifest_content_id if population else None,
        population_identity=(
            {k: v for k, v in population.identity().items() if k != "sites"}
            if population else {}
        ),
        environment=default_environment(),
        notes={
            "finetuned_state_dict": state_path,
            "use_multiview_confidence": False,
            "extra_edge_or_confidence_filter": False,
        },
    )
    return runner, contract


def command_infer(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    population = resolve_eval_population(config, limit=args.limit)
    samples = population.samples
    print(f"[infer] population: {_population_line(population)}")
    reader = build_reader(config)

    methods = config.get("methods", {})
    names = [args.method] if args.method else sorted(methods)
    out_root = Path(config["run"]["out_dir"]) / "predictions"

    overall: dict = {}
    for name in names:
        if name not in methods:
            raise SystemExit(f"unknown method {name!r}; configured: {sorted(methods)}")
        method = methods[name]
        out_dir = out_root / name

        # Legacy methods run as a subprocess in their own environment, then their
        # output is converted into the new artifact format. This is the branch the
        # audit found missing: previously any non-mapanything kind simply exited.
        if method.get("kind") == "legacy":
            overall[name] = _infer_legacy(
                name, method, config, population, reader, out_dir, args
            )
            continue

        runner, contract = build_method_runner(
            name, method, args.device, config, population
        )

        resumable = check_resumable(out_dir, contract)
        if not resumable["resumable"] and not args.overwrite:
            print(
                f"[infer] {name}: refusing to resume -- {resumable['reason']}\n"
                + "\n".join(f"    {d}" for d in resumable["differences"])
                + "\n  Pass --overwrite to discard, or point run.out_dir elsewhere."
            )
            overall[name] = {"status": "contract_mismatch", **resumable}
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        write_run_manifest(out_dir, contract, {"started": time.time()})

        timings, memories, failures = [], [], []
        n_written = 0
        n_reverified, n_recomputed = 0, 0
        for index, sample in enumerate(samples):
            path = out_dir / f"{sample.sample_token}.npz"
            if path.exists() and not args.overwrite:
                # A matching contract says the configuration is unchanged; it says
                # nothing about whether this particular file is intact. Verify
                # before skipping, or a truncated NPZ gets scored as finished.
                verdict = verify_prediction_artifact(
                    path, expected_grid_hw=contract.scoring_grid_hw
                )
                if verdict["ok"]:
                    n_reverified += 1
                    continue
                print(f"  [{name}] recomputing {sample.sample_token}: {verdict['reason']}")
                n_recomputed += 1
            try:
                loaded = reader.load(sample, cameras=list(runner.view_ids))
                result = runner.predict(loaded)
                save_prediction(
                    path, result["depth_z"], result["valid"],
                    raw_output=result["raw"], raw_space=result["raw_space"],
                    raw_valid=result.get("raw_valid"),
                    raw_transform=result.get("raw_transform"),
                )
                timings.append(result["timing_ms"])
                if result["peak_gpu_gb"]:
                    memories.append(result["peak_gpu_gb"])
                n_written += 1
            except Exception as exc:  # noqa: BLE001 - one frame must not kill the run
                failures.append({
                    "sample_token": sample.sample_token,
                    "error": f"{type(exc).__name__}: {exc}"[:400],
                })
                print(f"  [{name}] {sample.sample_token} failed: {exc}", flush=True)
            if (index + 1) % 10 == 0:
                print(f"  [{name}] {index + 1}/{len(samples)}", flush=True)

        cost = {
            "n_written": n_written,
            "n_reverified_and_skipped": n_reverified,
            "n_recomputed_after_failed_verification": n_recomputed,
            "n_failed": len(failures),
            "failures": failures[:20],
            "median_ms": float(np.median(timings)) if timings else None,
            "mean_ms": float(np.mean(timings)) if timings else None,
            "peak_gpu_gb": max(memories) if memories else None,
            "timing_scope": "synchronized forward calls, excluding data loading; no explicit warmup exclusion",
            "n_timed": len(timings),
            "ray_verification": runner.last_verification,
        }
        write_run_manifest(out_dir, contract, {"cost": cost, "finished": time.time()})
        overall[name] = cost
        print(f"[infer] {name}: {json.dumps({k: v for k, v in cost.items() if k not in ('failures', 'ray_verification')})}")

    summary_name = f"inference_summary_{args.method}.json" if args.method else "inference_summary.json"
    summary_path = Path(config["run"]["out_dir"]) / summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall, indent=2, default=str) + "\n")

    # A method that produced NOTHING must not exit 0. Measured on a rehearsal run:
    # 206,999 of 206,999 samples failed on a missing asset root and the command still
    # exited 0 with `n_written: 0`. In a batch sweep that reads as a completed step,
    # and the problem only resurfaces much later as an empty table cell -- by which
    # point the cause is many runs away. The failures were printed per sample, but a
    # scripted sweep checks the exit code, not the scrollback.
    # Legacy methods report `n_converted` rather than `n_written`, and both paths can
    # end in a status that produced nothing at all (`runner_failed`, `no_runner`,
    # `contract_mismatch`). All of them have to reach the exit code, or a sweep that
    # silently skipped a method still looks like it ran.
    BAD_STATUSES = {
        "runner_failed", "no_runner", "env_unavailable", "no_output",
        "contract_mismatch", "weights_missing", "dependency_missing",
    }
    problems = []
    for name, cost in sorted(overall.items()):
        produced = (
            cost.get("n_written", 0)
            + cost.get("n_converted", 0)
            + cost.get("n_reverified_and_skipped", 0)
        )
        status = cost.get("status")
        if status in BAD_STATUSES:
            problems.append((name, f"status={status}: "
                                   f"{str(cost.get('reason', ''))[:200]}"))
        elif cost.get("n_failed", 0) > 0 or cost.get("n_conversion_failures", 0) > 0:
            first = (cost.get("failures") or [{}])[0].get("error", "")
            problems.append((name, f"{produced} predictions available, "
                                   f"{cost.get('n_failed', cost.get('n_conversion_failures', 0))} samples failed. "
                                   f"First error: {first}"))
        elif not args.dry_run and produced != len(samples):
            problems.append((name, f"incomplete predictions: {produced}/{len(samples)}"))
    if problems:
        for name, detail in problems:
            print(f"[infer] FAILED {name}: {detail}", flush=True)
        return 1
    return 0


# --------------------------------------------------------------------- score

def _infer_legacy(
    name: str, method: dict, config: dict, population, reader, out_dir: Path, args
) -> dict:
    """Run one legacy method through the adapter and record what happened."""
    spec = E1_METHODS.get(method.get("spec", name))
    if spec is None:
        return {
            "status": "unknown_spec",
            "reason": (
                f"{name!r} is not in E1_METHODS; add a LegacyMethodSpec declaring its "
                "input views, output space and alignment"
            ),
        }
    contract = build_legacy_contract(spec, config, population)
    staging = Path(config["run"]["out_dir"]) / "legacy_staging" / name

    if args.dry_run:
        # A dry run must not write a contract. It did, and the contract it left
        # behind then made the real run refuse to resume against itself -- a
        # self-inflicted mismatch, since a dry run produces no predictions for the
        # contract to describe.
        result = run_legacy_method(
            spec, config, population, reader, staging, out_dir,
            device=args.device, limit=args.limit, dry_run=True,
        )
        print(f"[infer] {name}: {json.dumps({k: v for k, v in result.items() if k != 'availability'})}")
        return result

    resumable = check_resumable(out_dir, contract)
    if not resumable["resumable"] and not args.overwrite:
        print(f"[infer] {name}: refusing to resume -- {resumable['reason']}")
        for difference in resumable.get("differences", [])[:5]:
            print(f"    {difference}")
        return {"status": "contract_mismatch", **resumable}

    write_run_manifest(out_dir, contract, {"started": time.time()})
    result = run_legacy_method(
        spec, config, population, reader, staging, out_dir,
        device=args.device, limit=args.limit, dry_run=False,
        # Reaches the runner's own raw .npz cache, not just the converted
        # predictions -- otherwise a fixed runner silently re-reports old numbers.
        overwrite=bool(args.overwrite),
    )
    write_run_manifest(out_dir, contract, {"legacy_run": result, "finished": time.time()})
    print(f"[infer] {name}: {json.dumps({k: v for k, v in result.items() if k not in ('conversion_failures', 'availability')})}")
    return result


def command_score(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    methods = config.get("methods", {})
    if args.method and args.method not in methods:
        raise SystemExit(f"Unknown method: {args.method}")
    predictions_root = Path(config["run"]["out_dir"]) / "predictions"
    if args.method and not (predictions_root / args.method).is_dir():
        print(f"[score] {args.method}: missing prediction directory {predictions_root / args.method}",
              file=sys.stderr)
        return 1
    population = resolve_eval_population(config, limit=args.limit)
    samples = population.samples
    print(f"[score] population: {_population_line(population)}")
    reader = build_reader(config)

    annotation_config = config.get("annotation", {})
    records = (
        load_records(Path(annotation_config["dir"]), annotation_config.get("version"), annotation_config.get("asset_root"))
        if annotation_config.get("dir") and Path(annotation_config["dir"]).exists()
        else {}
    )
    print(f"[score] {len(records)} annotation records available")

    names = [args.method] if args.method else sorted(methods)
    results_dir = Path(config["run"]["out_dir"]) / "results"
    scored_any = False

    for name in names:
        method = methods.get(name, {})
        pred_dir = predictions_root / name
        if not pred_dir.exists():
            print(f"[score] {name}: no predictions at {pred_dir}; skipping")
            continue
        contract_path = pred_dir / "prediction_contract.json"
        if population.status == "formal":
            contract = read_formal_contract(contract_path, name, method, population.identity())
        elif contract_path.is_file():
            contract = PredictionContract.from_dict(json.loads(contract_path.read_text()))
        else:
            contract = PredictionContract(method=name)
        # The alignment comes from the declared spec for legacy methods, so a stale
        # YAML cannot override the protocol (Marigold is affine_depth, not
        # affine_inverse).
        spec = E1_METHODS.get(method.get("spec", name)) if method.get("kind") == "legacy" else None
        alignment = spec.alignment if spec else method.get("alignment", "none")
        scorer = MethodScorer(alignment_mode=alignment)

        sample_scores = []
        for sample in samples:
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

        population_payload = {
            "name": config["run"].get("population_name", "development_subset"),
            "n_annotated": len(records),
            # Identity and status come from the gate, so a config claiming `formal`
            # without its preconditions cannot stamp a result formal.
            **population.identity(),
        }
        document = build_result_document(name, contract, sample_scores, population_payload)
        path = write_result(results_dir / f"{name}.json", document)

        full = document["strata"].get("roi::full", {}).get("metrics", {})
        n_missing = document["sample_status_counts"].get("missing_prediction", 0)
        if n_missing == len(samples):
            # coverage=0 with error=NA is the correct scoring outcome, but printing it
            # like a score invites reading "0.0" as a measurement. Say what happened.
            print(
                f"[score] {name}: NO PREDICTIONS for any of {len(samples)} samples. "
                f"coverage=0, error=NA by protocol. Run `infer` first, or check why "
                f"the runner produced nothing. -> {path}"
            )
        else:
            scored_any = True
            suffix = f" ({n_missing} samples missing predictions)" if n_missing else ""
            print(
                f"[score] {name}: AbsRel={full.get('abs_rel')} "
                f"MAE_Z_cm={full.get('mae_z_cm')} coverage={full.get('coverage')}"
                f"{suffix} -> {path}"
            )
    return 0 if scored_any else 1


# --------------------------------------------------------------- E3 view sweep

def command_methods(args: argparse.Namespace) -> int:
    """Per-method wiring status: implemented, env present, weights reachable."""
    config = load_config(Path(args.config))
    rows = method_status_table()
    configured = set(config.get("methods", {}))
    for row in rows:
        row["in_config"] = row["method"] in configured
    payload = {
        "n_methods": len(rows),
        "n_wired": sum(1 for r in rows if r["wiring"] == "wired"),
        "n_no_runner": sum(1 for r in rows if r["wiring"] == "no_runner"),
        "n_env_missing": sum(1 for r in rows if r["wiring"] == "env_missing"),
        "methods": rows,
        "note": (
            "`no_runner` means the implementation does not exist; "
            "`weights_unavailable` means it exists but cannot be validated here. "
            "These are different states and are not merged."
        ),
    }
    out_dir = Path(config["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "method_status.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(f"  {row['method']:20s} {row['wiring']:12s} "
              f"align={row['alignment']:16s} in_config={row['in_config']} "
              f"{row['declared_status']}")
    print(json.dumps({k: payload[k] for k in
                      ("n_methods", "n_wired", "n_no_runner", "n_env_missing")}, indent=2))
    return 0


def command_views(args: argparse.Namespace) -> int:
    """E3: the same checkpoint at 1, 2 and 6 views, with cost."""
    config = load_config(Path(args.config))
    for name in ("mapanything_1view", "mapanything_2view", "mapanything_6view"):
        if name not in config.get("methods", {}):
            raise SystemExit(
                f"{name} is not configured; the view sweep needs all three "
                "(see configs/local.yaml)"
            )
        args.method = name
        if command_infer(args) or command_score(args):
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("preflight", command_preflight),
        ("methods", command_methods),
        ("infer", command_infer),
        ("score", command_score),
        ("views", command_views),
    ):
        sub = subparsers.add_parser(name, help=handler.__doc__ or name)
        sub.add_argument("--config", required=True)
        if name == "preflight":
            selection = sub.add_mutually_exclusive_group()
            selection.add_argument("--method", help="Check a single configured method")
            selection.add_argument("--methods", nargs="+", help="Check these configured methods together")
        else:
            sub.add_argument("--method", default=None)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--limit", type=int, default=None)
        sub.add_argument("--overwrite", action="store_true")
        sub.add_argument("--dry_run", action="store_true",
                         help="print the legacy runner command without running it")
        sub.set_defaults(func=handler)

    args = parser.parse_args()
    try:
        return args.func(args)
    except FormalRunBlocked as exc:
        print(f"\nFORMAL RUN BLOCKED\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
