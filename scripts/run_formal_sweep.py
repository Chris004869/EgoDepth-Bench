#!/usr/bin/env python3
"""Run formal inference and scoring on a local GPU pool, one method per worker."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--methods", nargs="+", help="Configured methods to evaluate (default: all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["run"]["status"] != "formal":
        parser.error("This command requires the formal population")
    methods = config["methods"]
    selected = args.methods or sorted(methods)
    if not selected or len(selected) != len(set(selected)):
        parser.error("Select at least one method and list each method only once")
    unknown = sorted(set(selected) - set(methods))
    if unknown:
        parser.error(f"Unknown methods: {', '.join(unknown)}")
    if args.export:
        paper = yaml.safe_load((ROOT / "configs/release.yaml").read_text())
        if set(selected) != set(paper["methods"]) or set(methods) != set(paper["methods"]):
            parser.error("--export requires the formal 24-configuration paper protocol")
    devices = [gpu.strip() for gpu in args.gpus.split(",")]
    if not all(devices) or len(devices) != len(set(devices)):
        parser.error("--gpus requires distinct, nonempty device IDs")

    pipeline = "scripts/run_eval_pipeline.py"
    config_args = ["--config", str(config_path)]
    subprocess.run([sys.executable, pipeline, "preflight", *config_args, "--methods", *selected],
                   cwd=ROOT, check=True)

    out_dir = ROOT / config["run"]["out_dir"]
    logs = out_dir / "execution_logs"
    logs.mkdir(parents=True, exist_ok=True)
    map_python = Path(config["paths"]["conda_prefix"]) / "envs/egodepth_mapanything/bin/python"
    pending = queue.Queue()
    for name in selected:
        pending.put(name)

    def worker(gpu):
        rows = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1")
        while True:
            try:
                name = pending.get_nowait()
            except queue.Empty:
                return rows

            inference_python = map_python if methods[name]["kind"] == "mapanything" else sys.executable
            started = time.monotonic()
            record = {"method": name, "gpu": gpu, "stages": {}}
            for stage in ("infer", "score"):
                interpreter = inference_python if stage == "infer" else sys.executable
                command = [str(interpreter), pipeline, stage, *config_args, "--method", name]
                if stage == "infer" and args.overwrite:
                    command.append("--overwrite")
                with (logs / f"{name}.{stage}.log").open("a") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["stages"][stage] = result.returncode
                if result.returncode:
                    break
            record["ok"] = record["stages"] == {"infer": 0, "score": 0}
            record["seconds"] = time.monotonic() - started
            rows.append(record)
            (logs / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        rows = [row for result in executor.map(worker, devices) for row in result]
    ok = len(rows) == len(selected) and all(row["ok"] for row in rows)
    summary = {"ok": ok, "config": str(config_path), "methods": rows}
    if ok and args.export:
        export = subprocess.run([sys.executable, "scripts/export_paper_artifacts.py", *config_args,
                                 "--require-complete"], cwd=ROOT)
        summary["export_returncode"] = export.returncode
        summary["ok"] = export.returncode == 0
    (out_dir / "execution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
