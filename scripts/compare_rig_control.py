#!/usr/bin/env python3
"""Recompute the paper's three-arm rig-control comparison from saved predictions."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egodepth_eval.data.mask_profile import get_profile
from egodepth_eval.data.population import PopulationRole, resolve_population
from egodepth_eval.data.reader import SampleReader
from egodepth_eval.report.rig_control import compare_rig_control
from egodepth_eval.score.external import load_annotations, release_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Unpacked benchmark release")
    parser.add_argument("--predictions-dir", required=True, type=Path, help="Parent of the three model output directories")
    parser.add_argument("--output", required=True, type=Path, help="New output JSON file")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.data_root, args.predictions_dir)):
        parser.error("Choose a new output file outside the release and predictions directories")
    config = release_config(args.data_root)
    population = resolve_population(config, PopulationRole.TEST)
    reader = SampleReader(config["data"]["asset_roots"], get_profile(config["data"]["mask_profile"]))
    result = compare_rig_control(population, reader, load_annotations(config), args.predictions_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8") as tmp:
        json.dump(result, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
        tmp.flush()
        os.link(tmp.name, output)
    for name, regions in result["methods"].items():
        print(f"{name}: aligned full-frame AbsRel = {regions['full']['macro']['aligned_abs_rel']:.6f}")
    print(output)


if __name__ == "__main__":
    main()
