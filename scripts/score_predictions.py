#!/usr/bin/env python3
"""Score user predictions. See --help or use the installed egodepth-score command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egodepth_eval.score.external import main

if __name__ == "__main__":
    raise SystemExit(main())
