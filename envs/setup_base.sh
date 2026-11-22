#!/usr/bin/env bash
# Run from an activated Python >=3.10 environment.
set -euo pipefail
TASK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${EGODEPTH_BOOTSTRAP_PYTHON:-python3}"
"$TASK_PYTHON" -m pip install -e "$TASK_REPO"
