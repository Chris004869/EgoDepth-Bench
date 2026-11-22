#!/usr/bin/env bash
# Recipes adapted from the research setup; fresh CUDA installs require validation.
# Usage: setup_method.sh mapanything [--config configs/local.yaml]
set -euo pipefail
TASK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_METHOD="${1:?Pass a model ID or environment group; see docs/MODELS.md}"
shift
TASK_BOOTSTRAP="${EGODEPTH_BOOTSTRAP_PYTHON:-python3}"
TASK_CONDA="${EGODEPTH_CONDA_PREFIX:-}"
TASK_THIRD="${EGODEPTH_THIRD_PARTY_DIR:-$TASK_REPO/third_party}"
if [[ "${1:-}" == --config ]]; then
    TASK_CONFIG="${2:?--config requires a YAML file}"
    mapfile -t TASK_PATHS < <("$TASK_BOOTSTRAP" - "$TASK_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
p = yaml.safe_load(Path(sys.argv[1]).read_text())['paths']
print(p.get('conda_prefix') or '')
print(p.get('third_party_dir') or 'third_party')
PY
)
    [[ ${#TASK_PATHS[@]} == 2 ]] || { echo 'Cannot read config. Use a Python with PyYAML via EGODEPTH_BOOTSTRAP_PYTHON.' >&2; exit 2; }
    TASK_CONDA="${TASK_PATHS[0]}"; TASK_THIRD="${TASK_PATHS[1]}"
    shift 2
fi
[[ $# == 0 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }
if [[ -z "$TASK_CONDA" ]]; then TASK_CONDA="$(conda info --base)"; fi
[[ -x "$TASK_CONDA/bin/conda" ]] || { echo 'Set EGODEPTH_CONDA_PREFIX to a Conda installation root.' >&2; exit 2; }
case "$TASK_METHOD" in
 unidepth_v2|unik3d|dac|unidac|metric3d_v2|mono) TASK_GROUP=mono ;;
 depth_pro|depthpro) TASK_GROUP=depthpro ;;
 depth_anything_3|da3) TASK_GROUP=da3 ;;
 foundation_stereo|fstereo) TASK_GROUP=fstereo ;;
 depth_anything_v2|marigold|group_d) TASK_GROUP=group_d ;;
 vggt|vggt_omega) TASK_GROUP=vggt ;;
 mapanything|cut3r|mast3r|zoedepth|geostereo|moge) TASK_GROUP="$TASK_METHOD" ;;
 *) echo "Unknown model/group: $TASK_METHOD. Install only the environments you need." >&2; exit 2 ;;
esac
TASK_ENV="$TASK_CONDA/envs/egodepth_$TASK_GROUP"
TASK_PYTHON="$TASK_ENV/bin/python"
TASK_VERSION=3.11
[[ "$TASK_GROUP" != zoedepth ]] || TASK_VERSION=3.10
if [[ ! -x "$TASK_PYTHON" ]]; then
    [[ ! -e "$TASK_ENV" ]] || { echo "Incomplete environment exists: $TASK_ENV; choose another Conda prefix." >&2; exit 2; }
    "$TASK_CONDA/bin/conda" create --prefix "$TASK_ENV" "python=$TASK_VERSION" pip -y -c conda-forge --override-channels
fi
TASK_NUMPY='numpy==1.26.4'
[[ "$TASK_GROUP" != mono && "$TASK_GROUP" != moge && "$TASK_GROUP" != mapanything ]] || TASK_NUMPY='numpy==2.4.6'
TASK_TIMM='timm==1.0.28'
[[ "$TASK_GROUP" != zoedepth ]] || TASK_TIMM='timm==0.6.13'
TASK_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$TASK_CONSTRAINTS"' EXIT
printf '%s\n' 'torch==2.4.1' 'torchvision==0.19.1' "$TASK_NUMPY" "$TASK_TIMM" \
    'opencv-python==4.10.0.84' 'opencv-python-headless==4.10.0.84' > "$TASK_CONSTRAINTS"
pip_install() { "$TASK_PYTHON" -m pip install -c "$TASK_CONSTRAINTS" "$@"; }
pip_install 'torch==2.4.1' 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cu124
pip_install "$TASK_NUMPY" 'opencv-python-headless==4.10.0.84' scipy pillow tqdm pyyaml einops matplotlib safetensors huggingface_hub
case "$TASK_GROUP" in
 mapanything) pip_install 'timm==1.0.28' -e "$TASK_THIRD/map-anything" ;;
 mono)
    pip_install 'timm==1.0.28' imageio h5py pandas tabulate termcolor protobuf mmengine mmcv-lite tensorboard wandb 'xformers==0.0.28.post1'
    for TASK_SOURCE in UniDepth UniK3D; do
      if [[ -d "$TASK_THIRD/$TASK_SOURCE" ]]; then pip_install --no-deps -e "$TASK_THIRD/$TASK_SOURCE"; fi
    done
    if [[ -d "$TASK_THIRD/DAC" ]]; then
      (cd "$TASK_THIRD/DAC/dac/models/ops" && "$TASK_PYTHON" setup.py build install)
    fi ;;
 depthpro) pip_install 'timm==1.0.28' pillow-heif -e "$TASK_THIRD/ml-depth-pro" ;;
 zoedepth) pip_install 'timm==0.6.13' h5py ;;
 da3) pip_install 'timm==1.0.28' imageio transformers trimesh -e "$TASK_THIRD/Depth-Anything-3" ;;
 fstereo) pip_install 'timm==1.0.28' omegaconf imageio ruamel.yaml trimesh joblib open3d ;;
 geostereo|group_d)
    pip_install 'diffusers==0.30.3' 'transformers==4.44.2' 'huggingface_hub==0.25.2' 'tokenizers<0.20' 'accelerate>=0.22.0' 'timm==1.0.28'
    if [[ "$TASK_GROUP" == geostereo ]]; then pip_install omegaconf imageio ruamel.yaml trimesh joblib open3d; fi ;;
 cut3r) pip_install 'numpy==1.26.4' roma trimesh tensorboard hydra-core 'pillow==10.3.0' h5py imageio 'transformers==4.44.2' 'huggingface_hub<1' accelerate ;;
 mast3r) pip_install scikit-learn roma trimesh ;;
 vggt) pip_install trimesh; [[ ! -d "$TASK_THIRD/vggt" ]] || pip_install -e "$TASK_THIRD/vggt"; [[ ! -d "$TASK_THIRD/vggt-omega" ]] || pip_install -e "$TASK_THIRD/vggt-omega" ;;
 moge) pip_install utils3d click trimesh -e "$TASK_THIRD/moge" ;;
esac
pip_install --no-deps -e "$TASK_REPO"
"$TASK_PYTHON" -m pip check
"$TASK_PYTHON" -c 'import torch; print("torch", torch.__version__, "CUDA available", torch.cuda.is_available())'
"$TASK_PYTHON" -m pip freeze > "$TASK_ENV/egodepth-installed.txt"
printf 'Interpreter: %s\nValidate a small inference run before a full benchmark.\n' "$TASK_PYTHON"
