"""Portable model locations and content identities; identity checks never download."""
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import json
import os
import subprocess
from egodepth_eval.score.prediction_io import file_content_digest

# Runner checkouts keep assets next to configs/. Wheel users can supply their
# own workspace through configure_paths or the EGODEPTH_* environment variables.
_module = Path(__file__).resolve()
_checkout = _module.parents[3]
if _module.parents[2] == _checkout / "src":
    ROOT = _checkout
    _registry = ROOT / "configs/model_sources.json"
else:
    ROOT = Path.cwd()
    _shared_registry = Path("share/egodepth-bench/model_sources.json")
    _dist = distribution("egodepth-bench")
    _registry = next(
        (entry.locate() for entry in _dist.files or () if entry.match(_shared_registry.as_posix())
         and entry.locate().is_file()),
        _dist.locate_file(_shared_registry),
    )
SOURCES = json.loads(_registry.read_text(encoding="utf-8"))["models"]


def configure_paths(config):
    """Apply explicit config paths to both this process and runner subprocesses."""
    for key, variable in [('conda_prefix', 'EGODEPTH_CONDA_PREFIX'),
                          ('third_party_dir', 'EGODEPTH_THIRD_PARTY_DIR'),
                          ('weights_dir', 'EGODEPTH_WEIGHTS_DIR')]:
        value = config.get('paths', {}).get(key)
        if value:
            os.environ[variable] = str(Path(value).expanduser().resolve())


def source_path(name):
    return Path(os.environ.get('EGODEPTH_THIRD_PARTY_DIR', ROOT / 'third_party')) / SOURCES[name]['directory']


def weights_root():
    return Path(os.environ.get('EGODEPTH_WEIGHTS_DIR', ROOT / 'weights'))


def torch_cache():
    return Path(os.environ.get('TORCH_HOME', Path.home() / '.cache/torch')) / 'hub/checkpoints'


def local_weight_path(name):
    locations = {
        'cut3r': 'cut3r/cut3r_512_dpt_4_64.pth',
        'vggt_omega': 'vggt_omega/vggt_omega_1b_512.pt',
        'depth_pro': 'depth_pro/depth_pro.pt',
        'unidac': 'unidac/unidac.pt', 'dac': 'dac/dac_swinl_indoor.pt',
        'foundation_stereo': 'foundation_stereo/23-51-11/model_best_bp2.pth',
        'geostereo': 'geostereo', 'depth_anything_3': 'depth_anything_3',
        'depth_anything_v2': 'depth_anything_v2/depth_anything_v2_vitl.pth',
    }
    if name == 'metric3d_v2': return torch_cache() / 'metric_depth_vit_large_800k.pth'
    if name == 'zoedepth': return torch_cache() / 'ZoeD_M12_N.pt'
    return weights_root() / locations[name]


def hf_model_options(name):
    revision = SOURCES[name].get('hf_revision')
    return {'revision': revision} if revision else {}


def snapshot(repo_id):
    hub = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
    root = hub / ('models--' + repo_id.replace('/', '--'))
    locked = next((v.get('hf_revision') for v in SOURCES.values() if v.get('hf_repo') == repo_id), None)
    ref = root / 'refs/main'
    revision = locked or (ref.read_text().strip() if ref.is_file() else None)
    path = root / 'snapshots' / revision if revision else root
    if not revision or not path.is_dir():
        raise FileNotFoundError(f'Local HF snapshot missing: {repo_id}@{revision or "main"}. Run scripts/prepare_models.py --download-weights for this method first.')
    return path


def weight_identity(name, explicit=None):
    base = 'mapanything' if name.startswith('mapanything') else name
    if explicit and Path(explicit).exists():
        path = Path(explicit)
    elif explicit and str(explicit) == SOURCES[base].get('hf_repo'):
        path = snapshot(explicit)
    elif explicit and ':' not in str(explicit) and (str(explicit).startswith(('/', '.', '~')) or str(explicit).endswith(('.pt', '.pth', '.safetensors'))):
        raise FileNotFoundError(explicit)
    elif base in {'unidepth_v2', 'unik3d', 'moge', 'mast3r', 'vggt', 'marigold', 'mapanything'}:
        path = snapshot(SOURCES[base]['hf_repo'])
    else:
        path = local_weight_path(base)
        if not path.exists() and base in {'depth_anything_v2', 'depth_anything_3'}:
            path = snapshot(SOURCES[base]['hf_repo'])
    if not path.exists(): raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in {'.pt', '.pth', '.bin', '.safetensors', '.json', '.yaml'})
    if not files: raise FileNotFoundError(f'No weights/configs in {path}')
    rows = []
    for p in files:
        digest = file_content_digest(p)
        if digest is None: raise OSError(f'Cannot hash weight {p}')
        rows.append({'file': p.name if path.is_file() else p.relative_to(path).as_posix(), 'sha256_16': digest, 'bytes': p.stat().st_size})
    content = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {'root': str(path), 'files': rows, 'digest': content}


def upstream_identity(name, third_party=None):
    base = 'mapanything' if name.startswith('mapanything') else name
    directory = Path(third_party) / SOURCES[base]['directory'] if third_party else source_path(base)
    if not directory.is_dir(): raise FileNotFoundError(directory)
    proc = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=directory, capture_output=True, text=True)
    head = proc.stdout.strip() if proc.returncode == 0 else 'non-git'
    rows = []
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts: continue
        if p.suffix not in {'.py', '.json', '.yaml', '.yml', '.cu', '.cpp', '.h'}: continue
        if any(x in p.parts for x in ('checkpoints', 'pretrained_models', 'logs', 'results')): continue
        rows.append((p.relative_to(directory).as_posix(), file_content_digest(p)))
    return {'commit': head, 'source_digest': hashlib.sha256(json.dumps(rows).encode()).hexdigest(), 'n_files': len(rows)}
