#!/usr/bin/env python3
"""Prepare selected official upstreams at recorded commits; weights are opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from egodepth_eval.infer.assets import SOURCES, configure_paths, source_path, local_weight_path, hf_model_options, weight_identity


def command(args, dry_run=False, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    if not dry_run: subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def configure_geostereo_backbones(target, spec, dry_run=False):
    """Recreate our two loader edits using the user's pinned upstream checkout.

    Line offsets refer to the exact commit in model_sources.json. Verify both
    outputs against the research checkout before changing either input.
    """
    if dry_run:
        print("Would configure GeoStereo's EdgeNeXt and DINOv2 loaders and verify both file digests.")
        return

    updates = {}
    for relative, expected in spec["patched_files_sha256"].items():
        path = target / relative
        current = path.read_bytes()
        if hashlib.sha256(current).hexdigest() == expected:
            continue
        original = subprocess.check_output(["git", "-C", str(target), "show", f"{spec['commit']}:{relative}"])
        if current != original:
            raise RuntimeError(f"Preserving modified source {path}; use a clean checkout")

        lines = original.splitlines(keepends=True)
        if relative == "GeoStereo/core/extractor.py":
            # Keep the upstream constructor and select timm's pretrained weights.
            lines[294] = lines[294].replace(b"pretrained=False", b"pretrained=True")
            lines[295:297] = [b"        # edgenext weights loaded by timm pretrained=True\n",
                              b"        # (see above)\n"]
        elif relative == "GeoStereo/depth_anything/dpt.py":
            lines[156:158] = [
                b'        self.pretrained = torch.hub.load("facebookresearch/dinov2", '
                b'"dinov2_{:}14".format(encoder), pretrained=False)\n'
            ]
            del lines[154]
        else:
            raise ValueError(f"Unknown GeoStereo integration file: {relative}")
        content = b"".join(lines)
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"{relative}: backbone edit differs from the recorded research file")
        updates[path] = content

    for path, content in updates.items():
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_bytes(content)
            shutil.copymode(path, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_source(name, dry_run=False):
    spec = SOURCES[name]
    target = source_path(name)
    if not target.exists():
        command(['git', 'init', target], dry_run)
        command(['git', '-C', target, 'remote', 'add', 'origin', spec['url']], dry_run)
        command(['git', '-C', target, 'fetch', '--depth', '1', 'origin', spec['commit']], dry_run)
        command(['git', '-C', target, 'checkout', '--detach', spec['commit']], dry_run)
    elif not dry_run:
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if head != spec['commit']: raise RuntimeError(f'{target}: expected commit {spec["commit"]}; preserve this checkout and choose an empty --third-party-dir')
    command(['git', '-C', target, 'submodule', 'update', '--init', '--recursive'], dry_run)
    if name == 'geostereo':
        configure_geostereo_backbones(target, spec, dry_run)
        return
    expected = spec.get('patched_files_sha256', {})
    def matches():
        return bool(expected) and all((target / f).is_file() and hashlib.sha256((target / f).read_bytes()).hexdigest() == digest for f, digest in expected.items())
    if not dry_run and matches():
        print('All recorded patches already match:', name)
        return
    for filename in spec['patches']:
        patch = ROOT / 'patches' / filename
        if dry_run:
            command(['git', '-C', target, 'apply', patch], True)
            continue
        # Reverse-check recognizes an already-applied patch without resetting user edits.
        reverse = subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
        if reverse.returncode == 0:
            print('Already applied:', filename)
        else:
            command(['git', '-C', target, 'apply', '--check', patch])
            command(['git', '-C', target, 'apply', patch])
    if expected and not dry_run and not matches():
        raise RuntimeError(f'{name}: patched source differs from the recorded files; preserve and inspect this checkout')


def prepare_weights(name, dry_run=False):
    spec = SOURCES[name]
    if name in {'cut3r', 'foundation_stereo'}:
        checkpoint = local_weight_path(name)
        required = [checkpoint]
        if name == "foundation_stereo":
            required.append(checkpoint.with_name("cfg.yaml"))
        if dry_run:
            print(f"{name}: provide the official files manually:", *required)
            return
        missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
        if missing:
            raise FileNotFoundError(
                f"{name}: obtain the official files listed in docs/MODELS.md: {', '.join(missing)}"
            )
        if weight_identity(name)["digest"] != spec["paper_checkpoint_content_id"]:
            raise ValueError(f"{name}: checkpoint differs from the model used in the paper: {checkpoint}")
        print(f"{name}: found required files and verified the paper checkpoint")
        return
    if dry_run:
        print('Would download official weights:', name, spec.get('hf_repo'), spec.get('hf_revision') or '(revision not recovered)')
        return
    if name == 'zoedepth':
        dest = local_weight_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_suffix('.download')
            urllib.request.urlretrieve('https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt', temporary)
            temporary.replace(dest)
        return
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:
        raise RuntimeError('Install huggingface_hub in the current interpreter to download weights') from error
    if name == 'metric3d_v2':
        dest = local_weight_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hf_hub_download('JUGGHM/Metric3D', dest.name), dest)
        return
    options = hf_model_options(name)
    single_files = {'dac': ['dac_swinl_indoor.pt', 'dac_swinl_indoor.json'], 'unidac': ['unidac.pt'], 'depth_pro': ['depth_pro.pt'], 'depth_anything_v2': ['depth_anything_v2_vitl.pth'], 'vggt_omega': ['vggt_omega_1b_512.pt']}
    if name in single_files:
        dest = local_weight_path(name).parent
        dest.mkdir(parents=True, exist_ok=True)
        for filename in single_files[name]:
            shutil.copy2(hf_hub_download(spec['hf_repo'], filename, **options), dest / filename)
    elif name in {'geostereo', 'depth_anything_3'}:
        snapshot_download(spec['hf_repo'], local_dir=local_weight_path(name), **options)
    else:
        snapshot_download(spec['hf_repo'], **options)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', default=['mapanything'], help='Model IDs from configs/model_sources.json, or all')
    p.add_argument('--config', type=Path)
    p.add_argument('--third-party-dir', type=Path)
    p.add_argument('--weights-dir', type=Path)
    p.add_argument('--download-weights', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.config:
        import yaml
        configure_paths(yaml.safe_load(args.config.read_text()))
    for value, key in [(args.third_party_dir, 'EGODEPTH_THIRD_PARTY_DIR'), (args.weights_dir, 'EGODEPTH_WEIGHTS_DIR')]:
        if value: os.environ[key] = str(value.expanduser().resolve())
    methods = list(SOURCES) if args.methods == ['all'] else args.methods
    unknown = set(methods) - set(SOURCES)
    if unknown: p.error('Unknown model IDs: ' + ', '.join(sorted(unknown)))
    failures = []
    for name in methods:
        try:
            prepare_source(name, args.dry_run)
            if args.download_weights: prepare_weights(name, args.dry_run)
        except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as error:
            failures.append(name)
            print(f'{name}: {error}', file=sys.stderr)
    if failures: raise SystemExit('Incomplete preparation: ' + ', '.join(failures))
    print('Preparation finished. This does not assert CUDA inference or paper-score reproduction.')


if __name__ == '__main__': main()
