#!/usr/bin/env python3
"""Generate machine-local configs without modifying the paper protocol templates."""
import argparse
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]


def configure(template, data_root, third_party, weights, conda_prefix, ft_checkpoint):
    config = yaml.safe_load(template.read_text())
    config['paths'] = {'third_party_dir': str(third_party), 'weights_dir': str(weights), 'conda_prefix': str(conda_prefix)}
    config['data'].update(samples_jsonl=str(data_root / 'protocols/all_official.jsonl'), asset_roots=[str(data_root)], site_map_path=str(data_root / 'splits/site_map.json'))
    config['annotation'].update(asset_root=str(data_root), dir=str(data_root / 'annotations_v6'), diagnostics_manifest=str(data_root / 'protocols/diagnostics.jsonl'), summary=str(data_root / 'annotations_v6/annotation_summary.json'))
    config['run']['out_dir'] = str(ROOT / 'outputs/benchmark')
    config['training']['out_dir'] = str(ROOT / 'outputs/mapanything')
    for method in config['methods'].values():
        for key in ['calibration_record', 'training_record']:
            if method.get(key): method[key] = str(ROOT / method[key])
        if method.get('finetuned_state_dict'):
            method['finetuned_state_dict'] = str(ft_checkpoint or weights / 'egodepth/mapanything_rig_ft.pt')
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--conda-prefix', required=True, type=Path, help='Conda installation root containing envs/; not an individual environment')
    p.add_argument('--third-party-dir', type=Path, default=ROOT / 'third_party')
    p.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    p.add_argument('--ft-checkpoint', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'configs/local.yaml')
    p.add_argument('--training-output', type=Path, default=ROOT / 'configs/local-training.yaml')
    p.add_argument('--force', action='store_true', help='Replace existing generated configs')
    args = p.parse_args()
    outputs = [args.output.expanduser().resolve(), args.training_output.expanduser().resolve()]
    if len(set(outputs)) != 2: p.error('Evaluation and training output paths must differ')
    if any(out in {ROOT / 'configs/release.yaml', ROOT / 'configs/training.yaml'} for out in outputs): p.error('Keep the checked-in templates unchanged')
    if not args.force and any(out.exists() for out in outputs): p.error('Generated config already exists; pass --force to replace it')
    values = [getattr(args, key).expanduser().resolve() for key in ['data_root','third_party_dir','weights_dir','conda_prefix']]
    checkpoint = args.ft_checkpoint.expanduser().resolve() if args.ft_checkpoint else None
    for template, out in zip(['release.yaml', 'training.yaml'], outputs):
        config = configure(ROOT / 'configs' / template, *values, checkpoint)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(config, sort_keys=False))
        print(out)
    print('Configs generated. Dataset/model files and CUDA environments are not downloaded or validated by this command.')
    if not checkpoint:
        print('To evaluate a local fine-tuned checkpoint, pass --ft-checkpoint PATH or place it at the configured finetuned_state_dict path.')


if __name__ == '__main__': main()
