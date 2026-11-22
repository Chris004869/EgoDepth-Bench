"""Create local-path annotation records for older evaluation consumers.

Portable public records remain unchanged. Regenerate this small directory after
moving the release; pass its path as annotation.dir to the existing evaluator.
"""
import argparse
import json
import os
from pathlib import Path


def materialize(root, output, directory=None):
    from annotations import annotation_directory
    root = Path(root).resolve()
    output = Path(output).resolve()
    source_dir = annotation_directory(root, directory).resolve()
    if output == source_dir or output == root / 'annotations':
        raise ValueError('Never overwrite portable annotations')
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sorted(source_dir.glob('s_*.json')):
        record = json.loads(source.read_text())
        def resolve(relative):
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Expected release-relative annotation path')
            actual = root / path
            if not actual.is_file():
                raise FileNotFoundError(actual)
            return str(actual)
        record['region_mask_path'] = resolve(record['region_mask_path'])
        record['overlay_paths'] = {k: resolve(v) for k, v in record.get('overlay_paths', {}).items()}
        target = output / source.name
        temporary = target.with_name(target.name + '.tmp.' + str(os.getpid()))
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(target)
        count += 1
    print(json.dumps({'local_annotation_records': count, 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output')
    parser.add_argument('--directory', help='Release-relative version directory; default follows annotation_version.json')
    args = parser.parse_args()
    materialize(args.root, args.output, args.directory)
