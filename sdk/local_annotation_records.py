"""Create local-path annotation records for older evaluation consumers.

Use the resulting absolute paths with consumers that require local mask files.
Regenerate these derived records after moving the release. The public scorer
reads the portable annotations directly.
"""
import argparse
import json
import secrets
from pathlib import Path

if __package__:
    from .annotations import annotation_directory
    from .release_paths import inside
else:
    from annotations import annotation_directory
    from release_paths import inside


def materialize(root, output, directory=None):
    root = Path(root).resolve()
    output = Path(output).resolve()
    source_dir = annotation_directory(root, directory).resolve()
    if output.is_relative_to(root):
        raise ValueError('Choose an output directory outside the release')
    sources = sorted(source_dir.glob('s_*.json'))
    if not sources:
        raise FileNotFoundError(f'No annotation records in {source_dir}')
    output.mkdir(parents=True, exist_ok=True)
    def resolve(relative):
        actual = inside(root, relative)
        if not actual.is_file():
            raise FileNotFoundError(actual)
        return str(actual)

    for source in sources:
        record = json.loads(inside(source_dir, source.name).read_text())
        record['region_mask_path'] = resolve(record['region_mask_path'])
        record['overlay_paths'] = {k: resolve(v) for k, v in record.get('overlay_paths', {}).items()}
        temporary = output / f".{source.name}.{secrets.token_hex(8)}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            temporary.replace(output / source.name)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'local_annotation_records': len(sources), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output')
    parser.add_argument('--directory', help='Release-relative version directory; default follows annotation_version.json')
    args = parser.parse_args()
    materialize(args.root, args.output, args.directory)
