"""Verify a relocated EgoDepth release without its construction workspace."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

if __package__:
    from .release_paths import inside
else:
    from release_paths import inside


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--assets', choices=['none', 'compressed', 'restored'], default='none')
    parser.add_argument('--split', choices=['train', 'val', 'test'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(inside(root, 'checksums/metadata_sha256.json').read_text())
    jobs = list(manifest.items())
    if args.assets != 'none':
        downloads = json.loads(inside(root, 'downloads/download_manifest.json').read_text())
        for shard in downloads['shards']:
            if args.split and Path(shard['restored_path']).parts[1] != args.split:
                continue
            prefix = 'restored_' if args.assets == 'restored' else ''
            jobs.append((shard[prefix + 'path'], shard[prefix + 'sha256']))

    def check(job):
        name, expected = job
        if sha(inside(root, name)) != expected:
            raise ValueError(f'SHA256 mismatch: {name}')
        return name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked = sum(1 for _ in pool.map(check, jobs))
    print(json.dumps({'status': 'passed', 'files_hashed': checked,
                      'assets': args.assets, 'split': args.split}))


if __name__ == '__main__':
    main()
