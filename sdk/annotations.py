"""Read portable AI diagnostic labels, preserving per-field acceptance."""
import json
from pathlib import Path

import numpy as np

if __package__:
    from .release_paths import inside
else:
    from release_paths import inside


def annotation_directory(root, directory=None):
    root = Path(root)
    if directory is None:
        pointer = inside(root, 'annotation_version.json')
        directory = json.loads(pointer.read_text())['directory'] if pointer.exists() else 'annotations'
    return inside(root, directory)


def read_annotation(root, token, directory=None):
    root = Path(root)
    if Path(token).name != token or not token.startswith('s_'):
        raise ValueError('Invalid sample token')
    record_path = inside(annotation_directory(root, directory), token + '.json')
    record = json.loads(record_path.read_text())
    if record['sample_token'] != token:
        raise ValueError('Annotation token mismatch')
    with np.load(inside(root, record['region_mask_path']), allow_pickle=False) as payload:
        region = payload['region']
    if region.shape != (416, 512) or not np.isin(region, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid annotation region map')
    usable = np.full(region.shape, -1, dtype=np.int8)
    for field, label in [('hand', 0), ('object', 1)]:
        if record.get(field + '_status') == 'accepted_ai':
            usable[region == label] = label
    if (record.get('roi_status') == 'accepted_ai' and
            record.get('automatic_checks', {}).get('background_confirmation', {}).get(
                'unpainted_is_background') is True):
        usable[region == 2] = 2
    return record, usable
