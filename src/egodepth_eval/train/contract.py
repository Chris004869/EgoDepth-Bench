"""Shared input contract for public, scale-only and fine-tuned MapAnything arms."""
import math


def adaptation_arms(config):
    names = config.get('adaptation', {})
    return (names.get('public_method', 'mapanything_6view'),
            names.get('scale_method', 'mapanything_scale_only'),
            names.get('ft_method', 'mapanything_ft'))


def method_inputs(method):
    return {'view_ids': list(method.get('view_ids', range(6))),
            'height': int(method.get('height', 406)), 'width': int(method.get('width', 518)),
            'pose_mode': method.get('pose_mode', 'none'),
            'amp_dtype': method.get('amp_dtype', 'bf16')}


def adaptation_inputs(config):
    public, scale, ft = adaptation_arms(config)
    base = config['methods'][public]
    expected = method_inputs(base)
    if expected['pose_mode'] not in ('none', 'rig'):
        raise ValueError('Invalid adaptation pose_mode')
    for name in (scale, ft):
        if name not in config['methods']: continue
        method = config['methods'][name]
        if method_inputs(method) != expected or method['checkpoint'] != base['checkpoint']:
            raise ValueError(f'{name}: adaptation inputs or public initialization differ')
    training = config.get('training', {})
    for key in ('height', 'width', 'amp_dtype', 'pose_mode'):
        if training.get(key, expected[key]) != expected[key]:
            raise ValueError(f'Training {key} differs from the public arm')
    return expected


def verify_calibration_inputs(record, expected):
    actual = record.get('input_contract')
    if actual is None:
        # Legacy records are known rays-only; never adopt them for the rig arm.
        if expected['pose_mode'] != 'none':
            raise ValueError('Rig calibration requires an explicit matching input contract')
    elif actual != expected:
        raise ValueError('Calibration and evaluation input contracts differ')


def validate_fitted_scalar(record, factor, inputs):
    """Check the archived 1,024-frame train fit before using its scalar."""
    fit = record["fit"]
    subset = record["calibration_subset"]
    if record["status"] != "formal" or fit["status"] != "ok" or subset["split"] != "train":
        raise ValueError("A successful formal train-split calibration is required")
    if not math.isfinite(factor) or factor <= 0 or factor != record["s_C"] or factor != fit["scale"]:
        raise ValueError("The scale factor must match the positive, finite fitted scalar")
    verify_calibration_inputs(record, inputs)
    if record["training_grid_hw"] != [inputs["height"], inputs["width"]]:
        raise ValueError("The calibration and evaluation input grids differ")
    if inputs["view_ids"] != list(range(6)):
        raise ValueError("The frozen calibration requires the six camera views in rig order")
    tokens = subset["sample_tokens"]
    if len(tokens) != 1024 or len(set(tokens)) != 1024 or subset["n_used"] != 1024:
        raise ValueError("The frozen calibration requires exactly 1,024 distinct samples")
