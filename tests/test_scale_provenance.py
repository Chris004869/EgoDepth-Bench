"""A release scalar and its source predictions must match the archived train fit."""
import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from egodepth_eval.score.prediction_io import PredictionContract
from egodepth_eval.train.contract import method_inputs
from scripts.derive_scale_only_predictions import validate_calibration, validate_source
from scripts.run_eval_pipeline import scale_is_measured

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["mapanything_scale_only", "mapanything_rig_scale_only"])
def test_only_recorded_train_scalar_is_accepted(name, tmp_path):
    method = yaml.safe_load((ROOT / "configs/release.yaml").read_text())["methods"][name]
    path = ROOT / method["calibration_record"]
    method["calibration_record"] = str(path)
    assert scale_is_measured(method)
    assert not scale_is_measured({**method, "calibration_record": None})
    assert not scale_is_measured({**method, "scale_factor": 42.0})

    fit = json.loads(path.read_text())
    tokens = set(fit["calibration_subset"]["sample_tokens"])
    inputs = method_inputs(method)
    validate_calibration(fit, method["scale_factor"], inputs, tokens)
    with pytest.raises(ValueError, match="outside the training population"):
        validate_calibration(fit, method["scale_factor"], inputs, set())

    wrong = copy.deepcopy(fit)
    entries = wrong["calibration_subset"]["sample_tokens"]
    entries[0] = entries[1]
    with pytest.raises(ValueError, match="distinct samples"):
        validate_calibration(wrong, method["scale_factor"], inputs, tokens)
    wrong_path = tmp_path / "calibration.json"
    wrong_path.write_text(json.dumps(wrong))
    assert not scale_is_measured({**method, "calibration_record": str(wrong_path)})
    wrong_path.write_text("{")
    assert not scale_is_measured({**method, "calibration_record": str(wrong_path)})


def test_derivation_rejects_mismatched_prediction_source():
    inputs = {"pose_mode": "rig", "width": 518, "height": 406, "amp_dtype": "bf16"}
    source = PredictionContract(
        method="public", checkpoint_content_id="weights", input_views=tuple(range(6)), precision="bf16",
        preprocessing={"pose_mode": "rig", "model_input_hw": [406, 518]},
        scale_contract="none (raw metric)", population_identity={"token_set_id": "tokens"},
    )
    calibration = {"weights_digest": "weights"}
    validate_source(source, "public", calibration, inputs, "tokens")
    for change in [
        {"method": "other"}, {"checkpoint_content_id": ""}, {"precision": "fp16"},
        {"preprocessing": {"pose_mode": "none"}}, {"input_views": (2,)},
        {"population_identity": {"token_set_id": "other"}}, {"scale_contract": "affine"},
    ]:
        with pytest.raises(ValueError):
            validate_source(dataclasses.replace(source, **change), "public", calibration, inputs, "tokens")


def test_optimized_python_also_enforces_calibration():
    code = """
import json
from pathlib import Path
import yaml
from egodepth_eval.train.contract import method_inputs
from scripts.derive_scale_only_predictions import validate_calibration
method = yaml.safe_load(Path('configs/release.yaml').read_text())['methods']['mapanything_rig_scale_only']
record = json.loads(Path(method['calibration_record']).read_text())
record['calibration_subset']['split'] = 'test'
validate_calibration(record, method['scale_factor'], method_inputs(method), set())
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "formal train-split calibration" in result.stderr
