"""Check saved baseline predictions against a formal run before scoring or export."""

import json
from pathlib import Path

from egodepth_eval.data.population import FormalRunBlocked
from egodepth_eval.infer.legacy_adapter import E1_METHODS
from egodepth_eval.score.prediction_io import PREDICTION_SCHEMA, PredictionContract


def read_formal_contract(path: Path, name: str, method: dict, identity: dict) -> PredictionContract:
    if not path.is_file():
        raise FormalRunBlocked(f"{name}: missing producer contract {path}. Run inference first.")
    payload = json.loads(path.read_text())
    contract = PredictionContract.from_dict(payload)
    if (
        payload.get("schema") != PREDICTION_SCHEMA
        or payload.get("contract_hash") != contract.identity_hash()
    ):
        raise FormalRunBlocked(f"{name}: invalid prediction contract schema or content digest")
    if (
        contract.method != name
        or not contract.checkpoint_content_id
        or not contract.upstream_revision
    ):
        raise FormalRunBlocked(
            f"{name}: producer method or model provenance is missing or inconsistent"
        )
    if (
        contract.data_version != identity["data_version"]
        or contract.target_camera_id != 2
        or contract.scoring_grid_hw != (416, 512)
    ):
        raise FormalRunBlocked(f"{name}: producer dataset or camera-2 scoring grid differs")
    for key in (
        "role",
        "status",
        "split",
        "token_set_id",
        "n_samples",
        "n_sites",
        "mask_profile",
        "data_version",
    ):
        if contract.population_identity.get(key) != identity[key]:
            raise FormalRunBlocked(f"{name}: producer population differs on {key}")

    if method["kind"] == "legacy":
        spec = E1_METHODS[method.get("spec", name)]
        views, output, scale = tuple(spec.input_views), spec.output_space, spec.alignment
    elif method["kind"] == "mapanything":
        views = tuple(method["view_ids"])
        output = "metric_depth_z"
        factor = float(method.get("scale_factor", 1.0))
        scale = (
            "none (raw metric)"
            if factor == 1
            else f"fixed positive scalar s_C={factor:.9f} from train"
        )
        # Historical rays-only runs predate the explicit pose_mode metadata field.
        if contract.preprocessing.get("pose_mode", "none") != method.get("pose_mode", "none"):
            raise FormalRunBlocked(f"{name}: producer pose mode differs")
        if contract.preprocessing.get("model_input_hw") != [
            method.get("height", 406),
            method.get("width", 518),
        ]:
            raise FormalRunBlocked(f"{name}: producer model input grid differs")
    else:
        raise FormalRunBlocked(f"{name}: unsupported baseline kind {method['kind']!r}")
    if (contract.input_views, contract.output_space, contract.scale_contract) != (
        views,
        output,
        scale,
    ):
        raise FormalRunBlocked(f"{name}: producer views, output representation or scale differ")
    return contract
