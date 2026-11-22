"""MapAnything six-view fine-tuning with camera-2-only supervision.

WHY THIS DOES NOT CALL `model.infer()`
`infer` is decorated `@torch.inference_mode()`, which produces tensors that can
never take part in a backward pass. Training therefore calls `model.forward()`
directly, and sets the geometric-input configuration deterministically itself.
That configuration matters: MapAnything randomly drops geometric inputs during its
own pretraining, and leaving that on would mean the rays reach the network on only
some steps.

WHAT IS EXPLICITLY TURNED OFF
  depth_prob = 0          no depth is ever fed as input (we have none to give,
                          and camera-2 depth is the target)
  cam_prob = 0 or 1       fixed by the shared public/scale/FT pose contract
  ray_dirs_prob = 1       the calibrated rays are always supplied
  dropout_prob = 0        deterministic, not sampled per step
  overall_prob = 1

WHAT RESUME RESTORES
Model, optimizer, scheduler, global step, epoch, and the torch/numpy/python RNG
states. Restoring weights alone would silently change the data order and the
augmentation stream, so a resumed run would not be the run it claims to continue.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from egodepth_eval.train.dataset import (
    TrainingBatchSpec,
    assert_no_reference_in_inputs,
    collate_views,
)
from egodepth_eval.train.loss import (
    DEFAULT_HUBER_DELTA,
    NonFiniteLoss,
    masked_log_huber_loss,
)

TRAINER_SCHEMA = "egodepth-training-1"


@dataclass
class TrainConfig:
    """Every training knob. Development defaults; a formal run must freeze these.

    None of these values is claimed to be frozen. They are what the feasibility
    run used, and the record says so.
    """

    run_name: str = "mapanything_ft_dev"
    checkpoint: str = "facebook/map-anything-apache"
    batch_size: int = 1
    grad_accum_steps: int = 4
    max_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10
    huber_delta: float = DEFAULT_HUBER_DELTA
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"          # bf16 | fp16 | fp32
    gradient_checkpointing: bool = True
    # Which parameter groups may update. "dpt_head" is the cheapest useful choice
    # and the default for the feasibility run; "all" is the full fine-tune.
    trainable: str = "dpt_head"
    seed: int = 20260912
    eval_every: int = 50
    save_every: int = 50
    num_workers: int = 2
    spec: TrainingBatchSpec = field(default_factory=TrainingBatchSpec)
    status: str = "development_not_frozen"
    notes: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        return payload


# Parameter-name prefixes per trainable group, matched against named_parameters().
#
# The names come from a gradient probe on the real Apache checkpoint (six views,
# camera-2 log-Huber loss), not from guessing at the architecture. That probe
# established which modules a depth-only loss can actually reach:
#
#   receive gradient : encoder, ray_dirs_encoder, info_sharing, fusion_norm_layer,
#                      dpt_feature_head, dpt_regressor_head, scale_head, scale_token
#   zero gradient in the original RAYS-ONLY probe: depth_encoder, cam_rot_encoder, cam_trans_encoder,
#                      depth_scale_encoder, cam_trans_scale_encoder
#                      -- exactly as intended: no depth and no poses are supplied,
#                      so those input encoders are inert. This is the E4 contract
#                      being verified rather than asserted. With known rig poses,
#                      camera encoders supply features but remain frozen by the
#                      selected group; they must no longer be described as inert.
#   NO gradient at all: pose_head. It is downstream of the pose output only, which
#                      the camera-2 depth loss never touches. Including it in a
#                      trainable group makes the group look larger than it is and
#                      leaves an optimizer holding a parameter that can never move.
TRAINABLE_GROUPS = {
    "all": None,
    # Dense depth path only. ~20.5M parameters, the cheapest group that learns.
    "dpt_head": ("dpt_feature_head", "dpt_regressor_head", "scale_head", "scale_token"),
    # Adds the multi-view transformer: ~474M parameters.
    "heads_and_info_sharing": (
        "dpt_feature_head", "dpt_regressor_head", "scale_head", "scale_token",
        "info_sharing", "fusion_norm_layer",
    ),
    # Adds the ray encoder, so the camera representation itself can adapt.
    "heads_info_sharing_and_rays": (
        "dpt_feature_head", "dpt_regressor_head", "scale_head", "scale_token",
        "info_sharing", "fusion_norm_layer", "ray_dirs_encoder",
    ),
}

# Modules a camera-2 depth loss cannot reach. Selecting one is a configuration
# error, not a preference, so it is refused rather than silently accepted.
UNREACHABLE_BY_DEPTH_LOSS = ("pose_head",)


def resolve_amp_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def set_deterministic_geometric_inputs(model, pose_mode='none') -> dict:
    """Force rays and optional rig poses on, no depth or geometric dropout.

    MapAnything's `infer` does this through `_configure_geometric_input_config`;
    training must do it explicitly, because the pretraining default samples the
    geometric inputs at random and the rays would then reach the network only
    part of the time.
    """
    if pose_mode not in ('none', 'rig'): raise ValueError('Invalid pose mode')
    previous = dict(model.geometric_input_config)
    model.geometric_input_config.update({
        "overall_prob": 1.0,
        "dropout_prob": 0.0,
        "ray_dirs_prob": 1.0,     # calibrated rays always supplied
        "depth_prob": 0.0,        # never feed depth: it is the target
        "cam_prob": float(pose_mode == 'rig'),
        "sparse_depth_prob": 0.0,
        "depth_scale_norm_all_prob": 0.0,
        "pose_scale_norm_all_prob": 0.0,
    })
    return previous


def select_trainable_parameters(model, group: str) -> dict:
    """Freeze everything outside the selected group.

    Returns a summary including the trainable parameter count, which is the check
    that "fine-tune the head" did not accidentally freeze everything (a run whose
    loss never moves) or unfreeze everything (an OOM at step one).
    """
    if group not in TRAINABLE_GROUPS:
        raise ValueError(
            f"unknown trainable group {group!r}; known: {sorted(TRAINABLE_GROUPS)}"
        )
    prefixes = TRAINABLE_GROUPS[group]
    if prefixes is not None:
        unreachable = [
            token for token in prefixes
            if any(token.startswith(bad) for bad in UNREACHABLE_BY_DEPTH_LOSS)
        ]
        if unreachable:
            raise ValueError(
                f"trainable group {group!r} includes {unreachable}, which the "
                "camera-2 depth loss cannot reach (verified by gradient probe). "
                "Those parameters would never update."
            )
    n_total, n_trainable, names = 0, 0, []
    for name, parameter in model.named_parameters():
        n_total += parameter.numel()
        if prefixes is None or any(token in name for token in prefixes):
            parameter.requires_grad_(True)
            n_trainable += parameter.numel()
            if len(names) < 20:
                names.append(name)
        else:
            parameter.requires_grad_(False)
    if n_trainable == 0:
        raise ValueError(
            f"trainable group {group!r} matched no parameters; the run would not "
            "learn anything. Check TRAINABLE_GROUPS against named_parameters()."
        )
    return {
        "group": group,
        "prefixes": list(prefixes) if prefixes else "all",
        "n_parameters_total": n_total,
        "n_parameters_trainable": n_trainable,
        "trainable_share": n_trainable / n_total if n_total else 0.0,
        "example_trainable_names": names,
    }


def checkpoint_dense_heads(model) -> list:
    """Recompute dense activations without changing module/state-dict names."""
    from torch.utils.checkpoint import checkpoint
    import torch
    names = []
    for name in ('dpt_feature_head', 'dpt_regressor_head'):
        head = getattr(model, name, None)
        if head is None or getattr(head, '_egodepth_checkpointed', False): continue
        original = head.forward
        def forward(*args, _original=original, **kwargs):
            if not torch.is_grad_enabled(): return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)
        head.forward = forward
        head._egodepth_checkpointed = True
        names.append(name)
    return names


def views_to_device(views: Sequence[dict], device, dtype=None, pose_mode='none') -> list:
    """Move batched views to the device in the layout `forward` expects.

    `ray_directions` is renamed to `ray_directions_cam` here -- that is the key
    `forward` reads. Passing `ray_directions` would be silently ignored and the
    model would run images-only while the run log claimed rays were supplied.
    """
    import torch

    if pose_mode not in ('none', 'rig'): raise ValueError('Invalid pose mode')
    present = ['camera_poses' in view for view in views]
    if pose_mode == 'rig':
        if not all(present): raise ValueError('Rig training requires every calibrated camera pose')
        from mapanything.utils.inference import preprocess_input_views_for_inference
        external = []
        for view in views:
            image = view['img'].to(device, non_blocking=True)
            if dtype is not None: image = image.to(dtype)
            external.append({'img': image,
                'ray_directions': view['ray_directions'].to(device, non_blocking=True),
                'camera_poses': view['camera_poses'].to(device, non_blocking=True),
                'data_norm_type': view['data_norm_type'],
                'is_metric_scale': torch.ones(image.shape[0], dtype=torch.bool, device=device)})
        # This helper only processes fixed calibration/image inputs, not model
        # inference. The subsequent model.forward remains fully differentiable.
        return preprocess_input_views_for_inference(external)
    if any(present): raise ValueError('Unexpected rig poses in a rays-only training arm')
    prepared = []
    for view in views:
        image = view["img"].to(device, non_blocking=True)
        rays = view["ray_directions"].to(device, non_blocking=True)
        if dtype is not None:
            image = image.to(dtype)
        # Unit-normalize exactly as the inference path does, so training and
        # evaluation feed identical tensors.
        rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-8)
        prepared.append({
            "img": image,
            "ray_directions_cam": rays.to(torch.float32),
            "data_norm_type": view["data_norm_type"],
            "is_metric_scale": torch.ones(
                image.shape[0], dtype=torch.bool, device=device
            ),
        })
    return prepared


def extract_target_depth_z(predictions: Sequence[dict], view_position: int):
    """Camera-2 optical-axis Z from the model output, differentiably.

    `pts3d_cam[..., 2]` is Z in the camera frame -- the GT's convention.
    `depth_along_ray` is range and would read long off-axis by up to 8x at this
    fisheye's periphery. The view is selected by its position in the supplied view
    list, not by a hardcoded index, so a 1- or 2-view run reads the right output.
    """
    prediction = predictions[view_position]
    if "pts3d_cam" in prediction:
        return prediction["pts3d_cam"][..., 2]
    if "depth_along_ray" in prediction and "ray_directions" in prediction:
        # Reconstruct Z = range * u_z when the head emits range plus directions.
        depth_along_ray = prediction["depth_along_ray"]
        if depth_along_ray.dim() == 4 and depth_along_ray.shape[-1] == 1:
            depth_along_ray = depth_along_ray[..., 0]
        return depth_along_ray * prediction["ray_directions"][..., 2]
    raise KeyError(
        f"prediction has no pts3d_cam or depth_along_ray+ray_directions; "
        f"keys: {sorted(prediction.keys())}"
    )


def rng_state() -> dict:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _as_byte_tensor(value):
    """Coerce a saved RNG state to the CPU ByteTensor the torch setters require."""
    import torch

    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    return tensor.cpu().to(torch.uint8)


def restore_rng_state(state: dict) -> None:
    import torch

    if not state:
        return
    if state.get("python") is not None:
        python_state = state["python"]
        # torch.save round-trips the tuple as a list; setstate needs a tuple.
        if isinstance(python_state, list):
            python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        random.setstate(python_state)
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    # torch.load may return the saved generator states as plain tensors of a
    # different dtype; both setters require a CPU ByteTensor and raise otherwise,
    # so coerce rather than trusting the round trip.
    if state.get("torch") is not None:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if state.get("torch_cuda") and torch.cuda.is_available():
        saved = state["torch_cuda"]
        available = torch.cuda.device_count()
        if len(saved) != available:
            # Resuming on a different GPU count: restoring a partial list would
            # silently leave some devices on a fresh seed, so say so.
            print(
                f"  [trainer] checkpoint holds {len(saved)} CUDA RNG states but "
                f"{available} devices are visible; restoring the first "
                f"{min(len(saved), available)}",
                flush=True,
            )
        torch.cuda.set_rng_state_all(
            [_as_byte_tensor(item) for item in saved[:available]]
        )


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MapAnythingTrainer:
    """Six-view fine-tuning with camera-2-only supervision."""

    def __init__(self, model, config: TrainConfig, device: str = "cuda", out_dir: Optional[Path] = None):
        import torch

        self.model = model
        self.config = config
        self.device = device
        self.out_dir = Path(out_dir or f"runs/{config.run_name}")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.previous_geometric_config = set_deterministic_geometric_inputs(model, config.spec.pose_mode)
        self.trainable_summary = select_trainable_parameters(model, config.trainable)
        # Always defined. Previously this attribute only existed when checkpointing
        # was enabled, so `gradient_checkpointing: false` made `write_run_record`
        # raise AttributeError -- a config value that could not actually be turned
        # off.
        self.gradient_checkpointing_applied: dict = {
            "requested": bool(config.gradient_checkpointing),
            "blocks_wrapped": {},
            "total_blocks_wrapped": 0,
            "note": "not requested" if not config.gradient_checkpointing else "",
        }
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        # Explicit warmup/cosine avoids OneCycle's zero-length warmup phase
        # (warmup_steps=1 raises ZeroDivisionError, including short acceptance runs).
        import math
        total_steps=max(1, config.max_steps)
        warmup=min(max(0,config.warmup_steps),max(0,total_steps-1))
        def lr_multiplier(step):
            if warmup and step < warmup: return (step + 1) / warmup
            progress=min(1.,max(0.,(step-warmup)/max(1,total_steps-1-warmup)))
            return .01 + .99*.5*(1.+math.cos(math.pi*progress))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_multiplier)
        self.scheduler.total_steps=total_steps
        self.amp_dtype = resolve_amp_dtype(config.amp_dtype)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.amp_dtype == "fp16" and torch.cuda.is_available()
            else None
        )
        self.global_step = 0
        self.epoch = 0
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.step_log: list = []
        # How many draws of the site-balanced stream have been consumed. Restored
        # from a checkpoint so a resumed run continues the data order instead of
        # replaying the first batches -- which the previous resume did, because only
        # the RNG state was saved and the stream was rebuilt from position zero.
        self.sampler_position = 0

    def _enable_gradient_checkpointing(self) -> dict:
        """Wrap the transformer blocks for gradient checkpointing.

        Setting `module.gradient_checkpointing = True` after construction does
        NOTHING: uniception reads that flag in `__init__` and wraps its blocks
        there, so a post-hoc flag leaves the blocks unwrapped while the run record
        claims checkpointing is on -- and the run then OOMs at a batch size the
        record says should fit. So the blocks are wrapped here using the same
        `wrap_module_with_gradient_checkpointing` helper, and the count of blocks
        actually wrapped is returned as the evidence.
        """
        applied = {
            "requested": True,
            "blocks_wrapped": {},
            "method": "explicit module wrapping",
        }

        def wrap_blocks(owner, attribute: str, label: str) -> None:
            blocks = getattr(owner, attribute, None)
            if blocks is None or not hasattr(owner, "wrap_module_with_gradient_checkpointing"):
                applied["blocks_wrapped"][label] = (
                    "unavailable: no block list or no wrapper helper"
                )
                return
            wrapped = 0
            for index in range(len(blocks)):
                blocks[index] = owner.wrap_module_with_gradient_checkpointing(blocks[index])
                wrapped += 1
            setattr(owner, attribute, blocks)
            if hasattr(owner, "gradient_checkpointing"):
                owner.gradient_checkpointing = True
            applied["blocks_wrapped"][label] = wrapped

        encoder = getattr(self.model, "encoder", None)
        inner = getattr(encoder, "model", None) if encoder is not None else None
        if inner is not None:
            wrap_blocks(encoder, "blocks", "encoder.blocks") if hasattr(encoder, "blocks") else None
            if hasattr(inner, "blocks") and hasattr(
                encoder, "wrap_module_with_gradient_checkpointing"
            ):
                blocks = inner.blocks
                for index in range(len(blocks)):
                    blocks[index] = encoder.wrap_module_with_gradient_checkpointing(
                        blocks[index]
                    )
                applied["blocks_wrapped"]["encoder.model.blocks"] = len(blocks)
                encoder.gradient_checkpointing = True

        info_sharing = getattr(self.model, "info_sharing", None)
        if info_sharing is not None:
            for attribute in ("self_attention_blocks", "cross_attention_blocks", "blocks"):
                if hasattr(info_sharing, attribute):
                    wrap_blocks(info_sharing, attribute, f"info_sharing.{attribute}")

        # Deterministic bilinear upsampling in the dense heads otherwise exceeds
        # a 24-GB GPU once Adam moments and DDP gradient buckets are allocated.
        applied['dense_heads_wrapped'] = checkpoint_dense_heads(self.model)

        total = sum(
            value for value in applied["blocks_wrapped"].values() if isinstance(value, int)
        )
        applied["total_blocks_wrapped"] = total
        if total == 0:
            # Loud, because the alternative is an OOM that looks like a bad batch
            # size rather than a checkpointing failure.
            applied["warning"] = (
                "gradient checkpointing was requested but no blocks were wrapped; "
                "memory use will be as if it were off"
            )
            print(f"  [trainer] WARNING: {applied['warning']}", flush=True)
        self.gradient_checkpointing_applied = applied
        return applied

    # ------------------------------------------------------------ one step

    def forward_batch(self, batch: dict, want_support: bool = False) -> tuple:
        """One differentiable forward + loss.

        Returns `(LossOutputs, prediction_z)` or, with `want_support=True`,
        `(LossOutputs, prediction_z, support)`. The support is the model's own
        `non_ambiguous_mask`, which validation needs so its P matches the P the
        scorer uses. The training LOSS deliberately does not use it: masking the
        loss by the model's own mask would let the network lower its loss by
        declining to predict the hard pixels.
        """
        import torch

        assert_no_reference_in_inputs(batch["views"])
        views = views_to_device(batch["views"], self.device, pose_mode=self.config.spec.pose_mode)
        assert_no_reference_in_inputs(views)

        with torch.autocast(
            "cuda" if str(self.device).startswith("cuda") else "cpu",
            dtype=self.amp_dtype,
            enabled=self.config.amp_dtype != "fp32",
        ):
            predictions = self.model(views)

        position = batch["target_view_position"]
        prediction_z = extract_target_depth_z(predictions, position).float()

        target = batch["target_depth_z"].to(self.device).float()
        valid = batch["target_valid"].to(self.device)
        if prediction_z.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction_z.shape)} != target shape "
                f"{tuple(target.shape)}: images and reference are on different grids"
            )

        outputs = masked_log_huber_loss(
            prediction_z, target, valid, huber_delta=self.config.huber_delta
        )
        if not want_support:
            return outputs, prediction_z

        support = None
        prediction = predictions[position]
        for key in ("non_ambiguous_mask", "mask"):
            if key in prediction:
                candidate = prediction[key]
                if candidate.dim() == 4 and candidate.shape[-1] == 1:
                    candidate = candidate[..., 0]
                if candidate.shape == prediction_z.shape:
                    support = candidate.detach().float()
                    break
        return outputs, prediction_z, support

    def train_step(self, batches: Sequence[dict]) -> dict:
        """One optimizer step over `grad_accum_steps` micro-batches."""
        import torch

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        records: list = []
        accumulated = 0.0

        for micro_batch in batches:
            outputs, _ = self.forward_batch(micro_batch)
            if outputs.status == "no_supervised_pixels":
                records.append(outputs.to_dict())
                continue
            loss = outputs.loss / len(batches)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += float(outputs.loss.detach().item())
            records.append(outputs.to_dict())

        usable = [r for r in records if r["status"] == "ok"]
        if not usable:
            raise NonFiniteLoss(
                "no micro-batch produced supervised pixels; this is a data problem "
                "and is not skipped"
            )

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.grad_clip_norm,
        )
        if not torch.isfinite(grad_norm):
            raise NonFiniteLoss(
                f"gradient norm is {grad_norm.item()!r} at step {self.global_step}"
            )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.global_step + 1 < self.scheduler.total_steps:
            self.scheduler.step()
        self.global_step += 1

        self.sampler_position += len(batches)
        record = {
            "step": self.global_step,
            "sampler_position": self.sampler_position,
            "loss": accumulated / max(1, len(usable)),
            "grad_norm": float(grad_norm.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "n_supervised": sum(r["n_supervised"] for r in usable),
            "n_pred_nonfinite": sum(r["n_pred_nonfinite"] for r in usable),
            "n_pred_floored": sum(r["n_pred_floored"] for r in usable),
            "micro_batches": len(batches),
            "site_ids": [site for b in batches for site in b["site_ids"]],
        }
        if torch.cuda.is_available():
            record["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.step_log.append(record)
        return record

    # ----------------------------------------------------------- validation

    def validate(self, val_dataset, val_indices: Optional[Sequence[int]] = None) -> dict:
        """Site-macro unaligned AbsRel on the VALIDATION population.

        WHAT CHANGED
        This used to be handed a few micro-batches sliced out of the *training*
        dataset, which meant the "validation" frames had all been trained on. It now
        takes the validation dataset explicitly -- built from the val split, asserted
        disjoint from train by `resolve_train_val` -- and iterates all of it (or a
        declared subset), so selection cannot read a training frame.

        The prediction support is the model's own `non_ambiguous_mask`, obtained the
        same way inference does, so the P used for selection is the P used for
        scoring rather than a looser "finite and positive".

        Unaligned on purpose: the criterion must not let a per-frame fit hide a
        metric-scale regression. Coverage is reported alongside, because a checkpoint
        can lower AbsRel by predicting fewer pixels.
        """
        import torch

        from egodepth_eval.score.accumulate import (
            ScoreAccumulator,
            accumulate_frame,
            prediction_domain,
            reference_domain,
        )
        from egodepth_eval.score.aggregate import site_macro

        self.model.eval()
        indices = (
            list(val_indices) if val_indices is not None else list(range(len(val_dataset)))
        )
        torch_dataset = val_dataset.torch_dataset()
        per_site: dict = {}
        n_frames = 0

        with torch.no_grad():
            for index in indices:
                batch = collate_views([torch_dataset[int(index)]])
                _, prediction_z, support = self.forward_batch(batch, want_support=True)
                prediction = prediction_z.detach().float().cpu().numpy()
                support_np = (
                    support.detach().float().cpu().numpy() > 0.5
                    if support is not None else None
                )
                from egodepth_eval.infer.mapanything_backend import resample_to_scoring_grid
                loaded = val_dataset.reader.load(val_dataset.samples[int(index)], cameras=[2])
                for position, site in enumerate(batch["site_ids"]):
                    site = site or "unmapped"
                    z, support_native = resample_to_scoring_grid(prediction[position],
                        support_np[position] if support_np is not None else np.ones(prediction[position].shape, bool),
                        loaded.reference_depth.shape)
                    G = reference_domain(loaded.reference_depth, loaded.reference_valid)
                    P = prediction_domain(z, support_native)
                    per_site.setdefault(site, ScoreAccumulator()).add(
                        accumulate_frame(z, loaded.reference_depth, G, P)
                    )
                    n_frames += 1

        result = site_macro(per_site)
        # Selection uses the complete macro when it is defined. Falling back to the
        # available-site score would let a checkpoint that produced nothing on a
        # hard site look better than one that produced a poor result there.
        complete = result.metrics["abs_rel"]
        available = result.available_site_metrics["abs_rel"]
        return {
            "step": self.global_step,
            "selection_metric": "site_macro_abs_rel_unaligned",
            "selection_value": complete,
            "selection_used_complete_macro": complete is not None,
            "abs_rel": complete,
            "abs_rel_available_sites": available,
            "coverage": result.metrics["coverage"],
            "mae_z_cm": result.metrics["mae_z_cm"],
            "n_sites": result.n_sites_total,
            "n_sites_with_predictions": result.n_sites_with_predictions,
            "n_sites_no_predictions": result.n_sites_no_predictions,
            "complete_macro_defined": result.complete_macro_defined,
            "n_frames_evaluated": n_frames,
            "prediction_support": "model non_ambiguous_mask, as in inference",
            "per_site": {k: {"abs_rel": v["abs_rel"], "coverage": v["coverage"],
                             "n_scored": v["n_scored"]}
                         for k, v in result.per_site.items()},
        }

    # ----------------------------------------------------------- checkpoint

    def save_checkpoint(self, path: Optional[Path] = None, tag: str = "last") -> Path:
        import torch

        path = Path(path or self.out_dir / f"checkpoint_{tag}.pt")
        payload = {
            "schema": TRAINER_SCHEMA,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "sampler_position": self.sampler_position,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "config": self.config.to_dict(),
            "trainable_summary": self.trainable_summary,
            "rng_state": rng_state(),
            "environment": {
                "hostname": platform.node(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path, restore_rng: bool = True) -> dict:
        """Restore weights, optimizer, scheduler, step, epoch and RNG state."""
        import torch

        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        saved_spec = dict(payload.get('config', {}).get('spec', {}))
        saved_spec.setdefault('pose_mode', 'none')
        if saved_spec != self.config.spec.to_dict():
            raise ValueError('Resume input contract changed (poses, cameras or resolution)')
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload.get("epoch", 0))
        self.sampler_position = int(payload.get("sampler_position", 0))
        self.best_metric = payload.get("best_metric")
        self.best_step = payload.get("best_step")
        if restore_rng:
            restore_rng_state(payload.get("rng_state") or {})
        return {
            "restored_from": str(path),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "sampler_position": self.sampler_position,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "restored": [
                "model", "optimizer", "scheduler", "global_step", "epoch",
                "sampler_position",
            ] + (["rng_state"] if restore_rng else []),
        }

    def write_run_record(self, extra: Optional[dict] = None) -> Path:
        import torch

        payload = {
            "schema": TRAINER_SCHEMA,
            "status": "development_feasibility_run",
            "run_name": self.config.run_name,
            "config": self.config.to_dict(),
            "trainable_summary": self.trainable_summary,
            "geometric_input_config_applied": dict(self.model.geometric_input_config),
            "geometric_input_config_before": self.previous_geometric_config,
            "gradient_checkpointing": self.gradient_checkpointing_applied,
            "supervision": {
                "target_camera_id": self.config.spec.target_camera_id,
                "loss": "Huber(log z_pred - log z*) on reference-valid camera-2 pixels",
                "reference_in_model_input": False,
                "other_views_supervised": False,
                "confidence_masking_of_loss": False,
            },
            "global_step": self.global_step,
            "sampler_position": self.sampler_position,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "step_log": self.step_log,
            "environment": {
                "hostname": platform.node(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "peak_gpu_gb": (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else None
                ),
            },
            **(extra or {}),
        }
        path = self.out_dir / "training_run.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
        return path
