#!/usr/bin/env python3
"""Production MapAnything adapter. Launch with torchrun; no scheduler-specific assumptions."""

import argparse, contextlib, json, os, sys, time, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from egodepth_eval.data.population import resolve_train_val
from egodepth_eval.data.reader import SampleReader
from egodepth_eval.data.mask_profile import get_profile
from egodepth_eval.train.dataset import (
    SixViewTrainingDataset,
    TrainingBatchSpec,
    collate_views,
    assert_no_reference_in_inputs,
)
from egodepth_eval.train.trainer import (
    MapAnythingTrainer,
    TrainConfig,
    views_to_device,
    extract_target_depth_z,
    seed_everything,
    rng_state,
    restore_rng_state,
)
from egodepth_eval.train.loss import masked_log_huber_loss
from egodepth_eval.train.contract import adaptation_arms, adaptation_inputs
from egodepth_eval.train.distributed import (
    init_distributed,
    rank_indices,
    TensorDataset,
    atomic_torch_save,
    canonical_digest,
    optimizer_config_identity,
    training_source_identity,
)
from egodepth_eval.infer.assets import (
    weight_identity,
    upstream_identity,
    configure_paths,
    hf_model_options,
)
from egodepth_eval.infer.mapanything_backend import resample_to_scoring_grid
from egodepth_eval.score.accumulate import (
    ScoreAccumulator,
    accumulate_frame,
    reference_domain,
    prediction_domain,
)
from egodepth_eval.score.aggregate import site_macro


class DepthObjective(torch.nn.Module):
    def __init__(self, model, device, dtype, delta, pose_mode="none"):
        super().__init__()
        self.base = model
        self.device = device
        self.dtype = dtype
        self.delta = delta
        self.pose_mode = pose_mode

    def predict(self, batch):
        assert_no_reference_in_inputs(batch["views"])
        views = views_to_device(batch["views"], self.device, pose_mode=self.pose_mode)
        with torch.autocast(
            "cuda",
            dtype=self.dtype,
            enabled=self.device.startswith("cuda") and self.dtype != torch.float32,
        ):
            outputs = self.base(views)
        # collate_views fixes the target at its declared sequence position.
        position = batch["target_view_position"]
        prediction = extract_target_depth_z(outputs, position).float()
        info = outputs[position]
        mask = info.get("non_ambiguous_mask", info.get("mask"))
        if mask is not None and mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return prediction, mask

    def forward(self, batch):
        prediction, _ = self.predict(batch)
        target = batch["target_depth_z"].to(self.device)
        valid = batch["target_valid"].to(self.device)
        losses = []
        for index in range(prediction.shape[0]):
            result = masked_log_huber_loss(
                prediction[index], target[index], valid[index], huber_delta=self.delta
            )
            if result.status != "ok":
                raise RuntimeError(f"No supervised target: {result.status}")
            losses.append(result.loss)
        return torch.stack(losses).mean()


def loader(dataset, indices, batch_size, workers, device):
    # Separate loader RNG avoids consuming model RNG during checkpoint resume.
    return DataLoader(
        TensorDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_views,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers else None,
        generator=torch.Generator().manual_seed(1001),
    )


def validate(objective, dataset, reader, rank, world, workers, device):
    objective.eval()
    per_site = {}
    frames = 0
    indices = np.arange(rank, len(dataset), world, dtype=np.int64)
    with torch.no_grad():
        for batch in loader(dataset, indices, 1, workers, device):
            pred, mask = objective.predict(batch)
            token = batch["sample_tokens"][0]
            # Native targets are read independently: selection uses EXACT scoring grid/domain.
            idx = int(indices[frames])
            sample = dataset.samples[idx]
            loaded = reader.load(sample, cameras=[2])
            z = pred[0].cpu().numpy()
            support = (
                np.ones(z.shape, bool) if mask is None else mask[0].float().cpu().numpy() > 0.5
            )
            z, support = resample_to_scoring_grid(z, support, loaded.reference_depth.shape)
            G = reference_domain(loaded.reference_depth, loaded.reference_valid)
            P = prediction_domain(z, support)
            per_site.setdefault(sample.physical_site_id, ScoreAccumulator()).add(
                accumulate_frame(z, loaded.reference_depth, G, P)
            )
            frames += 1
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, (per_site, frames))
    else:
        gathered = [(per_site, frames)]
    merged = {}
    total = 0
    for rows, count in gathered:
        total += count
        for site, acc in rows.items():
            merged.setdefault(site, ScoreAccumulator()).add(acc)
    if total != len(dataset):
        raise RuntimeError(f"Validation count {total} != {len(dataset)}")
    result = site_macro(merged)
    objective.train()
    if result.metrics["abs_rel"] is None:
        raise RuntimeError("Incomplete validation site macro; no checkpoint selected")
    return {
        "abs_rel": result.metrics["abs_rel"],
        "coverage": result.metrics["coverage"],
        "n_frames": total,
        "n_sites": result.n_sites_total,
        "per_site": result.per_site,
        "grid": "native camera2",
        "alignment": "none",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--benchmark-steps", type=int, default=0)
    p.add_argument(
        "--benchmark-validation",
        action="store_true",
        help="Measure the complete validation pass after benchmark steps",
    )
    p.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help="Intentional checkpoint interruption; never declares completion",
    )
    p.add_argument("--output")
    args = p.parse_args()
    rank, world, local, device = init_distributed()
    config = yaml.safe_load(Path(args.config).read_text())
    configure_paths(config)
    tc = dict(config["training"])
    tc["seed"] = args.seed
    global_batch = int(tc["global_batch_size"])
    batch_size = int(tc["batch_size"])
    steps = int(tc["max_steps"])
    if global_batch % (world * batch_size):
        raise ValueError("global_batch_size must be divisible by world_size * batch_size")
    accum = global_batch // (world * batch_size)
    if min(steps, global_batch, batch_size, int(tc["eval_every"])) < 1:
        raise ValueError("Steps, batches and eval interval must be positive")
    if tc.get("deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    if tc.get("amp_dtype", "bf16") not in ["bf16", "fp32"]:
        raise ValueError("Production supports bf16/fp32; fp16 requires separate scaler validation")
    train_pop, val_pop = resolve_train_val(config)
    reader = SampleReader(
        [Path(x) for x in config["data"]["asset_roots"]],
        get_profile(config["data"]["mask_profile"]),
    )
    inputs = adaptation_inputs(config)
    public_name = adaptation_arms(config)[0]
    spec = TrainingBatchSpec(
        height=inputs["height"],
        width=inputs["width"],
        view_ids=tuple(inputs["view_ids"]),
        pose_mode=inputs["pose_mode"],
    )
    train_data = SixViewTrainingDataset(train_pop.samples, reader, spec, seed=args.seed)
    val_data = SixViewTrainingDataset(val_pop.samples, reader, spec, seed=args.seed)
    identity = {
        "optimization": optimizer_config_identity(tc, world),
        "train_tokens": train_pop.token_set_id(),
        "val_tokens": val_pop.token_set_id(),
        "manifest": train_pop.manifest_content_id,
        "data_version": config["data"]["data_version"],
        "weights": weight_identity("mapanything", config["methods"][public_name]["checkpoint"])[
            "digest"
        ],
        "input_contract": inputs,
        "upstream": upstream_identity("mapanything", config["paths"]["third_party_dir"]),
        "training_source": training_source_identity(Path(__file__).resolve().parents[1]),
        "torch": torch.__version__,
    }
    out = Path(args.output or tc["out_dir"]) / f"seed{args.seed}"
    if args.benchmark_steps:
        out = out / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint_last.pt").exists() and not args.resume and not args.benchmark_steps:
        raise FileExistsError("Existing run; pass --resume or a new output directory")
    seed_everything(args.seed)
    sys.path.insert(0, str(Path(config["paths"]["third_party_dir"]) / "map-anything"))
    from mapanything.models import MapAnything

    checkpoint = config["methods"][public_name]["checkpoint"]
    options = {} if Path(checkpoint).exists() else hf_model_options("mapanything")
    model = MapAnything.from_pretrained(checkpoint, strict=True, **options).to(device)
    base = MapAnythingTrainer(
        model,
        TrainConfig(
            run_name=f"seed{args.seed}",
            max_steps=steps,
            checkpoint=config["methods"][public_name]["checkpoint"],
            learning_rate=float(tc["learning_rate"]),
            weight_decay=float(tc.get("weight_decay", 0.01)),
            warmup_steps=int(tc.get("warmup_steps", 200)),
            trainable=tc["trainable"],
            gradient_checkpointing=bool(tc.get("gradient_checkpointing", True)),
            amp_dtype=tc.get("amp_dtype", "bf16"),
            spec=spec,
        ),
        device=device,
        out_dir=out,
    )
    objective = DepthObjective(
        model, device, base.amp_dtype, float(tc.get("huber_delta", 0.1)), pose_mode=spec.pose_mode
    )
    optimizer, scheduler = base.optimizer, base.scheduler
    start_step = 0
    best = float("inf")
    best_step = None
    if args.resume:
        if Path(args.resume).resolve().parent != out.resolve():
            raise ValueError(
                "Resume in the original seed directory so the selected best checkpoint is retained"
            )
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["identity"] != identity:
            raise ValueError(
                "Resume identity mismatch: data, budget, batch, world size or seed changed"
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best = ckpt["best"]
        best_step = ckpt["best_step"]
        if start_step >= steps:
            raise ValueError(
                "Checkpoint already completed this budget; no additional steps requested"
            )
        if best_step is not None and not (out / "checkpoint_best.pt").is_file():
            raise FileNotFoundError("Selected best checkpoint is missing")
        restore_rng_state(ckpt["rank_rng"][rank])
        del ckpt
    wrapped = (
        DDP(
            objective,
            device_ids=[local] if device != "cpu" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        if world > 1
        else objective
    )
    draw = rank_indices(train_data, steps, global_batch, rank, world, start_step)
    batches = iter(loader(train_data, draw, batch_size, int(tc.get("num_workers", 4)), device))
    timings = []
    validation_times = []
    last_val = None
    started = time.monotonic()
    completed = start_step

    def checkpoint(step):
        nonlocal best, best_step
        state = rng_state()
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states = [state]
        payload = {
            "schema": "egodepth-ddp-1",
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best": best,
            "best_step": best_step,
            "rank_rng": states,
            "global_samples_consumed": step * global_batch,
            "training_config": tc,
        }
        if rank == 0:
            atomic_torch_save(payload, out / "checkpoint_last.pt")
            if best_step == step:
                atomic_torch_save(payload, out / "checkpoint_best.pt")
        if world > 1:
            dist.barrier()

    for step in range(start_step, steps):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(accum):
            b = next(batches)
            context = (
                wrapped.no_sync() if world > 1 and micro + 1 < accum else contextlib.nullcontext()
            )
            with context:
                loss = wrapped(b) / accum
                loss.backward()
                total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(
            [x for x in model.parameters() if x.requires_grad], float(tc.get("grad_clip_norm", 1.0))
        )
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite distributed gradient")
        optimizer.step()
        if step + 1 < scheduler.total_steps:
            scheduler.step()
        if device != "cpu":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        if world > 1:
            clock = torch.tensor(elapsed, device=device)
            dist.all_reduce(clock, op=dist.ReduceOp.MAX)
            elapsed = float(clock)
        timings.append(elapsed)
        completed = step + 1
        if rank == 0:
            record = {
                "step": completed,
                "loss_rank0": total_loss,
                "step_seconds": elapsed,
                "global_batch": global_batch,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0,
            }
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if completed % 10 == 0 or completed == 1:
                print(json.dumps(record), flush=True)
        benchmark_done = args.benchmark_steps and len(timings) >= args.benchmark_steps
        interrupted = args.stop_after_step and completed >= args.stop_after_step
        if not args.benchmark_steps and (
            completed % int(tc["eval_every"]) == 0 or completed == steps
        ):
            vt = time.monotonic()
            last_val = validate(
                objective, val_data, reader, rank, world, int(tc.get("num_workers", 4)), device
            )
            validation_times.append(time.monotonic() - vt)
            if last_val["abs_rel"] < best:
                best = last_val["abs_rel"]
                best_step = completed
            if rank == 0:
                with (out / "validation.jsonl").open("a") as f:
                    f.write(json.dumps({"step": completed, **last_val}) + "\n")
            checkpoint(completed)
        elif not args.benchmark_steps and (
            completed % int(tc.get("save_every", 500)) == 0 or interrupted
        ):
            checkpoint(completed)
        if benchmark_done or interrupted:
            break
    if args.benchmark_steps and args.benchmark_validation:
        vt = time.monotonic()
        last_val = validate(
            objective, val_data, reader, rank, world, int(tc.get("num_workers", 4)), device
        )
        validation_times.append(time.monotonic() - vt)
    if rank == 0:
        steady = timings[min(5, max(0, len(timings) - 1)) :]
        record = {
            "status": "benchmark"
            if args.benchmark_steps
            else ("complete" if completed == steps and best_step is not None else "interrupted"),
            "step": completed,
            "best_step": best_step,
            "best_metric": best if math.isfinite(best) else None,
            "world_size": world,
            "global_batch": global_batch,
            "batch_per_gpu": batch_size,
            "gradient_accumulation": accum,
            "trainable": base.trainable_summary,
            "identity": identity,
            "wall_seconds": time.monotonic() - started,
            "steady_step_p50_s": float(np.median(steady)),
            "steady_step_p90_s": float(np.quantile(steady, 0.9)),
            "validation_seconds": validation_times,
            "hardware": torch.cuda.get_device_name(local) if device != "cpu" else "cpu",
            "validation_frame_count": None if last_val is None else last_val["n_frames"],
            "torch": torch.__version__,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0,
            "note": "Benchmark timings are hardware-specific; no formal metric is inferred from a benchmark.",
        }
        (out / "training_run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
