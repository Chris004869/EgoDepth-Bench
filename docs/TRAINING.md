# MapAnything adaptation

The primary experiment fine-tunes the Apache MapAnything initialization using
six RGB views, six calibrated ray fields and metric camera-to-rig extrinsics.
Camera-2 reference depth and validity masks are targets only. Public, Scale-only
and fine-tuned arms use the same inputs and initialization.

`configs/training.yaml` preserves the final recipe. `results/training_run.json`
records the completed original run; `results/official/` contains its held-out
test result. The code repository does not bundle the dataset or model weights.

| Setting | Final recipe |
|---|---|
| Data | 61,085 training anchors; 4,491 validation anchors; disjoint physical sites |
| GPUs in the reported run | Six NVIDIA RTX 4090 |
| Global batch | 36: one sample/GPU, six accumulation steps |
| Budget and seed | 6,000 optimizer steps; seed 20260912 |
| Optimizer | AdamW; learning rate 1e-5; weight decay 0.01; 200 warmup steps |
| Loss | Masked log-depth Huber, delta 0.1; reference support determines supervision |
| Precision | bfloat16; gradient checkpointing; deterministic settings |
| Model input | H406 × W518; six views; rays and rig poses |
| Validation | Every 500 steps; native camera-2 grid; unaligned equal-site AbsRel |
| Selected original checkpoint | Step 6,000 |
| Original software recorded | PyTorch 2.4.1+cu124; upstream commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` |

The original run took 43,412.66 seconds (about 12.06 hours). This is a measurement
on that hardware, not a runtime guarantee for other systems. Results involve one
training seed.

## Prepare a new training run

Run commands from the repository root. Restore the train and validation shards
as described in [DATA.md](DATA.md). Keep the full metadata/protocol package; the
split-aware resolver selects training and validation populations independently.

```bash
python -m pip install -e . huggingface_hub
python scripts/configure.py \
  --data-root /path/to/egodepth_v1_20260912 \
  --conda-prefix /path/to/conda
python scripts/prepare_models.py --methods mapanything \
  --config configs/local.yaml --download-weights
EGODEPTH_BOOTSTRAP_PYTHON=python bash envs/setup_method.sh mapanything --config configs/local.yaml
```

If you already prepared MapAnything using [the baseline guide](MODELS.md), use
the generated training config and activate the same environment:

```bash
conda activate /path/to/conda/envs/egodepth_mapanything
torchrun --standalone --nproc_per_node=6 scripts/train_distributed.py \
  --config configs/local-training.yaml \
  --seed 20260912 --output outputs/mapanything
```

The distributed entrypoint is the paper-recipe training path. It verifies split
separation, input contracts and initialization; nonfinite gradients or invalid
supervision fail explicitly. Training writes into
`outputs/mapanything/seed20260912/`, including `checkpoint_best.pt`,
`checkpoint_last.pt`, `training_run.json`, `steps.jsonl` and `validation.jsonl`.
Validation must complete and select a checkpoint before a run is marked complete.
No test score selects the checkpoint.

A different GPU count can preserve the global batch when it divides 36, but
changes accumulation and the execution environment. Report that as a new run;
the timing and exact numerical trajectory are not inherited from the original.

## Scale-only control and test binding

The archived rig-conditioned control uses
`configs/scale_only_rig_calibration.json`: one positive scalar
`s_C = 0.8807126587853854`, fitted on 1,024 unique training anchors. The older
rays-only scale is a separate historical condition.

To fit the control again with the configured public initialization:

```bash
python scripts/run_training.py calibrate --config configs/local-training.yaml \
  --device cuda --out_dir outputs/calibration
```

This creates `outputs/calibration/scale_only_calibration.json`. Its input,
initialization and training-population identities must match the trained model.
To bind a completed new distributed run to an evaluation configuration:

```bash
python scripts/bind_adaptation.py \
  --config configs/local.yaml \
  --seed-dir outputs/mapanything/seed20260912 \
  --calibration outputs/calibration/scale_only_calibration.json \
  --out configs/local-trained.yaml
```

The binder checks budget, seed, batch, data, public weights, upstream source,
training source and the selected checkpoint. It does not bypass a mismatch.
Evaluate the new configuration in a fresh output directory using the evaluation
workflow. The paper archive remains unchanged.

## Resume and weight distribution

Resume an interrupted **new distributed run** using its original seed directory:

```bash
torchrun --standalone --nproc_per_node=6 scripts/train_distributed.py \
  --config configs/local-training.yaml --seed 20260912 \
  --output outputs/mapanything \
  --resume outputs/mapanything/seed20260912/checkpoint_last.pt
```

Strict resume requires the same source, upstream code, public initialization,
data identities, optimization budget, GPU topology and RNG state. A completed
6,000-step checkpoint is not an authorization to extend the budget. The original
research checkpoint is not guaranteed to resume under this reorganized public
checkout: its recorded training-source identity refers to the original code.
The archived identities remain historical evidence rather than being rewritten
to match the public tree.

The original best checkpoint is an approximately 8.96 GB training recovery file
containing model, optimizer, scheduler and per-rank RNG states. For checkpoint
availability and compatible formats, see the [model guide](MODELS.md).
