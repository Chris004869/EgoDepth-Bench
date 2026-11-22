# Models and environments

The paper evaluates **18 public models through 24 configurations**: 17 legacy subprocess runners and seven MapAnything conditions (1/2/6 views, rays-only scale calibration, six-view rig, rig scale calibration, and rig fine-tuning). `configs/release.yaml` is the complete protocol template. `configs/training.yaml` contains the final batch-36, 6,000-step training recipe. The old OmniVGGT, RAFT-Stereo and legacy MapAnything runners are not part of this release.

## Prepare one model

Install this repository's Python dependencies first, then generate local paths:

```bash
python scripts/configure.py --data-root /path/to/egodepth_v1 --conda-prefix /path/to/miniconda3
python -m pip install huggingface_hub
python scripts/prepare_models.py --methods mapanything --config configs/local.yaml --download-weights
EGODEPTH_BOOTSTRAP_PYTHON=python bash envs/setup_method.sh mapanything --config configs/local.yaml
```

`configure.py` generates `configs/local.yaml` and `configs/local-training.yaml`; `--output` and `--training-output` override those destinations. It preserves protocol settings and replaces machine paths only. `--third-party-dir`, `--weights-dir`, and `--ft-checkpoint` select other asset locations. Existing generated configs require `--force` to replace them. Use an activated Python with PyYAML for these commands.

`prepare_models.py` clones only the selected official repositories at the commits recorded in `configs/model_sources.json`, initializes their submodules, and applies the listed patches. It does not download weights unless `--download-weights` is supplied. `--dry-run` prints the operations. `--methods all` selects all 18 models; manual and gated downloads remain subject to their upstream requirements. Environment setup is per model/group; it does not implicitly install every method.

The setup scripts are release recipes adapted from the research environments. **Fresh environment installation and all-model CUDA inference have not been rerun as part of packaging.** The original paper runs are archived separately. Validate a small inference run before launching a complete benchmark. The installer constrains Torch, torchvision, NumPy, timm and both OpenCV distribution names throughout dependency resolution; the final first-party editable install does not resolve dependencies again. Some upstream packages explicitly require `opencv-python`, while this project uses `opencv-python-headless`; if both are installed their release versions are constrained to match. The script runs `pip check` and prints CUDA availability, but that alone does not establish model correctness. DAC's extension additionally needs a compatible CUDA toolkit/compiler.

## Run inference and scoring

After preparing the dataset, source and weights, run MapAnything with its own environment. These commands use the full formal test population; the frozen protocol settings stay in `configs/local.yaml`.

```bash
TASK_MAP=/path/to/miniconda3/envs/egodepth_mapanything/bin/python
"$TASK_MAP" scripts/run_eval_pipeline.py preflight --config configs/local.yaml --method mapanything_6view_rig
"$TASK_MAP" scripts/run_eval_pipeline.py infer --config configs/local.yaml --method mapanything_6view_rig
"$TASK_MAP" scripts/run_eval_pipeline.py score --config configs/local.yaml --method mapanything_6view_rig
"$TASK_MAP" scripts/export_paper_artifacts.py --config configs/local.yaml --out_dir outputs/benchmark/paper_artifacts --skip_aligned --skip_examples --no_plots
```

A single-method export leaves uncomputed rows empty. To regenerate complete artifacts, prepare and run all 24 configured conditions, retain their predictions, then export with `--require-complete`. Aligned/paired diagnostics and qualitative examples additionally need those predictions and reference arrays. The already archived official scores and paper artifacts are available under `results/` without rerunning inference.

For a short smoke run, copy your config, set `run.status: development` and choose a separate `run.out_dir`. To test real held-out samples, point `data.samples_jsonl` in that copy at `protocols/test_1hz.jsonl`. Pass the same `--limit N`, config and method to both `infer` and `score`. The formal protocol requires all 6,574 test anchors.

For a full-population sweep of selected conditions, use the base evaluation interpreter:

```bash
python scripts/run_formal_sweep.py --config configs/local.yaml --gpus 0 --methods depth_anything_v2 mapanything_6view_rig
```

The sweep checks the dataset and annotations once, then checks the selected conditions. The public MapAnything arms use the upstream weights. A scale-only arm additionally requires its matching train calibration record, and `mapanything_rig_ft` needs the fine-tuned checkpoint. Checkpoint references resolved by a backend at model load time still require the corresponding files or cache. Run `preflight --methods NAME [NAME ...]` for the same subset checks before inference. The sweep launches MapAnything in its configured environment and other methods through their individual subprocess adapters. Use `--export` with the full method set for complete paper tables.

## Exact methods

| Model ID | Upstream source | Environment | Weights used |
|---|---|---|---|
| `unik3d` | [UniK3D](https://github.com/lpiccinelli-eth/UniK3D/tree/29f7862f3c3ee79c53ef0153dfc901ce511df575) | `egodepth_mono` | `lpiccinelli/unik3d-vitl` |
| `dac` | [DAC](https://github.com/yuliangguo/depth_any_camera/tree/371ee299429257bb9a27d1e23b7dc53670e37023) | `egodepth_mono` | `yuliangguo/depth-any-camera`: `dac_swinl_indoor.pt` + `.json` |
| `unidac` | [UniDAC](https://github.com/girish1511/UniDAC/tree/9ddfc1f4cea68e08273ec9bca037f2ef9e1aa90e) | `egodepth_mono` | `girish1511/UniDAC`: `unidac.pt` |
| `unidepth_v2` | [UniDepth](https://github.com/lpiccinelli-eth/UniDepth/tree/8d8cfe4c7ee15297099983607febf0d4f32eb3d6) | `egodepth_mono` | `lpiccinelli/unidepth-v2-vitl14` |
| `depth_pro` | [ml-depth-pro](https://github.com/apple/ml-depth-pro/tree/9efe5c1def37a26c5367a71df664b18e1306c708) | `egodepth_depthpro` | `apple/DepthPro`: `depth_pro.pt` |
| `metric3d_v2` | [Metric3D](https://github.com/YvanYin/Metric3D/tree/eb5b6fac0dc155e4e52f576e304fbf11655ff339) | `egodepth_mono` | `JUGGHM/Metric3D`: `metric_depth_vit_large_800k.pth` |
| `depth_anything_3` | [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) | `egodepth_da3` | `depth-anything/DA3METRIC-LARGE` |
| `zoedepth` | [ZoeDepth](https://github.com/isl-org/ZoeDepth/tree/d87f17b2f5fdcb174cf4fb115491f4a6c60de152) | `egodepth_zoedepth` | ZoeDepth GitHub release v1.0: `ZoeD_M12_N.pt` (NYU head) |
| `foundation_stereo` | [FoundationStereo](https://github.com/NVlabs/FoundationStereo/tree/6e8806816b533e4d13ddbb95ffa907b797060a62) | `egodepth_fstereo` | Official `23-51-11/model_best_bp2.pth` + `cfg.yaml` |
| `geostereo` | [geostereo](https://github.com/qz-wei/GeoStereo/tree/9b13879530d5a23eb2557da17f5b9c747b673130) | `egodepth_geostereo` | Complete `qz-wei/GeoStereo` snapshot |
| `mast3r` | [mast3r](https://github.com/naver/mast3r/tree/f5209afc300cec36239a7ac992263f36847bbba0) | `egodepth_mast3r` | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` |
| `cut3r` | [cut3r](https://github.com/CUT3R/CUT3R/tree/8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf) | `egodepth_cut3r` | Official `cut3r_512_dpt_4_64.pth` |
| `vggt` | [vggt](https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce) | `egodepth_vggt` | `facebook/VGGT-1B` |
| `vggt_omega` | [vggt-omega](https://github.com/facebookresearch/vggt-omega/tree/282ec70363edeff59424bf43731658092fba3d37) | `egodepth_vggt` | `facebook/VGGT-Omega`: `vggt_omega_1b_512.pt` (upstream access required) |
| `depth_anything_v2` | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf) | `egodepth_group_d` | `depth-anything/Depth-Anything-V2-Large` |
| `marigold` | [Marigold](https://github.com/prs-eth/Marigold/tree/ac915287aaa4e5fd5fe578322735c7c411a1f263) | `egodepth_group_d` | `prs-eth/marigold-depth-lcm-v1-0`, fp16, LCM |
| `moge` | [moge](https://github.com/microsoft/moge/tree/74fbce054ebed49800de42d0ad0e83495065719a) | `egodepth_moge` | `Ruicheng/moge-vitl` (v1 model class) |
| `mapanything` | [map-anything](https://github.com/facebookresearch/map-anything/tree/3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9) | `egodepth_mapanything` | `facebook/map-anything-apache` |

The twelve environment groups follow the original isolation strategy. `mono` includes UniDepth, UniK3D, Metric3D, DAC and UniDAC; `vggt` includes VGGT and VGGT-Omega; `group_d` includes Depth Anything V2 and Marigold. Other methods have separate environments. To prepare all sources needed by a shared environment, list all those method IDs in `prepare_models.py`. The installer uses Torch 2.4.1/cu124 and torchvision 0.19.1 as its baseline; upstream package resolution and hardware compatibility still require the fresh-install checks described above. It is not a full lockfile of every historical environment.

## Weight locations and version evidence

`configs/model_sources.json` records all 18 source commits, the original result's checkpoint content identity, and the eight HF revisions recoverable from local cache records. The recovered revision is used by both model loading and identity checks. Missing HF revisions are explicitly `null`; source commit pinning does not imply those weights or every transitive dependency are fully pinned. A local cache ref recovered during release preparation is evidence of the available asset version, not proof that an entire clean install has been reproduced. The archived checkpoint content identities remain available for comparison.

HF-backed models use the normal `HF_HOME` / `HF_HUB_CACHE`. Local checkpoint files go under `paths.weights_dir` (default `weights/`):

```text
weights/
  cut3r/cut3r_512_dpt_4_64.pth
  vggt_omega/vggt_omega_1b_512.pt
  foundation_stereo/23-51-11/{model_best_bp2.pth,cfg.yaml}
  depth_pro/depth_pro.pt
  dac/{dac_swinl_indoor.pt,dac_swinl_indoor.json}
  unidac/unidac.pt
  depth_anything_v2/depth_anything_v2_vitl.pth
  depth_anything_3/                  # full model directory
  geostereo/                        # full model directory
  egodepth/mapanything_rig_ft.pt     # separately released fine-tuned model
```

Metric3D and ZoeDepth use `${TORCH_HOME:-~/.cache/torch}/hub/checkpoints/`, matching their upstream Torch Hub loaders. `prepare_models.py --download-weights` fills these locations for supported official HF/GitHub downloads. It requires `huggingface_hub` in the calling interpreter. It never substitutes an unofficial mirror for a gated model. For manual downloads use [FoundationStereo's official 23-51-11 folder](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing) and [CUT3R's official final checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link). FoundationStereo needs the accompanying `cfg.yaml`. VGGT-Omega requires the upstream access grant before downloading with your own HF credentials.

The EgoDepth-Bench fine-tuned model is distributed separately; no model-hosting URL is claimed here until that release exists. The original `checkpoint_best.pt` is a training recovery package, not an inference-only artifact. `--ft-checkpoint` can point at the supported exported model or a compatible recovery checkpoint. The training recipe creates a new run identity; the archived training record must not be relabeled as a run from this cleaned code.

UniDAC extracts its DINOv3 backbone from the complete released `unidac.pt` into the same local weights directory; it does not require downloading the original authors' private DINOv3 path. GeoStereo needs both DINOv2 code in Torch Hub and pretrained EdgeNeXt Small weights through timm. The first run needs access to their official sources or pre-populated caches. The preparation script configures the two backbone loaders in your pinned checkout and verifies both resulting files against the research SHA-256 digests. The historical revisions of these two transitive assets remain unknown.

All runners and identity checks share `src/egodepth_eval/infer/assets.py`. Configured `paths.conda_prefix`, `paths.third_party_dir`, and `paths.weights_dir` populate `EGODEPTH_CONDA_PREFIX`, `EGODEPTH_THIRD_PARTY_DIR`, and `EGODEPTH_WEIGHTS_DIR`; explicit config paths take precedence. Run the documented entrypoints so those paths reach subprocesses. Direct runner invocation can set those environment variables itself.
