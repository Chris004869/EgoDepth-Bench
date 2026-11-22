# EgoDepth-Bench

**Benchmarking Metric Depth for Egocentric Manipulation**

Yuan Gao, Dongchen Wei, Xinyang Lin, Shuhua Cao, Xueliang Guo, Zhuohan Li, Yihao Wang · **GenRobot**

[Project page](https://chris004869.github.io/EgoDepth-Bench/) · [Paper (PDF)](https://chris004869.github.io/EgoDepth-Bench/site-assets/egodepth-bench.pdf)

EgoDepth-Bench evaluates metric depth from head-worn RGB cameras during real
manipulation. The dataset contains **941,233 synchronized timestamps**, six
calibrated RGB views per timestamp, and camera-2 fisheye Z-depth across **37
physical sites**. The benchmark reports depth accuracy together with prediction
coverage, including on hands and manipulated objects.

This repository provides data readers, the official scoring protocol, adapters
for **18 models / 24 evaluated configurations**, the MapAnything fine-tuning
recipe, and archived paper results. The paper evaluates **6,574 test anchors from eight physical sites**. An additional
diagnostic protocol covers **800 of those anchors** with reviewed annotations.
The [data guide](docs/DATA.md) describes the release format. Model environments
and checkpoints are documented in the [baseline guide](docs/MODELS.md).

## Start here

| Goal | Entry point |
| --- | --- |
| Check installation without data or a GPU | Synthetic example below |
| Read the real starter or released dataset | [Data guide](docs/DATA.md) |
| Evaluate predictions from your own model | `egodepth-score` and [evaluation protocol](docs/EVALUATION.md) |
| Recreate tables from the archived 24 results | [Archived results](results/README.md) |
| Re-score saved predictions on the frozen test split | [Evaluation and common-support controls](docs/EVALUATION.md) |
| Run a supported baseline | [Models and environments](docs/MODELS.md) |
| Train the MapAnything adaptation | [Training guide](docs/TRAINING.md) |

## Install and run a CPU example

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,download,plot]"
python examples/synthetic_smoke.py
python -m pytest -q
```

The synthetic example exercises the actual data reader and scorer with known
geometry and partially missing prediction support. It downloads no model or
dataset. Its output is an installation check, not a benchmark result.

The base installation, `python -m pip install -e .`, supports reading data and
scoring predictions on a CPU. GPU model inference uses separate environments;
install only the method you intend to run.

## Read real data

After obtaining and extracting the starter package:

```bash
python examples/read_starter.py --data /path/to/egodepth-starter
```

The reader checks RGB, depth, validity and calibration. For full splits, follow
the checksum and restoration instructions in [docs/DATA.md](docs/DATA.md).
Depth is optical-axis **Z in meters**, and released masks use **True = valid**.

## Score your own predictions

Write one `<sample_token>.npz` per official test anchor. Each file contains
`depth_z` as a floating-point **416 × 512** array in meters and `valid` as a
boolean array on the same camera-2 grid. Then run:

```bash
egodepth-score \
  --release-root /path/to/egodepth_v1 \
  --predictions-dir /path/to/predictions \
  --method my_method \
  --output outputs/my_method.json
```

The default evaluates metric depth without reference-based alignment. Missing
files are rejected unless `--allow-missing` is explicitly supplied; with that
option they remain in the population and contribute zero coverage. See
[the evaluation guide](docs/EVALUATION.md) for oracle-aligned outputs, ROI
annotations, aggregation and prediction contracts.

## Recreate the archived result tables

```bash
python scripts/export_archived_results.py --output outputs/archived_results
```

This reads the bundled 24 result JSON files and exports tables without data,
weights or a GPU. It does not rerun inference. The paper's metric MapAnything
comparison is:

| Six-view, camera-to-rig condition | Full-image AbsRel ↓ |
| --- | ---: |
| Public checkpoint | 0.708 |
| Training-set scale calibration | 0.601 |
| Fine-tuned | 0.091 |

Values above are rounded as in the paper. Full statistics, supports and
provenance are retained under [results/](results/README.md).

## Configure baseline inference

```bash
python scripts/configure.py \
  --data-root /path/to/egodepth_v1 \
  --conda-prefix /path/to/miniconda3
```

This creates `configs/local.yaml` and `configs/local-training.yaml`. The Conda
prefix is the installation root containing `envs/`. Model source, weight and
output paths can be configured without editing the adapters. Continue with the
method-specific instructions in [docs/MODELS.md](docs/MODELS.md).

The scoring implementation preserves site-wise aggregation, coverage, native
fisheye geometry and the distinction between metric and oracle-aligned methods.
Do not reuse old predictions after changing their source, checkpoint or
preprocessing identity. The archived experiments and validation of this public
package are documented separately in [docs/VALIDATION.md](docs/VALIDATION.md).

## Citation

```bibtex
@misc{gao2026egodepthbench,
  title={EgoDepth-Bench: Benchmarking Metric Depth for Egocentric Manipulation},
  author={Yuan Gao and Dongchen Wei and Xinyang Lin and Shuhua Cao and
          Xueliang Guo and Zhuohan Li and Yihao Wang},
  year={2026},
  note={Manuscript}
}
```

An arXiv identifier will be added after posting. Machine-readable citation
metadata is available in [CITATION.cff](CITATION.cff).

## License and contact

First-party code: **Apache-2.0**, copyright **2026 GenRobot**. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). Third-party projects, patches, weights and datasets retain
their own terms; consult [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

For questions, open an issue in this repository or contact Yuan Gao at
**gaoyuan100@outlook.com**. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributions.
