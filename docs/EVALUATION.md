# Evaluation protocol

Use `egodepth-score` for your own predictions and `scripts/run_eval_pipeline.py`
for the provided model adapters. Both use `src/egodepth_eval/score/`, the scoring
implementation used for the archived 24 configurations.

## Evaluate your own model

Start with the restored test shards and metadata described in the
[data guide](DATA.md). Export one `<sample_token>.npz` per official test
anchor. File names must match `sample_token` in `protocols/test_1hz.jsonl`.
For a metric-depth model the file contains two arrays on the native camera-2
grid, in row/column order (416, 512):

| Key | Type and meaning |
| --- | --- |
| `depth_z` | Floating-point optical-axis Z in metres |
| `valid` | Boolean model support, true where the model returns usable depth |

Use the support emitted by your model. A dense model with a prediction at every
pixel may declare all pixels supported; numeric finiteness and positivity are
applied by the scorer. Keep the reference validity mask out of the export.

```python
from pathlib import Path
import numpy as np

def save_prediction(directory, sample_token, depth_z, model_valid):
    z = np.asarray(depth_z, dtype=np.float32)
    valid = np.asarray(model_valid)
    if z.shape != (416, 512) or valid.shape != z.shape or valid.dtype != np.bool_:
        raise ValueError("Expected camera-2 depth and a boolean mask on the native grid")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{sample_token}.npz", depth_z=z, valid=valid)
```

Run the installed CLI from the checkout:

```bash
egodepth-score --release-root /path/to/release \
  --predictions-dir /path/to/predictions --method my_model \
  --output outputs/my_model.json
```

The JSON records the number of anchors in `n_samples` and site IDs under
`population.sites`. The headline numbers are in `strata["roi::full"]["metrics"]`:

| Metric | Interpretation |
| --- | --- |
| `abs_rel` ↓ | Mean absolute depth error divided pixelwise by reference Z |
| `mae_z_cm` ↓ | Mean absolute Z error in centimetres |
| `coverage` ↑ | Share of reference-domain pixels with a valid prediction |
| `a_5cm` ↑ | Share of all reference-domain pixels predicted within 5 cm |

The scorer first computes each metric within a physical site, then averages
across sites. If you want to inspect a complete prediction folder and result
JSON before obtaining data, run `python examples/synthetic_smoke.py --output-dir outputs/first-score`
with a new output directory.

The formal v1 scorer reads the complete test manifest, camera calibrations and
frozen v6 annotation files, including all 800 diagnostic masks. It checks their
identity before it scores the first prediction. To score a custom subset or
only the full image with no ROI files, copy the generated configuration, set
`run.status: development`, point `data.samples_jsonl` to the chosen anchors,
and omit the `annotation` block if ROI scoring is unnecessary. Use
`egodepth-score --config <your-config.yaml>` in place of `--release-root`;
the result records its development status.

For an explicitly oracle-aligned evaluation, also provide a native-grid `raw`
array and select `--alignment positive_scale`, `affine_depth` or
`affine_inverse`. These modes use the test reference depth for fitting and must
be identified as oracle results. `--alignment none` is the default.

The scorer rejects unexpected tokens, missing files, wrong array shapes and
implicit validity conventions. `--allow-missing` explicitly retains absent
predictions as zero coverage. Outputs are not overwritten unless `--overwrite`
is supplied. `--prediction-contract` can record producer provenance supplied by
you; this command does not independently verify how the predictions were made.

For the supplied adapters, first generate the local configuration with
`scripts/configure.py` and follow [MODELS.md](MODELS.md).

## Data and scoring coordinates

- The target is camera 2's native fisheye image at **512 × 416** pixels.
- Depth is optical-axis **Z in meters**, not Euclidean range along a ray.
- Six RGB views have independent calibration. Camera-to-rig transforms are
  metric poses; supplying rays alone does not supply these poses.
- The released reference mask uses `True = valid`.
- Official train, validation and test protocols contain 61,085, 4,491 and 6,574
  anchors across 24, 5 and 8 physical sites, respectively.

The reference domain is the intersection of the declared reference-valid mask,
finite reference depth, and the inclusive **[0.25, 2.0] m** range. Prediction
support requires the method's declared valid support and finite, positive
metric depth. Predictions are **not clipped** into the scoring range.

Errors are evaluated on the intersection of reference and prediction support.
Coverage is that intersection divided by the reference domain. Missing
predictions remain in the coverage denominator. A method with no prediction on
a site has zero coverage and undefined error on that site; an official result
must not silently drop that site to obtain a better mean.

## Aggregation and regions

Pixels are pooled within each physical site, then sites receive equal weight.
The benchmark does not use a single global pixel mean or an unweighted mean of
frame errors. Report coverage alongside depth error.

Hand and object masks are partial annotations. Unannotated pixels remain
unknown; the complement of an object mask is not automatically a labeled
background. The frozen v6 diagnostic set has 800 anchors, with reference-supported
hand and object regions on 291 and 257 frames. Full-image and region comparisons
must use the intended frame population and shared support stated in the paper.

Field angle is computed from calibrated camera rays. Pixel distance from the
image center is not a substitute for a fisheye viewing angle. Depth-angle bins
and diagnostic support counts are preserved by the scoring implementation.

## Metric and oracle-aligned outputs

| Mode | Interpretation |
| --- | --- |
| `none` | Score metric Z directly; no reference-based fitting |
| `positive_scale` | Fit one positive scale on the target frame |
| `affine_inverse` | Fit an affine transform to raw inverse depth, then invert |
| `affine_depth` | Fit an affine transform in raw depth space |

Oracle alignment is fitted once on the full target frame and reused for all
regions and strata. Do not fit each ROI separately. Raw zero inverse depth can
be a valid representation value before alignment; validity must be checked in
the appropriate output representation.

The paper's **scale-only control** is different: its factor
`0.8807126587853854` was calibrated using training data under the six-view,
camera-to-rig condition. It is a fixed calibration, not a test-frame oracle fit.
The public, scale-only and fine-tuned MapAnything arms use the same rig inputs.

## Reproduction boundaries

`results/official/` and `results/paper_artifacts/` contain archived research
results with public path references.
The [provenance record](../results/provenance.json) tracks changes made for release.
Exporting those results reproduces tables from saved statistics; it does not
rerun model inference.

New inference writes prediction contracts that bind the checkpoint, upstream
revision, preprocessing, input views, output representation, support, dataset
population and relevant source contents. Portability changes give this public
code a different source identity from the internal research snapshot. Do not
edit old contracts to bypass reuse checks.

The original Marigold inference did not fix a random seed, so a new run is not
expected to reproduce its archived predictions bit for bit. The reported
fine-tuning uses one fixed training seed; site-bootstrap intervals measure
variation across sites, not across training seeds.

## Six-view rig control

To reproduce the paper's rig comparison on the official test split, point the
following command at the unpacked release and the parent of the three prediction
directories. The comparison runs on CPU with saved predictions.

```bash
python scripts/compare_rig_control.py \
  --data-root /path/to/egodepth_v1 \
  --predictions-dir outputs/benchmark/predictions \
  --output outputs/rig_common_support.json
```

The three directory names are `mapanything_6view`, `mapanything_6view_rig`
and `vggt_omega`. Each contains its runner's `prediction_contract.json` and a
`<sample_token>.npz` for every test anchor. The comparison reads `depth_z`
(float32) and `valid` (bool) at 416 × 512. MapAnything supplies optical-axis
metres. VGGT-Ω supplies positive optical-axis depth up to scale. Its unaligned
errors have no metric interpretation.

For each frame, the three arms share the intersection of their positive, finite
prediction support and valid camera-2 reference Z in [0.25, 2.0] m. Each arm
fits one positive least-squares scale on those pixels and reuses it for the
hand and object regions. A frame with fewer than 32 common pixels fails the
comparison. Within each site, errors are summed over pixels before division.
Sites then receive equal weight. Paired 95% intervals resample the eight sites
with replacement (2,000 draws, seed 20260912).

The archived full-frame aligned AbsRel values are **0.334** (no poses),
**0.327** (known rig) and **0.284** (VGGT-Ω). The rig-minus-no-pose difference,
computed before rounding, is **−0.008** with a 95% interval of **[−0.014,
−0.002]**. The output includes per-site values, ROI results, paired differences,
per-site support counts, the population identity, prediction contracts and
source digests. SHA-256 hashes bind the scored reference and prediction arrays
in manifest order. The formal paper-artifact exporter
uses this same computation when the configuration includes all three arms.

## Independently verify the full-image metrics

For a saved run, the following audit opens the release shards with the SDK,
reads the model's NPZ files, fits the declared alignment independently in
NumPy and compares five equal-site metrics with the archived result.

```bash
python scripts/verify_formal_scores_independent.py \
  --config configs/local.yaml --method mapanything_6view_rig \
  --result results/official/mapanything_6view_rig.json \
  --out outputs/audit/mapanything_6view_rig.json
```

The configuration locates the released data and the run's predictions.
A complete comparison verifies the frozen test token-to-site mapping and
reports a difference for each metric. A mismatch returns a nonzero exit
status. Use `--limit 4` for an evenly spaced probe. Its output is marked
`computed_only` because four frames cannot verify a paper table.
