# Release validation

The checks below were run against the public release and its source archives on
November 22, 2026. Archived experiment IDs trace back to the original paper
runs. Re-scoring saved predictions and rerunning a model are listed separately.

| Check | Inputs and observed result |
| --- | --- |
| CPU checks | 164 tests passed on Python 3.12. The optional PyTorch suite passed all five tests in the training environment. |
| Installed package | The wheel installed into a clean Python 3.12 environment. Outside the source checkout, the CLI scored the synthetic fixture and the package loaded the pinned registry for all 18 models. |
| CPU scoring example | Three synthetic frames from two sites. AbsRel 0.1875, MAE 12.5 cm, coverage 0.75. |
| Release reader | Twelve training anchors from four sites. All six RGB views and their calibrated rays were decoded. Using reference Z as the prediction gave zero error and full coverage. |
| Formal test identity | Both official manifest entry points resolve to the same 6,574 anchors from eight sites. The check binds recording, timestamp, physical site, camera calibration and the frozen v6 diagnostic annotations. Swapping recording-to-site assignments caused the external scorer to reject the run even though both counts stayed the same. |
| Archived paper tables | All 38 artifact checksums matched. Re-aggregating 1,104 strata across 24 configurations passed 43,056 scalar checks. |
| Full-image score from saved predictions | For all 6,574 anchors, an independent SDK/NumPy implementation reproduced five MapAnything six-view rig metrics. AbsRel 0.7077561770, MAE 49.07482595 cm, coverage 0.9999714655. All five differences from the archived JSON were zero at the reported floating-point precision. |
| Public scoring CLI on saved predictions | The external CLI scored all 6,574 MapAnything six-view rig predictions. All 12,381 pre-existing numeric fields checked across 46 strata agreed exactly with the archive. The 6,574 file hashes in the result matched a second read. Hand and object MAE were 36.51 cm and 33.81 cm. |
| Public scoring CLI on sparse stereo predictions | Re-scored all 6,574 FoundationStereo files and independently checked their hashes. Across 46 strata, all 11,154 archived numeric fields compared in the site averages and per-site metrics matched exactly. Full-image MAE was 7.3532 cm with 35.4174% coverage. |
| Rig-pose control from saved predictions | Recomputed the three arms on their per-frame pixel intersection, including hand/object regions and paired site bootstrap intervals. All 463 archived numeric fields agreed within an absolute difference of 3.0 × 10⁻¹³. Full-image aligned AbsRel was 0.334190 without poses, 0.326521 with known poses and 0.284004 for VGGT-Ω. |
| Fresh baseline inference | Depth Anything V2 Large was run on CPU using the original upstream revision and weight digest. Four fixed test anchors were scored as a development probe. Compared with re-scoring the saved predictions on those same anchors, support masks matched pixel for pixel, AbsRel differed by 0.00000418 and MAE by 0.00056 cm. |
| Training code | The five optional PyTorch rig/input and gradient-checkpoint tests passed in a separate environment. The original six-GPU training run and the full 24-configuration inference sweep were outside this release check. |

The SDK also validated the identities of all 6,574 test anchors, read the 12-anchor
starter and verified 4,862 metadata checksums on the local release. A four-anchor
development run contains one physical site. A single-method MapAnything export
also ran against the full test protocol and the 800-anchor diagnostic subset.

## Repeat the checks

From the repository root, Python 3.10 or newer:

```bash
python -m pip install -e ".[dev,download,plot]"
python -m pip check
python -m pytest -q
python examples/synthetic_smoke.py
python scripts/export_archived_results.py --output outputs/archived_results
```

The tests exercise the reference and prediction domains, undefined metrics
at zero support, site aggregation, alignment, camera geometry, frozen
population identities, annotations, configuration and release restoration.
The CPU example and the archived-table export can run without dataset
downloads or model weights. The PyTorch tests run when that optional
dependency is installed. GitHub Actions tests CPU scoring on Python 3.10 and 3.12,
and the training contracts on Python 3.11 with CPU PyTorch 2.4.1.

For checks that use real inputs, see [full-image and rig-control verification](EVALUATION.md),
[model-specific inference and dependencies](MODELS.md) and
[the distributed training recipe](TRAINING.md). The four-frame inference
check used Python 3.11, PyTorch 2.4.1+cu124 on CPU, NumPy 1.26.4 and
OpenCV 4.11.0.86. Its score describes those four frames only.
