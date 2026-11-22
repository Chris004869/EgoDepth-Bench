# Archived paper results

Every configuration uses the same 6,574 camera-2 test anchors across eight physical sites.
The 800-anchor diagnostic subset supplies hand and object annotations from the frozen v6 release.

| Find a result | Source |
| --- | --- |
| Accuracy and coverage across public checkpoints | [Main table](paper_artifacts/main_table.tex) · [Per-method JSON with site-level support](official/) |
| How view count and known rig poses affect MapAnything | [View ablation](paper_artifacts/views_table.tex) |
| Public checkpoint, training-set scale calibration and fine-tuning | [Adaptation table](paper_artifacts/adaptation_table.tex) · [Training record](training_run.json) |
| Geometry, interaction regions and motion | [Diagnostic figure](paper_artifacts/interaction_diagnostics.png) · [Underlying statistics](paper_artifacts/interaction_diagnostics.json) |

For example, FoundationStereo has a full-image AbsRel of 0.068 at **35.4%** coverage. Its object
MAE of 2.52 cm is measured at **81.0%** object coverage. The hand, object and full-image
denominators differ. A comparison on a common set of pixels is a separate experiment.

## Verify and regenerate tabular summaries

```bash
python scripts/export_archived_results.py --output outputs/archived_results
```

The command uses only the Python standard library. It verifies the checksums of the
38 archived artifacts and recomputes site averages for 1,104 strata. It writes:

| Output | Contents |
| --- | --- |
| `all_configurations.csv` and `.md` | Metrics for all 24 configurations. The Markdown pairs each ROI error with its coverage and identifies the scale contract. |
| `views.csv` and `.md` | The same public MapAnything checkpoint with one, two or six views, including the six-view rig condition. |
| `adaptation.csv` and `.md` | Direct metric scores for the three rig-conditioned arms, plus the archived per-frame scale-aligned comparison on common support. |
| `verification.json` | Number of files, strata and scalar comparisons verified. |

CSV coverage values are fractions. Markdown coverage values are percentages. An
empty CSV error or a Markdown dash denotes an undefined complete site average.
For the adaptation comparison, the aligned full-image AbsRel values are
0.327 (public), 0.327 (training-set calibration) and 0.072 (fine-tuned).
Each arm fits one positive scale per frame using the reference and prediction
support shared by all three. Object scoring reuses that full-frame fit.
The exporter reads those diagnostic values from the [archived export manifest](paper_artifacts/export_manifest.json).

[Pixel-level verification](../docs/EVALUATION.md) uses the data and saved predictions.
The table export alone checks archived statistics.

## Provenance and public metadata

`provenance.json` maps each original artifact SHA256 to its public derivative
SHA256 and lists every modified field. All finite scientific values, sample
identifiers, model/input contracts and recorded content identities are preserved.
Machine hostnames and process IDs are removed; internal absolute paths are replaced
by public relative paths or logical resource identifiers. Six undefined `NaN`
stereo diagnostic cells are encoded as JSON `null` so standard JSON readers can
load the file. They have not been filled or estimated.

`dataset://`, `artifact://`, `model://` and `archive://` values identify resources;
they are not live download URLs. Recorded code identities, checkpoint digests and
contract hashes still describe the **original experiment**. Metadata redaction
does not recreate that original byte sequence, and these historical identities
are not rewritten to pretend that the current public source produced the scores.

**Do not use this public archive as a resumable inference run.** Start a new output
directory for current-code inference or training, which creates its own contracts
and identities. The archive is for inspecting, tabulating and tracing the reported
statistics. Model weights and dataset assets are not bundled here.
