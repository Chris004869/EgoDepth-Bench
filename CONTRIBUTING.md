# Contributing

For a bug report, include the command, configuration, package versions, the
dataset release and annotation version, and the relevant error message. A small
synthetic reproduction is preferable to uploading dataset frames or checkpoints.

Use the same CPU checks as GitHub Actions before submitting a change. From the code checkout, run

```bash
python -m pip install -e ".[dev,download,plot]"
python -m pytest -q
python examples/synthetic_smoke.py
python scripts/export_archived_results.py --output outputs/archived_results
```

Changes to scoring, camera geometry, masks, alignment, or aggregation should
include a numerical regression test. Preserve prediction-contract validation and
the separation between metric and oracle-aligned results. Do not overwrite the
archived paper results when adding a method or changing the implementation.

New model integrations should document their official source, revision, weights,
license, preprocessing, input views, output representation, and valid support.
Keep model dependencies in a separate environment and include any required
upstream patches.

Contributions to first-party code are provided under the repository's
Apache-2.0 license. Retain upstream notices for third-party material.
