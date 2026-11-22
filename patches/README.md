# Integration patches

The DAC, UniDAC and CUT3R patches apply through `python scripts/prepare_models.py --methods MODEL`, which pins the upstream commit, applies patches in order, and verifies the resulting changed-file hashes. Do not apply the pixel-center patches before their DS-projection prerequisites. See [third-party terms and attribution](../docs/THIRD_PARTY.md) and the complete available license texts in `../third_party_licenses/`. Patch context remains governed by upstream terms; the root license does not relicense it.

GeoStereo uses checkout-local loader configuration in `scripts/prepare_models.py`. Its resulting source hashes are recorded alongside the other models in `configs/model_sources.json`.
