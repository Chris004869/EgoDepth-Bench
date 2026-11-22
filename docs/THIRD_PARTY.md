# Third-party components

The root Apache-2.0 license covers GenRobot's first-party benchmark code. It does not relicense upstream models, their weights, datasets, or patch context. This repository does not bundle upstream source trees or model weights. Download them from the official sources listed in `configs/model_sources.json` and follow their own terms. Code and weights can have different licenses.

The table below reports the license text or README statement found in the local upstream checkouts at the recorded commits. It is an evidence inventory, not a substitute for the full upstream terms. Weight terms not explicitly established in those files still need to be read at the official model source.

| Component | Local license evidence |
|---|---|
| `unik3d` | CC-BY-NC-SA-4.0 (LICENSE; pyproject metadata differs) |
| `dac` | MIT |
| `unidac` | MIT |
| `unidepth_v2` | CC-BY-NC-4.0 |
| `depth_pro` | Apple custom LICENSE |
| `metric3d_v2` | BSD-2-Clause |
| `depth_anything_3` | Apache-2.0 |
| `zoedepth` | MIT |
| `foundation_stereo` | NVIDIA custom LICENSE |
| `geostereo` | No project-wide license found. The `depth_anything` subdirectory includes Apache-2.0. |
| `mast3r` | CC-BY-NC-SA-4.0 + NOTICE/CHECKPOINTS_NOTICE |
| `cut3r` | CC-BY-NC-SA-4.0 |
| `vggt` | VGGT License (2025-07-29) |
| `vggt_omega` | FAIR Noncommercial Research License |
| `depth_anything_v2` | Apache-2.0 code; Large weights CC-BY-NC-4.0 |
| `marigold` | Apache-2.0 code; OpenRAIL++-M model |
| `moge` | MIT |
| `mapanything` | Apache-2.0 code and selected apache checkpoint |

In particular, Depth Anything V2 **Large** weights are CC-BY-NC-4.0 even though its source-tree LICENSE is Apache-2.0. The selected DA3METRIC-LARGE and `facebook/map-anything-apache` variants are the Apache variants stated by their upstream READMEs. UniK3D's LICENSE says CC-BY-NC-SA-4.0 while its package metadata says CC-BY-NC; do not silently collapse that discrepancy. No top-level GeoStereo license was found in the inspected checkout, so no permission or license has been inferred for it.

## Integration changes

Three upstreams use patch files. The preparation script applies each series in order and verifies the SHA-256 of the resulting files against the research checkout.

- **DAC.** `dac_ds_projection_and_ray_reverse.patch` and `dac_ds_pixel_centres.patch` add double-sphere forward projection, use the supplied ray table for reverse projection and match pixel-center normalization to `align_corners=True`.
- **UniDAC.** `unidac_ds_projection.patch`, `unidac_ray_table_reverse.patch` and `unidac_ds_pixel_centres.patch` implement the corresponding camera integration in its separate source tree.
- **CUT3R.** `cut3r_rope_negative_positions.patch` makes the PyTorch RoPE fallback handle negative token positions using the arithmetic of the compiled kernel.

License notices for this patch context are included in `third_party_licenses/`. DAC credits Sparsh Garg (2025) and UniDAC credits Girish Chandar Ganesan (2026). CUT3R supplies its copyright and license notice in the upstream source. The geometry checks cover resampling in the DAC and UniDAC adapters. They do not establish upstream validation of full model outputs.

For **GeoStereo**, the distribution contains the benchmark's own instructions for configuring the two backbone loaders in a user-supplied checkout. `scripts/prepare_models.py` checks the pinned commit, verifies the resulting files against their archived digests and preserves any unrecognized local edits. The public package carries no GeoStereo diff or vendor files. The historical DINOv2 and EdgeNeXt revisions were not recorded. Consult the authors for GeoStereo project-wide terms. [Pinned source tree](https://github.com/qz-wei/GeoStereo/tree/9b13879530d5a23eb2557da17f5b9c747b673130) · [Depth Anything subdirectory license](https://github.com/qz-wei/GeoStereo/blob/9b13879530d5a23eb2557da17f5b9c747b673130/GeoStereo/depth_anything/LICENSE.txt)
