# Reading EgoDepth-Bench

The dataset pairs six synchronized fisheye RGB images with one camera-2 reference
depth map per timestamp. The code repository contains the SDK and protocol
configuration; dataset images, depth arrays and split manifests are separate
data assets. Raw IR/MCAP recordings and the GT production pipeline are not
included in this code release. The code's Apache-2.0 license does not license
data assets. Dataset hosting and distribution terms are not assigned in this
repository yet.

| Split | Physical sites | Recordings | Dense timestamps | Official anchors |
|---|---:|---:|---:|---:|
| Train | 24 | 755 | 609,652 | 61,085 |
| Validation | 5 | 163 | 134,637 | 4,491 |
| Test | 8 | 237 | 196,944 | 6,574 |

There are 941,233 unique timestamps and 5,647,398 RGB images. All recordings and
revisits of a physical site stay in one split. Test sites were held out from
fine-tuning; some were seen in earlier public-model development evaluations.

## Start with twelve training anchors

The separate `egodepth-starter.tar.gz` package contains twelve anchors from four
training sites (6,510,607 bytes). It is a reader example, not a benchmark subset.
Its original SHA256 is
`ca71afbc73ea7805f6dbc123b8ed3d8660856b85acce68d6643e225788fbc54f`.
No download URL is implied by this filename. When the dataset is distributed,
obtain the package and its manifest from the dataset's official host.

Use the current repository SDK with the extracted starter. The starter's older
embedded annotation helpers are not the current v6 annotation interface.

```bash
python -m pip install numpy Pillow
tar -xzf /path/to/egodepth-starter.tar.gz -C /path/to/data
python examples/read_starter.py \
  --data /path/to/data/egodepth-starter \
  --output outputs/starter_check.json
```

The CPU example reads all twelve anchors, checks six independent calibrated ray
fields, and feeds identity predictions through the scoring functions. No PyTorch,
model weights or GPU are required. A passing identity check verifies the reader
and metric conventions; it does not measure GT accuracy or model performance.
It writes only the optional report, outside the dataset.

## Full-release layout and restoration

Keep the published metadata and byte-offset index beside the corresponding
uncompressed tar shards:

```text
egodepth_v1_20260912/
  metadata/{samples.jsonl,index.sqlite,calibrations.json,sites.json}
  protocols/{train_3hz,val_1hz,test_1hz,all_official,diagnostics}.jsonl
  protocols/sampling.json
  splits/site_map.json
  annotation_version.json
  annotations_v6/
  downloads/download_manifest.json
  downloads/.../*.tar.zst
  checksums/metadata_sha256.json
  shards/{train,val,test}/*.tar
```

The complete compressed release is approximately 454.75 GB; restored indexed
shards occupy 882.62 GB across 1,195 files (decimal units). Keep space for both
representations during restoration, or download one split at a time. Preserve
the exact uncompressed tar bytes: the SDK seeks directly to indexed byte ranges
and does not extract millions of files.

After obtaining the metadata package and a split's compressed shards:

```bash
python -m pip install -r sdk/requirements.txt
python sdk/verify_checksums.py /path/to/release --assets compressed --split test
python sdk/restore_downloads.py /path/to/release --split test
python sdk/verify_checksums.py /path/to/release --assets restored --split test
python sdk/egodepth_dataset.py /path/to/release \
  --split test --protocol protocols/test_1hz.jsonl
```

The restore command consumes files already downloaded; it is not a network
downloader. It verifies compressed and restored SHA256, rejects paths escaping
the dataset root, and atomically installs each verified shard. Interrupted or
failed restoration leaves any existing destination intact and removes its own
temporary partial file. Rerunning skips already verified destinations.
`verify_checksums.py` always verifies the metadata listed in the release manifest;
keep that complete metadata package even when downloading only one split.

## Python interface and geometric contract

```python
import sys
sys.path.insert(0, "sdk")
from egodepth_dataset import EgoDepthDataset, unit_rays

data = EgoDepthDataset(
    "/path/to/release", split="train", protocol="protocols/train_3hz.jsonl"
)
try:
    sample = data[0]
    images = sample["images"]       # camera id -> H x W x 3 uint8 RGB
    z = sample["target_depth_z"]    # camera-2 optical-axis Z, metres, float32
    valid = sample["target_valid"]  # bool; True means reference-valid
    rays = {c: unit_rays(sample["calibrations"][str(c)]) for c in range(6)}
finally:
    data.close()
```

- All released grids are W512 × H416. Each camera has its own double-sphere
  calibration. Resizing or cropping RGB requires the same geometry transform.
- Depth is optical-axis **Z**, not ray distance. For a unit ray `u`, distance is
  `Z / u_z` only where `u_z > 0`. RGB FoV is not GT coverage.
- Stored depth is float16, promoted to float32 when read. The boolean mask is
  already corrected to **True = valid**; do not invert it again.
- Only camera 2 has a released reference target. RGB, rays and rig calibration
  are model inputs; reference depth/masks are supervision or scoring targets.
- The GT retains its full range. The scorer selects reference pixels in
  `[0.25, 2.0]` metres; it does not clip predictions to that interval.
- Prediction support and reference validity are distinct. Pixel statistics pool
  within each site, then sites receive equal weight. Missing predictions lower
  coverage and cannot silently remove a site.

Omit `protocol` for the dense split. The reader works as a map-style PyTorch
Dataset without importing torch, and reopens handles in worker processes. Its
indexed I/O uses `os.pread`; Linux is the supported execution environment.

## Frozen protocols and diagnostic labels

Training uses 3 Hz anchors and validation/test use 1 Hz anchors. Within each
recording, the fixed grid is `start + (k + 0.5) / Hz`; choose the nearest
score-eligible timestamp within 50 ms and prefer the earlier one on ties.
Missing frames never shift the grid. `all_official.jsonl` concatenates the three
partitions for split-aware consumers and must not be used as one training set.

`annotation_version.json` selects `annotations_v6/` (`ai-rgb-reviewed-v6`). The
800 fixed diagnostic anchors contain reference-bearing accepted hand masks on
291 frames and object masks on 257 frames, spanning eight sites. These are
AI-generated and AI-reviewed diagnostic labels with uncertainty, not independently
validated human ground truth. Unknown or unaccepted regions stay unknown; an
unpainted pixel is not automatically background. Use
`sdk/annotations.py::read_annotation` to enforce per-field acceptance.

For older consumers that require ordinary local mask paths, materialize derived
records outside the dataset:

```bash
python sdk/local_annotation_records.py /path/to/release outputs/annotation_records
```

The code starts from calibrated RGB, camera-2 Z and released validity masks.
The paper describes the upstream IR reconstruction and physical validation. Its
1.27 mm result is the median absolute error in recovered ball-bar center spacing
over 1,220 accepted measurements. [Paper, Geometric Accuracy Validation](https://chris004869.github.io/EgoDepth-Bench/site-assets/egodepth-bench.pdf)
