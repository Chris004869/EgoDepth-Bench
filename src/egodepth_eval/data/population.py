"""Resolve split-specific populations and validate the inputs of formal runs.

Training and calibration use training sites, validation uses validation sites,
and formal test scoring uses the complete frozen main test population.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from egodepth_eval.data.contract import CanonicalSample
from egodepth_eval.data.paper_protocol import verify_paper_population
from egodepth_eval.data.mask_profile import get_profile, is_formal_profile
from egodepth_eval.data.site_map import SiteMapping, require_formal_sites


_RECORDING_ID = re.compile(rb'"recording_id"\s*:\s*("(?:\\.|[^"\\])*")')


def recording_id_from_line(raw):
    match = _RECORDING_ID.search(raw)
    if match is None:
        raise ValueError('Manifest row lacks a string recording_id')
    return json.loads(match.group(1))


class PopulationRole(str, Enum):
    """What the samples are for. Determines which split may be loaded."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    CALIBRATION = "calibration"     # subset C: train only
    ANNOTATION = "annotation"       # the diagnostic subset, drawn from test
    DEVELOPMENT = "development"     # no split constraint; never formal


# Role -> the only split it may read. A role is not free to pick.
ROLE_SPLIT = {
    PopulationRole.TRAIN: "train",
    PopulationRole.VAL: "val",
    PopulationRole.TEST: "test",
    PopulationRole.CALIBRATION: "train",
    PopulationRole.ANNOTATION: "test",
    PopulationRole.DEVELOPMENT: None,
}


class FormalRunBlocked(RuntimeError):
    """Raised when a formal run is requested without every precondition met.

    Never caught to downgrade the run: a formal request that cannot be honoured
    must fail, not silently become a development run.
    """


@dataclass
class Population:
    """A resolved, split-filtered sample set with a verifiable identity."""

    role: PopulationRole
    split: Optional[str]
    samples: list
    status: str                       # formal | development
    site_mapping: Optional[SiteMapping]
    mask_profile_name: str
    data_version: Optional[str]
    manifest_path: str
    manifest_content_id: str
    annotation_version: Optional[str] = None
    formal_ready: bool = False
    blockers: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)
    # A derived population must establish its own scientific identity.
    paper_protocol: Optional[dict] = field(default=None, init=False)
    _token_cache: Optional[tuple] = field(default=None, init=False, repr=False)

    @property
    def sites(self) -> list:
        # Reuse the split summary to avoid decoding large training manifests.
        summary = self.notes.get("site_mapping_summary")
        if isinstance(self.samples, (IndexBackedSampleList, LazySampleList)) and summary and "sites" in summary:
            return list(summary["sites"])
        return sorted({s.physical_site_id for s in self.samples if s.physical_site_id})

    def _tokens_and_id(self):
        cached = self._token_cache
        if cached is not None and cached[0] is self.samples:
            return cached[1], cached[2]
        direct = getattr(self.samples, "tokens", None)
        tokens = list(direct) if direct is not None else [s.sample_token for s in self.samples]
        digest = hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()[:16]
        # Ordinary lists used by derived populations may change without a resize.
        if isinstance(self.samples, (IndexBackedSampleList, LazySampleList)):
            self._token_cache = (self.samples, tokens, digest)
        return tokens, digest

    @property
    def sample_tokens(self) -> list:
        tokens, _ = self._tokens_and_id()
        return list(tokens)

    def token_set_id(self) -> str:
        """Return the historical digest used by the archived paper results."""
        _, digest = self._tokens_and_id()
        return digest

    def identity(self) -> dict:
        return {
            "role": self.role.value,
            "split": self.split,
            "status": self.status,
            "n_samples": len(self.samples),
            "n_sites": len(self.sites),
            "sites": self.sites,
            "mask_profile": self.mask_profile_name,
            "data_version": self.data_version,
            "manifest_path": self.manifest_path,
            "manifest_content_id": self.manifest_content_id,
            "token_set_id": self.token_set_id(),
            "annotation_version": self.annotation_version,
            "paper_protocol": self.paper_protocol,
            "formal_ready": self.formal_ready,
            "blockers": self.blockers,
            "site_mapping_status": (
                self.site_mapping.status if self.site_mapping else None
            ),
            **{k: v for k, v in self.notes.items() if not k.startswith("_")},
        }

    def assert_disjoint_from(self, other: "Population") -> None:
        """Fail if two populations share a sample or a site.

        Called between train and val, and between train/val and test. Sharing a
        *site* is the failure that matters: two different timestamps from the same
        room are not independent, so token-level disjointness alone is not enough.
        """
        shared_tokens = set(self.sample_tokens) & set(other.sample_tokens)
        if shared_tokens:
            raise FormalRunBlocked(
                f"{self.role.value} and {other.role.value} share "
                f"{len(shared_tokens)} samples, e.g. {sorted(shared_tokens)[:3]}"
            )
        shared_sites = set(self.sites) & set(other.sites)
        if shared_sites:
            raise FormalRunBlocked(
                f"{self.role.value} and {other.role.value} share "
                f"{len(shared_sites)} physical sites, e.g. {sorted(shared_sites)[:3]}. "
                "Every recording, date and camera of a site must be in one split."
            )


def manifest_content_id(path: Path) -> str:
    """Hash the manifest bytes so relocation preserves its content identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def apply_site_mapping(
    samples: Sequence[CanonicalSample], mapping: SiteMapping
) -> dict:
    """Apply the authoritative recording map in place and summarize coverage."""
    summary = {"n_mapped": 0, "n_unmapped": 0, "unmapped_examples": [],
               "splits": {}, "sites": set()}
    for sample in samples:
        site = mapping.recording_to_site.get(sample.recording_id)
        if site is None:
            summary["n_unmapped"] += 1
            if len(summary["unmapped_examples"]) < 10:
                summary["unmapped_examples"].append(sample.recording_id)
            sample.physical_site_id = None
            sample.split = None
            continue
        split = mapping.site_to_split.get(site)
        sample.physical_site_id = site
        sample.split = split
        summary["n_mapped"] += 1
        summary["sites"].add(site)
        summary["splits"][split] = summary["splits"].get(split, 0) + 1
    summary["sites"] = sorted(summary["sites"])
    return summary


class IndexBackedSampleList(Sequence):
    """Resolve sample tokens through a SQLite byte-offset index beside samples.jsonl."""

    def __init__(self, index_dir: Path, tokens: list, mapping=None,
                 recording_filter: Optional[set] = None, cache_size: int = 512):
        from egodepth_eval.data.manifest_store import ManifestStore

        self._store = ManifestStore(Path(index_dir))
        self._tokens = list(tokens)
        self._mapping = mapping
        # The recordings this population was filtered to, so the mapping summary can
        # aggregate in SQL instead of streaming every row.
        self.recording_filter = recording_filter
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_size = int(cache_size)

    def __len__(self) -> int:
        return len(self._tokens)

    def _decode(self, index: int) -> CanonicalSample:
        sample = self._store.get(self._tokens[index])
        if self._mapping is not None:
            site = self._mapping.recording_to_site.get(sample.recording_id)
            sample.physical_site_id = site
            sample.split = self._mapping.site_to_split.get(site) if site else None
        return sample

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        sample = self._decode(index)
        self._cache[index] = sample
        self._cache_order.append(index)
        if len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def tokens(self) -> list:
        return list(self._tokens)

    def materialize(self) -> list:
        return [self._decode(index) for index in range(len(self))]


def tokens_for_recordings(index_dir: Path, recordings: Optional[set]) -> list:
    """Sample tokens for the given recordings, via SQL, in a deterministic order."""
    from egodepth_eval.data.manifest_store import ManifestStore

    store = ManifestStore(Path(index_dir))
    if recordings is None:
        return store.tokens()
    tokens: list = []
    # Chunked IN clauses: SQLite's default variable limit is 999.
    ordered = sorted(recordings)
    for start in range(0, len(ordered), 900):
        chunk = ordered[start:start + 900]
        placeholders = ",".join("?" * len(chunk))
        rows = store.connection.execute(
            f"SELECT sample_token FROM samples WHERE recording_id IN ({placeholders}) "
            "ORDER BY recording_id, timestamp_ns",
            chunk,
        )
        tokens.extend(row[0] for row in rows)
    return tokens


class LazySampleList(Sequence):
    """Decode selected raw JSONL rows on access and apply the physical-site mapping."""

    def __init__(self, lines: list, mapping=None, cache_size: int = 512):
        self._lines = lines
        self._mapping = mapping
        self._cache: dict = {}
        self._cache_order: list = []
        self._cache_size = int(cache_size)

    def _decode(self, index: int) -> CanonicalSample:
        sample = CanonicalSample.from_dict(json.loads(self._lines[index]))
        if self._mapping is not None:
            site = self._mapping.recording_to_site.get(sample.recording_id)
            sample.physical_site_id = site
            sample.split = self._mapping.site_to_split.get(site) if site else None
        return sample

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        sample = self._decode(index)
        self._cache[index] = sample
        self._cache_order.append(index)
        if len(self._cache_order) > self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        return sample

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def materialize(self) -> list:
        """A real list, for callers that need to mutate the samples in place."""
        return [self._decode(index) for index in range(len(self))]


def summarize_mapping_for_samples(samples, mapping) -> dict:
    """Count recordings and sites from this selection, including a materialized subset."""
    counts: dict = {}
    if isinstance(samples, IndexBackedSampleList):
        store = samples._store
        # Count the rows belonging to the selected recordings inside SQLite.
        recordings = getattr(samples, "recording_filter", None)
        if recordings is not None:
            ordered = sorted(recordings)
            for start in range(0, len(ordered), 900):
                chunk = ordered[start:start + 900]
                placeholders = ",".join("?" * len(chunk))
                for recording, n in store.connection.execute(
                    "SELECT recording_id, COUNT(*) FROM samples "
                    f"WHERE recording_id IN ({placeholders}) GROUP BY recording_id",
                    chunk,
                ):
                    counts[recording] = n
        else:
            for recording, n in store.connection.execute(
                "SELECT recording_id, COUNT(*) FROM samples GROUP BY recording_id"
            ):
                counts[recording] = n
    elif isinstance(samples, LazySampleList):
        counts = Counter(recording_id_from_line(raw) for raw in samples._lines)
    else:
        counts = Counter(sample.recording_id for sample in samples)

    summary = {"n_mapped": 0, "n_unmapped": 0, "unmapped_examples": [],
               "splits": {}, "sites": set()}
    for recording, n in counts.items():
        site = mapping.recording_to_site.get(recording)
        if site is None:
            summary["n_unmapped"] += n
            if len(summary["unmapped_examples"]) < 10:
                summary["unmapped_examples"].append(recording)
            continue
        split = mapping.site_to_split.get(site)
        summary["n_mapped"] += n
        summary["sites"].add(site)
        summary["splits"][split] = summary["splits"].get(split, 0) + n
    summary["sites"] = sorted(summary["sites"])
    summary["n_recordings"] = len(counts)
    return summary


def summarize_mapping_over_lines(lines: list, mapping) -> dict:
    """Mapping coverage over the retained lines, without decoding the samples."""
    summary = {"n_mapped": 0, "n_unmapped": 0, "unmapped_examples": [],
               "splits": {}, "sites": set()}
    for raw in lines:
        recording = recording_id_from_line(raw)
        site = mapping.recording_to_site.get(recording)
        if site is None:
            summary["n_unmapped"] += 1
            if len(summary["unmapped_examples"]) < 10:
                summary["unmapped_examples"].append(recording)
            continue
        split = mapping.site_to_split.get(site)
        summary["n_mapped"] += 1
        summary["sites"].add(site)
        summary["splits"][split] = summary["splits"].get(split, 0) + 1
    summary["sites"] = sorted(summary["sites"])
    return summary


def load_manifest_lines(
    path: Path,
    keep_recordings: Optional[set] = None,
) -> list:
    """Read selected JSONL rows, filtering by recording before sample decoding."""
    lines: list = []
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            if keep_recordings is not None:
                if recording_id_from_line(raw) not in keep_recordings:
                    continue
            lines.append(raw)
    return lines


def load_manifest_samples(
    path: Path,
    keep_recordings: Optional[set] = None,
) -> list:
    """Stream canonical samples, optionally retaining selected recordings only."""
    samples = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            if keep_recordings is not None:
                if payload.get("recording_id") not in keep_recordings:
                    continue
            samples.append(CanonicalSample.from_dict(payload))
    return samples


def recordings_for_split(mapping: SiteMapping, split: Optional[str]) -> Optional[set]:
    """Resolve recording membership from the authoritative physical-site split."""
    if split is None:
        return None
    return {
        recording for recording, site in mapping.recording_to_site.items()
        if mapping.site_to_split.get(site) == split
    }


def resolve_population(
    config: dict,
    role: PopulationRole,
    limit: Optional[int] = None,
    require_annotations: bool = False,
    precomputed_manifest_id: Optional[str] = None,
) -> Population:
    """THE gate. Every entry point obtains samples through this and nothing else.

    Applies the site mapping, filters to the role's split, and -- when
    `run.status` is `formal` -- raises `FormalRunBlocked` unless every formal
    precondition holds. There is deliberately no flag to skip the check, and no
    path that downgrades a failed formal request to a development run.
    """
    data = config.get("data", {})
    intended_status = str(config.get("run", {}).get("status", "development"))
    wants_formal = intended_status == "formal"

    manifest_path = Path(data.get("samples_jsonl", ""))
    if role == PopulationRole.ANNOTATION and config.get("annotation", {}).get("diagnostics_manifest"):
        manifest_path = Path(config["annotation"]["diagnostics_manifest"])
        precomputed_manifest_id = None
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing samples manifest: {manifest_path}")
    if wants_formal and role == PopulationRole.DEVELOPMENT:
        raise FormalRunBlocked("The development role requires run.status=development")
    if limit is not None:
        if wants_formal:
            raise FormalRunBlocked(f"--limit {limit} is restricted to development runs")
        if limit <= 0:
            raise ValueError("--limit must be a positive number of samples")
    content_id = precomputed_manifest_id or manifest_content_id(manifest_path)
    blockers: list = []
    notes: dict = {}

    # --- site mapping -------------------------------------------------
    # Loaded BEFORE the manifest scan, so the role's split can be turned into a
    # recording filter and the other splits are never materialized.
    mapping: Optional[SiteMapping] = None
    site_map_path = data.get("site_map_path")
    if site_map_path and Path(site_map_path).exists():
        mapping = SiteMapping.load(Path(site_map_path))

    wanted_split_early = ROLE_SPLIT[role]
    keep = (
        recordings_for_split(mapping, wanted_split_early)
        if mapping is not None else None
    )
    # Prefer the SQLite index: the split filter becomes a SQL query and only the
    # token list is retained. Fall back to scanning the JSONL for a bare subset
    # directory that has no index.
    index_dir = manifest_path.parent
    if manifest_path.name == 'samples.jsonl' and (index_dir / "index.sqlite").exists():
        tokens = tokens_for_recordings(index_dir, keep)
        samples = IndexBackedSampleList(
            index_dir, tokens, mapping=mapping, recording_filter=keep
        )
        notes["sample_storage"] = (
            f"sqlite-backed: {len(tokens)} tokens retained, samples decoded on access"
        )
        recording_source = "sqlite"
    else:
        lines = load_manifest_lines(manifest_path, keep_recordings=keep)
        samples = LazySampleList(lines, mapping=mapping)
        notes["sample_storage"] = (
            f"jsonl-backed: {len(lines)} raw lines retained "
            f"({sum(len(line) for line in lines) / 1e6:.0f} MB), decoded on access"
        )
        recording_source = "jsonl"

    notes["n_samples_loaded"] = len(samples)
    notes["recording_prefilter"] = (
        None if keep is None
        else {"split": wanted_split_early, "n_recordings": len(keep),
              "source": recording_source}
    )

    if mapping is not None:
        if wants_formal:
            # `require_formal_sites` rejects a non-frozen mapping, a wrong site
            # count, a site spanning two splits, or a test site with recorded
            # development history. Its ValueError becomes FormalRunBlocked so every
            # entry point reports it through one handler rather than as a traceback.
            try:
                require_formal_sites(mapping)
            except ValueError as exc:
                raise FormalRunBlocked(str(exc)) from exc
    else:
        blockers.append(
            f"no site mapping at {site_map_path!r}: recording IDs are not physical "
            "sites and cannot be substituted"
        )
        if not any(s.physical_site_id for s in samples):
            notes["site_mapping_summary"] = {
                "n_mapped": 0,
                "note": "manifest carries no site ids and no mapping was supplied",
            }

    # --- mask profile -------------------------------------------------
    profile_name = str(data.get("mask_profile", ""))
    try:
        profile = get_profile(profile_name)      # raises on an unknown name
    except KeyError as exc:
        raise FormalRunBlocked(str(exc)) from exc
    if not is_formal_profile(profile_name):
        blockers.append(
            f"mask profile {profile_name!r} has validity_source="
            f"{profile.validity_source}: a formal run needs validity from a real "
            "mask file, not derived from the depth array"
        )

    # --- annotation ---------------------------------------------------
    annotation = config.get("annotation", {})
    annotation_version = annotation.get("version")
    if require_annotations:
        if not (annotation.get("dir") and Path(annotation["dir"]).exists()):
            blockers.append(f"no annotation directory at {annotation.get('dir')!r}")
        if str(annotation.get("status")) != "frozen":
            blockers.append(
                f"annotation status is {annotation.get('status')!r}, not 'frozen'"
            )

    # --- split filter -------------------------------------------------
    # The recording prefilter above already removed other splits when a mapping was
    # available; this still runs, because a manifest may carry its own split fields
    # and the assertion below must hold either way.
    wanted_split = wanted_split_early
    if wanted_split is None:
        selected = samples
    elif keep is not None:
        # The recording prefilter selects the split before sample decoding.
        selected = samples
        notes["split_filter"] = (
            f"applied as a recording prefilter over {len(keep)} recordings; "
            "the split is a property of the site mapping, not of the sample rows"
        )
    else:
        selected = [s for s in samples if s.split == wanted_split]
        present = sorted({str(s.split) for s in samples})
        notes["splits_present_in_manifest"] = present
        notes["n_filtered_out"] = len(samples) - len(selected)
        if not selected:
            message = (
                f"role {role.value} needs split {wanted_split!r}, and the manifest "
                f"holds none. Splits present: {present}."
            )
            if wants_formal:
                raise FormalRunBlocked(message)
            blockers.append(message)

    if limit is not None:
        notes["n_samples_before_limit"] = len(selected)
        notes["limit_applied"] = limit
        selected = selected[:limit]

    if mapping is not None:
        summary = summarize_mapping_for_samples(selected, mapping)
        notes["site_mapping_summary"] = summary
        if keep is not None:
            notes["splits_present_after_filter"] = sorted(str(k) for k in summary["splits"])

    if wants_formal and not len(selected):
        raise FormalRunBlocked(f'Role {role.value} resolved to an empty population from {manifest_path}')

    # Check decoded edge/interior rows as well as the recording map. Formal
    # main-test identities are checked exhaustively after this common split gate.
    if wanted_split is not None and len(selected) > 0:
        n_probe = min(len(selected), max(64, min(1024, len(selected) // 200)))
        stride = max(1, len(selected) // n_probe)
        probe_indices = list(range(0, len(selected), stride))[:n_probe]
        # Always include the ends, where an off-by-one in a filter shows up.
        for edge in (0, len(selected) - 1):
            if edge not in probe_indices:
                probe_indices.append(edge)

        wrong, unsited = [], []
        for index in probe_indices:
            sample = selected[index]
            if sample.split != wanted_split:
                wrong.append(sample.sample_token)
            if not sample.physical_site_id:
                unsited.append(sample.sample_token)
        notes["split_assertion"] = {
            "n_probed": len(probe_indices),
            "n_retained": len(selected),
            "stride": stride,
            "includes_first_and_last": True,
            "n_wrong_split": len(wrong),
            "n_without_site": len(unsited),
            "exhaustive_over": (
                f"{len(mapping.recording_to_site)} recording assignments; decoded rows sampled"
                if mapping is not None else "decoded rows sampled; no recording map"
            ),
        }
        if wrong:
            raise FormalRunBlocked(
                f"{len(wrong)} of {len(probe_indices)} probed samples have a split "
                f"other than {wanted_split!r}, e.g. {wrong[:3]}"
            )
        if unsited:
            message = (
                f"{len(unsited)} of {len(probe_indices)} probed samples in split "
                f"{wanted_split!r} have no physical_site_id, e.g. {unsited[:3]}"
            )
            if wants_formal:
                raise FormalRunBlocked(message)
            blockers.append(message)

    formal_ready = wants_formal and not blockers
    if wants_formal and not formal_ready:
        raise FormalRunBlocked(
            "run.status is 'formal' but these preconditions are unmet:\n  - "
            + "\n  - ".join(blockers)
            + "\nRun `run_eval_pipeline.py preflight` for the full report."
        )

    population = Population(
        role=role,
        split=wanted_split,
        samples=selected,
        # The status a result is stamped with is derived, never copied from the
        # config: a config claiming `formal` without the preconditions cannot
        # reach this point, and a development config cannot mislabel itself.
        status="formal" if (wants_formal and formal_ready) else "development",
        site_mapping=mapping,
        mask_profile_name=profile_name,
        data_version=data.get("data_version"),
        manifest_path=str(manifest_path),
        manifest_content_id=content_id,
        annotation_version=annotation_version,
        formal_ready=formal_ready,
        blockers=blockers,
        notes=notes,
    )
    if wants_formal and role in (PopulationRole.TEST, PopulationRole.ANNOTATION):
        try:
            population.paper_protocol = verify_paper_population(
                config, selected, diagnostic=(role == PopulationRole.ANNOTATION)
            )
        except (ValueError, KeyError, OSError) as exc:
            raise FormalRunBlocked(f"Formal v1 protocol validation failed: {exc}") from exc
    return population


def resolve_train_val(config: dict, limit: Optional[int] = None) -> tuple:
    """Train and val populations, asserted disjoint in both samples and sites.

    Model selection uses the validation split.
    """
    manifest_id = manifest_content_id(Path(config["data"]["samples_jsonl"]))
    train = resolve_population(
        config, PopulationRole.TRAIN, limit=limit,
        precomputed_manifest_id=manifest_id,
    )
    val = resolve_population(
        config, PopulationRole.VAL, limit=limit,
        precomputed_manifest_id=manifest_id,
    )
    train.assert_disjoint_from(val)
    return train, val
