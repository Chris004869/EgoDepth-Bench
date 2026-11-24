"""Map recordings to physical sites and assign each site to one split.

Formal runs require an explicit physical-site mapping. Development manifests
may group samples by recording for testing without claiming site generalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SPLITS = ("train", "val", "test")

# The author-confirmed count. Used only as a consistency check on a supplied
# mapping, never to generate names.
CONFIRMED_SITE_COUNT = 37


@dataclass
class SiteMapping:
    """recording_id -> physical_site_id, plus site -> split."""

    recording_to_site: dict
    site_to_split: dict
    source: str
    status: str = "draft"
    # Track previous use when validating held-out test sites.
    development_history: dict = field(default_factory=dict)
    notes: str = ""
    history_policy: str = "fully_unseen"

    @property
    def sites(self) -> list:
        return sorted(set(self.recording_to_site.values()))

    def split_of_recording(self, recording_id: str) -> Optional[str]:
        site = self.recording_to_site.get(recording_id)
        return self.site_to_split.get(site) if site else None

    def validate(self, strict_site_count: bool = False) -> list:
        """Structural problems as a list of strings; empty means clean."""
        problems: list = []
        for site, split in self.site_to_split.items():
            if split not in SPLITS:
                problems.append(f"site {site!r} has invalid split {split!r}")
        unmapped = sorted({s for s in self.sites if s not in self.site_to_split})
        if unmapped:
            problems.append(
                f"{len(unmapped)} sites have no split: {unmapped[:5]}"
            )
        # One site must not straddle two splits: every recording, date and camera
        # of a site belongs to the same split, or the unseen-site claim collapses.
        by_site: dict = {}
        for recording, site in self.recording_to_site.items():
            by_site.setdefault(site, []).append(recording)
        for site, recordings in by_site.items():
            splits = {self.site_to_split.get(site) for _ in recordings}
            if len(splits) > 1:
                problems.append(f"site {site!r} maps to multiple splits: {splits}")
        if strict_site_count and len(self.sites) != CONFIRMED_SITE_COUNT:
            problems.append(
                f"mapping holds {len(self.sites)} sites, but {CONFIRMED_SITE_COUNT} "
                "physical locations are confirmed; reconcile before a formal run"
            )
        if self.history_policy not in {
            "fully_unseen", "held_out_from_training_with_disclosed_pilot_evaluation"
        }:
            problems.append(f"unknown history_policy: {self.history_policy!r}")
        def allowed_pilot(history):
            return (
                self.history_policy == "held_out_from_training_with_disclosed_pilot_evaluation"
                and isinstance(history, list) and bool(history)
                and all(isinstance(event, dict)
                        and event.get("kind") == "public_model_pilot_evaluation"
                        and event.get("counts_as_training") is False
                        and bool(event.get("source")) for event in history)
            )
        leaked = sorted(
            site for site, split in self.site_to_split.items()
            if split == "test" and self.development_history.get(site)
            and not allowed_pilot(self.development_history[site])
        )
        if leaked:
            problems.append(
                f"{len(leaked)} test sites have recorded development history "
                f"({leaked[:5]}); they are not fully unseen"
            )
        return problems

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "status": self.status,
            "notes": self.notes,
            "history_policy": self.history_policy,
            "n_sites": len(self.sites),
            "n_recordings": len(self.recording_to_site),
            "site_to_split": dict(sorted(self.site_to_split.items())),
            "development_history": dict(sorted(self.development_history.items())),
            "recording_to_site": dict(sorted(self.recording_to_site.items())),
        }

    @classmethod
    def load(cls, path: Path) -> "SiteMapping":
        payload = json.loads(Path(path).read_text())
        recording_to_site = payload.get("recording_to_site")
        if not isinstance(recording_to_site, dict) or not recording_to_site:
            raise ValueError(f"{path}: recording_to_site must be a non-empty object")
        site_to_split = payload.get("site_to_split")
        if not isinstance(site_to_split, dict):
            raise ValueError(f"{path}: site_to_split must be an object")
        return cls(
            recording_to_site={str(k): str(v) for k, v in recording_to_site.items()},
            site_to_split={str(k): str(v) for k, v in site_to_split.items()},
            source=str(payload.get("source", str(path))),
            status=str(payload.get("status", "draft")),
            development_history=payload.get("development_history", {}) or {},
            notes=str(payload.get("notes", "")),
            history_policy=str(payload.get("history_policy", "fully_unseen")),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")


def pseudo_site_mapping(recording_ids: list, split: str = "dev") -> SiteMapping:
    """Development-only mapping where each recording is its own pseudo-site.

    Explicitly NOT a site partition. It exists so smoke tests can exercise the
    site-macro aggregation path (which needs more than one site to be
    meaningful) while remaining rejected by `require_formal_sites`.
    """
    recording_to_site = {
        recording: f"pseudo_site::{recording}" for recording in sorted(set(recording_ids))
    }
    return SiteMapping(
        recording_to_site=recording_to_site,
        site_to_split={site: split for site in recording_to_site.values()},
        source="pseudo_site_mapping(development only)",
        status="development_pseudo_sites",
        notes=(
            "Each recording is treated as its own pseudo-site. This is a "
            "development placeholder for exercising site-macro aggregation. It is "
            "not a physical-site partition and must not be reported as one."
        ),
    )


def require_formal_sites(mapping: Optional[SiteMapping]) -> SiteMapping:
    """Gate for formal runs. Raises unless a real, frozen site mapping is present."""
    if mapping is None:
        raise ValueError(
            "No site mapping supplied. A formal run needs recording -> "
            "physical_site_id and site -> split. Provide "
            "`data.site_map_path` in the config; see "
            "configs/templates/site_map.example.json."
        )
    if mapping.status != "frozen":
        raise ValueError(
            f"site mapping status is {mapping.status!r}, not 'frozen'. A formal run "
            "requires a frozen mapping so the test sites cannot move after scores "
            "are seen."
        )
    problems = mapping.validate(strict_site_count=True)
    if problems:
        raise ValueError(
            "site mapping is not usable for a formal run:\n  - "
            + "\n  - ".join(problems)
        )
    return mapping


def summarize_site_support(mapping: SiteMapping, recording_ids: list) -> dict:
    """Per-split site and recording counts for the recordings actually present."""
    per_split: dict = {split: {"sites": set(), "recordings": 0} for split in SPLITS}
    unmapped: list = []
    for recording in recording_ids:
        site = mapping.recording_to_site.get(recording)
        if site is None:
            unmapped.append(recording)
            continue
        split = mapping.site_to_split.get(site)
        if split not in per_split:
            unmapped.append(recording)
            continue
        per_split[split]["sites"].add(site)
        per_split[split]["recordings"] += 1
    return {
        "per_split": {
            split: {"n_sites": len(value["sites"]), "n_recordings": value["recordings"],
                    "sites": sorted(value["sites"])}
            for split, value in per_split.items()
        },
        "n_unmapped_recordings": len(unmapped),
        "unmapped_examples": unmapped[:10],
    }
