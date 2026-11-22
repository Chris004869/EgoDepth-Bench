"""The annotation record: what the AI pipeline may assert, and what it must not.

STATUS IS PER FIELD, NOT PER FRAME.
A frame can have an accepted hand/object ROI and an unresolved motion label. The
ROI must still be usable for the region diagnostics -- dropping the whole frame
because motion was ambiguous would preferentially discard exactly the busy
manipulation frames the paper is about. So `roi_status` and `motion_status` are
separate fields with separate histories.

UNKNOWN IS A REGION, NOT A GAP.
`unknown` marks the un-annotated domain. It is reported, never folded into
background, and never scored as the model being right or wrong. The complement of
the candidate masks is NOT background: an unclaimed pixel is unknown until a
review confirms it holds no hand and no manipulated object.

TWO VLMs AGREEING IS NOT ACCURACY.
The review model is a *related* model trained on overlapping data. Agreement is
recorded as `review_agreement`, described as pipeline stability, and never
reported as a measured label-accuracy figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

ANNOTATION_SCHEMA = "egodepth-annotation-1"

# Region label values as stored in the mask array.
LABEL_UNKNOWN = -1
LABEL_HAND = 0
LABEL_OBJECT = 1
LABEL_BACKGROUND = 2

LABEL_NAMES = {
    LABEL_UNKNOWN: "unknown",
    LABEL_HAND: "hand",
    LABEL_OBJECT: "object",
    LABEL_BACKGROUND: "background",
}


class FieldStatus(str, Enum):
    """Lifecycle of one annotation field."""

    PROPOSED = "proposed"           # candidate generated, not yet reviewed
    ACCEPTED_AI = "accepted_ai"     # AI-generated and AI-reviewed, checks passed
    UNCERTAIN = "uncertain"         # review or checks unresolved after retries
    FAILED = "failed"               # the pipeline could not produce a candidate
    NOT_APPLICABLE = "not_applicable"   # e.g. no visible hand in this frame


class MotionLabel(str, Enum):
    """Motion condition. `UNKNOWN` never means static.

    The three components are recorded separately because "the camera moved" and
    "the object moved relative to the room" are different claims, and at 3 Hz the
    second is often undecidable. Active manipulation requires visible object
    motion *during* hand manipulation; low motion requires all three small.
    """

    ACTIVE_MANIPULATION = "active_manipulation"
    LOW_MOTION = "low_motion"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class MotionEvidence:
    """The three separately-recorded motion components."""

    background_image_motion: Optional[str] = None   # low | high | unknown
    hand_activity: Optional[str] = None             # none | present | unknown
    object_motion_vs_environment: Optional[str] = None  # low | high | unknown
    flow_summary: dict = field(default_factory=dict)
    reasoning: str = ""

    def derive_label(self) -> MotionLabel:
        """Combine the components. Any unknown component blocks a decision.

        Optical flow under a 200-degree fisheye is not a reliable global motion
        estimator, so flow only ever contributes to a component, and a missing
        component yields UNKNOWN rather than a default of static.
        """
        background = self.background_image_motion
        hand = self.hand_activity
        obj = self.object_motion_vs_environment
        if None in (background, hand, obj) or "unknown" in (background, hand, obj):
            return MotionLabel.UNKNOWN
        if hand == "present" and obj == "high":
            return MotionLabel.ACTIVE_MANIPULATION
        if background == "low" and hand in ("none",) and obj == "low":
            return MotionLabel.LOW_MOTION
        return MotionLabel.OTHER


@dataclass
class InstanceRecord:
    """One tracked instance: which region it belongs to and how it was obtained."""

    instance_id: str
    label: str                        # hand | object | background
    prompt: str = ""
    prompt_kind: str = "text"         # text | box | point
    score: Optional[float] = None
    n_pixels: int = 0
    first_seen_token: Optional[str] = None
    last_seen_token: Optional[str] = None
    # Set when the instance persists across a pause or a release-and-regrasp.
    identity_continuity: str = "unverified"   # verified | unverified | broken
    notes: dict = field(default_factory=dict)


@dataclass
class ReviewRound:
    """One visual review call. The raw structured reply is kept verbatim."""

    round_index: int
    reviewer_model: str
    reviewer_revision: Optional[str] = None
    inputs: dict = field(default_factory=dict)   # frames/overlays actually shown
    decision: str = "unknown"        # accept | revise | reject | unknown
    reasons: list = field(default_factory=list)
    objects_to_revise: list = field(default_factory=list)
    new_prompts: list = field(default_factory=list)
    raw_response: str = ""
    parse_ok: bool = True
    parse_error: str = ""
    # Deliberately named so nobody reads it as an accuracy figure.
    review_agreement: Optional[str] = None


@dataclass
class AnnotationRecord:
    """The frozen annotation for one sample.

    `region_mask_path` points at an int8 array on the scoring grid holding the
    LABEL_* values. Statuses gate what the diagnostics may use it for.
    """

    sample_token: str
    site_id: Optional[str]
    annotation_version: str
    region_mask_path: Optional[str] = None
    roi_status: str = FieldStatus.PROPOSED.value
    # Hand and object are reviewed separately and accepted separately. A frame
    # whose hand mask is confirmed while its manipulated object stays ambiguous
    # still contributes a usable hand ROI; collapsing both into one status would
    # throw away the confirmed region -- the same reason motion has its own status.
    hand_status: str = FieldStatus.PROPOSED.value
    object_status: str = FieldStatus.PROPOSED.value
    motion_status: str = FieldStatus.PROPOSED.value
    motion_label: str = MotionLabel.UNKNOWN.value
    motion_evidence: dict = field(default_factory=dict)
    instances: list = field(default_factory=list)
    label_pixel_counts: dict = field(default_factory=dict)
    candidate_backend: dict = field(default_factory=dict)
    review_rounds: list = field(default_factory=list)
    automatic_checks: dict = field(default_factory=dict)
    retry_history: list = field(default_factory=list)
    context_frames: list = field(default_factory=list)
    overlay_paths: dict = field(default_factory=dict)
    reason: str = ""
    schema: str = ANNOTATION_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AnnotationRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    @property
    def roi_usable(self) -> bool:
        """True when at least one region was accepted, so the frame has usable ROI."""
        return (
            self.roi_status == FieldStatus.ACCEPTED_AI.value
            or self.hand_usable
            or self.object_usable
        )

    @property
    def hand_usable(self) -> bool:
        return self.hand_status == FieldStatus.ACCEPTED_AI.value

    @property
    def object_usable(self) -> bool:
        return self.object_status == FieldStatus.ACCEPTED_AI.value

    @property
    def motion_usable(self) -> bool:
        """Motion-condition comparisons may use this frame.

        An unknown motion label keeps the frame in the main test set and in the
        ROI diagnostics; it only withdraws it from the motion contrast.
        """
        return (
            self.motion_status == FieldStatus.ACCEPTED_AI.value
            and self.motion_label != MotionLabel.UNKNOWN.value
        )


def summarize_statuses(records: list) -> dict:
    """Acceptance / uncertainty / failure counts, broken out by site and motion.

    Reported so a reader can see whether the automatic pipeline concentrated its
    rejections on particular sites or on the harder manipulations -- a pipeline
    that only accepts easy frames would otherwise look like a clean success.
    """
    summary: dict = {
        "n_records": len(records),
        "roi_status": {},
        "hand_status": {},
        "object_status": {},
        "motion_status": {},
        "motion_label": {},
        "by_site": {},
        "n_review_rounds": {},
    }
    for record in records:
        summary["roi_status"][record.roi_status] = (
            summary["roi_status"].get(record.roi_status, 0) + 1
        )
        for key in ("hand_status", "object_status"):
            value = getattr(record, key)
            summary[key][value] = summary[key].get(value, 0) + 1
        summary["motion_status"][record.motion_status] = (
            summary["motion_status"].get(record.motion_status, 0) + 1
        )
        summary["motion_label"][record.motion_label] = (
            summary["motion_label"].get(record.motion_label, 0) + 1
        )
        rounds = str(len(record.review_rounds))
        summary["n_review_rounds"][rounds] = summary["n_review_rounds"].get(rounds, 0) + 1

        site = record.site_id or "unmapped"
        bucket = summary["by_site"].setdefault(
            site,
            {"n": 0, "roi_accepted": 0, "roi_uncertain": 0, "roi_failed": 0,
             "hand_accepted": 0, "object_accepted": 0,
             "motion_accepted": 0, "motion_unknown": 0},
        )
        bucket["n"] += 1
        if record.hand_usable:
            bucket["hand_accepted"] += 1
        if record.object_usable:
            bucket["object_accepted"] += 1
        if record.roi_status == FieldStatus.ACCEPTED_AI.value:
            bucket["roi_accepted"] += 1
        elif record.roi_status == FieldStatus.UNCERTAIN.value:
            bucket["roi_uncertain"] += 1
        elif record.roi_status == FieldStatus.FAILED.value:
            bucket["roi_failed"] += 1
        if record.motion_usable:
            bucket["motion_accepted"] += 1
        if record.motion_label == MotionLabel.UNKNOWN.value:
            bucket["motion_unknown"] += 1

    summary["note"] = (
        "AI-generated and AI-reviewed. Review agreement describes pipeline "
        "stability between two related vision-language models; it is not a "
        "measured label-accuracy rate and the two models are not statistically "
        "independent."
    )
    return summary
