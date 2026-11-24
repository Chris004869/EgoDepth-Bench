"""Reference validity for released and historical development samples.

Released v1 masks use True for valid measurements. A prediction is evaluated
where the mask is true and depth is finite, positive and in the scoring range.
Older data snapshots have explicit development profiles with different mask
semantics. The reader never infers mask direction from the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

VALIDITY_SOURCES = ("finite_positive", "declared_mask")


@dataclass(frozen=True)
class MaskProfile:
    """How to turn stored depth + mask files into the reference-valid domain."""

    name: str
    validity_source: str
    # For declared_mask: does a stored 1 mean valid (per mask_encoding) or invalid?
    one_is_valid: bool = True
    # Fail if the declared mask file is absent, rather than falling back.
    require_mask_file: bool = False
    # Compare the NPY mask against finite-positive and record the disagreement.
    audit_inverse_semantics: bool = True
    # Refuse a frame whose mask disagrees with the profile beyond this share.
    max_audit_disagreement: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.validity_source not in VALIDITY_SOURCES:
            raise ValueError(
                f"unknown validity_source {self.validity_source!r}; "
                f"expected one of {VALIDITY_SOURCES}"
            )


PROFILE_336L_LEGACY_20260828 = MaskProfile(
    name="336l_legacy_20260828",
    validity_source="finite_positive",
    one_is_valid=False,
    require_mask_file=False,
    audit_inverse_semantics=True,
    notes=(
        "Legacy finite-positive depth. The optional stored mask uses zero for "
        "valid pixels and is used only for the per-frame mask audit."
    ),
)

# Development-only compatibility with an earlier export whose stored NPY mask was inverted.
PROFILE_336L_20260912_INVERTED_NPY = MaskProfile(
    name="336l_20260912_inverted_npy",
    validity_source="declared_mask",
    one_is_valid=False,
    require_mask_file=True,
    audit_inverse_semantics=True,
    max_audit_disagreement=1.0,
    notes=(
        "Legacy inverted mask. A pixel is valid when its stored mask is zero "
        "and its depth is finite and positive."
    ),
)

# Frozen v1 release mask, shared by all formal runs.
PROFILE_RELEASE_DECLARED = MaskProfile(
    name="release_declared_mask",
    validity_source="declared_mask",
    one_is_valid=True,
    require_mask_file=True,
    audit_inverse_semantics=True,
    max_audit_disagreement=1.0,
    notes=(
        "The release requires an explicit True-is-valid mask. A pixel is valid "
        "when the mask is true and its depth is finite and positive."
    ),
)

PROFILES = {
    PROFILE_336L_LEGACY_20260828.name: PROFILE_336L_LEGACY_20260828,
    PROFILE_336L_20260912_INVERTED_NPY.name: PROFILE_336L_20260912_INVERTED_NPY,
    PROFILE_RELEASE_DECLARED.name: PROFILE_RELEASE_DECLARED,
}

def is_formal_profile(name: str) -> bool:
    """The official v1 release stores an explicit True-is-valid mask."""
    return name == PROFILE_RELEASE_DECLARED.name


def get_profile(name: str) -> MaskProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown mask profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


def decode_packed_mask(
    payload: bytes, height: int, width: int, bitorder: str = "little"
) -> np.ndarray:
    """Decode a packbits validity mask to a bool array of shape (height, width).

    The last byte can contain up to seven zero padding bits after the final
    pixel. Reject nonzero padding or extra bytes before reshaping the mask.
    """
    if bitorder not in ("little", "big"):
        raise ValueError(f"bitorder must be little or big, got {bitorder!r}")
    n_expected = int(height) * int(width)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder=bitorder)
    if bits.size < n_expected:
        raise ValueError(
            f"packed mask holds {bits.size} bits, need {n_expected} for "
            f"{height}x{width}"
        )
    padding = bits.size - n_expected
    if padding >= 8:
        raise ValueError(
            f"packed mask has {padding} trailing bits, more than one byte of "
            f"padding: the declared shape {height}x{width} does not match the file"
        )
    if padding and bits[n_expected:].any():
        raise ValueError(
            f"packed mask padding bits are not zero ({padding} bits); refusing to "
            "guess the bit layout"
        )
    return bits[:n_expected].astype(bool).reshape(int(height), int(width))


def resolve_reference_validity(
    depth: np.ndarray,
    profile: MaskProfile,
    mask_array: Optional[np.ndarray] = None,
    packed_payload: Optional[bytes] = None,
    packed_bitorder: str = "little",
) -> tuple[np.ndarray, dict]:
    """Reference-valid domain for one frame, plus an audit record.

    Returns `(valid, audit)`. `valid` is the *reference* validity only -- the
    main scoring Z interval [0.25, 2.0] is applied later and separately, so
    "outside the benchmark range" never gets confused with "no reference here".
    """
    depth = np.asarray(depth, dtype=np.float32)
    finite_positive = np.isfinite(depth) & (depth > 0)

    declared: Optional[np.ndarray] = None
    if mask_array is not None:
        declared = np.asarray(mask_array).astype(bool)
        if declared.shape != depth.shape:
            raise ValueError(
                f"mask shape {declared.shape} != depth shape {depth.shape}"
            )
    elif packed_payload is not None:
        declared = decode_packed_mask(
            packed_payload, *depth.shape, bitorder=packed_bitorder
        )

    audit: dict = {
        "profile": profile.name,
        "validity_source": profile.validity_source,
        "n_finite_positive": int(finite_positive.sum()),
        "n_declared_mask_true": int(declared.sum()) if declared is not None else None,
        "mask_file_present": declared is not None,
    }

    if declared is not None and profile.audit_inverse_semantics:
        as_valid = declared
        as_invalid = ~declared
        audit["n_mask_true_and_positive_depth"] = int((as_valid & finite_positive).sum())
        audit["n_mask_false_and_positive_depth"] = int((as_invalid & finite_positive).sum())
        # A file whose 1s never coincide with positive depth is inverted with
        # respect to one_is_valid=true. Report it; do not act on it.
        audit["looks_inverted"] = bool(
            audit["n_mask_true_and_positive_depth"] == 0
            and audit["n_finite_positive"] > 0
        )

    if profile.validity_source == "finite_positive":
        valid = finite_positive
        audit["applied"] = "isfinite(depth) & (depth > 0)"
    else:
        if declared is None:
            if profile.require_mask_file:
                raise FileNotFoundError(
                    f"profile {profile.name!r} requires a declared validity mask, "
                    "but none was supplied for this frame"
                )
            valid = finite_positive
            audit["applied"] = "fallback isfinite & positive (no mask file)"
        else:
            as_declared = declared if profile.one_is_valid else ~declared
            valid = as_declared & finite_positive
            audit["applied"] = (
                f"{'mask' if profile.one_is_valid else '~mask'} & finite & positive"
            )
            n_ref = int(finite_positive.sum())
            if n_ref:
                dropped = int((finite_positive & ~as_declared).sum()) / n_ref
                audit["share_finite_positive_dropped_by_mask"] = float(dropped)
                if dropped > profile.max_audit_disagreement:
                    raise ValueError(
                        f"declared mask drops {dropped:.1%} of finite positive depth, "
                        f"above the profile limit "
                        f"{profile.max_audit_disagreement:.1%}. Verify the mask "
                        "direction on this batch before scoring it."
                    )

    audit["n_valid"] = int(valid.sum())
    return valid, audit


def load_mask_array(path: Path) -> np.ndarray:
    """Read a `.valid.npy` mask. Raises rather than returning a default."""
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D mask, got shape {array.shape}")
    unique = np.unique(array)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"{path}: mask holds values beyond 0/1: {unique[:8]}")
    return array
