"""Versioned, content-addressed golden sets.

The finding that forced this module
-----------------------------------
Every eval and calibration story in this repository was built on driving
footage while the product is indoor offices, shops and industrial floors.
Depth priors, person-scale distributions, lighting, occlusion patterns and
INT8 activation ranges are all domain-specific, and all of them were pointed
at the wrong domain. The harness transfers; the footage does not.

So golden sets are versioned rather than edited. ``v1`` (driving) is retained
and marked ``legacy`` — it still catches pipeline regressions, and deleting it
would throw away a working regression signal — but it is excluded from product
metrics. ``v2`` is indoor, and is the only set a product number may cite.

Immutability
------------
A golden set is a measuring instrument. Editing one silently invalidates every
number ever reported against it, and the invalidation is undetectable after the
fact: the old scores and the new scores are both "the golden set score". So
:class:`GoldenSet` is frozen, adding clips means minting a new version, and
:func:`write_golden_set` refuses to overwrite an existing manifest. The
``set_sha`` over the member clips is what makes a reported metric traceable to
the exact instrument that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

MANIFEST_SUFFIX = ".golden.json"
SCHEMA_VERSION: Literal["1.0"] = "1.0"


class Domain(str, Enum):
    """What kind of scene a golden set is drawn from.

    Not cosmetic metadata. A depth model evaluated on the wrong domain reports
    a number that is real, reproducible, and about nothing the product does.
    """

    DRIVING = "driving"
    INDOOR = "indoor"


class Status(str, Enum):
    """Whether a set may back a product claim."""

    ACTIVE = "active"
    LEGACY = "legacy"
    """Retained for pipeline regression only. Excluded from product metrics."""


class GoldenSetError(RuntimeError):
    """Raised on an attempt to edit, overwrite, or misuse a golden set."""


# Condition taxonomy, taken from the Site Zero capture protocol. Stratifying
# by these is what stops an eval set from being 30 clips of the same bright
# empty corridor and reporting that as coverage.
class Condition(str, Enum):
    # Lighting
    DAYLIGHT = "daylight"
    EVENING_ARTIFICIAL = "evening_artificial"
    BLINDS_DRAWN = "blinds_drawn"
    LIGHTS_TRANSIENT = "lights_transient"
    """Lights switching on or off mid-clip — the auto-exposure recovery case."""
    GLARE = "glare"

    # Occupancy
    EMPTY = "empty"
    SINGLE_PERSON = "single_person"
    CROWDED = "crowded"

    # Difficulty
    OCCLUSION_CROSSING = "occlusion_crossing"
    OCCLUSION_FURNITURE = "occlusion_furniture"
    BLIND_SPOT_TRAVERSAL = "blind_spot_traversal"
    FAR_FIELD = "far_field"
    """Subject at the far edge of the capability envelope — the long corridor."""

    # Re-ID stressors
    CLOTHING_CHANGE = "clothing_change"
    SIMILAR_CLOTHING = "similar_clothing"
    BAG_CARRIED = "bag_carried"

    # Degradation
    CAMERA_BUMP = "camera_bump"
    LENS_SMUDGE = "lens_smudge"


@dataclass(frozen=True)
class GoldenClip:
    """One member of a golden set.

    Attributes:
        clip_id: Stable identifier.
        content_sha: Hash of the clip bytes. The set is content-addressed
            through its members, so a clip swapped underneath the manifest
            changes the set's own sha.
        conditions: Which taxonomy conditions this clip exercises.
        source_dataset: Registry name, so the lane travels with the clip and a
            product metric can be checked for lane-R contamination.
        notes: Free text.
    """

    clip_id: str
    content_sha: str
    conditions: tuple[Condition, ...] = ()
    source_dataset: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise GoldenSetError("GoldenClip.clip_id must not be empty")
        if not self.content_sha:
            raise GoldenSetError(
                f"{self.clip_id!r} has no content_sha; an unhashed member "
                "makes the whole set untraceable"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "content_sha": self.content_sha,
            "conditions": [c.value for c in self.conditions],
            "source_dataset": self.source_dataset,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GoldenSet:
    """An immutable, content-addressed evaluation set."""

    version: str
    domain: Domain
    status: Status
    clips: tuple[GoldenClip, ...] = ()
    description: str = ""
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise GoldenSetError("a golden set needs a version")
        seen: set[str] = set()
        for clip in self.clips:
            if clip.clip_id in seen:
                raise GoldenSetError(
                    f"duplicate clip_id {clip.clip_id!r} in {self.version}: a "
                    "clip counted twice silently weights the metric toward it"
                )
            seen.add(clip.clip_id)

    @property
    def set_sha(self) -> str:
        """Hash over the member ids and their content hashes.

        This is what a reported metric cites. Two runs quoting the same
        ``set_sha`` measured the same instrument; two quoting different ones
        did not, whatever the version strings say.
        """
        payload = json.dumps(
            sorted((c.clip_id, c.content_sha) for c in self.clips),
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def usable_for_product_metrics(self) -> bool:
        return self.status is Status.ACTIVE

    def require_product_usable(self) -> None:
        """Raise unless this set may back a product claim.

        Raises:
            GoldenSetError: for a legacy set, naming why. A driving-domain
                number reported as a product metric is not a small error — it
                is a real, reproducible measurement of something the product
                does not do.
        """
        if not self.usable_for_product_metrics:
            raise GoldenSetError(
                f"golden set {self.version!r} is {self.status.value} "
                f"({self.domain.value} domain) and must not back a product "
                "metric. It is retained for pipeline regression only. Use the "
                "active indoor set."
            )

    def condition_coverage(self) -> dict[Condition, int]:
        """How many clips exercise each condition.

        Zeroes are included deliberately: an absent condition is the thing you
        want to see, and a dict that only lists what is present hides it.
        """
        counts = {condition: 0 for condition in Condition}
        for clip in self.clips:
            for condition in clip.conditions:
                counts[condition] += 1
        return counts

    def missing_conditions(self) -> tuple[Condition, ...]:
        """Conditions no clip covers."""
        return tuple(c for c, n in self.condition_coverage().items() if n == 0)

    def with_clips(self, clips: tuple[GoldenClip, ...], version: str) -> "GoldenSet":
        """Derive a NEW version containing additional clips.

        The only supported way to grow a set. Mutating one in place would
        invalidate every previously reported number without any way to detect
        it afterwards.

        Raises:
            GoldenSetError: if ``version`` matches this set's own.
        """
        if version == self.version:
            raise GoldenSetError(
                f"cannot add clips under the same version {version!r}: adding "
                "means a new version, editing is an error"
            )
        return GoldenSet(
            version=version,
            domain=self.domain,
            status=self.status,
            clips=self.clips + clips,
            description=self.description,
            supersedes=self.version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "domain": self.domain.value,
            "status": self.status.value,
            "supersedes": self.supersedes,
            "description": self.description,
            "set_sha": self.set_sha,
            "clip_count": len(self.clips),
            "clips": [c.as_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoldenSet":
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise GoldenSetError(
                f"golden-set manifest schema_version {version!r} is not "
                f"{SCHEMA_VERSION!r}"
            )
        loaded = cls(
            version=payload["version"],
            domain=Domain(payload["domain"]),
            status=Status(payload["status"]),
            clips=tuple(
                GoldenClip(
                    clip_id=c["clip_id"],
                    content_sha=c["content_sha"],
                    conditions=tuple(Condition(x) for x in c.get("conditions", [])),
                    source_dataset=c.get("source_dataset", ""),
                    notes=c.get("notes", ""),
                )
                for c in payload.get("clips", [])
            ),
            description=payload.get("description", ""),
            supersedes=payload.get("supersedes"),
        )
        recorded = payload.get("set_sha")
        if recorded is not None and recorded != loaded.set_sha:
            raise GoldenSetError(
                f"golden set {loaded.version!r} does not match its recorded "
                f"set_sha: manifest says {recorded[:12]}..., contents hash to "
                f"{loaded.set_sha[:12]}.... The set was edited after it was "
                "written, which invalidates every metric reported against it."
            )
        return loaded


def manifest_path(root: Path, version: str) -> Path:
    return root / f"{version}{MANIFEST_SUFFIX}"


def write_golden_set(root: Path, golden: GoldenSet, overwrite: bool = False) -> Path:
    """Write a golden-set manifest, refusing to overwrite by default.

    Raises:
        GoldenSetError: if the manifest already exists. ``overwrite=True``
            exists for tests and for correcting a manifest before any metric
            has been reported against it; using it afterwards is the thing
            this module exists to prevent.
    """
    path = manifest_path(root, golden.version)
    if path.exists() and not overwrite:
        raise GoldenSetError(
            f"{path} already exists. Golden sets are immutable: adding clips "
            f"means a new version (see GoldenSet.with_clips), and editing "
            f"{golden.version!r} would invalidate every number already "
            "reported against it."
        )
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden.as_dict(), indent=2, sort_keys=True) + "\n")
    return path


def available_versions(root: Path) -> list[str]:
    """Versions with a manifest under ``root``, sorted."""
    if not root.exists():
        return []
    return sorted(
        path.name.removesuffix(MANIFEST_SUFFIX)
        for path in root.glob(f"*{MANIFEST_SUFFIX}")
    )


def load_golden_set(root: Path, version: str) -> GoldenSet:
    """Load one version's manifest.

    Raises:
        GoldenSetError: if it is missing or has been edited since it was
            written.
    """
    path = manifest_path(root, version)
    if not path.exists():
        available = available_versions(root)
        raise GoldenSetError(
            f"golden set {version!r} not found at {path}. Available: "
            f"{available or 'none'}"
        )
    return GoldenSet.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True)
class SiteZeroPlan:
    """Target composition for the v2 indoor set, from the capture protocol.

    Held as data rather than prose so ``make eval`` can report the gap between
    what Site Zero has captured and what the set needs, instead of someone
    re-reading a document and estimating.
    """

    target_clips: int = 30
    required_conditions: tuple[Condition, ...] = field(
        default_factory=lambda: tuple(Condition)
    )
    min_clips_per_condition: int = 1

    def gap(self, golden: GoldenSet) -> dict[str, Any]:
        """What still has to be captured for this set to be complete."""
        coverage = golden.condition_coverage()
        short = {
            c.value: self.min_clips_per_condition - coverage[c]
            for c in self.required_conditions
            if coverage[c] < self.min_clips_per_condition
        }
        return {
            "clips_present": len(golden.clips),
            "clips_target": self.target_clips,
            "clips_short": max(0, self.target_clips - len(golden.clips)),
            "conditions_short": short,
            "complete": not short and len(golden.clips) >= self.target_clips,
        }
