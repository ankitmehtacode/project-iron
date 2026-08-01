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
        hard_coverage: This clip is deliberately hard to observe — a coverage
            gap, a blind spot, a camera pointed away from the action. Excluded
            from the mint-time observability floor and reported on its own
            line, so the hard case stays represented instead of being authored
            away to make the aggregate look good.
    """

    clip_id: str
    content_sha: str
    conditions: tuple[Condition, ...] = ()
    source_dataset: str = ""
    notes: str = ""
    hard_coverage: bool = False

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
            "hard_coverage": self.hard_coverage,
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
                    hard_coverage=c.get("hard_coverage", False),
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


MIN_OBSERVABLE_FRACTION = 0.80
"""Share of a set's motion frames that must be observable to its own cameras.

v2-indoor scored 0.4444: agents spent most of their frames behind a furniture
slab or off the bottom edge, so a recall of 1.0 rode on 41 moving frames and
described the set rather than the gate. A set below this floor cannot support
the numbers it will be asked to produce, and the cheapest moment to find that
out is before it is minted — after minting, its sha is quoted and every metric
computed against it inherits the problem.

Clips tagged ``hard_coverage`` are excluded from the aggregate, so keeping a
genuine blind-spot scenario does not push a set below the floor. That
exclusion is the reason the floor can be strict.
"""


def mint_golden_set(
    root: Path,
    golden: GoldenSet,
    observable_fractions: dict[str, float],
    overwrite: bool = False,
) -> Path:
    """Write a golden set, refusing one whose ground truth is not observable.

    Args:
        root: Directory of manifests.
        golden: The set to mint.
        observable_fractions: Per ``clip_id``, the share of that clip's GT
            motion frames its camera could act on — from
            ``scorecard.observability_partition``. Measured, never estimated:
            this is the number the floor exists to enforce.
        overwrite: Passed through to :func:`write_golden_set`.

    Raises:
        GoldenSetError: if a clip has no measurement, or if the aggregate over
            the non-``hard_coverage`` clips falls below
            :data:`MIN_OBSERVABLE_FRACTION`.
    """
    missing = sorted(
        c.clip_id for c in golden.clips if c.clip_id not in observable_fractions
    )
    if missing:
        raise GoldenSetError(
            f"{len(missing)} clip(s) have no measured observable fraction: "
            f"{', '.join(missing[:5])}. A set cannot be accepted on clips "
            "whose observability was never measured — that is how v2-indoor "
            "reached 0.4444 without anyone noticing."
        )

    scored = [c for c in golden.clips if not c.hard_coverage]
    if not scored:
        raise GoldenSetError(
            "every clip is tagged hard_coverage, so the observability floor "
            "has nothing to check. A set of nothing but blind spots measures "
            "nothing."
        )

    aggregate = sum(observable_fractions[c.clip_id] for c in scored) / len(scored)
    if aggregate < MIN_OBSERVABLE_FRACTION:
        worst = sorted(scored, key=lambda c: observable_fractions[c.clip_id])[:5]
        detail = ", ".join(
            f"{c.clip_id}={observable_fractions[c.clip_id]:.2f}" for c in worst
        )
        raise GoldenSetError(
            f"{golden.version} has observable_fraction {aggregate:.4f}, below "
            f"the {MIN_OBSERVABLE_FRACTION:.2f} floor over {len(scored)} "
            f"non-hard_coverage clips. Worst: {detail}. Re-author the scenes "
            "so agents stay in frustum; do not lower the floor, and do not tag "
            "clips hard_coverage to get under it — that tag is for scenarios "
            "that are deliberately hard, not for ones that came out badly."
        )

    return write_golden_set(root, golden, overwrite)


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
