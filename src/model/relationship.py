"""Bitemporal Relationship, Correction, and the invalidation closure.

Why bitemporal
--------------
A badge swipe at 14:02 can resolve which enrolled identity an anonymous
track has been since 13:58 — the fact was true from 13:58, but the system
did not *know* it until 14:02. A relationship type with one timestamp can
represent only one of those instants and is forced to pick, silently
overwriting the other. :class:`Relationship` carries both axes as
independently required fields: ``valid_from_ns``/``valid_to_ns`` (when the
fact was true in the world) and ``asserted_at_ns``/``asserted_by`` (when
and how the system learned it). Neither axis has a default, so a
single-timestamp relationship — one axis supplied, the other silently
omitted — is a ``TypeError`` at the call site, not a validation gap.

Corrections and the invalidation closure
------------------------------------------
A :class:`Correction` is how a fact changes after artifacts have already
been derived from the old version of it: it names what changed
(``target``, ``kind``, ``reason``, ``evidence``, ``actor``) and, required,
which derived artifacts it ``invalidates``. :class:`DerivedArtifact`
(a scorecard, a digest, a behaviour model, an episode) must declare its
``input_closure`` — every id it was derived from — before
:class:`ArtifactRegistry` will register it; an artifact with no recorded
inputs can never be found by :meth:`ArtifactRegistry.apply_correction`'s
graph walk, so it is rejected at registration instead of silently becoming
un-invalidatable. Applying a correction marks the named artifacts AND
everything transitively downstream of them invalid, without deleting any
of them: the pre-correction record remains retrievable by id, with its
original ``input_closure`` and ``manifest_sha`` intact, so a query made
before the correction landed stays reproducible and explainable — only its
validity flag changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, get_args

from src.events.schema import EntityRef

RelationshipBasis = Literal["asserted", "observed", "inferred"]
RELATIONSHIP_BASES: tuple[RelationshipBasis, ...] = get_args(RelationshipBasis)

CorrectionKind = Literal[
    "identity_resolution",
    "revocation",
    "track_merge",
    "split",
    "verb_retraction",
    "recalibration",
]
CORRECTION_KINDS: tuple[CorrectionKind, ...] = get_args(CorrectionKind)


class RelationshipError(ValueError):
    """Raised when a Relationship, Correction, or DerivedArtifact is malformed."""


@dataclass(frozen=True)
class Relationship:
    """A bitemporal fact linking two entities.

    Attributes:
        subject: The relationship's subject.
        predicate: What the relationship asserts (e.g. ``"same_identity_as"``,
            ``"contains"``, ``"assigned_to"``).
        object: The relationship's object.
        valid_from_ns: When the relationship became true in the world.
        valid_to_ns: When it stopped being true, or ``None`` if it still
            holds. Required as a field (no default) even though ``None``
            is a legal value — the caller must say "still open" on
            purpose, not by omission.
        asserted_at_ns: When the system learned this relationship.
        asserted_by: What learned it (a sensor id, a correction actor, a
            reconciliation job name).
        basis: How the assertion was made.
        confidence: In [0, 1].
        evidence: Opaque references into the evidence layer.
    """

    subject: EntityRef
    predicate: str
    object: EntityRef
    valid_from_ns: int
    valid_to_ns: int | None
    asserted_at_ns: int
    asserted_by: str
    basis: RelationshipBasis
    confidence: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predicate:
            raise RelationshipError("Relationship.predicate must not be empty")
        if not self.asserted_by:
            raise RelationshipError("Relationship.asserted_by must not be empty")
        if self.basis not in RELATIONSHIP_BASES:
            raise RelationshipError(
                f"unknown relationship basis {self.basis!r}; expected one of "
                f"{RELATIONSHIP_BASES}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise RelationshipError(
                f"Relationship.confidence must lie in [0, 1], got {self.confidence}"
            )
        if self.valid_to_ns is not None and self.valid_to_ns <= self.valid_from_ns:
            raise RelationshipError(
                f"valid_to_ns {self.valid_to_ns} must be after valid_from_ns "
                f"{self.valid_from_ns}"
            )

    @property
    def is_retroactive(self) -> bool:
        """True when the system learned this fact after it became true.

        The canonical case: a badge swipe at 14:02 asserting a relationship
        that has been true since 13:58.
        """
        return self.asserted_at_ns > self.valid_from_ns


# ---------------------------------------------------------------------------
# Correction and the invalidation closure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Correction:
    """A record that a prior fact has changed, and what it invalidates.

    Attributes:
        target: The id of the thing corrected (an entity, relationship, or
            event id).
        kind: What kind of correction this is.
        reason: Human-readable explanation.
        evidence: Opaque references into the evidence layer supporting
            the correction.
        actor: Who or what made the correction.
        invalidates: Derived artifact ids directly invalidated by this
            correction. Required and non-empty: a correction that
            invalidates nothing downstream has nothing for
            :class:`ArtifactRegistry` to act on and is not representable.
    """

    target: str
    kind: CorrectionKind
    reason: str
    evidence: tuple[str, ...]
    actor: str
    invalidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.target:
            raise RelationshipError("Correction.target must not be empty")
        if self.kind not in CORRECTION_KINDS:
            raise RelationshipError(
                f"unknown correction kind {self.kind!r}; expected one of "
                f"{CORRECTION_KINDS}"
            )
        if not self.reason:
            raise RelationshipError("Correction.reason must not be empty")
        if not self.actor:
            raise RelationshipError("Correction.actor must not be empty")
        if not self.invalidates:
            raise RelationshipError(
                "Correction.invalidates is required and must not be empty: a "
                "correction that names no downstream artifact cannot be "
                "propagated"
            )


@dataclass(frozen=True)
class DerivedArtifact:
    """A scorecard, digest, behaviour model, or episode derived from inputs.

    Attributes:
        artifact_id: Unique id.
        kind: What kind of artifact this is (free text: "scorecard",
            "digest", "behaviour_model", "episode", ...).
        input_closure: Every id (event, relationship, observation, or other
            artifact) this was derived from. STRUCTURAL: required and must
            be non-empty — see :meth:`ArtifactRegistry.register`.
        manifest_sha: The run that produced this artifact.
    """

    artifact_id: str
    kind: str
    input_closure: tuple[str, ...]
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise RelationshipError("DerivedArtifact.artifact_id must not be empty")
        if not self.kind:
            raise RelationshipError("DerivedArtifact.kind must not be empty")
        if not self.manifest_sha:
            raise RelationshipError("DerivedArtifact.manifest_sha must not be empty")


class ArtifactRegistry:
    """Tracks derived artifacts and propagates corrections transitively.

    Nothing is ever deleted: :meth:`apply_correction` only flips a validity
    flag. :meth:`get` always returns the original record, invalid or not,
    so a pre-correction answer stays retrievable with its original
    provenance.
    """

    def __init__(self) -> None:
        self._artifacts: dict[str, DerivedArtifact] = {}
        self._invalid: set[str] = set()
        # input_id -> set of artifact_ids whose input_closure names it.
        self._dependents: dict[str, set[str]] = defaultdict(set)

    def register(self, artifact: DerivedArtifact) -> None:
        """Register a derived artifact.

        Raises:
            RelationshipError: if ``artifact.input_closure`` is empty. This
                is the STRUCTURAL guard: an artifact with no declared
                inputs can never be reached by a correction's graph walk,
                so it is rejected up front rather than silently becoming
                permanently valid no matter what changes beneath it.
        """
        if not artifact.input_closure:
            raise RelationshipError(
                f"artifact {artifact.artifact_id!r} has no input_closure and "
                "cannot be registered: an artifact whose inputs are not "
                "recorded can never be found when one of them is corrected"
            )
        self._artifacts[artifact.artifact_id] = artifact
        for input_id in artifact.input_closure:
            self._dependents[input_id].add(artifact.artifact_id)

    def get(self, artifact_id: str) -> DerivedArtifact:
        """Return the original record for ``artifact_id``, valid or not."""
        return self._artifacts[artifact_id]

    def is_valid(self, artifact_id: str) -> bool:
        return artifact_id not in self._invalid

    def apply_correction(self, correction: Correction) -> frozenset[str]:
        """Mark ``correction.invalidates`` and everything downstream invalid.

        Walks the dependency graph built from every registered artifact's
        ``input_closure``: an artifact that consumed a now-invalid artifact
        as an input becomes invalid too, recursively.

        Returns:
            The full set of artifact ids newly marked invalid by this call
            (already-invalid artifacts are not repeated).
        """
        newly_invalid: set[str] = set()
        frontier = list(correction.invalidates)
        while frontier:
            artifact_id = frontier.pop()
            if artifact_id in self._invalid or artifact_id in newly_invalid:
                continue
            newly_invalid.add(artifact_id)
            frontier.extend(self._dependents.get(artifact_id, ()))
        self._invalid |= newly_invalid
        return frozenset(newly_invalid)
