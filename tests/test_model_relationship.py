"""Objective 4 — bitemporal Relationship, Correction, and invalidation closure.

The canonical case (spec-required): a badge swipe at 14:02 resolves an
anonymous track from 13:58. Identity resolves retroactively, affected
derived artifacts are marked invalid transitively, and the pre-correction
answer stays retrievable with its original provenance.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.events.schema import EntityRef
from src.model.relationship import (
    ArtifactRegistry,
    Correction,
    DerivedArtifact,
    Relationship,
    RelationshipError,
)

SECOND_NS = 1_000_000_000
T_13_58 = 1_785_000_000 * SECOND_NS
T_14_02 = T_13_58 + 4 * 60 * SECOND_NS

SESSION = EntityRef("session", "sess-4f2a91")
EMPLOYEE = EntityRef("enrolled", "employee-42")


def _relationship(**overrides: object) -> Relationship:
    kwargs: dict[str, object] = dict(
        subject=SESSION,
        predicate="same_identity_as",
        object=EMPLOYEE,
        valid_from_ns=T_13_58,
        valid_to_ns=None,
        asserted_at_ns=T_14_02,
        asserted_by="badge-reader-03",
        basis="observed",
        confidence=0.98,
        evidence=("obs-badge-1",),
    )
    kwargs.update(overrides)
    return Relationship(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# STRUCTURAL: both time axes required — a single-timestamp relationship
# is unconstructable.
# ---------------------------------------------------------------------------


def test_relationship_requires_all_four_temporal_fields() -> None:
    fields = {f.name for f in dataclasses.fields(Relationship)}
    assert {"valid_from_ns", "valid_to_ns", "asserted_at_ns", "asserted_by"} <= fields
    # None of the four temporal fields has a default: dataclasses.fields()
    # reports MISSING for a field with no default.
    for name in ("valid_from_ns", "valid_to_ns", "asserted_at_ns", "asserted_by"):
        field = next(f for f in dataclasses.fields(Relationship) if f.name == name)
        no_factory: object = field.default_factory
        assert field.default is dataclasses.MISSING
        assert no_factory is dataclasses.MISSING


def test_relationship_missing_valid_time_axis_is_unconstructable() -> None:
    with pytest.raises(TypeError):
        Relationship(  # type: ignore[call-arg]
            subject=SESSION,
            predicate="same_identity_as",
            object=EMPLOYEE,
            asserted_at_ns=T_14_02,
            asserted_by="badge-reader-03",
            basis="observed",
            confidence=0.98,
            evidence=(),
        )


def test_relationship_missing_assertion_time_axis_is_unconstructable() -> None:
    with pytest.raises(TypeError):
        Relationship(  # type: ignore[call-arg]
            subject=SESSION,
            predicate="same_identity_as",
            object=EMPLOYEE,
            valid_from_ns=T_13_58,
            valid_to_ns=None,
            basis="observed",
            confidence=0.98,
            evidence=(),
        )


def test_relationship_valid_to_must_be_explicit_even_when_open() -> None:
    # valid_to_ns has no default: an "open" relationship must say so with
    # an explicit None, not by omitting the keyword.
    field = next(f for f in dataclasses.fields(Relationship) if f.name == "valid_to_ns")
    assert field.default is dataclasses.MISSING
    ok = _relationship(valid_to_ns=None)
    assert ok.valid_to_ns is None


def test_relationship_is_retroactive_when_asserted_after_valid_from() -> None:
    rel = _relationship()
    assert rel.is_retroactive
    assert rel.valid_from_ns == T_13_58
    assert rel.asserted_at_ns == T_14_02


def test_relationship_rejects_valid_to_before_valid_from() -> None:
    with pytest.raises(RelationshipError):
        _relationship(valid_from_ns=100, valid_to_ns=50)


def test_relationship_rejects_out_of_range_confidence() -> None:
    with pytest.raises(RelationshipError):
        _relationship(confidence=1.5)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------


def test_correction_requires_nonempty_invalidates() -> None:
    with pytest.raises(RelationshipError):
        Correction(
            target="sess-4f2a91",
            kind="identity_resolution",
            reason="badge swipe matched",
            evidence=("obs-badge-1",),
            actor="reconciliation-job",
            invalidates=(),
        )


def test_correction_rejects_unknown_kind() -> None:
    with pytest.raises(RelationshipError):
        Correction(
            target="sess-4f2a91",
            kind="teleportation",  # type: ignore[arg-type]
            reason="x",
            evidence=(),
            actor="a",
            invalidates=("artifact-1",),
        )


# ---------------------------------------------------------------------------
# DerivedArtifact — STRUCTURAL: no input_closure, cannot be registered.
# ---------------------------------------------------------------------------


def test_derived_artifact_without_input_closure_cannot_be_registered() -> None:
    orphan = DerivedArtifact(
        artifact_id="scorecard-orphan",
        kind="scorecard",
        input_closure=(),
        manifest_sha="sha-1",
    )
    registry = ArtifactRegistry()
    with pytest.raises(RelationshipError):
        registry.register(orphan)


def test_derived_artifact_construction_also_rejects_empty_ids() -> None:
    with pytest.raises(RelationshipError):
        DerivedArtifact(
            artifact_id="", kind="scorecard", input_closure=("x",), manifest_sha="s"
        )


# ---------------------------------------------------------------------------
# The canonical case: retroactive badge-swipe identity resolution.
# ---------------------------------------------------------------------------


def test_canonical_badge_swipe_resolves_identity_and_invalidates_transitively() -> None:
    registry = ArtifactRegistry()

    # A scorecard was computed at 13:59, while the track was still anonymous,
    # over an input closure that names the anonymous session.
    scorecard = DerivedArtifact(
        artifact_id="scorecard-lobby-1358",
        kind="scorecard",
        input_closure=("sess-4f2a91", "obs-cam-1"),
        manifest_sha="sha-scorecard-1",
    )
    registry.register(scorecard)

    # A daily digest was computed later, downstream of that scorecard —
    # this is the transitive hop the closure must reach.
    digest = DerivedArtifact(
        artifact_id="digest-2026-07-31",
        kind="digest",
        input_closure=("scorecard-lobby-1358",),
        manifest_sha="sha-digest-1",
    )
    registry.register(digest)

    # An unrelated artifact must NOT be invalidated.
    unrelated = DerivedArtifact(
        artifact_id="scorecard-dock-1358",
        kind="scorecard",
        input_closure=("sess-other-track",),
        manifest_sha="sha-scorecard-2",
    )
    registry.register(unrelated)

    assert registry.is_valid("scorecard-lobby-1358")
    assert registry.is_valid("digest-2026-07-31")

    # The badge swipe at 14:02 resolves the identity retroactively to
    # 13:58 — the fact was true before the system knew it.
    resolution = _relationship()
    assert resolution.is_retroactive

    correction = Correction(
        target="sess-4f2a91",
        kind="identity_resolution",
        reason="badge swipe at cam-lobby-01 door matched session sess-4f2a91",
        evidence=("obs-badge-1",),
        actor="reconciliation-job-v1",
        invalidates=("scorecard-lobby-1358",),
    )
    invalidated = registry.apply_correction(correction)

    # Direct target plus its transitive downstream, and nothing else.
    assert invalidated == frozenset({"scorecard-lobby-1358", "digest-2026-07-31"})
    assert not registry.is_valid("scorecard-lobby-1358")
    assert not registry.is_valid("digest-2026-07-31")
    assert registry.is_valid("scorecard-dock-1358")

    # The pre-correction answer remains retrievable with its original
    # provenance — nothing about the record itself changed.
    original = registry.get("scorecard-lobby-1358")
    assert original.manifest_sha == "sha-scorecard-1"
    assert original.input_closure == ("sess-4f2a91", "obs-cam-1")
    assert original == scorecard


def test_apply_correction_is_idempotent_on_already_invalid_artifacts() -> None:
    registry = ArtifactRegistry()
    registry.register(
        DerivedArtifact(
            artifact_id="a", kind="scorecard", input_closure=("x",), manifest_sha="s"
        )
    )
    correction = Correction(
        target="x",
        kind="recalibration",
        reason="r",
        evidence=(),
        actor="a",
        invalidates=("a",),
    )
    first = registry.apply_correction(correction)
    second = registry.apply_correction(correction)
    assert first == frozenset({"a"})
    assert second == frozenset()  # nothing NEW invalidated the second time
    assert not registry.is_valid("a")
