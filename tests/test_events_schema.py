"""Tests for event schema v1.

Two things are under test: that a record survives every round trip unchanged,
and that every illegal record is rejected at construction. The second matters
more. These records are the only place facts are allowed to originate, so a
malformed one that gets stored is a false fact with a timestamp on it.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.events import (
    GEOMETRIC_VERBS,
    INTERACTION_VERBS,
    POSE_VERBS,
    SCHEMA_VERSION,
    ClipRef,
    EntityRef,
    Event,
    SchemaError,
    Verb,
    events_arrow_schema,
    read_events_parquet,
    write_events_parquet,
)

MANIFEST = "a" * 64
SITE = "site-hq-1"


def make_event(**overrides: object) -> Event:
    """A valid baseline event; override one field per test to make it illegal."""
    defaults: dict[str, object] = {
        "event_id": uuid.uuid4(),
        "site_id": SITE,
        "ts_ns": 1_700_000_000_000_000_000,
        "subject": EntityRef("session", "track-7"),
        "verb": Verb.ENTERED,
        "confidence": 0.9,
        "observed": True,
        "importance": 0.4,
        "manifest_sha": MANIFEST,
        "zone": EntityRef("zone", "lobby"),
        "clip": ClipRef(
            "cam-3-20260731",
            1_700_000_000_000_000_000,
            1_700_000_005_000_000_000,
            "c" * 64,
        ),
    }
    defaults.update(overrides)
    return Event(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_v1_vocabulary_is_exactly_nineteen_verbs() -> None:
    """Pin the closed vocabulary. Adding a verb must be a deliberate act."""
    assert len(list(Verb)) == 19
    assert len(GEOMETRIC_VERBS) == 8
    assert len(POSE_VERBS) == 5
    assert len(INTERACTION_VERBS) == 6


def test_every_verb_belongs_to_exactly_one_group() -> None:
    """A verb in no group would bypass the coherence rules entirely."""
    groups = (GEOMETRIC_VERBS, POSE_VERBS, INTERACTION_VERBS)
    for verb in Verb:
        assert sum(verb in group for group in groups) == 1, verb


def test_group_membership_matches_the_specified_vocabulary() -> None:
    assert {v.value for v in GEOMETRIC_VERBS} == {
        "entered",
        "exited",
        "dwelled",
        "approached",
        "followed",
        "loitered",
        "ran",
        "fell",
    }
    assert {v.value for v in POSE_VERBS} == {
        "sat",
        "stood",
        "bent_down",
        "reached",
        "lying",
    }
    assert {v.value for v in INTERACTION_VERBS} == {
        "picked_up",
        "put_down",
        "carried",
        "handed_over",
        "opened",
        "closed",
    }


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_dict_round_trip_preserves_everything() -> None:
    original = make_event(
        verb=Verb.PICKED_UP,
        object=EntityRef("asset", "laptop-114"),
        observed=False,
        confidence=0.71,
        importance=0.95,
    )
    assert Event.from_dict(original.to_dict()) == original


def test_dict_round_trip_with_all_optionals_absent() -> None:
    minimal = Event(
        event_id=uuid.uuid4(),
        site_id=SITE,
        ts_ns=1,
        subject=EntityRef("session", "t1"),
        verb=Verb.RAN,
        confidence=0.5,
        observed=True,
        importance=0.5,
        manifest_sha=MANIFEST,
    )
    restored = Event.from_dict(minimal.to_dict())
    assert restored == minimal
    assert restored.object is None
    assert restored.zone is None
    assert restored.clip is None


def test_parquet_round_trip_preserves_everything(tmp_path: Path) -> None:
    events = [
        make_event(verb=Verb.ENTERED),
        make_event(verb=Verb.PICKED_UP, object=EntityRef("asset", "laptop-114")),
        make_event(verb=Verb.EXITED, observed=False, clip=None, zone=None),
    ]
    path = write_events_parquet(events, tmp_path / "events.parquet")
    assert read_events_parquet(path) == events


def test_parquet_preserves_the_observed_flag(tmp_path: Path) -> None:
    """The flag that separates what was seen from what was guessed.

    If this ever round-trips wrong, inferred events become indistinguishable
    from observations, which is the failure the schema exists to prevent.
    """
    events = [make_event(observed=True), make_event(observed=False)]
    restored = read_events_parquet(write_events_parquet(events, tmp_path / "e.parquet"))
    assert [e.observed for e in restored] == [True, False]
    assert restored[1].is_inferred


def test_parquet_preserves_exact_timestamps(tmp_path: Path) -> None:
    """Nanoseconds must survive as the exact integer the detector produced."""
    ts = 1_700_000_000_123_456_789
    restored = read_events_parquet(
        write_events_parquet([make_event(ts_ns=ts)], tmp_path / "e.parquet")
    )
    assert restored[0].ts_ns == ts


def test_parquet_preserves_threshold_sensitive_floats(tmp_path: Path) -> None:
    """float32 would turn 0.8 into 0.7999999523 and break an alert threshold."""
    restored = read_events_parquet(
        write_events_parquet(
            [make_event(confidence=0.8, importance=0.15)], tmp_path / "e.parquet"
        )
    )
    assert restored[0].confidence == 0.8
    assert restored[0].importance == 0.15


def test_empty_file_is_valid_and_distinguishable_from_no_file(tmp_path: Path) -> None:
    path = write_events_parquet([], tmp_path / "empty.parquet")
    assert path.exists()
    assert read_events_parquet(path) == []


def test_written_schema_matches_the_declared_schema(tmp_path: Path) -> None:
    path = write_events_parquet([make_event()], tmp_path / "e.parquet")
    assert (
        pq.read_table(path)
        .schema.remove_metadata()
        .equals(events_arrow_schema().remove_metadata())
    )


# ---------------------------------------------------------------------------
# Validation — each illegal case rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", sorted(INTERACTION_VERBS, key=lambda v: v.value))
def test_interaction_verb_without_an_object_is_rejected(verb: Verb) -> None:
    """'picked_up' is not a fact until it says what was picked up."""
    with pytest.raises(SchemaError, match="requires an object"):
        make_event(verb=verb, object=None)


@pytest.mark.parametrize("verb", sorted(POSE_VERBS, key=lambda v: v.value))
def test_pose_verb_with_an_object_is_rejected(verb: Verb) -> None:
    """'sat a laptop' is not a claim about anything."""
    with pytest.raises(SchemaError, match="must not carry an object"):
        make_event(verb=verb, object=EntityRef("asset", "laptop-114"))


@pytest.mark.parametrize("verb", sorted(GEOMETRIC_VERBS, key=lambda v: v.value))
def test_geometric_verbs_accept_an_object_either_way(verb: Verb) -> None:
    assert make_event(verb=verb, object=None).object is None
    assert make_event(verb=verb, object=EntityRef("asset", "door-2")).object is not None


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0, -5.0])
def test_confidence_outside_the_unit_interval_is_rejected(value: float) -> None:
    with pytest.raises(SchemaError, match="confidence must lie in"):
        make_event(confidence=value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_importance_outside_the_unit_interval_is_rejected(value: float) -> None:
    with pytest.raises(SchemaError, match="importance must lie in"):
        make_event(importance=value)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unit_interval_endpoints_are_accepted(value: float) -> None:
    assert make_event(confidence=value, importance=value).confidence == value


def test_clip_with_non_positive_interval_is_rejected() -> None:
    with pytest.raises(SchemaError, match="positive interval"):
        ClipRef("cam-1", 1000, 1000)
    with pytest.raises(SchemaError, match="positive interval"):
        ClipRef("cam-1", 2000, 1000)


def test_missing_manifest_sha_is_rejected() -> None:
    """An event that cannot be traced to a run cannot be defended."""
    with pytest.raises(SchemaError, match="manifest_sha is required"):
        make_event(manifest_sha="")


def test_empty_site_id_is_rejected() -> None:
    with pytest.raises(SchemaError, match="site_id"):
        make_event(site_id="")


def test_unknown_entity_kind_is_rejected() -> None:
    with pytest.raises(SchemaError, match="unknown entity kind"):
        EntityRef("employee", "ankit")  # type: ignore[arg-type]


def test_empty_entity_id_is_rejected() -> None:
    with pytest.raises(SchemaError, match="id must not be empty"):
        EntityRef("session", "")


def test_zone_field_must_reference_a_zone() -> None:
    """Otherwise 'where' could silently hold a person."""
    with pytest.raises(SchemaError, match="must reference a zone"):
        make_event(zone=EntityRef("asset", "laptop-114"))


def test_verb_outside_the_enum_is_rejected() -> None:
    with pytest.raises(SchemaError, match="closed v1 vocabulary"):
        make_event(verb="sprinted")


# ---------------------------------------------------------------------------
# Schema version enforcement
# ---------------------------------------------------------------------------


def test_constructing_with_a_foreign_schema_version_is_rejected() -> None:
    with pytest.raises(SchemaError, match="unsupported schema_version"):
        make_event(schema_version="2.0")


def test_reading_a_foreign_schema_version_is_rejected() -> None:
    """Reinterpreting another version's fields under v1 names is how a log
    silently starts meaning something else."""
    payload = make_event().to_dict()
    payload["schema_version"] = "0.9"
    with pytest.raises(SchemaError, match="cannot read a record written against"):
        Event.from_dict(payload)


def test_reading_an_unknown_verb_is_rejected() -> None:
    payload = make_event().to_dict()
    payload["verb"] = "levitated"
    with pytest.raises(SchemaError, match="unknown verb"):
        Event.from_dict(payload)


def test_reading_a_record_missing_a_required_field_is_rejected() -> None:
    payload = make_event().to_dict()
    del payload["manifest_sha"]
    with pytest.raises(SchemaError, match="missing required field"):
        Event.from_dict(payload)


def test_parquet_with_a_foreign_schema_version_is_rejected(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "foreign.parquet"
    schema = events_arrow_schema().with_metadata({b"iron_schema_version": b"9.9"})
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    with pytest.raises(SchemaError, match="written against schema_version"):
        read_events_parquet(path)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_events_are_immutable() -> None:
    """The log is append-only; a record editable in place is not evidence."""
    event = make_event()
    with pytest.raises(Exception):
        event.importance = 0.99  # type: ignore[misc]


def test_rescoring_importance_returns_a_copy() -> None:
    original = make_event(importance=0.2)
    rescored = original.with_importance(0.8)
    assert original.importance == 0.2
    assert rescored.importance == 0.8
    assert rescored.event_id == original.event_id


def test_schema_version_constant_is_one_point_zero() -> None:
    assert SCHEMA_VERSION == "1.0"
