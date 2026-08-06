"""Objective 2 — the v2 event-class hierarchy.

STRUCTURAL rules under test:
  - PredictedEvent cannot be admitted as evidence (closed-world dispatch).
  - HypothesisEvent cannot trigger an alert (closed-world dispatch).
  - A confirmed PredictedEvent produces a NEW ObservedEvent; the
    PredictedEvent itself is frozen and never mutated.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from src.events.schema import ClipRef, EntityRef, Verb
from src.model.events import (
    EVENT_SCHEMA_VERSION,
    EventError,
    HypothesisEvent,
    InferredEvent,
    ObservedEvent,
    PredictedEvent,
    assemble_evidence,
    confirm_prediction,
    event_v2_from_dict,
    event_v2_to_dict,
)

SUBJECT = EntityRef("session", "sess-1")
LAPTOP = EntityRef("asset", "laptop-1")
LOBBY = EntityRef("zone", "lobby")


def _observed(**overrides: object) -> ObservedEvent:
    kwargs: dict[str, object] = dict(
        event_id=uuid.uuid4(),
        site_id="site-0",
        ts_ns=1,
        subject=SUBJECT,
        verb=Verb.ENTERED,
        confidence=0.9,
        importance=0.5,
        manifest_sha="sha-1",
        zone=LOBBY,
    )
    kwargs.update(overrides)
    return ObservedEvent(**kwargs)  # type: ignore[arg-type]


def _predicted(**overrides: object) -> PredictedEvent:
    kwargs: dict[str, object] = dict(
        event_id=uuid.uuid4(),
        site_id="site-0",
        ts_ns=100,
        subject=SUBJECT,
        verb=Verb.EXITED,
        confidence=0.6,
        importance=0.3,
        manifest_sha="sha-1",
        predicted_by="trajectory-extrapolator-v1",
    )
    kwargs.update(overrides)
    return PredictedEvent(**kwargs)  # type: ignore[arg-type]


def _hypothesis(**overrides: object) -> HypothesisEvent:
    kwargs: dict[str, object] = dict(
        event_id=uuid.uuid4(),
        site_id="site-0",
        ts_ns=50,
        subject=SUBJECT,
        verb=Verb.LOITERED,
        confidence=0.2,
        importance=0.1,
        manifest_sha="sha-1",
        rationale="dwell time exceeded threshold but track is fragmented",
    )
    kwargs.update(overrides)
    return HypothesisEvent(**kwargs)  # type: ignore[arg-type]


def _inferred(**overrides: object) -> InferredEvent:
    kwargs: dict[str, object] = dict(
        event_id=uuid.uuid4(),
        site_id="site-0",
        ts_ns=75,
        subject=SUBJECT,
        verb=Verb.EXITED,
        confidence=0.7,
        importance=0.4,
        manifest_sha="sha-1",
        basis="trajectory extrapolated across a 3-frame blind spot",
    )
    kwargs.update(overrides)
    return InferredEvent(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and coherence (shared with v1)
# ---------------------------------------------------------------------------


def test_observed_event_constructs() -> None:
    event = _observed()
    assert event.event_class == "observed"
    assert event.schema_version == EVENT_SCHEMA_VERSION


def test_interaction_verb_requires_object_across_all_classes() -> None:
    for factory in (_observed, _inferred, _predicted, _hypothesis):
        with pytest.raises(EventError):
            factory(verb=Verb.PICKED_UP, object=None)
        ok = factory(verb=Verb.PICKED_UP, object=LAPTOP)
        assert ok.object == LAPTOP


def test_pose_verb_forbids_object() -> None:
    with pytest.raises(EventError):
        _observed(verb=Verb.SAT, object=LAPTOP)


def test_inferred_requires_basis() -> None:
    with pytest.raises(EventError):
        _inferred(basis="")


def test_predicted_requires_predicted_by() -> None:
    with pytest.raises(EventError):
        _predicted(predicted_by="")


def test_hypothesis_requires_rationale() -> None:
    with pytest.raises(EventError):
        _hypothesis(rationale="")


def test_all_four_classes_are_frozen() -> None:
    for event in (_observed(), _inferred(), _predicted(), _hypothesis()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.confidence = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# STRUCTURAL: evidence admission rejects PredictedEvent by type
# ---------------------------------------------------------------------------


def test_predicted_event_cannot_be_admitted_as_evidence() -> None:
    with pytest.raises(EventError, match="cannot be admitted as evidence"):
        assemble_evidence([_predicted()])


def test_observed_inferred_hypothesis_are_all_evidence_eligible() -> None:
    admitted = assemble_evidence([_observed(), _inferred(), _hypothesis()])
    assert len(admitted) == 3


def test_evidence_assembly_rejects_predicted_even_mixed_with_eligible() -> None:
    with pytest.raises(EventError):
        assemble_evidence([_observed(), _predicted()])


def test_no_isinstance_predicted_event_check_in_evidence_admission_source() -> None:
    """The rejection mechanism is dispatch, not a written conditional.

    Regression guard against someone "fixing" the dispatch by adding back
    an ``if isinstance(event, PredictedEvent): raise`` — that would still
    pass the behavioural tests above, so this asserts the implementation
    shape directly.
    """
    import inspect

    from src.model import events as events_module

    source = inspect.getsource(events_module._admit_as_evidence)
    assert "isinstance" not in source
    assert "PredictedEvent" not in source


# ---------------------------------------------------------------------------
# Alert triggering moved to src.model.alert.emit_alert (Day 15).
#
# Day 13's events.raise_alert() lived here as a permissive type-eligibility
# check with no evidence requirement; Day 15 deleted it (see
# src/model/events.py's module docstring and src/model/alert.py's) because
# its one distinguishing feature -- letting a PredictedEvent trigger an
# alert -- could never be paired with an evidence chain, since
# PredictedEvent is excluded from evidence eligibility too. What alert
# triggering rejects and admits by type is now tested against emit_alert
# in tests/test_model_alert.py: test_emit_alert_rejects_hypothesis_event,
# test_emit_alert_rejects_predicted_event,
# test_emit_alert_succeeds_for_observed_and_inferred.
# ---------------------------------------------------------------------------
# STRUCTURAL: confirming a prediction produces a NEW ObservedEvent
# ---------------------------------------------------------------------------


def test_confirm_prediction_produces_new_observed_event_with_supersedes() -> None:
    predicted = _predicted()
    confirmed = confirm_prediction(
        predicted,
        event_id=uuid.uuid4(),
        ts_ns=predicted.ts_ns + 5,
        manifest_sha="sha-2",
    )
    assert isinstance(confirmed, ObservedEvent)
    assert confirmed.supersedes == predicted.event_id
    assert confirmed.event_id != predicted.event_id
    # The original prediction is untouched and still independently valid.
    assert predicted.supersedes is None
    assert predicted.event_class == "predicted"


def test_confirm_prediction_does_not_mutate_the_predicted_event() -> None:
    predicted = _predicted()
    before = dataclasses.astuple(predicted)
    confirm_prediction(
        predicted,
        event_id=uuid.uuid4(),
        ts_ns=predicted.ts_ns + 1,
        manifest_sha="sha-2",
    )
    assert dataclasses.astuple(predicted) == before


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_observed, _inferred, _predicted, _hypothesis])
def test_event_v2_dict_round_trip(factory: object) -> None:
    event = factory()  # type: ignore[operator]
    payload = event_v2_to_dict(event)
    restored = event_v2_from_dict(payload)
    assert restored == event


def test_event_v2_round_trip_with_clip_and_supersedes() -> None:
    clip = ClipRef(
        video_id="cam-1/clip", start_ts_ns=0, end_ts_ns=10, content_sha="abc"
    )
    predicted = _predicted()
    confirmed = confirm_prediction(
        predicted, event_id=uuid.uuid4(), ts_ns=200, manifest_sha="sha-2"
    )
    confirmed = dataclasses.replace(confirmed, clip=clip)
    payload = event_v2_to_dict(confirmed)
    restored = event_v2_from_dict(payload)
    assert restored == confirmed
    assert restored.supersedes == predicted.event_id


def test_event_v2_from_dict_rejects_wrong_schema_version() -> None:
    payload = event_v2_to_dict(_observed())
    payload["schema_version"] = "1.0"
    with pytest.raises(EventError):
        event_v2_from_dict(payload)


def test_event_v2_from_dict_rejects_unknown_event_class() -> None:
    payload = event_v2_to_dict(_observed())
    payload["event_class"] = "speculated"
    with pytest.raises(EventError):
        event_v2_from_dict(payload)
