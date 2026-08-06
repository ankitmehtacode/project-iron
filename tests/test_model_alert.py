"""Objective 2 (Day 14) — closing falsification test 4, the alert-evidence hole.

STRUCTURAL rules under test:
  - Alert.evidence_chain is required and non-empty — no alert without
    resolvable evidence.
  - emit_alert() is a closed-world singledispatch: only ObservedEvent and
    InferredEvent are eligible. PredictedEvent and HypothesisEvent both
    raise.
  - explain() resolves the full chain or raises naming the exact broken
    hop.
  - Carried from Day 13: ActivityMode still cannot attach to alerts or
    evidence.
  - Day 15: emit_alert() is the ONLY public way to emit an alert.
    events.raise_alert(), the Day-13 permissive path, is deleted — see
    test_alert_package_exposes_no_alternative_emission_entry_point below.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from src.events.schema import EntityRef, Verb
from src.model.alert import (
    Alert,
    AlertError,
    ExplainabilityError,
    emit_alert,
    explain,
)
from src.model.episode import ActivityMode, attach_to_alert, attach_to_evidence
from src.model.evidence import DerivationStep, Evidence, UncalibratedScore
from src.model.events import (
    HypothesisEvent,
    InferredEvent,
    ObservedEvent,
    PredictedEvent,
)

BASE_TS = 1_785_000_000 * 1_000_000_000


def _observed_event(evidence_refs: tuple[str, ...] = ("ev-1",)) -> ObservedEvent:
    return ObservedEvent(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=BASE_TS,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.PICKED_UP,
        object=EntityRef("asset", "laptop-1"),
        confidence=0.9,
        importance=0.8,
        manifest_sha="sha-1",
        evidence_refs=evidence_refs,
    )


def _inferred_event(evidence_refs: tuple[str, ...] = ("ev-1",)) -> InferredEvent:
    return InferredEvent(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=BASE_TS,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.EXITED,
        confidence=0.7,
        importance=0.5,
        manifest_sha="sha-1",
        basis="trajectory extrapolated across a blind spot",
        evidence_refs=evidence_refs,
    )


def _predicted_event(evidence_refs: tuple[str, ...] = ()) -> PredictedEvent:
    return PredictedEvent(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=BASE_TS,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.EXITED,
        confidence=0.6,
        importance=0.3,
        manifest_sha="sha-1",
        predicted_by="trajectory-extrapolator-v1",
        evidence_refs=evidence_refs,
    )


def _hypothesis_event(evidence_refs: tuple[str, ...] = ()) -> HypothesisEvent:
    return HypothesisEvent(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=BASE_TS,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.LOITERED,
        confidence=0.2,
        importance=0.1,
        manifest_sha="sha-1",
        rationale="dwell exceeded threshold on a fragmented track",
        evidence_refs=evidence_refs,
    )


def _evidence(evidence_id: str = "ev-1", **overrides: object) -> Evidence:
    kwargs: dict[str, object] = dict(
        evidence_id=evidence_id,
        clip_refs=("cam-1/clip-1",),
        state_refs=("graph_rev=3",),
        observation_refs=("obs-1", "obs-2"),
        derivation_chain=(
            DerivationStep(stage="detector:yolov8", producer_sha="sha-det"),
            DerivationStep(stage="hand_object_continuity", producer_sha="sha-cont"),
        ),
        producer_shas=("sha-det", "sha-cont"),
        reproducible=True,
        reproduce_command="python scripts/rerun_claim.py --evidence ev-1",
    )
    kwargs.update(overrides)
    return Evidence(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# STRUCTURAL: Alert.evidence_chain required and non-empty
# ---------------------------------------------------------------------------


def test_alert_evidence_chain_is_required_field() -> None:
    fields = {f.name for f in dataclasses.fields(Alert)}
    assert "evidence_chain" in fields
    field = next(f for f in dataclasses.fields(Alert) if f.name == "evidence_chain")
    assert field.default is dataclasses.MISSING


def test_alert_rejects_empty_evidence_chain() -> None:
    with pytest.raises(AlertError):
        Alert(
            alert_id="alert-1",
            event=_observed_event(),
            evidence_chain=(),
            manifest_sha="sha-alert-1",
        )


def test_alert_constructs_with_nonempty_chain() -> None:
    alert = Alert(
        alert_id="alert-1",
        event=_observed_event(),
        evidence_chain=(_evidence(),),
        manifest_sha="sha-alert-1",
    )
    assert alert.evidence_chain


def test_alert_is_frozen() -> None:
    alert = Alert(
        alert_id="alert-1",
        event=_observed_event(),
        evidence_chain=(_evidence(),),
        manifest_sha="sha-alert-1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        alert.alert_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# STRUCTURAL: emit_alert is closed-world dispatch, only Observed/Inferred
# ---------------------------------------------------------------------------


def test_emit_alert_succeeds_for_observed_and_inferred() -> None:
    for factory in (_observed_event, _inferred_event):
        alert = emit_alert(
            "alert-1",
            factory(),
            evidence_chain=[_evidence()],
            manifest_sha="sha-alert-1",
        )
        assert isinstance(alert, Alert)


def test_emit_alert_rejects_predicted_event() -> None:
    with pytest.raises(AlertError, match="cannot emit a fully-explainable alert"):
        emit_alert(
            "alert-1",
            _predicted_event(),
            evidence_chain=[_evidence()],
            manifest_sha="sha-alert-1",
        )


def test_emit_alert_rejects_hypothesis_event() -> None:
    with pytest.raises(AlertError, match="cannot emit a fully-explainable alert"):
        emit_alert(
            "alert-1",
            _hypothesis_event(),
            evidence_chain=[_evidence()],
            manifest_sha="sha-alert-1",
        )


def test_emit_alert_rejects_empty_evidence_even_for_eligible_event() -> None:
    with pytest.raises(AlertError):
        emit_alert(
            "alert-1", _observed_event(), evidence_chain=[], manifest_sha="sha-alert-1"
        )


def test_no_isinstance_or_predicted_hypothesis_check_in_dispatch_source() -> None:
    """Regression guard: the rejection is dispatch, not a written conditional."""
    import inspect

    from src.model import alert as alert_module

    source = inspect.getsource(alert_module._admit_for_alert_emission)
    assert "isinstance" not in source
    assert "PredictedEvent" not in source
    assert "HypothesisEvent" not in source


# ---------------------------------------------------------------------------
# STRUCTURAL (Day 15): exactly one public way to emit an alert.
#
# Day 13 shipped a second entry point, events.raise_alert() -- permissive,
# no evidence requirement. Auditing its callers found none in production
# and found its one distinguishing capability (letting a PredictedEvent
# alert) could never be paired with an evidence chain, so it was deleted
# rather than migrated. These tests assert the deletion on the module's
# actual public surface, not on a convention a later edit could reopen by
# adding a new function and forgetting to route it through emit_alert.
# ---------------------------------------------------------------------------


def test_alert_package_exposes_no_alternative_emission_entry_point() -> None:
    from src.model import alert as alert_module

    assert not hasattr(alert_module, "raise_alert")

    public_emitters = [
        name
        for name in alert_module.__all__
        if "alert" in name.lower()
        and callable(getattr(alert_module, name))
        and not isinstance(getattr(alert_module, name), type)
    ]
    assert public_emitters == ["emit_alert"], (
        "alert.py's public surface must expose exactly one alert-emitting "
        f"callable; found {public_emitters}"
    )


def test_raise_alert_deleted_from_events_module_and_package_root() -> None:
    import src.model as model_package
    from src.model import events as events_module

    assert not hasattr(events_module, "raise_alert")
    assert not hasattr(model_package, "raise_alert")
    assert "raise_alert" not in model_package.__all__
    assert not hasattr(events_module, "AlertEligibleEvent")


# ---------------------------------------------------------------------------
# explain() — full resolution, or the exact broken hop
# ---------------------------------------------------------------------------


def test_explain_resolves_the_full_chain() -> None:
    event = _observed_event()
    evidence = _evidence()
    alert = emit_alert(
        "alert-1", event, evidence_chain=[evidence], manifest_sha="sha-alert-1"
    )
    store = {"alert-1": alert}

    explained = explain("alert-1", store)

    assert explained.alert_id == "alert-1"
    assert explained.event_id == str(event.event_id)
    assert explained.event_class == "observed"
    assert len(explained.hops) == 1
    hop = explained.hops[0]
    assert hop.evidence_id == "ev-1"
    assert hop.observation_refs == ("obs-1", "obs-2")
    assert hop.clip_refs == ("cam-1/clip-1",)
    assert hop.producer_shas == ("sha-det", "sha-cont")
    assert hop.derivation_stages == ("detector:yolov8", "hand_object_continuity")


def test_explain_unknown_alert_id_names_that_hop() -> None:
    with pytest.raises(ExplainabilityError, match="alert_id -> alert"):
        explain("does-not-exist", {})


def test_explain_event_with_no_evidence_refs_names_that_hop() -> None:
    event = _observed_event(evidence_refs=())
    # Bypass emit_alert (which would also reject this) to test explain()
    # in isolation against a hand-built Alert -- still must go through
    # Alert's own non-empty evidence_chain check, so give it one evidence
    # record the event itself does not reference.
    alert = Alert(
        alert_id="alert-1",
        event=event,
        evidence_chain=(_evidence(),),
        manifest_sha="sha-alert-1",
    )
    with pytest.raises(ExplainabilityError, match="event -> evidence"):
        explain("alert-1", {"alert-1": alert})


def test_explain_evidence_ref_not_in_chain_names_that_hop() -> None:
    event = _observed_event(evidence_refs=("ev-does-not-exist",))
    alert = Alert(
        alert_id="alert-1",
        event=event,
        evidence_chain=(_evidence(evidence_id="ev-1"),),
        manifest_sha="sha-alert-1",
    )
    with pytest.raises(ExplainabilityError, match="event -> evidence"):
        explain("alert-1", {"alert-1": alert})


def test_explain_missing_observation_in_store_names_that_hop() -> None:
    event = _observed_event()
    evidence = _evidence()
    alert = emit_alert(
        "alert-1", event, evidence_chain=[evidence], manifest_sha="sha-alert-1"
    )
    incomplete_store = {"obs-1": object()}  # missing obs-2

    with pytest.raises(ExplainabilityError, match="observation -> store"):
        explain("alert-1", {"alert-1": alert}, observation_store=incomplete_store)


def test_explain_succeeds_when_observation_store_is_complete() -> None:
    event = _observed_event()
    evidence = _evidence()
    alert = emit_alert(
        "alert-1", event, evidence_chain=[evidence], manifest_sha="sha-alert-1"
    )
    complete_store = {"obs-1": object(), "obs-2": object()}

    explained = explain("alert-1", {"alert-1": alert}, observation_store=complete_store)
    assert explained.hops[0].observation_refs == ("obs-1", "obs-2")


# ---------------------------------------------------------------------------
# Carried from Day 13: ActivityMode remains attachable to neither.
# ---------------------------------------------------------------------------


def test_activity_mode_still_cannot_attach_to_alert_or_evidence() -> None:
    mode = ActivityMode(
        scope_id="ep-1",
        scope_kind="episode",
        label="browsing",
        score=UncalibratedScore(0.5),
    )
    with pytest.raises(Exception):
        attach_to_alert(mode)
    with pytest.raises(Exception):
        attach_to_evidence(mode, _evidence())
