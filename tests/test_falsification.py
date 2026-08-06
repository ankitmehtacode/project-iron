"""The five falsification tests from IRON_DATA_MODEL, run as integration tests.

Each test exercises several `src/model` primitives together, the way a
real query or write path would, rather than one type in isolation. Per
the Day-13 prompt: "A data model that passes all five on day one probably
wasn't tested hard enough" — so where a test finds a genuine gap rather
than confirming a guarantee, it says so in its own docstring and asserts
the gap explicitly (not just leaves it unasserted), and the finding is
also written up in FOUNDATION_REPORT.md's Day-13 section.

Summary (see the Day-14 report for the full discussion; Day-13 status in
parens):
    1. Absence under degraded coverage    -- PASSES (unchanged since Day 13)
    2. Retroactive badge resolution       -- PASSES (unchanged since Day 13)
    3. Twin re-version                    -- PASSES as of Day 14
       (src/model/world.py — WorldPosition requires twin_rev; cross-rev
       distance/reproject raise without an explicit TwinRevTransform).
       Day 13: PARTIAL, staleness detectable but not enforced.
    4. Alert explainability               -- PASSES as of Day 14
       (src/model/alert.py — emit_alert()/explain() require and resolve a
       full evidence chain; raise_alert() stays the deliberately broader
       Day-13 type-eligibility check for forecast-only paging).
       Day 13: PARTIAL, an alert-eligible event could carry zero evidence.
    5. Behaviour-query shape              -- PASSES at the type level
       (an ActivityMode can never be confused with a fact); still blocked
       on the unimplemented estimator for producing a real answer to a
       live query — unchanged since Day 13, by scope, not by oversight.
"""

from __future__ import annotations

import uuid

import pytest

from src.contracts.frames import AffineTransform, FrameGeometry
from src.events.schema import EntityRef, Verb
from src.model.alert import AlertError, emit_alert, explain
from src.model.coverage import (
    Absence,
    CannotEstablish,
    Coverage,
    Interval,
    prove_absence,
)
from src.model.episode import ActivityMode, attach_to_alert, attach_to_evidence
from src.model.evidence import DerivationStep, Evidence, UncalibratedScore
from src.model.events import (
    EventError,
    HypothesisEvent,
    InferredEvent,
    ObservedEvent,
    PredictedEvent,
    assemble_evidence,
    raise_alert,
)
from src.model.frame_of_reference import FrameOfReference
from src.model.relationship import (
    ArtifactRegistry,
    Correction,
    DerivedArtifact,
    Relationship,
)
from src.model.world import (
    RigidTransform3D,
    TwinRevError,
    TwinRevTransform,
    WorldPosition,
)

SECOND_NS = 1_000_000_000
BASE_TS = 1_785_000_000 * SECOND_NS


# ---------------------------------------------------------------------------
# 1. Absence under degraded coverage -- PASSES
# ---------------------------------------------------------------------------


def test_falsification_absence_under_degraded_coverage() -> None:
    """A camera that goes 'degraded' partway through the query window must
    block a proven absence, even though it saw nothing the whole time.

    This is the failure this system cannot afford: reporting "no one
    entered the loading dock" from a camera that was struggling to see
    for half the window. The model must refuse to prove absence there.
    """
    query = Interval(BASE_TS, BASE_TS + 3600 * SECOND_NS)
    live_half = Coverage(
        subject_id="cam-dock-02",
        subject_kind="camera",
        interval=Interval(query.start_ns, query.start_ns + 1800 * SECOND_NS),
        status="live",
        envelope_ref="motion_gate@cam-dock-02@twin_rev=1",
        gaps=(),
        manifest_sha="sha-cov-1",
    )
    degraded_half = Coverage(
        subject_id="cam-dock-02",
        subject_kind="camera",
        interval=Interval(query.start_ns + 1800 * SECOND_NS, query.end_ns),
        status="degraded",
        envelope_ref="motion_gate@cam-dock-02@twin_rev=1",
        gaps=(),
        manifest_sha="sha-cov-2",
    )

    def never_entered(subject_id: str, interval: Interval) -> bool:
        return False

    result = prove_absence(
        "cam-dock-02", query, never_entered, coverage_log=[live_half, degraded_half]
    )

    assert isinstance(result, CannotEstablish)
    assert not isinstance(result, Absence)
    assert result.reason == "coverage_insufficient"
    assert any("degraded" in v for v in result.envelope_violations)


# ---------------------------------------------------------------------------
# 2. Retroactive badge resolution -- PASSES
# ---------------------------------------------------------------------------


def test_falsification_retroactive_badge_resolution() -> None:
    """Identity resolves retroactively; derived artifacts invalidate
    transitively; the pre-correction answer stays retrievable.

    Full scenario also covered directly in test_model_relationship.py;
    reproduced here as an end-to-end integration path because this is one
    of the five falsification tests the data model must survive.
    """
    registry = ArtifactRegistry()
    scorecard = DerivedArtifact(
        artifact_id="scorecard-lobby-1358",
        kind="scorecard",
        input_closure=("sess-4f2a91",),
        manifest_sha="sha-1",
    )
    registry.register(scorecard)

    resolution = Relationship(
        subject=EntityRef("session", "sess-4f2a91"),
        predicate="same_identity_as",
        object=EntityRef("enrolled", "employee-42"),
        valid_from_ns=BASE_TS,  # 13:58
        valid_to_ns=None,
        asserted_at_ns=BASE_TS + 4 * 60 * SECOND_NS,  # 14:02
        asserted_by="badge-reader-03",
        basis="observed",
        confidence=0.98,
        evidence=("obs-badge-1",),
    )
    assert resolution.is_retroactive

    correction = Correction(
        target="sess-4f2a91",
        kind="identity_resolution",
        reason="badge swipe matched",
        evidence=("obs-badge-1",),
        actor="reconciliation-job",
        invalidates=("scorecard-lobby-1358",),
    )
    invalidated = registry.apply_correction(correction)

    assert invalidated == frozenset({"scorecard-lobby-1358"})
    assert not registry.is_valid("scorecard-lobby-1358")
    # The pre-correction record is untouched and still retrievable.
    assert registry.get("scorecard-lobby-1358") == scorecard


# ---------------------------------------------------------------------------
# 3. Twin re-version -- PASSES as of Day 14 (see module docstring)
# ---------------------------------------------------------------------------


def test_falsification_twin_reversion_staleness_is_detectable() -> None:
    """A camera recalibration bumps twin_rev; old-rev data must be flaggable.

    FrameOfReference.is_current_for correctly distinguishes pre-remount
    from post-remount pixel-frame data (unchanged since Day 13).
    """
    pre_remount = FrameOfReference(
        geometry=FrameGeometry(1920, 1080),
        to_canonical=AffineTransform.identity(),
        twin_rev=1,
    )
    current_twin_rev_after_remount = 2

    assert pre_remount.is_current_for(1)
    assert not pre_remount.is_current_for(current_twin_rev_after_remount)


def test_falsification_twin_reversion_world_position_now_raises_across_revs() -> None:
    """FIXED on Day 14 (src/model/world.py): a WorldPosition cannot be
    compared, or have its distance measured, against a position from a
    different twin_rev without an explicit TwinRevTransform.

    Day 13's version of this test asserted the gap directly: nothing
    stopped a caller from combining a stale-rev position with current
    data, no exception, no warning. That is no longer true.
    """
    pre_remount = WorldPosition(x_m=1.0, y_m=2.0, z_m=0.0, twin_rev=1)
    post_remount = WorldPosition(x_m=1.0, y_m=2.0, z_m=0.0, twin_rev=2)

    with pytest.raises(TwinRevError):
        pre_remount.distance_to(post_remount)  # no transform supplied

    wrong_direction = TwinRevTransform(
        from_twin_rev=2, to_twin_rev=1, transform=RigidTransform3D.identity()
    )
    with pytest.raises(TwinRevError):
        pre_remount.reproject(wrong_direction)  # pre_remount is rev 1, not 2

    # With the correct, explicit transform, the historical position
    # remains interpretable — the falsification claim itself.
    remount_shift = TwinRevTransform(
        from_twin_rev=1,
        to_twin_rev=2,
        transform=RigidTransform3D(
            rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), translation=(0.1, 0.0, 0.0)
        ),
    )
    reprojected = pre_remount.reproject(remount_shift)
    assert reprojected.twin_rev == 2
    assert pre_remount.twin_rev == 1  # original record untouched


# ---------------------------------------------------------------------------
# 4. Alert explainability -- PASSES as of Day 14 (see module docstring)
# ---------------------------------------------------------------------------


def _explainable_event(evidence_refs: tuple[str, ...] = ("ev-1",)) -> ObservedEvent:
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


def test_falsification_alert_explainability_when_evidence_is_populated() -> None:
    """PASSES: an alert-eligible event with evidence_refs resolves to a
    real, non-empty derivation chain — the alert can be explained.
    """
    event = _explainable_event()
    triggerable = raise_alert(event)
    assert triggerable is event

    evidence_store = {
        "ev-1": Evidence(
            evidence_id="ev-1",
            clip_refs=("cam-1/clip-1",),
            state_refs=(),
            observation_refs=("obs-1", "obs-2"),
            derivation_chain=(
                DerivationStep(stage="detector:yolov8", producer_sha="sha-det"),
                DerivationStep(stage="hand_object_continuity", producer_sha="sha-cont"),
            ),
            producer_shas=("sha-det", "sha-cont"),
            reproducible=True,
            reproduce_command="python scripts/rerun_claim.py --evidence ev-1",
        )
    }

    chain = [evidence_store[ref] for ref in event.evidence_refs]
    assert all(e.derivation_chain for e in chain)
    assert all(e.observation_refs for e in chain)


def test_falsification_raise_alert_alone_still_permits_empty_evidence() -> None:
    """UNCHANGED, by design: events.raise_alert() (Day 13) is deliberately
    still the broader type-eligibility check, and still permits an event
    with empty evidence_refs to pass it — that path exists for forecast
    paging (test below), not for the fully-explainable production alert
    surface. src/model/alert.py's emit_alert() is the strict path, tested
    next, and it is what actually closes the falsification gap.
    """
    unexplainable = _explainable_event(evidence_refs=())
    triggerable = raise_alert(unexplainable)  # still succeeds -- unchanged
    assert triggerable.evidence_refs == ()


def test_falsification_emit_alert_now_requires_a_resolvable_evidence_chain() -> None:
    """FIXED on Day 14 (src/model/alert.py): the strict alert-emission path
    requires a non-empty, resolved evidence chain, and explain() resolves
    it end to end -- alert -> event -> evidence -> observations/clips,
    with producer_sha checked at every derivation step.
    """
    unexplainable = _explainable_event(evidence_refs=())
    with pytest.raises(AlertError):
        emit_alert("alert-1", unexplainable, evidence_chain=[], manifest_sha="sha-1")

    explainable = _explainable_event(evidence_refs=("ev-1",))
    evidence = Evidence(
        evidence_id="ev-1",
        clip_refs=("cam-1/clip-1",),
        state_refs=(),
        observation_refs=("obs-1", "obs-2"),
        derivation_chain=(
            DerivationStep(stage="detector:yolov8", producer_sha="sha-det"),
            DerivationStep(stage="hand_object_continuity", producer_sha="sha-cont"),
        ),
        producer_shas=("sha-det", "sha-cont"),
        reproducible=True,
        reproduce_command="python scripts/rerun_claim.py --evidence ev-1",
    )
    alert = emit_alert(
        "alert-1", explainable, evidence_chain=[evidence], manifest_sha="sha-1"
    )
    explained = explain("alert-1", {"alert-1": alert})
    assert explained.hops[0].observation_refs == ("obs-1", "obs-2")
    assert explained.hops[0].producer_shas == ("sha-det", "sha-cont")


def test_falsification_predicted_event_can_alert_but_is_not_evidence_eligible() -> None:
    """A PredictedEvent may trigger an alert (paging on a forecast) but can
    never itself be admitted as evidence (ADR 0003). Explaining such an
    alert therefore requires rendering the InferredEvent/ObservedEvent
    chain that fed the predictor, not the PredictedEvent record itself --
    a UI/rendering requirement for whichever Day builds the alert surface.
    """
    predicted = PredictedEvent(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=BASE_TS + 100 * SECOND_NS,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.EXITED,
        confidence=0.6,
        importance=0.3,
        manifest_sha="sha-1",
        predicted_by="trajectory-extrapolator-v1",
    )
    assert raise_alert(predicted) is predicted
    with pytest.raises(EventError):
        assemble_evidence([predicted])


# ---------------------------------------------------------------------------
# 5. Behaviour-query shape -- PASSES at the type level
# ---------------------------------------------------------------------------


def test_falsification_behaviour_query_shape_cannot_be_confused_with_a_fact() -> None:
    """An ActivityMode (the shape a behaviour query would answer with)
    can never satisfy an alert or evidence path, and is never one of the
    four fact-bearing event classes.

    PASSES structurally. What remains BLOCKED on the unimplemented
    estimator: there is no live path that actually answers a behaviour
    query yet (src/model/episode.py::solve_state raises
    NotImplementedError) -- this test proves the answer's shape is safe
    once that estimator exists, not that the estimator exists.
    """
    mode = ActivityMode(
        scope_id="traj-1",
        scope_kind="trajectory",
        label="browsing",
        score=UncalibratedScore(0.55),
    )

    assert not isinstance(
        mode, (ObservedEvent, InferredEvent, PredictedEvent, HypothesisEvent)
    )

    with pytest.raises(Exception):
        attach_to_alert(mode)

    dummy_evidence = Evidence(
        evidence_id="ev-x",
        clip_refs=(),
        state_refs=(),
        observation_refs=("obs-1",),
        derivation_chain=(DerivationStep(stage="s", producer_sha="p"),),
        producer_shas=(),
        reproducible=True,
        reproduce_command="cmd",
    )
    with pytest.raises(Exception):
        attach_to_evidence(mode, dummy_evidence)

    # The estimator that would produce a live ActivityMode from a real
    # trajectory does not exist yet -- confirming the boundary this
    # falsification test is checking.
    from src.model.episode import StateGraph, StateQuery, solve_state

    graph = StateGraph()
    graph.append_factor("f1", "motion_prior", ("obs-1",), "sha-1")
    with pytest.raises(NotImplementedError):
        solve_state(
            StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev), graph
        )
