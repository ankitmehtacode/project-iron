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
    4. Alert explainability               -- PASSES as of Day 14; seam
       closed Day 15
       (src/model/alert.py — emit_alert()/explain() require and resolve a
       full evidence chain. Day 15 deleted events.raise_alert(), the
       Day-13 permissive path with no evidence requirement: auditing its
       callers found none in production, and found its one distinguishing
       feature -- letting a PredictedEvent trigger an alert -- could never
       be paired with an evidence chain, since PredictedEvent is excluded
       from evidence eligibility too. emit_alert() is now the only public
       alert path.)
       Day 13: PARTIAL, an alert-eligible event could carry zero evidence.
    5. Behaviour-query shape              -- PARTIAL as of Day 20
       (an ActivityMode can never be confused with a fact -- unchanged);
       solve_state now genuinely resolves a single-entity StateQuery
       (src/estimator, Day 20) instead of raising NotImplementedError
       unconditionally, so the "blocked on the unimplemented estimator"
       half of Day 13's finding is closed for the single-entity case. What
       remains blocked: nothing yet turns a resolved StateEstimate into an
       ActivityMode label (that is behaviour modeling, still out of
       scope), so this test proves state estimation now answers live
       queries, not that a live behaviour query can be answered end to
       end. Multi-entity graphs and horizon_kind="smoothed" remain
       NotImplementedError, tested explicitly.
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
from src.model.episode import (
    ActivityMode,
    EpisodeError,
    StateGraph,
    StateQuery,
    attach_to_alert,
    attach_to_evidence,
    solve_state,
)
from src.model.evidence import DerivationStep, Evidence, UncalibratedScore
from src.model.events import (
    EventError,
    HypothesisEvent,
    InferredEvent,
    ObservedEvent,
    PredictedEvent,
    assemble_evidence,
)
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.relationship import (
    ArtifactRegistry,
    Correction,
    DerivedArtifact,
    Relationship,
)
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid
from src.model.world import (
    RigidTransform3D,
    TwinRevError,
    TwinRevTransform,
    WorldPosition,
)
from src.estimator.filter import run_single_entity_filter
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import motion_model_for

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
    real, non-empty derivation chain, and emit_alert()/explain() actually
    walk that chain end to end — the alert can be explained, not just
    shown to carry a ref that points somewhere.
    """
    event = _explainable_event()
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
        "alert-1", event, evidence_chain=[evidence], manifest_sha="sha-1"
    )
    explained = explain("alert-1", {"alert-1": alert})
    assert explained.hops[0].observation_refs == ("obs-1", "obs-2")
    assert explained.hops[0].derivation_stages == (
        "detector:yolov8",
        "hand_object_continuity",
    )


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


def test_falsification_predicted_event_can_neither_alert_nor_be_evidence() -> None:
    """Day 15: a PredictedEvent can trigger NEITHER emit_alert() nor
    assemble_evidence() (ADR 0003).

    Day 13/14's version of this test asserted a PredictedEvent COULD
    trigger an alert via the permissive events.raise_alert() -- a
    "forecast paging" path that Day 15 deleted, because it could never be
    paired with an evidence chain: assemble_evidence() has no handler for
    PredictedEvent either, so that path could only ever alert on nothing.
    A future forecast-paging feature must render the InferredEvent/
    ObservedEvent chain that fed the predictor and be built as its own
    explicit capability, not by relaxing emit_alert()'s dispatch.
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
    with pytest.raises(AlertError):
        emit_alert("alert-x", predicted, evidence_chain=[], manifest_sha="sha-1")
    with pytest.raises(EventError):
        assemble_evidence([predicted])


# ---------------------------------------------------------------------------
# 5. Behaviour-query shape -- PARTIAL as of Day 20
# ---------------------------------------------------------------------------


def _position_observation(x_m: float, ts_ns: int) -> Observation:
    return Observation(
        observation_id=generate_ulid(now_ns=ts_ns),
        sensor_id="cam-falsification-5",
        ts_ns=ts_ns,
        frame_ref="cam-falsification-5/frame",
        measurement=WorldPositionMeasurement(x_m=x_m, y_m=0.0, z_m=0.0),
        uncertainty=Uncertainty(kind="gaussian_3d", params=(("sigma_m", 0.05),)),
        frame_of_reference=FrameOfReference(
            geometry=FrameGeometry(1920, 1080),
            to_canonical=AffineTransform.identity(),
            twin_rev=1,
        ),
        producer_shas=("estimator-test",),
        envelope_status="within_envelope",
    )


def test_falsification_behaviour_query_shape_cannot_be_confused_with_a_fact() -> None:
    """An ActivityMode (the shape a behaviour query would answer with)
    can never satisfy an alert or evidence path, and is never one of the
    four fact-bearing event classes -- PASSES structurally, unchanged.

    Day 13/17-19: solve_state raised NotImplementedError unconditionally,
    so this test could only prove the answer's shape was safe, not that a
    live query could be answered. Day 20 fills solve_state for the
    single-entity case: this test now builds a REAL single-entity graph
    (src/estimator's filter, not a bare append_factor with no payload) and
    confirms solve_state returns an actual StateEstimate rather than
    raising. Still PARTIAL, not PASSES: nothing turns that StateEstimate
    into an ActivityMode label yet (behaviour modeling is out of scope),
    so a live *behaviour* query is still unanswerable end to end -- only
    the state-estimation half that would feed one now works.
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

    # The single-entity estimator now exists and genuinely resolves a
    # StateQuery -- this is the boundary that moved since Day 13.
    graph = StateGraph()
    observations = [
        _position_observation(1.0, BASE_TS),
        _position_observation(1.5, BASE_TS + SECOND_NS),
    ]
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-falsification-5"),
        manifest_sha="falsification-test-5",
    )
    estimate = solve_state(
        StateQuery(
            at_ts_ns=BASE_TS + SECOND_NS, horizon_ns=0, graph_rev=graph.graph_rev
        ),
        graph,
    )
    assert estimate.observed is True
    assert estimate.residuals  # STRUCTURAL: never absent, see src.estimator.state

    # What is STILL blocked: turning a StateEstimate into an ActivityMode.
    # No function in this codebase does that -- there is nothing to call
    # and assert NotImplementedError on, which is itself the honest report:
    # this seam has not been built yet, not merely stubbed.
    assert not hasattr(estimate, "activity_mode")

    # And multi-entity / smoothed resolution are still real gaps, not
    # silently-wrong answers:
    smoothed_query = StateQuery(
        at_ts_ns=BASE_TS + SECOND_NS,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
        horizon_kind="smoothed",
    )
    with pytest.raises(NotImplementedError):
        solve_state(smoothed_query, graph)

    # A bare Day-13-style graph (no estimator payload) still cannot be
    # resolved -- confirms the refusal is about missing evidence, not a
    # blanket "unimplemented" any longer.
    bare_graph = StateGraph()
    bare_graph.append_factor("f1", "motion_prior", ("obs-1",), "sha-1")
    with pytest.raises(EpisodeError, match="no resolvable state"):
        solve_state(
            StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=bare_graph.graph_rev),
            bare_graph,
        )
