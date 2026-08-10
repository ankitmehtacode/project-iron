"""Objective 6 — Episode, ActivityMode, StateGraph skeleton.

STRUCTURAL rules under test:
  - is_interaction is derived, not a separate Interaction type.
  - ActivityMode.admissible is always False; attach_to_alert and
    attach_to_evidence both unconditionally raise.
  - StateGraph is append-only with a monotonic graph_rev.
  - solve_state: Day 20 fills the single-entity case (see
    tests/test_estimator_filter.py for the real end-to-end path); this file
    keeps the boundary tests that belong to the model layer itself — a bare
    Day-13-style graph with no estimator payload still refuses (just with a
    more specific error now), and horizon_kind="smoothed" still raises
    NotImplementedError.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.model.episode import (
    ActivityMode,
    Episode,
    EpisodeError,
    Factor,
    Participant,
    StateGraph,
    StateQuery,
    attach_to_alert,
    attach_to_evidence,
    solve_state,
)
from src.model.evidence import DerivationStep, Evidence, UncalibratedScore

BASE_TS = 1_785_000_000 * 1_000_000_000


def _episode(**overrides: object) -> Episode:
    kwargs: dict[str, object] = dict(
        episode_id="ep-1",
        site_id="site-hq-1",
        start_ns=BASE_TS,
        end_ns=BASE_TS + 30_000_000_000,
        participants=(
            Participant(entity_id="sess-1", role="primary"),
            Participant(entity_id="laptop-1", role="object"),
        ),
        boundary_confidence=0.8,
        evidence_refs=("ev-1",),
        manifest_sha="sha-ep-1",
    )
    kwargs.update(overrides)
    return Episode(**kwargs)  # type: ignore[arg-type]


def _mode(**overrides: object) -> ActivityMode:
    kwargs: dict[str, object] = dict(
        scope_id="ep-1",
        scope_kind="episode",
        label="browsing",
        score=UncalibratedScore(0.6),
    )
    kwargs.update(overrides)
    return ActivityMode(**kwargs)  # type: ignore[arg-type]


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        clip_refs=(),
        state_refs=(),
        observation_refs=("obs-1",),
        derivation_chain=(DerivationStep(stage="s", producer_sha="p"),),
        producer_shas=(),
        reproducible=True,
        reproduce_command="cmd",
    )


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def test_episode_requires_participants() -> None:
    with pytest.raises(EpisodeError):
        _episode(participants=())


def test_episode_requires_positive_span() -> None:
    with pytest.raises(EpisodeError):
        _episode(start_ns=100, end_ns=100)


def test_episode_boundary_confidence_range() -> None:
    with pytest.raises(EpisodeError):
        _episode(boundary_confidence=1.2)


def test_participant_rejects_unknown_role() -> None:
    with pytest.raises(EpisodeError):
        Participant(entity_id="x", role="villain")  # type: ignore[arg-type]


def test_is_interaction_derived_not_a_separate_type() -> None:
    solo = _episode(
        participants=(
            Participant(entity_id="sess-1", role="primary"),
            Participant(entity_id="laptop-1", role="object"),
        )
    )
    assert not solo.is_interaction

    two_people = _episode(
        participants=(
            Participant(entity_id="sess-1", role="primary"),
            Participant(entity_id="sess-2", role="counterpart"),
            Participant(entity_id="laptop-1", role="object"),
        )
    )
    assert two_people.is_interaction

    # No separate Interaction class exists in the module.
    import src.model.episode as episode_module

    assert not hasattr(episode_module, "Interaction")


def test_episode_is_frozen() -> None:
    ep = _episode()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.boundary_confidence = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# STRUCTURAL: ActivityMode is never admissible.
# ---------------------------------------------------------------------------


def test_activity_mode_score_is_typed_uncalibrated() -> None:
    mode = _mode()
    assert isinstance(mode.score, UncalibratedScore)
    assert not hasattr(mode.score, "probability")


def test_activity_mode_admissible_is_always_false() -> None:
    mode = _mode()
    assert mode.admissible is False


def test_activity_mode_rejects_true_admissible_even_bypassing_typing() -> None:
    with pytest.raises(EpisodeError):
        ActivityMode(
            scope_id="ep-1",
            scope_kind="episode",
            label="browsing",
            score=UncalibratedScore(0.5),
            admissible=True,  # type: ignore[arg-type]
        )


def test_activity_mode_scope_kind_has_no_identity_option() -> None:
    fields = {f.name for f in dataclasses.fields(ActivityMode)}
    assert "scope_kind" in fields
    with pytest.raises(EpisodeError):
        _mode(scope_kind="enrolled_identity")


def test_activity_mode_rejects_unknown_label() -> None:
    with pytest.raises(EpisodeError):
        _mode(label="plotting_a_heist")


def test_attach_to_alert_always_raises() -> None:
    mode = _mode()
    with pytest.raises(EpisodeError, match="cannot trigger an alert"):
        attach_to_alert(mode)


def test_attach_to_evidence_always_raises() -> None:
    mode = _mode()
    with pytest.raises(EpisodeError, match="cannot be admitted into evidence"):
        attach_to_evidence(mode, _evidence())


# ---------------------------------------------------------------------------
# StateGraph skeleton
# ---------------------------------------------------------------------------


def test_state_graph_append_only_and_monotonic_graph_rev() -> None:
    graph = StateGraph()
    assert graph.graph_rev == 0
    f1 = graph.append_factor("f1", "observation_likelihood", ("obs-1",), "sha-1")
    f2 = graph.append_factor("f2", "motion_prior", ("obs-2",), "sha-1")
    assert f1.graph_rev == 1
    assert f2.graph_rev == 2
    assert graph.graph_rev == 2
    assert graph.factors_as_of(1) == (f1,)
    assert graph.factors_as_of(2) == (f1, f2)


def test_state_graph_has_no_mutation_method() -> None:
    """append_factor is the only mutator; payload_for (Day 20) is a pure
    read accessor, same category as factors_as_of, not a second mutator."""
    public_methods = {
        name
        for name in dir(StateGraph)
        if not name.startswith("_") and callable(getattr(StateGraph, name))
    }
    assert public_methods == {"append_factor", "factors_as_of", "payload_for"}


def test_factor_is_frozen() -> None:
    graph = StateGraph()
    factor = graph.append_factor("f1", "kind", ("x",), "sha")
    with pytest.raises(dataclasses.FrozenInstanceError):
        factor.graph_rev = 99  # type: ignore[misc]


def test_factor_requires_nonempty_inputs() -> None:
    with pytest.raises(EpisodeError):
        Factor(factor_id="f", factor_kind="k", inputs=(), graph_rev=1, manifest_sha="s")


def test_state_query_rejects_negative_horizon() -> None:
    with pytest.raises(EpisodeError):
        StateQuery(at_ts_ns=1, horizon_ns=-1, graph_rev=0)


def test_solve_state_on_bare_skeleton_graph_raises() -> None:
    """A graph built the Day-13 way (append_factor with no payload) still
    cannot be resolved -- there is no numeric estimate attached to any
    factor for the filter to find. Day 20 makes this a specific,
    diagnosable EpisodeError instead of a blanket NotImplementedError; see
    tests/test_estimator_filter.py for the real, payload-bearing path."""
    graph = StateGraph()
    graph.append_factor("f1", "kind", ("x",), "sha")
    query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    with pytest.raises(EpisodeError, match="no resolvable state"):
        solve_state(query, graph)


def test_solve_state_smoothed_horizon_raises_not_implemented() -> None:
    """Day 20's scope is single-entity FILTERING only. horizon_kind
    exists and is exercised (StateQuery accepts and validates it), but
    'smoothed' names a real, not-yet-built capability rather than silently
    falling back to a filtered answer nobody asked for."""
    graph = StateGraph()
    graph.append_factor("f1", "kind", ("x",), "sha")
    query = StateQuery(
        at_ts_ns=BASE_TS,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
        horizon_kind="smoothed",
    )
    with pytest.raises(NotImplementedError):
        solve_state(query, graph)


def test_state_query_rejects_unknown_horizon_kind() -> None:
    bogus_kind: Any = "bogus"
    with pytest.raises(EpisodeError):
        StateQuery(at_ts_ns=1, horizon_ns=0, graph_rev=0, horizon_kind=bogus_kind)


def test_state_graph_rejects_duplicate_factor_id() -> None:
    graph = StateGraph()
    graph.append_factor("f1", "kind", ("x",), "sha")
    with pytest.raises(EpisodeError):
        graph.append_factor("f1", "kind2", ("y",), "sha")


def test_payload_for_returns_none_when_absent() -> None:
    graph = StateGraph()
    graph.append_factor("f1", "kind", ("x",), "sha")
    assert graph.payload_for("f1") is None
    assert graph.payload_for("does-not-exist") is None


def test_append_factor_stores_and_returns_payload() -> None:
    graph = StateGraph()
    sentinel = object()
    graph.append_factor("f1", "kind", ("x",), "sha", payload=sentinel)
    assert graph.payload_for("f1") is sentinel
