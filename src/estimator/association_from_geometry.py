"""Real, geometry-grounded ``AssociationCandidate`` construction.

Extracted from ``scripts/build_association_demo.py`` (Day 35, retired Day 37
Objective 3) so the same measurement model backs a real, systematic sweep of
a golden set (``scripts/associate_golden_set.py``) instead of two hand-picked
components a human already knew the answer to.

Where the numbers come from
----------------------------
Every log-likelihood is a real 3-D isotropic Gaussian log-density, computed
from REAL agent positions in an existing golden-set clip. For a chosen frame
and a "detected" agent, each candidate is "this detection continues track
for agent K": its log-likelihood is the Gaussian log-density of the ACTUAL
frame position under a linear-velocity prediction of agent K's own track
from the two prior frames.

The measurement-noise sigma is not invented here: it is derived from
``src.estimator.motion_model.PERSON_SIGMA_A_MPS2`` (1.5 m/s^2) and
``PEDESTRIAN_STOP_DURATION_S`` (1.0 s), whose product (1.5 m/s) is this
project's own declared, cited velocity-uncertainty floor for a walking
person. Propagated forward by one assumed frame interval, that gives a
one-frame-ahead POSITION sigma of ``SIGMA_V_FLOOR_MPS * dt``.

Hard-constraint checking is NOT implemented here: ``resolve_data_association``
requires a ``hard_violation_check`` callable, and every caller here passes
one that always returns ``None`` (no physical-consistency evaluation exists
in this path). Every death recorded is therefore either
``DominatedByLikelihood`` or ``PrunedByBudget`` -- never
``RefutedByHardConstraint``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from src.estimator.joint import (
    DECISIVE_LOG_BAYES_FACTOR,
    Ambiguous,
    AssociationBudgetConfig,
    AssociationCandidate,
    Decisive,
    build_identity_event,
    resolve_data_association,
)
from src.events.schema import EntityRef, Verb
from src.model.hypothesis import DominatedByLikelihood, HypothesisStore, PrunedByBudget

FloatArray = npt.NDArray[np.float64]

SIGMA_V_FLOOR_MPS = 1.5
"""PERSON_SIGMA_A_MPS2 (1.5 m/s^2) * PEDESTRIAN_STOP_DURATION_S (1.0 s) --
src.estimator.motion_model's own cited pedestrian velocity-uncertainty
floor, re-derived here (integrated over a time step to get a position
uncertainty), not reused for the wrong quantity."""


def position_sigma_m(dt_s: float) -> float:
    return SIGMA_V_FLOOR_MPS * dt_s


def band(margin_nats: float) -> str | None:
    """Kass & Raftery (1995, JASA 90(430), p.777, Table 4) / Jeffreys (1961)
    band for a Bayes factor of exp(margin_nats), in the SAME thresholds
    src.estimator.joint's DECISIVE_LOG_BAYES_FACTOR docstring cites. A
    caller below the decisiveness threshold (an Ambiguous verdict) has no
    band -- it sits in the "bare mention" region by construction, so this
    returns None rather than a band name a viewer could mistake for
    support."""
    if margin_nats < DECISIVE_LOG_BAYES_FACTOR:
        return None
    if margin_nats < math.log(20):
        return "positive"
    if margin_nats < math.log(150):
        return "strong"
    return "very strong"


def predicted_position(
    xyz: FloatArray, agent: int, frame: int, dt_s: float
) -> FloatArray:
    velocity = (xyz[frame - 1, agent] - xyz[frame - 2, agent]) / dt_s
    result: FloatArray = xyz[frame - 1, agent] + velocity * dt_s
    return result


def log_likelihood(actual: FloatArray, predicted: FloatArray, sigma_m: float) -> float:
    squared_distance = float(np.sum((actual - predicted) ** 2))
    return -0.5 * (
        3 * math.log(2 * math.pi * sigma_m**2) + squared_distance / sigma_m**2
    )


def cause_payload(cause: Any) -> dict[str, Any] | None:
    """Every DeathCause kind, rendered with a human-readable ``detail`` so
    the UI never needs its own copy of this dispatch -- a death cause a
    caller here never produces (e.g. RefutedByHardConstraint, since no
    hard-constraint check is implemented in this path) still renders
    correctly if it ever appears."""
    if cause is None:
        return None
    payload: dict[str, Any] = {"kind": cause.kind}
    if isinstance(cause, PrunedByBudget):
        payload["budget"] = cause.budget
        payload["detail"] = (
            f"not ruled out — resource-limited: lost a competition for "
            f"{cause.budget} budget slot(s) before evidence ever spoke "
            "to whether it was right (Day 28's PRUNED_BY_BUDGET rule)"
        )
    elif isinstance(cause, DominatedByLikelihood):
        payload["dominant_hypothesis_id"] = cause.dominant_hypothesis_id
        payload["detail"] = (
            "evaluated and found decisively less likely than "
            f"{cause.dominant_hypothesis_id.split('::')[-1]}"
        )
    else:
        payload["detail"] = str(cause)
    return payload


def resolve_geometric_component(
    xyz: FloatArray,
    *,
    component_id: str,
    clip_id: str,
    source_path: str,
    site_id: str,
    frame: int,
    detected_agent: int,
    budget: int,
    assumed_fps: float,
    verb: Verb,
    importance: float,
    caller_confidence: float,
    manifest_sha: str,
    config_sha: str,
    event_namespace: str,
    selection: str,
) -> dict[str, Any]:
    """Resolve one real component: every agent in ``xyz`` at ``frame`` is a
    candidate continuation of ``detected_agent``'s track, scored by a real
    Gaussian log-density over measured geometry -- see module docstring.

    ``selection`` is written into the output payload verbatim and must say
    HOW this component was chosen (e.g. ``"systematic_sweep"`` vs
    ``"boundary_fixture_hand_selected"``) -- self-labelling, the same
    discipline Day 36 added to ``PromotionResult`` after finding an
    unlabelled SELF_TEST output could be mistaken for a real result read
    out of context. A component chosen to exercise a rendering path must
    never be mistaken for one that occurred in a systematic sweep."""
    import uuid

    dt_s = 1.0 / assumed_fps
    sigma_m = position_sigma_m(dt_s)
    n_agents = xyz.shape[1]
    actual = xyz[frame, detected_agent]

    candidates = []
    hyp_to_agent: dict[str, int] = {}
    for candidate_agent in range(n_agents):
        hyp_id = f"agent-{candidate_agent}"
        hyp_to_agent[hyp_id] = candidate_agent
        ll = log_likelihood(
            actual, predicted_position(xyz, candidate_agent, frame, dt_s), sigma_m
        )
        candidates.append(
            AssociationCandidate(
                hypothesis_id=hyp_id,
                proposition=(
                    f"detection at {clip_id} frame {frame} continues the "
                    f"track for agent-{candidate_agent}"
                ),
                log_likelihood=ll,
            )
        )

    store = HypothesisStore()
    budget_config = AssociationBudgetConfig(per_component_budget=budget)
    resolution = resolve_data_association(
        store,
        component_id,
        candidates,
        hard_violation_check=lambda _candidate: None,
        budget_config=budget_config,
    )

    def subject_for(hypothesis: Any) -> EntityRef:
        agent = hyp_to_agent[hypothesis.id.split("::", 1)[1]]
        return EntityRef(kind="session", id=f"{clip_id}::agent-{agent}")

    event = build_identity_event(
        resolution.verdict,
        subject_for,
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{event_namespace}/{component_id}"),
        site_id=site_id,
        ts_ns=int(frame * dt_s * 1e9),
        verb=verb,
        manifest_sha=manifest_sha,
        importance=importance,
        caller_confidence=caller_confidence,
    )

    verdict = resolution.verdict
    verdict_payload: dict[str, Any]
    if isinstance(verdict, Decisive):
        verdict_payload = {
            "kind": "decisive",
            "winner": verdict.winner.id,
            "margin_nats": verdict.margin,
            "kass_raftery_band": (
                "very strong" if math.isinf(verdict.margin) else band(verdict.margin)
            ),
            "competitors": [c.id for c in verdict.competitors],
        }
    elif isinstance(verdict, Ambiguous):
        verdict_payload = {
            "kind": "ambiguous",
            "margin_nats": verdict.margin,
            "competitors": [c.id for c in verdict.competitors],
        }
    else:  # pragma: no cover - exhaustive by AssociationVerdict's definition
        raise AssertionError(f"unhandled verdict {verdict!r}")

    from src.model.events import event_v2_to_dict

    return {
        "component_id": component_id,
        "selection": selection,
        "source_clip": clip_id,
        "source": source_path,
        "frame_index": frame,
        "detected_agent": detected_agent,
        "assumed_fps": assumed_fps,
        "measurement_sigma_m": sigma_m,
        "measurement_sigma_derivation": (
            "PERSON_SIGMA_A_MPS2 (1.5 m/s^2) * PEDESTRIAN_STOP_DURATION_S "
            "(1.0 s) = 1.5 m/s velocity-uncertainty floor "
            "(src.estimator.motion_model), propagated over one assumed "
            f"frame interval (dt={dt_s:.6f} s at {assumed_fps} fps) to a "
            "position sigma."
        ),
        "candidates": [
            {
                "hypothesis_id": c.hypothesis_id,
                "proposition": c.proposition,
                "log_likelihood_nats": c.log_likelihood,
            }
            for c in candidates
        ],
        "decisions": [
            {
                "hypothesis_id": d.hypothesis_id,
                "kept": d.kept,
                "cause": cause_payload(d.cause),
                "log_likelihood_nats": d.log_likelihood,
            }
            for d in resolution.decisions
        ],
        "verdict": verdict_payload,
        "budget_config": {
            "per_component_budget": budget_config.per_component_budget,
            "sha": budget_config.sha,
        },
        "graph_rev": None,
        "graph_rev_note": (
            "no src.model.episode.StateGraph is constructed for this sweep "
            "-- the verdict is computed directly from candidate "
            "log-likelihoods, not replayed against an append-only factor "
            "graph, so there is no graph revision to report. Stated as an "
            "absence rather than a fabricated 0 or 1."
        ),
        "config_sha": config_sha,
        "model_shas": None,
        "model_shas_note": (
            "no learned model produced these numbers -- the log-likelihoods "
            "are a closed-form Gaussian density over measured geometry, so "
            "there is no model artifact to hash. Stated as an absence "
            "rather than omitted."
        ),
        "event": event_v2_to_dict(event),
    }


__all__ = [
    "SIGMA_V_FLOOR_MPS",
    "position_sigma_m",
    "band",
    "predicted_position",
    "log_likelihood",
    "cause_payload",
    "resolve_geometric_component",
]
