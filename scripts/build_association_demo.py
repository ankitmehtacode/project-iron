"""Produce a real, on-disk AssociationVerdict artifact for the Inspector.

Day 35, Objective 2's Association/Identity view requires "real hypothesis-
store contents only, served from real artifacts on disk" — but as of Day 34,
``resolve_data_association``/``compute_association_verdict``/
``build_identity_event`` (src/estimator/joint.py) are exercised only by unit
tests. No script or pipeline stage ever calls them against real data and
writes the result anywhere; there was nothing on disk for a UI to serve.
This script closes exactly that gap and no more: it is a measurement/demo
script in the same family as ``measure_envelope.py`` or
``measure_substream_hypothesis.py``, not a production pipeline stage. Full
multi-hypothesis state propagation — both competing hypotheses coexisting
as a genuine mixture in the joint state — remains explicitly deferred (see
``src/estimator/joint.py``'s own module docstring); this script only makes
one already-shipped capability (the verdict computation) visible.

Where the numbers come from
----------------------------
Every log-likelihood below is a real 3-D isotropic Gaussian log-density,
computed from REAL agent positions in an existing golden-set clip
(``data/synthetic/synthetic-indoor-v3/crowded_6agents__cam_a.npz``, golden
set ``v3-indoor``) — never a literal placeholder. For a chosen frame and a
"detected" agent, each candidate is "this detection continues track for
agent K": its log-likelihood is the Gaussian log-density of the ACTUAL
frame position under a linear-velocity prediction of agent K's own track
from the two prior frames.

The measurement-noise sigma is not invented for this script: it is derived
from ``src.estimator.motion_model.PERSON_SIGMA_A_MPS2`` (1.5 m/s^2) and
``PEDESTRIAN_STOP_DURATION_S`` (1.0 s), whose product (1.5 m/s) is this
project's own declared, cited velocity-uncertainty floor for a walking
person (see that module's docstring). Propagated forward by one assumed
frame interval (dt = 1/12 s — this repo's synthetic clips carry no explicit
frame-rate field, so 12 fps is a stated assumption, not a measured one) that
gives a one-frame-ahead POSITION sigma of ``sigma_v_floor * dt`` ~= 0.125 m.
This is a deliberate re-derivation (velocity uncertainty integrated over a
time step to get position uncertainty), not the "reuse a sigma for the
wrong quantity" mistake the noise-floor discipline warns against.

Hard-constraint checking is NOT implemented here: ``resolve_data_association``
requires a ``hard_violation_check`` callable, and this script passes one
that always returns ``None`` (no physical-consistency evaluation exists in
this demo path). Every death recorded below is therefore either
``DominatedByLikelihood`` or ``PrunedByBudget`` — never
``RefutedByHardConstraint``.

Two real components are resolved, chosen (by searching the clip's own real
distances, not by editing them) to exercise both verdict types and the
PRUNED_BY_BUDGET-vs-DOMINATED_BY_LIKELIHOOD split the Inspector view must
render distinctly:

* frame 13, detected agent 0, budget=1 — real geometry makes agents 0 and 3
  nearly co-located (1.8 cm apart at frame 15; already close at frame 13),
  so the verdict is genuinely AMBIGUOUS (margin 0.52 nats, below the
  ln(3)-nat decisiveness threshold). Choosing budget=1 for this component
  additionally cuts agent 3 — the verdict's own near-tied competitor — with
  a real boundary margin (0.52 nats) below threshold too, so its death cause
  is PRUNED_BY_BUDGET: a resource decision, not an evidentiary one, on the
  exact hypothesis the verdict itself could not rule out. The remaining
  four agents are DOMINATED_BY_LIKELIHOOD, clearly excluded by evidence.
* frame 39, detected agent 1, budget=3 — agent 1's own continuation
  dominates every other agent by tens of nats (the nearest competitor is
  3.9 m away), a genuinely DECISIVE case landing far into the "very
  strong" Kass & Raftery band.

Outputs
-------
* ``outputs/associations/<component_id>.json`` — one per component: full
  candidate list with real scores, the verdict, every death cause, and the
  resulting event from ``build_identity_event``.
* ``outputs/events/events.parquet`` — the two identity events above, PLUS
  one real, geometrically-grounded ``PredictedEvent`` and one
  ``HypothesisEvent`` (see their construction below), giving Objective 3's
  fixed Events view real, non-Absent, all-four-classes data to render. This
  is a demonstration of correct event CONSTRUCTION and STORAGE — it is not
  a verb-detection pipeline (adding a production verb detector is its own
  day per the closed-vocabulary policy in the events skill) and must not be
  read as one.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.config import IronConfig
from src.estimator.joint import (
    DECISIVE_LOG_BAYES_FACTOR,
    AssociationBudgetConfig,
    AssociationCandidate,
    Ambiguous,
    Decisive,
    build_identity_event,
    resolve_data_association,
)
from src.events.schema import EntityRef, Verb
from src.model.events import (
    EventV2,
    HypothesisEvent,
    PredictedEvent,
    event_v2_to_dict,
    write_events_v2_parquet,
)
from src.model.hypothesis import DominatedByLikelihood, HypothesisStore, PrunedByBudget

CLIP_ID = "crowded_6agents__cam_a"
CLIP_PATH = Path("data/synthetic/synthetic-indoor-v3/crowded_6agents__cam_a.npz")
SITE_ID = "demo-crowded-6agents"
ASSUMED_FPS = 12.0
DT_S = 1.0 / ASSUMED_FPS

# Derived, not invented -- see module docstring. A one-frame-ahead position
# sigma from this project's own cited pedestrian velocity-uncertainty floor.
SIGMA_V_FLOOR_MPS = 1.5  # PERSON_SIGMA_A_MPS2 * PEDESTRIAN_STOP_DURATION_S
SIGMA_P_M = SIGMA_V_FLOOR_MPS * DT_S


def _band(margin_nats: float) -> str | None:
    """Kass & Raftery (1995, JASA 90(430), p.777, Table 4) / Jeffreys (1961)
    band for a Bayes factor of exp(margin_nats), in the SAME thresholds
    src.estimator.joint's DECISIVE_LOG_BAYES_FACTOR docstring cites: below
    Bayes-factor 3 is "not worth more than a bare mention", 3-20 is
    "positive", 20-150 is "strong", above 150 is "very strong". A caller
    below the decisiveness threshold (an Ambiguous verdict) has no band --
    it sits in the "bare mention" region by construction, so this returns
    None rather than a band name a viewer could mistake for support.
    """
    if margin_nats < DECISIVE_LOG_BAYES_FACTOR:
        return None
    if margin_nats < math.log(20):
        return "positive"
    if margin_nats < math.log(150):
        return "strong"
    return "very strong"


def _predicted_position(xyz: np.ndarray, agent: int, frame: int) -> np.ndarray:
    velocity = (xyz[frame - 1, agent] - xyz[frame - 2, agent]) / DT_S
    return xyz[frame - 1, agent] + velocity * DT_S


def _log_likelihood(actual: np.ndarray, predicted: np.ndarray) -> float:
    squared_distance = float(np.sum((actual - predicted) ** 2))
    return -0.5 * (
        3 * math.log(2 * math.pi * SIGMA_P_M**2) + squared_distance / SIGMA_P_M**2
    )


def _resolve_component(
    xyz: np.ndarray,
    *,
    component_id: str,
    frame: int,
    detected_agent: int,
    budget: int,
) -> dict[str, Any]:
    n_agents = xyz.shape[1]
    actual = xyz[frame, detected_agent]

    candidates = []
    hyp_to_agent: dict[str, int] = {}
    for candidate_agent in range(n_agents):
        hyp_id = f"agent-{candidate_agent}"
        hyp_to_agent[hyp_id] = candidate_agent
        log_likelihood = _log_likelihood(
            actual, _predicted_position(xyz, candidate_agent, frame)
        )
        candidates.append(
            AssociationCandidate(
                hypothesis_id=hyp_id,
                proposition=(
                    f"detection at {CLIP_ID} frame {frame} continues the "
                    f"track for agent-{candidate_agent}"
                ),
                log_likelihood=log_likelihood,
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
        return EntityRef(kind="session", id=f"{CLIP_ID}::agent-{agent}")

    event = build_identity_event(
        resolution.verdict,
        subject_for,
        event_id=uuid.uuid5(
            uuid.NAMESPACE_URL, f"iron://day35-demo/{component_id}"
        ),
        site_id=SITE_ID,
        ts_ns=int(frame * DT_S * 1e9),
        verb=Verb.APPROACHED,
        manifest_sha=IronConfig.load().config_sha(),
        importance=0.5,
        caller_confidence=0.9,
    )

    verdict = resolution.verdict
    verdict_payload: dict[str, Any]
    if isinstance(verdict, Decisive):
        verdict_payload = {
            "kind": "decisive",
            "winner": verdict.winner.id,
            "margin_nats": verdict.margin,
            "kass_raftery_band": (
                "very strong" if math.isinf(verdict.margin) else _band(verdict.margin)
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

    def cause_payload(cause: Any) -> dict[str, Any] | None:
        """Every DeathCause kind, rendered with a human-readable ``detail``
        so the UI never needs its own copy of this dispatch -- a death
        cause the store can construct that this script never produces
        (e.g. RefutedByHardConstraint, since no hard-constraint check is
        implemented here) still renders correctly if it ever appears."""
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

    return {
        "component_id": component_id,
        "source_clip": CLIP_ID,
        "source": str(CLIP_PATH),
        "frame_index": frame,
        "detected_agent": detected_agent,
        "assumed_fps": ASSUMED_FPS,
        "measurement_sigma_m": SIGMA_P_M,
        "measurement_sigma_derivation": (
            "PERSON_SIGMA_A_MPS2 (1.5 m/s^2) * PEDESTRIAN_STOP_DURATION_S "
            "(1.0 s) = 1.5 m/s velocity-uncertainty floor "
            "(src.estimator.motion_model), propagated over one assumed "
            f"frame interval (dt={DT_S:.6f} s at {ASSUMED_FPS} fps) to a "
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
            "no src.model.episode.StateGraph is constructed for this demo "
            "-- the verdict is computed directly from candidate "
            "log-likelihoods, not replayed against an append-only factor "
            "graph, so there is no graph revision to report. Stated as an "
            "absence rather than a fabricated 0 or 1."
        ),
        "config_sha": IronConfig.load().config_sha(),
        "model_shas": None,
        "model_shas_note": (
            "no learned model produced these numbers -- the log-likelihoods "
            "are a closed-form Gaussian density over measured geometry, so "
            "there is no model artifact to hash. Stated as an absence "
            "rather than omitted."
        ),
        "event": event_v2_to_dict(event),
    }


def _predicted_event(xyz: np.ndarray, *, agent_a: int, agent_b: int, frame: int) -> PredictedEvent:
    """A real forward-looking claim: will agent_a's and agent_b's tracks be
    closer K frames from now than they are at ``frame``? Genuinely computed
    by linear extrapolation of each agent's OWN measured velocity -- not a
    verb-detection claim (see module docstring)."""
    horizon = 5
    v_a = (xyz[frame, agent_a] - xyz[frame - 1, agent_a]) / DT_S
    v_b = (xyz[frame, agent_b] - xyz[frame - 1, agent_b]) / DT_S
    now = float(np.linalg.norm(xyz[frame, agent_a] - xyz[frame, agent_b]))
    future_a = xyz[frame, agent_a] + v_a * DT_S * horizon
    future_b = xyz[frame, agent_b] + v_b * DT_S * horizon
    future = float(np.linalg.norm(future_a - future_b))
    return PredictedEvent(
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, "iron://day35-demo/predicted"),
        site_id=SITE_ID,
        ts_ns=int(frame * DT_S * 1e9),
        subject=EntityRef(kind="session", id=f"{CLIP_ID}::agent-{agent_a}"),
        verb=Verb.APPROACHED,
        confidence=0.6,
        importance=0.3,
        manifest_sha=IronConfig.load().config_sha(),
        predicted_by=(
            f"linear extrapolation of agent-{agent_a}'s and agent-{agent_b}'s "
            f"measured velocity at frame {frame} over the next {horizon} "
            f"frames ({horizon * DT_S:.3f} s at {ASSUMED_FPS} fps): distance "
            f"{now:.3f} m now, {future:.3f} m projected"
        ),
    )


def _hypothesis_event(
    xyz: np.ndarray, *, agent_a: int, agent_b: int, frame: int
) -> HypothesisEvent:
    """An unconfirmed possibility this clip's real geometry actually raises:
    at their closest real approach, agent_a and agent_b are close enough
    that a re-identification or track-merge error is a live, uninvestigated
    possibility -- distinct from the PredictedEvent above (a forward
    projection of where they will be) and from the Ambiguous association
    verdict already computed for this same pair (a resolved, typed decision
    over candidate tracks). A HypothesisEvent is neither: it is a lead
    raised FOR investigation, not a claim (STRUCTURAL: cannot trigger an
    alert -- src.model.events.HypothesisEvent's own docstring).

    (Every window checked in this clip has a per-agent net-displacement-to-
    path-length ratio of ~1.0 -- every agent here moves in a straight line
    at constant velocity, so a loitering/pause hypothesis has no honest
    basis in this clip's actual data and was deliberately not used.)
    """
    distance_m = float(np.linalg.norm(xyz[frame, agent_a] - xyz[frame, agent_b]))
    return HypothesisEvent(
        event_id=uuid.uuid5(uuid.NAMESPACE_URL, "iron://day35-demo/hypothesis"),
        site_id=SITE_ID,
        ts_ns=int(frame * DT_S * 1e9),
        subject=EntityRef(kind="session", id=f"{CLIP_ID}::agent-{agent_a}"),
        verb=Verb.APPROACHED,
        confidence=0.3,
        importance=0.4,
        manifest_sha=IronConfig.load().config_sha(),
        rationale=(
            f"agent-{agent_a} and agent-{agent_b} measured {distance_m:.4f} m "
            f"apart at frame {frame}, close enough that a track merge or "
            "identity swap between the two is a live, unconfirmed "
            "possibility -- consistent with the genuinely Ambiguous "
            "association verdict already computed for this pair (see "
            "outputs/associations/ambiguous-demo.json), not yet resolved "
            "by any re-identification cue"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    with np.load(CLIP_PATH) as data:
        xyz = np.asarray(data["agent_xyz"])

    components = [
        _resolve_component(
            xyz, component_id="ambiguous-demo", frame=13, detected_agent=0, budget=1
        ),
        _resolve_component(
            xyz, component_id="decisive-demo", frame=39, detected_agent=1, budget=3
        ),
    ]

    out_dir = Path("outputs/associations")
    out_dir.mkdir(parents=True, exist_ok=True)
    for component in components:
        path = out_dir / f"{component['component_id']}.json"
        path.write_text(json.dumps(component, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")

    identity_events: list[EventV2] = []
    for component in components:
        from src.model.events import event_v2_from_dict

        identity_events.append(event_v2_from_dict(component["event"]))

    predicted = _predicted_event(xyz, agent_a=0, agent_b=3, frame=15)
    hypothesis = _hypothesis_event(xyz, agent_a=0, agent_b=3, frame=15)

    events_path = Path("outputs/events/events.parquet")
    write_events_v2_parquet([*identity_events, predicted, hypothesis], events_path)
    print(f"wrote {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
