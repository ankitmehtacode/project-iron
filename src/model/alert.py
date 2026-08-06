"""Alert emission and end-to-end explainability.

The hole this closes (Day-13 falsification test 4)
------------------------------------------------------
Day 13 built type-eligibility for alerting (``events.raise_alert``, a
``singledispatch`` registry excluding ``HypothesisEvent``) but never
required that an alert-eligible event actually carry a *resolvable*
evidence chain. `test_falsification_alert_explainability_is_not_
structurally_required` (Day 13) showed an ``ObservedEvent`` with empty
``evidence_refs`` still passing ``raise_alert()`` — an alert that can
fire with nothing behind it is the seed of "the system said so and
nobody can explain why", which ends credibility with a Tier-3 customer
the first time an investigator asks.

Two alert paths now, on purpose
----------------------------------
``events.raise_alert`` (Day 13, unchanged) is deliberately still broader
than this module: it stays the type-eligibility check used when a
forecast alone is allowed to page someone (a ``PredictedEvent`` can
still trigger a fast-lane notification). This module's :func:`emit_alert`
is the stricter, fully-explainable path, and its eligible set is exactly
the *intersection* of Day 13's ``AlertEligibleEvent``
(``Observed | Inferred | Predicted``) and ``EvidenceEligibleEvent``
(``Observed | Inferred | Hypothesis``) — which is ``Observed | Inferred``.
A ``PredictedEvent`` cannot go through :func:`emit_alert` because it can
never itself be admitted as evidence (ADR 0003); explaining a
forecast-triggered notification means rendering the ``InferredEvent``/
``ObservedEvent`` chain that fed the predictor, never the prediction
record itself. A ``HypothesisEvent`` cannot go through either path — a
lead is not a claim strong enough to page anyone, in either framing.

Both eligibility checks are closed-world ``singledispatch`` registries,
not an ``if``, matching Day 13's pattern exactly: an unregistered event
type fails by the absence of a handler, which a later edit cannot
accidentally invert the way it can invert a conditional.

explain() resolves the whole chain or names the exact broken hop
----------------------------------------------------------------------
:func:`explain` walks ``alert -> event -> evidence -> observation_refs
/ clip_refs / state_refs``, with ``producer_sha`` checked at every
derivation step, and raises :class:`ExplainabilityError` naming the
precise hop that failed to resolve rather than returning a partial or
best-effort explanation. Most of the individual hops are already
guaranteed non-empty by :class:`~src.model.evidence.Evidence`'s own
Day-13 invariants (``observation_refs``, ``derivation_chain``, each
``DerivationStep.producer_sha``); :func:`explain` checks them again
explicitly here so a future relaxation of those invariants elsewhere
cannot silently reopen this hole without a test in *this* module
catching it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Mapping, Sequence

from src.model.evidence import Evidence
from src.model.events import EventV2, InferredEvent, ObservedEvent

AlertEmissionEligibleEvent = ObservedEvent | InferredEvent
"""Exactly the intersection of Day 13's AlertEligibleEvent and
EvidenceEligibleEvent — see the module docstring."""


class AlertError(ValueError):
    """Raised when an Alert cannot be constructed or emitted."""


class ExplainabilityError(ValueError):
    """Raised by :func:`explain`, naming the exact hop that failed to resolve."""


@dataclass(frozen=True)
class Alert:
    """An emitted alert. STRUCTURAL: always carries a resolvable evidence chain.

    Attributes:
        alert_id: Unique id.
        event: The triggering event. Typed as ``AlertEmissionEligibleEvent``
            — a ``PredictedEvent`` or ``HypothesisEvent`` here is a type
            error, not a runtime check.
        evidence_chain: The resolved :class:`~src.model.evidence.Evidence`
            records backing ``event``. Required and non-empty: there is no
            way to construct an ``Alert`` with an empty or missing chain.
            :func:`emit_alert` is the only constructor used in practice and
            enforces this before returning; a caller who bypasses it and
            builds an ``Alert`` directly still hits this check.
        manifest_sha: The run that produced this alert.
    """

    alert_id: str
    event: AlertEmissionEligibleEvent
    evidence_chain: tuple[Evidence, ...]
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.alert_id:
            raise AlertError("Alert.alert_id must not be empty")
        if not self.evidence_chain:
            raise AlertError(
                "Alert.evidence_chain must not be empty: an alert with no "
                "resolvable evidence is exactly the 'the system said so and "
                "nobody can explain why' failure this type exists to "
                "prevent"
            )
        if not self.manifest_sha:
            raise AlertError("Alert.manifest_sha must not be empty")


# ---------------------------------------------------------------------------
# Closed-world dispatch: only ObservedEvent/InferredEvent may emit an Alert.
# ---------------------------------------------------------------------------


@singledispatch
def _admit_for_alert_emission(event: object) -> AlertEmissionEligibleEvent:
    raise AlertError(
        f"{type(event).__name__} cannot emit a fully-explainable alert: no "
        "handler is registered for it in "
        "src.model.alert._admit_for_alert_emission. A forecast is not "
        "self-evidencing and a lead is not a claim strong enough to page "
        "anyone — see the module docstring for what each excluded class "
        "should do instead."
    )


@_admit_for_alert_emission.register
def _(event: ObservedEvent) -> AlertEmissionEligibleEvent:
    return event


@_admit_for_alert_emission.register
def _(event: InferredEvent) -> AlertEmissionEligibleEvent:
    return event


def emit_alert(
    alert_id: str,
    event: EventV2,
    evidence_chain: Sequence[Evidence],
    manifest_sha: str,
) -> Alert:
    """Emit an :class:`Alert` for ``event``, requiring a resolved evidence chain.

    Raises:
        AlertError: if ``event``'s type has no registered handler (a
            ``PredictedEvent``, a ``HypothesisEvent``, or any future event
            class not wired in here), or if ``evidence_chain`` is empty.
    """
    eligible = _admit_for_alert_emission(event)
    return Alert(
        alert_id=alert_id,
        event=eligible,
        evidence_chain=tuple(evidence_chain),
        manifest_sha=manifest_sha,
    )


# ---------------------------------------------------------------------------
# explain() — full-chain resolution, or the exact broken hop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainedHop:
    """One resolved evidence record in an alert's chain, for rendering."""

    evidence_id: str
    observation_refs: tuple[str, ...]
    clip_refs: tuple[str, ...]
    state_refs: tuple[str, ...]
    producer_shas: tuple[str, ...]
    derivation_stages: tuple[str, ...]


@dataclass(frozen=True)
class ExplainedAlert:
    """The full, resolved explanation for one alert."""

    alert_id: str
    event_id: str
    event_class: str
    hops: tuple[ExplainedHop, ...]


def explain(
    alert_id: str,
    alert_store: Mapping[str, Alert],
    observation_store: Mapping[str, object] | None = None,
) -> ExplainedAlert:
    """Resolve ``alert_id``'s full evidence chain end to end.

    Walks alert -> event -> evidence -> observation_refs / clip_refs /
    state_refs, checking producer_sha at every derivation step. When
    ``observation_store`` is supplied, each observation_ref must actually
    resolve in it; omitted, that hop is checked for presence only (matches
    :class:`~src.model.evidence.Evidence`'s own requirement that
    ``observation_refs`` be non-empty).

    Raises:
        ExplainabilityError: naming the exact hop that failed to resolve —
            unknown alert_id, an event evidence_ref with no matching
            Evidence in the chain, an evidence record with no observations
            or derivation chain, a derivation step with no producer_sha,
            or (when ``observation_store`` is given) an observation_ref
            absent from it.
    """
    alert = alert_store.get(alert_id)
    if alert is None:
        raise ExplainabilityError(
            f"alert_id {alert_id!r} not found in alert_store -- broken hop: "
            "alert_id -> alert"
        )

    if not alert.event.evidence_refs:
        raise ExplainabilityError(
            f"alert {alert_id!r}: event {alert.event.event_id} has no "
            "evidence_refs -- broken hop: event -> evidence (nothing to "
            "explain)"
        )

    by_id = {ev.evidence_id: ev for ev in alert.evidence_chain}
    hops: list[ExplainedHop] = []

    for ref in alert.event.evidence_refs:
        evidence = by_id.get(ref)
        if evidence is None:
            raise ExplainabilityError(
                f"alert {alert_id!r}: event references evidence_ref {ref!r} "
                "but no Evidence with that id was resolved into the "
                "alert's evidence_chain -- broken hop: event -> evidence"
            )
        if not evidence.observation_refs:
            raise ExplainabilityError(
                f"alert {alert_id!r}: evidence {ref!r} has no "
                "observation_refs -- broken hop: evidence -> observation"
            )
        if observation_store is not None:
            for obs_ref in evidence.observation_refs:
                if obs_ref not in observation_store:
                    raise ExplainabilityError(
                        f"alert {alert_id!r}: evidence {ref!r} references "
                        f"observation {obs_ref!r}, not found in "
                        "observation_store -- broken hop: observation -> "
                        "store"
                    )
        if not evidence.derivation_chain:
            raise ExplainabilityError(
                f"alert {alert_id!r}: evidence {ref!r} has no "
                "derivation_chain -- broken hop: evidence -> derivation"
            )
        for step in evidence.derivation_chain:
            if not step.producer_sha:
                raise ExplainabilityError(
                    f"alert {alert_id!r}: evidence {ref!r} derivation stage "
                    f"{step.stage!r} has no producer_sha -- broken hop: "
                    "derivation -> producer_sha"
                )
        hops.append(
            ExplainedHop(
                evidence_id=evidence.evidence_id,
                observation_refs=evidence.observation_refs,
                clip_refs=evidence.clip_refs,
                state_refs=evidence.state_refs,
                producer_shas=evidence.producer_shas,
                derivation_stages=tuple(s.stage for s in evidence.derivation_chain),
            )
        )

    return ExplainedAlert(
        alert_id=alert.alert_id,
        event_id=str(alert.event.event_id),
        event_class=alert.event.event_class,
        hops=tuple(hops),
    )
