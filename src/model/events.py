"""Event schema v2 — four classes, not a flag on one record.

Schema v1 (:mod:`src.events.schema`) marks an inferred fact with
``observed: bool``. That collapses four genuinely different epistemic
states into one bit: a fact seen directly, a fact derived from other
evidence, a forward-looking claim that has not happened yet, and an
unconfirmed possibility raised for investigation. Conflating "inferred"
and "predicted" is how a system ends up alerting on a forecast, and
conflating "hypothesis" with either is how it ends up treating a guess as
evidence. Four classes make each state a distinct, statically-checkable
type instead of a branch on a boolean that every consumer has to
remember to check correctly.

The verb vocabulary is unchanged and not duplicated
----------------------------------------------------
:class:`~src.events.schema.Verb` and its GEOMETRIC/POSE/INTERACTION
groups are reused directly from schema v1. The vocabulary's closedness and
versioning policy live there; v2 only changes how a fact's epistemic
status is represented, not what verbs exist.

Evidence admission is a closed-world dispatch, not a conditional
---------------------------------------------------------------------
:func:`assemble_evidence` is built on ``functools.singledispatch`` with
handlers registered only for the event classes that are actually
eligible. A :class:`PredictedEvent` passed to :func:`assemble_evidence`
hits no registered handler and the dispatch itself raises — there is no
``if event_class == "predicted": reject`` anywhere in this module for a
future edit to accidentally invert or forget.

Alert triggering lives in :mod:`src.model.alert`, not here (Day 15)
---------------------------------------------------------------------
Day 13 shipped a second, permissive type-eligibility check in this
module, ``raise_alert``, alongside the strict ``src.model.alert.
emit_alert`` added Day 14 — two alert paths with nothing steering
callers to the strict one. Auditing every caller (Day 15) found none in
production code, and found that ``raise_alert``'s one distinguishing
feature — letting a :class:`PredictedEvent` trigger an alert — could
never be given an evidence chain, because :func:`assemble_evidence`
excludes ``PredictedEvent`` from evidence eligibility too. That is not a
caller to migrate; it is a permissive path that could only ever alert on
nothing. ``raise_alert`` and ``AlertEligibleEvent`` are deleted.
``src.model.alert.emit_alert`` is now the only public way to emit an
alert, and it is a closed-world dispatch requiring a resolved evidence
chain (see that module's docstring). A future forecast-paging feature
must be built as its own explicit capability, not by relaxing this one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal, Sequence

from src.events.schema import (
    INTERACTION_VERBS,
    POSE_VERBS,
    ClipRef,
    EntityRef,
    Verb,
)

EVENT_SCHEMA_VERSION: Literal["2.0"] = "2.0"

EventClassName = Literal["observed", "inferred", "predicted", "hypothesis"]


class EventError(ValueError):
    """Raised when a v2 event record or operation violates its contract."""


def _validate_common(event: "_EventCommon") -> None:
    kind = type(event).__name__
    if not event.site_id:
        raise EventError(f"{kind}.site_id must not be empty")
    if not event.manifest_sha:
        raise EventError(
            f"{kind}.manifest_sha is required: an event that cannot be "
            "traced to the run that produced it cannot be defended in a "
            "review"
        )
    if not isinstance(event.verb, Verb):
        raise EventError(
            f"{kind}.verb must be a Verb from the closed v1 vocabulary, "
            f"got {event.verb!r}"
        )
    for name in ("confidence", "importance"):
        value = getattr(event, name)
        if not 0.0 <= value <= 1.0:
            raise EventError(f"{kind}.{name} must lie in [0, 1], got {value}")
    if event.verb in INTERACTION_VERBS and event.object is None:
        raise EventError(
            f"{event.verb.value!r} is an INTERACTION verb and requires an "
            "object. 'picked_up' is not a fact until it says what was "
            "picked up."
        )
    if event.verb in POSE_VERBS and event.object is not None:
        raise EventError(
            f"{event.verb.value!r} is a POSE verb and must not carry an "
            f"object, got {event.object}"
        )
    if event.zone is not None and event.zone.kind != "zone":
        raise EventError(
            f"{kind}.zone must reference a zone entity, got kind "
            f"{event.zone.kind!r}"
        )


@dataclass(frozen=True)
class _EventCommon:
    """Fields shared by every v2 event class.

    Attributes:
        event_id: Stable unique identity for this record.
        site_id: Which installation; part of the timeline key with ``ts_ns``.
        ts_ns: When the claim's instant is, nanoseconds since the epoch.
        subject: Who or what the claim is about.
        verb: What happened, from the closed vocabulary in
            :mod:`src.events.schema`.
        confidence: Detector/model posterior in [0, 1].
        importance: Triage weight in [0, 1].
        manifest_sha: Which run produced this record.
        object: What was acted upon (required for INTERACTION verbs,
            forbidden for POSE verbs).
        zone: Where, when bound to a named region.
        clip: The footage backing this record, if any.
        evidence_refs: Opaque references into the evidence layer (see
            :mod:`src.model.evidence`) that back this claim.
        state_refs: Opaque references into the state graph (see
            :mod:`src.model.episode`) this claim was derived against.
        supersedes: The event this record replaces or confirms, if any —
            e.g. the :class:`PredictedEvent` a :class:`ObservedEvent`
            confirms via :func:`confirm_prediction`.
        schema_version: The contract this record was written against.
    """

    event_id: uuid.UUID
    site_id: str
    ts_ns: int
    subject: EntityRef
    verb: Verb
    confidence: float
    importance: float
    manifest_sha: str
    object: EntityRef | None = None
    zone: EntityRef | None = None
    clip: ClipRef | None = None
    evidence_refs: tuple[str, ...] = ()
    state_refs: tuple[str, ...] = ()
    supersedes: uuid.UUID | None = None
    schema_version: Literal["2.0"] = EVENT_SCHEMA_VERSION


@dataclass(frozen=True)
class ObservedEvent(_EventCommon):
    """A fact directly seen: subject, verb, and object were all observed."""

    event_class: Literal["observed"] = "observed"

    def __post_init__(self) -> None:
        _validate_common(self)


@dataclass(frozen=True)
class InferredEvent(_EventCommon):
    """A fact not directly seen but derived, with a stated basis.

    Examples: a trajectory extrapolated across a dropped-frame gap, an
    identity resolved retroactively by a later badge swipe.
    """

    event_class: Literal["inferred"] = "inferred"
    basis: str = ""

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.basis:
            raise EventError("InferredEvent.basis must not be empty")


@dataclass(frozen=True)
class PredictedEvent(_EventCommon):
    """A forward-looking claim: this has not happened, or may not.

    STRUCTURAL: cannot be admitted as evidence (:func:`assemble_evidence`
    raises for it). If it comes true, :func:`confirm_prediction` produces
    a brand new :class:`ObservedEvent` with ``supersedes`` pointing back
    here — this record is frozen and is never rewritten into one.
    """

    event_class: Literal["predicted"] = "predicted"
    predicted_by: str = ""

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.predicted_by:
            raise EventError("PredictedEvent.predicted_by must not be empty")


@dataclass(frozen=True)
class HypothesisEvent(_EventCommon):
    """An unconfirmed possibility raised for investigation.

    STRUCTURAL: cannot trigger an alert (:func:`src.model.alert.emit_alert`
    raises for it). A hypothesis is a lead, not a claim strong enough to
    page anyone.
    """

    event_class: Literal["hypothesis"] = "hypothesis"
    rationale: str = ""

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.rationale:
            raise EventError("HypothesisEvent.rationale must not be empty")


EventV2 = ObservedEvent | InferredEvent | PredictedEvent | HypothesisEvent
EvidenceEligibleEvent = ObservedEvent | InferredEvent | HypothesisEvent


# ---------------------------------------------------------------------------
# Evidence admission — closed-world dispatch, no PredictedEvent handler.
# ---------------------------------------------------------------------------


@singledispatch
def _admit_as_evidence(event: object) -> EvidenceEligibleEvent:
    raise EventError(
        f"{type(event).__name__} cannot be admitted as evidence: no handler "
        "is registered for it in src.model.events._admit_as_evidence. "
        "Evidence admission is a closed-world dispatch by design — an "
        "unconfirmed forward-looking event class is rejected by the "
        "absence of a registration, not by a conditional that a later "
        "edit could invert."
    )


@_admit_as_evidence.register
def _(event: ObservedEvent) -> EvidenceEligibleEvent:
    return event


@_admit_as_evidence.register
def _(event: InferredEvent) -> EvidenceEligibleEvent:
    return event


@_admit_as_evidence.register
def _(event: HypothesisEvent) -> EvidenceEligibleEvent:
    return event


def assemble_evidence(events: Sequence[EventV2]) -> tuple[EvidenceEligibleEvent, ...]:
    """Admit ``events`` as evidence.

    Raises:
        EventError: on the first :class:`PredictedEvent` (or any type with
            no registered handler) encountered.
    """
    return tuple(_admit_as_evidence(e) for e in events)


# ---------------------------------------------------------------------------
# Prediction confirmation — a new record, never a mutation.
# ---------------------------------------------------------------------------


def confirm_prediction(
    predicted: PredictedEvent,
    *,
    event_id: uuid.UUID,
    ts_ns: int,
    manifest_sha: str,
    evidence_refs: tuple[str, ...] = (),
    state_refs: tuple[str, ...] = (),
    confidence: float | None = None,
) -> ObservedEvent:
    """Return a NEW :class:`ObservedEvent` confirming ``predicted`` came true.

    ``predicted`` is a frozen dataclass and is never rewritten in place —
    the only representation of "this prediction came true" is a new
    record whose ``supersedes`` points back to it, so both the original
    prediction and its confirmation remain independently retrievable with
    their own provenance.
    """
    return ObservedEvent(
        event_id=event_id,
        site_id=predicted.site_id,
        ts_ns=ts_ns,
        subject=predicted.subject,
        verb=predicted.verb,
        confidence=predicted.confidence if confidence is None else confidence,
        importance=predicted.importance,
        manifest_sha=manifest_sha,
        object=predicted.object,
        zone=predicted.zone,
        clip=predicted.clip,
        evidence_refs=evidence_refs,
        state_refs=state_refs,
        supersedes=predicted.event_id,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_V2_CLASS_BY_NAME: dict[str, type] = {
    "observed": ObservedEvent,
    "inferred": InferredEvent,
    "predicted": PredictedEvent,
    "hypothesis": HypothesisEvent,
}


def _entity_to_dict(ref: EntityRef | None) -> dict[str, str] | None:
    return None if ref is None else {"kind": ref.kind, "id": ref.id}


def _entity_from_dict(raw: dict[str, str] | None) -> EntityRef | None:
    if raw is None:
        return None
    return EntityRef(kind=raw["kind"], id=raw["id"])  # type: ignore[arg-type]


def _clip_to_dict(clip: ClipRef | None) -> dict[str, Any] | None:
    if clip is None:
        return None
    return {
        "video_id": clip.video_id,
        "start_ts_ns": clip.start_ts_ns,
        "end_ts_ns": clip.end_ts_ns,
        "content_sha": clip.content_sha,
    }


def _clip_from_dict(raw: dict[str, Any] | None) -> ClipRef | None:
    if raw is None:
        return None
    return ClipRef(
        video_id=raw["video_id"],
        start_ts_ns=raw["start_ts_ns"],
        end_ts_ns=raw["end_ts_ns"],
        content_sha=raw.get("content_sha"),
    )


def event_v2_to_dict(event: EventV2) -> dict[str, Any]:
    """Serialize any v2 event to a plain JSON-compatible mapping."""
    payload: dict[str, Any] = {
        "event_class": event.event_class,
        "event_id": str(event.event_id),
        "site_id": event.site_id,
        "ts_ns": event.ts_ns,
        "subject": _entity_to_dict(event.subject),
        "verb": event.verb.value,
        "object": _entity_to_dict(event.object),
        "zone": _entity_to_dict(event.zone),
        "confidence": event.confidence,
        "importance": event.importance,
        "clip": _clip_to_dict(event.clip),
        "manifest_sha": event.manifest_sha,
        "evidence_refs": list(event.evidence_refs),
        "state_refs": list(event.state_refs),
        "supersedes": str(event.supersedes) if event.supersedes is not None else None,
        "schema_version": event.schema_version,
    }
    if isinstance(event, InferredEvent):
        payload["basis"] = event.basis
    elif isinstance(event, PredictedEvent):
        payload["predicted_by"] = event.predicted_by
    elif isinstance(event, HypothesisEvent):
        payload["rationale"] = event.rationale
    return payload


def event_v2_from_dict(payload: dict[str, Any]) -> EventV2:
    """Rebuild a v2 event from :func:`event_v2_to_dict` output.

    Raises:
        EventError: if the payload names an unknown event_class or was
            written against a different schema version.
    """
    version = payload.get("schema_version")
    if version != EVENT_SCHEMA_VERSION:
        raise EventError(
            f"cannot read a record written against schema_version "
            f"{version!r}; this build supports {EVENT_SCHEMA_VERSION!r}"
        )
    event_class = payload.get("event_class")
    if event_class not in _V2_CLASS_BY_NAME:
        raise EventError(
            f"unknown event_class {event_class!r}; expected one of "
            f"{sorted(_V2_CLASS_BY_NAME)}"
        )

    subject = _entity_from_dict(payload["subject"])
    if subject is None:
        raise EventError("subject is required")

    try:
        common: dict[str, Any] = dict(
            event_id=uuid.UUID(str(payload["event_id"])),
            site_id=payload["site_id"],
            ts_ns=int(payload["ts_ns"]),
            subject=subject,
            verb=Verb(payload["verb"]),
            confidence=float(payload["confidence"]),
            importance=float(payload["importance"]),
            manifest_sha=payload["manifest_sha"],
            object=_entity_from_dict(payload.get("object")),
            zone=_entity_from_dict(payload.get("zone")),
            clip=_clip_from_dict(payload.get("clip")),
            evidence_refs=tuple(payload.get("evidence_refs", ())),
            state_refs=tuple(payload.get("state_refs", ())),
            supersedes=(
                uuid.UUID(payload["supersedes"]) if payload.get("supersedes") else None
            ),
        )
    except KeyError as exc:
        raise EventError(f"event record is missing required field {exc}") from exc
    except ValueError as exc:
        raise EventError(f"unknown verb in event record: {exc}") from exc

    if event_class == "observed":
        return ObservedEvent(**common)
    if event_class == "inferred":
        return InferredEvent(**common, basis=payload["basis"])
    if event_class == "predicted":
        return PredictedEvent(**common, predicted_by=payload["predicted_by"])
    return HypothesisEvent(**common, rationale=payload["rationale"])


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventClassName",
    "EventError",
    "ObservedEvent",
    "InferredEvent",
    "PredictedEvent",
    "HypothesisEvent",
    "EventV2",
    "EvidenceEligibleEvent",
    "assemble_evidence",
    "confirm_prediction",
    "event_v2_to_dict",
    "event_v2_from_dict",
]
