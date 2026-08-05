"""IRON_DATA_MODEL v0.3 as types, validation, and serialization.

Built Day 13 from three rounds of architecture review. This package is
primitives and rules — construction, validation, serialization, migration,
structural guards — not inference. The factor-graph solver, the estimator,
behaviour models, and the LLM layer are out of scope; where this package
needs a placeholder for one, it is a function that raises
``NotImplementedError`` with a docstring naming what will fill it, never a
stub that silently returns a plausible-looking answer.

Every rule marked STRUCTURAL in the Day-13 prompt is enforced by the type
system or a constructor-time raise, proven by a test that attempts the
violation. See ``tests/test_model_*.py``.
"""

from __future__ import annotations

from src.model.entity import (
    ENTITY_KINDS,
    IDENTITY_CLASSES,
    AnonymousSessionEntity,
    Entity,
    EntityError,
    EntityKind,
    EnrolledEntity,
    FixtureEntity,
    IdentityClass,
    RegisteredAssetEntity,
    TimeVaryingAttribute,
)
from src.model.envelope import (
    ENVELOPE_STATUSES,
    Envelope,
    EnvelopeCurve,
    EnvelopeError,
    EnvelopeStatus,
)
from src.model.events import (
    EVENT_SCHEMA_VERSION,
    AlertEligibleEvent,
    EventClassName,
    EventError,
    EventV2,
    EvidenceEligibleEvent,
    HypothesisEvent,
    InferredEvent,
    ObservedEvent,
    PredictedEvent,
    assemble_evidence,
    confirm_prediction,
    event_v2_from_dict,
    event_v2_to_dict,
    raise_alert,
)
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import (
    BadgeSwipeMeasurement,
    CameraFrameMeasurement,
    DoorContactMeasurement,
    HRISSyncMeasurement,
    Measurement,
)
from src.model.observation import Observation, ObservationError
from src.model.uncertainty import ALL_UNCERTAINTY_KINDS, Uncertainty, UncertaintyError
from src.model.ulid import ULID, InvalidULID, generate_ulid

__all__ = [
    "ULID",
    "InvalidULID",
    "generate_ulid",
    "Uncertainty",
    "UncertaintyError",
    "ALL_UNCERTAINTY_KINDS",
    "CameraFrameMeasurement",
    "BadgeSwipeMeasurement",
    "DoorContactMeasurement",
    "HRISSyncMeasurement",
    "Measurement",
    "FrameOfReference",
    "Observation",
    "ObservationError",
    "Entity",
    "EntityError",
    "EntityKind",
    "ENTITY_KINDS",
    "IdentityClass",
    "IDENTITY_CLASSES",
    "AnonymousSessionEntity",
    "EnrolledEntity",
    "RegisteredAssetEntity",
    "FixtureEntity",
    "TimeVaryingAttribute",
    "EnvelopeStatus",
    "ENVELOPE_STATUSES",
    "EnvelopeCurve",
    "Envelope",
    "EnvelopeError",
    "EVENT_SCHEMA_VERSION",
    "EventClassName",
    "EventError",
    "EventV2",
    "EvidenceEligibleEvent",
    "AlertEligibleEvent",
    "ObservedEvent",
    "InferredEvent",
    "PredictedEvent",
    "HypothesisEvent",
    "assemble_evidence",
    "raise_alert",
    "confirm_prediction",
    "event_v2_to_dict",
    "event_v2_from_dict",
]
