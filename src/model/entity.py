"""Entity: something an observation, event, or relationship is about.

The identity_class STRUCTURAL rule
-----------------------------------
An anonymous session must never acquire a cross-session persistence field —
that is exactly how a system starts asserting "this unknown person is the
same unknown person from yesterday" without ever having said so on
purpose. The rule is enforced by giving each ``identity_class`` its own
frozen dataclass instead of one class with an optional field:
:class:`AnonymousSessionEntity` has no ``persistent_identity_ref`` slot to
populate, so ``AnonymousSessionEntity(persistent_identity_ref=...)`` is a
``TypeError`` at the call site — a type error, not a validation failure
caught later. See ``test_model_primitives.py`` for the proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

EntityKind = Literal[
    "person", "asset", "vehicle", "door", "zone", "camera", "container"
]
ENTITY_KINDS: tuple[EntityKind, ...] = get_args(EntityKind)

IdentityClass = Literal["anonymous_session", "enrolled", "registered_asset", "fixture"]
IDENTITY_CLASSES: tuple[IdentityClass, ...] = get_args(IdentityClass)


class EntityError(ValueError):
    """Raised when an entity or one of its attributes violates its contract."""


@dataclass(frozen=True)
class TimeVaryingAttribute:
    """A named fact about an entity that holds only over an interval.

    Attributes:
        name: Attribute name (e.g. ``"role"``, ``"color"``).
        value: Its value over the interval, as a string — attributes are
            heterogeneous and typing each one individually would require a
            registry that provides no safety this layer needs; consumers
            that need a typed value parse it at the point of use.
        valid_from_ns: When this value started holding.
        valid_to_ns: When it stopped, or ``None`` if it still holds.
    """

    name: str
    value: str
    valid_from_ns: int
    valid_to_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise EntityError("TimeVaryingAttribute.name must not be empty")
        if self.valid_to_ns is not None and self.valid_to_ns <= self.valid_from_ns:
            raise EntityError(
                f"TimeVaryingAttribute {self.name!r} has valid_to_ns "
                f"{self.valid_to_ns} <= valid_from_ns {self.valid_from_ns}"
            )


@dataclass(frozen=True)
class _EntityCommon:
    entity_id: str
    kind: EntityKind
    first_seen_ns: int
    last_seen_ns: int
    attributes: tuple[TimeVaryingAttribute, ...] = field(default_factory=tuple)


def _validate_common(entity: "_EntityCommon") -> None:
    if not entity.entity_id:
        raise EntityError("entity_id must not be empty")
    if entity.kind not in ENTITY_KINDS:
        raise EntityError(
            f"unknown entity kind {entity.kind!r}; expected one of {ENTITY_KINDS}"
        )
    if entity.last_seen_ns < entity.first_seen_ns:
        raise EntityError(
            f"last_seen_ns {entity.last_seen_ns} precedes first_seen_ns "
            f"{entity.first_seen_ns}"
        )


@dataclass(frozen=True)
class AnonymousSessionEntity(_EntityCommon):
    """A within-camera-session identity with no persistence across sessions.

    Structurally cannot carry a cross-session identity reference — there is
    no field for one. If a track needs to be re-linked across a gap, that
    is a :class:`~src.model.relationship.Correction`
    (``kind="identity_resolution"``), never a mutation of this record.
    """

    identity_class: Literal["anonymous_session"] = "anonymous_session"

    def __post_init__(self) -> None:
        _validate_common(self)


@dataclass(frozen=True)
class EnrolledEntity(_EntityCommon):
    """A known, cross-session identity."""

    identity_class: Literal["enrolled"] = "enrolled"
    persistent_identity_ref: str = ""

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.persistent_identity_ref:
            raise EntityError(
                "EnrolledEntity.persistent_identity_ref must not be empty"
            )


@dataclass(frozen=True)
class RegisteredAssetEntity(_EntityCommon):
    """A tracked, tagged physical object (a badge, a laptop, a vehicle)."""

    identity_class: Literal["registered_asset"] = "registered_asset"
    asset_tag: str = ""

    def __post_init__(self) -> None:
        _validate_common(self)
        if not self.asset_tag:
            raise EntityError("RegisteredAssetEntity.asset_tag must not be empty")


@dataclass(frozen=True)
class FixtureEntity(_EntityCommon):
    """A permanent, site-fixed entity: a zone, a door, a camera."""

    identity_class: Literal["fixture"] = "fixture"

    def __post_init__(self) -> None:
        _validate_common(self)


Entity = AnonymousSessionEntity | EnrolledEntity | RegisteredAssetEntity | FixtureEntity
