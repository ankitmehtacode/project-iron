"""Typed measurement payloads — camera and non-camera sensors are peers.

A monitoring product that only models cameras cannot represent the events
that most reliably ground identity: a badge swipe, a door contact, an
HRIS sync. Each sensor kind gets its own frozen payload type rather than a
shared dict, so a consumer pattern-matching on
:data:`Measurement` gets a mypy-checked exhaustiveness error the day a new
sensor kind is added and not handled — a bare dict would let that slip
through silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CameraFrameMeasurement:
    """A detection or region reported from a video frame."""

    sensor_kind: Literal["camera"] = "camera"
    bbox_px: tuple[float, float, float, float] | None = None
    """(x0, y0, x1, y1) in the frame's own pixel space, or None for a
    frame-level (not object-level) observation such as a scene embedding."""
    label: str | None = None

    def __post_init__(self) -> None:
        if self.bbox_px is not None:
            x0, y0, x1, y1 = self.bbox_px
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"degenerate bbox {self.bbox_px}")


@dataclass(frozen=True)
class BadgeSwipeMeasurement:
    """An access-control badge read."""

    sensor_kind: Literal["badge"] = "badge"
    badge_id: str = ""
    door_id: str = ""
    granted: bool = False

    def __post_init__(self) -> None:
        if not self.badge_id:
            raise ValueError("BadgeSwipeMeasurement.badge_id must not be empty")
        if not self.door_id:
            raise ValueError("BadgeSwipeMeasurement.door_id must not be empty")


@dataclass(frozen=True)
class DoorContactMeasurement:
    """A magnetic door-contact sensor reading."""

    sensor_kind: Literal["door_contact"] = "door_contact"
    door_id: str = ""
    state: Literal["open", "closed"] = "closed"

    def __post_init__(self) -> None:
        if not self.door_id:
            raise ValueError("DoorContactMeasurement.door_id must not be empty")


@dataclass(frozen=True)
class HRISSyncMeasurement:
    """A record pulled from HR/identity-system sync (roster, badge issuance)."""

    sensor_kind: Literal["hris_sync"] = "hris_sync"
    employee_id: str = ""
    event_type: Literal["provisioned", "revoked", "role_changed"] = "provisioned"

    def __post_init__(self) -> None:
        if not self.employee_id:
            raise ValueError("HRISSyncMeasurement.employee_id must not be empty")


@dataclass(frozen=True)
class WorldPositionMeasurement:
    """A 3D world-frame position reading, already unprojected/triangulated.

    Day 20: the state estimator's measurement models (:mod:`src.estimator.
    measurement_model`) consume this — the estimator works in metric world
    space, not pixel space, so it needs a position measurement as a first-
    class sensor kind rather than reaching back into a
    :class:`CameraFrameMeasurement`'s ``bbox_px`` and re-deriving geometry
    this type already assumes was done upstream.

    No ``twin_rev`` field here: the containing
    :class:`~src.model.observation.Observation`'s ``frame_of_reference.
    twin_rev`` is the single authoritative source for which twin revision
    ``x_m``/``y_m``/``z_m`` were computed under. Duplicating it here would
    create two fields that could disagree.
    """

    sensor_kind: Literal["world_position"] = "world_position"
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    def __post_init__(self) -> None:
        import math

        for name, value in (("x_m", self.x_m), ("y_m", self.y_m), ("z_m", self.z_m)):
            if not math.isfinite(value):
                raise ValueError(
                    f"WorldPositionMeasurement.{name} must be finite, got {value}"
                )


Measurement = (
    CameraFrameMeasurement
    | BadgeSwipeMeasurement
    | DoorContactMeasurement
    | HRISSyncMeasurement
    | WorldPositionMeasurement
)
