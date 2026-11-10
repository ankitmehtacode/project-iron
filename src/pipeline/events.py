"""Event compiler: three geometric verbs over world-coordinate zones.

``entered``, ``exited`` and ``dwelled`` only. Input is a subject's floor
trajectory, ``(ts_ns, x, y)`` samples in time order; output is
:class:`~src.model.events.ObservedEvent` records (schema v2, what the
Inspector renders).

The same function compiles the product's tracks and the ground truth's
trajectories, so the event precision/recall measures the pipeline's error,
not a difference between two compilers.

Rules (stated before the first run, Day 43):

- Inside/outside is the trajectory point in the zone polygon (even-odd rule).
- A change of side counts only once it has persisted ``min_presence_s``; the
  event is stamped at the first sample of the new side. Flicker across an
  edge produces nothing.
- The side a subject is first seen on is its starting state, not an event:
  a track that begins inside a zone did not observably enter it.
- ``dwelled`` fires once per stay, at the first sample ``dwell_s`` after the
  stay began (after entry, or after the track's first sample if it began
  inside, a lower bound on the stay that is still true).
- A gap longer than ``max_gap_s`` between samples resets the state: nothing
  was observed across it, so no entry or exit is claimed across it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from src.events.schema import ClipRef, EntityKind, EntityRef, Verb
from src.model.events import ObservedEvent

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

COMPILED_VERBS = (Verb.ENTERED, Verb.EXITED, Verb.DWELLED)


@dataclass(frozen=True)
class Zone:
    """A named floor polygon in world metres (``X_Y_Z_UP``, the scene's frame)."""

    name: str
    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a zone needs a name")
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.name!r} needs at least three vertices")

    def contains(self, xy: FloatArray) -> npt.NDArray[np.bool_]:
        """Even-odd point-in-polygon for ``[N, 2]`` points."""
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        poly = np.asarray(self.polygon, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        inside = np.zeros(len(pts), dtype=bool)
        for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                at_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < at_x)
        return inside


@dataclass(frozen=True)
class EventRules:
    min_presence_s: float = 0.5
    dwell_s: float = 5.0
    max_gap_s: float = 2.0

    def __post_init__(self) -> None:
        if min(self.min_presence_s, self.dwell_s, self.max_gap_s) <= 0:
            raise ValueError("event rule durations must be positive")


@dataclass(frozen=True)
class ZoneConfig:
    site_id: str
    zones: tuple[Zone, ...]
    rules: EventRules


def load_zone_config(path: Path) -> ZoneConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) - {"site_id", "zones", "rules"}:
        raise ValueError(f"{path}: expected keys site_id, zones, rules")
    zones = tuple(
        Zone(
            name=z["name"], polygon=tuple((float(x), float(y)) for x, y in z["polygon"])
        )
        for z in raw["zones"]
    )
    if len({z.name for z in zones}) != len(zones):
        raise ValueError(f"{path}: zone names must be unique")
    return ZoneConfig(
        site_id=str(raw["site_id"]),
        zones=zones,
        rules=EventRules(**(raw.get("rules") or {})),
    )


@dataclass(frozen=True)
class Trajectory:
    """One subject's floor samples, strictly increasing in time."""

    subject_id: str
    ts_ns: IntArray
    xy: FloatArray
    frame_refs: tuple[str, ...]
    """One per sample: what a person opens to check it (``camera/frame-N``)."""
    score: float
    """Carried into ``confidence``; see :func:`compile_events`."""

    def __post_init__(self) -> None:
        n = len(self.ts_ns)
        if self.xy.shape != (n, 2) or len(self.frame_refs) != n:
            raise ValueError(f"{self.subject_id}: ts, xy and frame_refs disagree")
        if n and np.any(np.diff(self.ts_ns) <= 0):
            raise ValueError(f"{self.subject_id}: samples must be strictly increasing")


@dataclass(frozen=True)
class CompiledEvent:
    """What the compiler decided, before it becomes a schema record."""

    subject_id: str
    verb: Verb
    zone: str
    ts_ns: int
    frame_ref: str


def compile_trajectory(
    trajectory: Trajectory, zones: Sequence[Zone], rules: EventRules
) -> list[CompiledEvent]:
    out: list[CompiledEvent] = []
    ts, xy = trajectory.ts_ns, trajectory.xy
    if len(ts) == 0:
        return out
    min_ns = int(rules.min_presence_s * 1e9)
    dwell_ns = int(rules.dwell_s * 1e9)
    gap_ns = int(rules.max_gap_s * 1e9)
    for zone in zones:
        inside = zone.contains(xy)
        state: bool | None = None  # debounced side; None = not yet established
        candidate_start: int | None = None  # index where the other side began
        stay_start_ns: int | None = None
        dwelled = False
        for i in range(len(ts)):
            if i and ts[i] - ts[i - 1] > gap_ns:
                state, candidate_start, stay_start_ns, dwelled = None, None, None, False
            side = bool(inside[i])
            if state is None:
                if candidate_start is None or bool(inside[candidate_start]) != side:
                    candidate_start = i
                if ts[i] - ts[candidate_start] >= min_ns:
                    state = side
                    stay_start_ns = int(ts[candidate_start]) if side else None
                    candidate_start = None
            elif side == state:
                candidate_start = None
            else:
                if candidate_start is None:
                    candidate_start = i
                if ts[i] - ts[candidate_start] >= min_ns:
                    k = candidate_start
                    verb = Verb.ENTERED if side else Verb.EXITED
                    out.append(
                        CompiledEvent(
                            trajectory.subject_id,
                            verb,
                            zone.name,
                            int(ts[k]),
                            trajectory.frame_refs[k],
                        )
                    )
                    state = side
                    stay_start_ns = int(ts[k]) if side else None
                    dwelled = False
                    candidate_start = None
            if (
                state
                and not dwelled
                and stay_start_ns is not None
                and candidate_start is None
                and ts[i] - stay_start_ns >= dwell_ns
            ):
                out.append(
                    CompiledEvent(
                        trajectory.subject_id,
                        Verb.DWELLED,
                        zone.name,
                        int(ts[i]),
                        trajectory.frame_refs[i],
                    )
                )
                dwelled = True
    out.sort(key=lambda e: (e.ts_ns, e.subject_id, e.zone, e.verb.value))
    return out


def compile_events(
    trajectories: Sequence[Trajectory],
    config: ZoneConfig,
    *,
    manifest_sha: str,
    clip_ids: Mapping[str, tuple[str, str | None]],
    subject_kind: EntityKind = "session",
) -> tuple[list[CompiledEvent], list[ObservedEvent]]:
    """Compile every trajectory; return the decisions and their records.

    ``confidence`` carries the track's mean UNCALIBRATED detector score (1.0
    for the oracle): no calibration record exists for any producer, and the
    run manifest says so. ``importance`` is 0.0: no triage policy is applied.
    ``clip_ids`` maps a frame_ref's camera to ``(video_id, content_sha)``.
    """
    decisions: list[CompiledEvent] = []
    records: list[ObservedEvent] = []
    for trajectory in trajectories:
        for d in compile_trajectory(trajectory, config.zones, config.rules):
            decisions.append(d)
            camera = d.frame_ref.split("/", 1)[0]
            video_id, content_sha = clip_ids[camera]
            records.append(
                ObservedEvent(
                    event_id=uuid.uuid4(),
                    site_id=config.site_id,
                    ts_ns=d.ts_ns,
                    subject=EntityRef(kind=subject_kind, id=d.subject_id),
                    verb=d.verb,
                    confidence=float(min(max(trajectory.score, 0.0), 1.0)),
                    importance=0.0,
                    manifest_sha=manifest_sha,
                    zone=EntityRef(kind="zone", id=d.zone),
                    clip=ClipRef(
                        video_id=video_id,
                        start_ts_ns=max(d.ts_ns - 1_000_000_000, 0),
                        end_ts_ns=d.ts_ns + 1_000_000_000,
                        content_sha=content_sha,
                    ),
                    evidence_refs=(d.frame_ref,),
                )
            )
    return decisions, records
