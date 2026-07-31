"""Event schema v1 — the product's core data contract.

The log is truth; language is rendering
---------------------------------------
This system's output may end up in an internal investigation or a security
audit. One hallucinated sentence presented as fact ends the product category.
The defence is architectural, not a prompt: **facts originate only in typed
event records**. An LLM may format an :class:`Event` into a sentence or compile
a question into a query over these records, but it may never author a fact. If
the language layer is switched off entirely, the product still works — that is
the test of whether the layering is right.

Every event should carry a :class:`ClipRef`. An event is evidence because
someone can watch it.

The verb vocabulary is closed and versioned
-------------------------------------------
v1 has 19 verbs in three groups. Adding one is a schema version bump plus a
detection recipe, labelled evaluation data, and a published precision/recall
floor *before* it may fire in production. Roughly thirty verbs done reliably is
worth far more than an open vocabulary at 60% precision, because a verb nobody
trusts poisons every answer it appears in.

Verb / object coherence
-----------------------
============  ================================================  ==================
Group         Verbs                                             ``object``
============  ================================================  ==================
GEOMETRIC     entered, exited, dwelled, approached, followed,    optional
              loitered, ran, fell
POSE          sat, stood, bent_down, reached, lying              must be absent
INTERACTION   picked_up, put_down, carried, handed_over,         **required**
              opened, closed
============  ================================================  ==================

An interaction with no object is not an under-specified event, it is an
incoherent one: "picked up" is not a fact until you say what was picked up. A
pose verb with an object is equally incoherent — "sat a laptop" is not a claim
about anything. The constructor rejects both.

The ``observed`` flag
---------------------
``observed=False`` marks an *inferred* fact: a trajectory predicted across a
blind spot, an identity resolved retroactively. Downstream code must render
inferred events distinctly, and must not let them satisfy a query asking what
was actually seen unless that query opts in. A system that renders predictions
identically to observations fabricates evidence, which is the specific failure
this whole module exists to prevent.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION: Literal["1.0"] = "1.0"

EntityKind = Literal["session", "enrolled", "asset", "zone"]
ENTITY_KINDS: tuple[EntityKind, ...] = ("session", "enrolled", "asset", "zone")


class Verb(str, enum.Enum):
    """The closed v1 verb vocabulary.

    Inherits from ``str`` so a verb serialises as its own name in JSON and
    Parquet without a conversion step that could drift between writer and
    reader.
    """

    # GEOMETRIC — derived from tracks, zones, and time.
    ENTERED = "entered"
    EXITED = "exited"
    DWELLED = "dwelled"
    APPROACHED = "approached"
    FOLLOWED = "followed"
    LOITERED = "loitered"
    RAN = "ran"
    FELL = "fell"

    # POSE — derived from body configuration.
    SAT = "sat"
    STOOD = "stood"
    BENT_DOWN = "bent_down"
    REACHED = "reached"
    LYING = "lying"

    # INTERACTION — subject acts on an object.
    PICKED_UP = "picked_up"
    PUT_DOWN = "put_down"
    CARRIED = "carried"
    HANDED_OVER = "handed_over"
    OPENED = "opened"
    CLOSED = "closed"


GEOMETRIC_VERBS: frozenset[Verb] = frozenset(
    {
        Verb.ENTERED,
        Verb.EXITED,
        Verb.DWELLED,
        Verb.APPROACHED,
        Verb.FOLLOWED,
        Verb.LOITERED,
        Verb.RAN,
        Verb.FELL,
    }
)

POSE_VERBS: frozenset[Verb] = frozenset(
    {Verb.SAT, Verb.STOOD, Verb.BENT_DOWN, Verb.REACHED, Verb.LYING}
)

INTERACTION_VERBS: frozenset[Verb] = frozenset(
    {
        Verb.PICKED_UP,
        Verb.PUT_DOWN,
        Verb.CARRIED,
        Verb.HANDED_OVER,
        Verb.OPENED,
        Verb.CLOSED,
    }
)


def _assert_vocabulary_is_partitioned() -> None:
    """Fail at import if the groups drift out of sync with the enum.

    A verb that belongs to no group would silently bypass the coherence rules;
    one in two groups would get contradictory treatment.
    """
    groups = (GEOMETRIC_VERBS, POSE_VERBS, INTERACTION_VERBS)
    union: set[Verb] = set().union(*groups)
    missing = set(Verb) - union
    if missing:
        raise RuntimeError(f"verbs in no group: {sorted(v.value for v in missing)}")
    if len(union) != sum(len(group) for group in groups):
        raise RuntimeError("a verb appears in more than one group")


_assert_vocabulary_is_partitioned()


class SchemaError(ValueError):
    """Raised when an event or record violates the v1 contract."""


@dataclass(frozen=True)
class EntityRef:
    """A reference to something an event is about.

    One typed keyspace for four different kinds of thing, so that an anonymous
    session id can never be silently compared against an enrolled identity.
    They look alike — both are opaque strings — and conflating them is how a
    system ends up asserting that an unidentified person *is* a named employee.

    Attributes:
        kind: ``session`` for an anonymous within-camera track, ``enrolled``
            for a known identity, ``asset`` for a tracked object, ``zone`` for
            a named region.
        id: Opaque identifier, unique within ``kind``.
    """

    kind: EntityKind
    id: str

    def __post_init__(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise SchemaError(
                f"unknown entity kind {self.kind!r}; expected one of {ENTITY_KINDS}"
            )
        if not self.id:
            raise SchemaError("EntityRef.id must not be empty")

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}"


@dataclass(frozen=True)
class ClipRef:
    """A pointer to the footage backing an event.

    Attributes:
        video_id: Identifier of the source recording.
        start_ts_ns: Interval start, nanoseconds since the epoch.
        end_ts_ns: Interval end. Must be strictly after the start.
        content_sha: Hash of the clip bytes, when available. Optional because
            an event may be emitted live before the segment is sealed, but
            without it the clip cannot be proved unmodified later.
    """

    video_id: str
    start_ts_ns: int
    end_ts_ns: int
    content_sha: str | None = None

    def __post_init__(self) -> None:
        if not self.video_id:
            raise SchemaError("ClipRef.video_id must not be empty")
        if self.end_ts_ns <= self.start_ts_ns:
            raise SchemaError(
                f"ClipRef must cover a positive interval: start_ts_ns="
                f"{self.start_ts_ns} end_ts_ns={self.end_ts_ns}"
            )

    @property
    def duration_ns(self) -> int:
        return self.end_ts_ns - self.start_ts_ns


@dataclass(frozen=True)
class Event:
    """One typed fact about the world, at one instant, at one site.

    Attributes:
        event_id: Stable unique identity for this record.
        site_id: Which installation. Part of the timeline key together with
            ``ts_ns``; timestamps from two sites are not comparable on their
            own.
        ts_ns: When it happened, nanoseconds since the epoch. Not a frame
            index — frame indices reset on reconnect, do not survive a dropped
            frame, and are not comparable across cameras.
        subject: Who or what acted.
        verb: What happened, from the closed v1 vocabulary.
        object: What was acted upon. Required for INTERACTION verbs, forbidden
            for POSE verbs, optional otherwise.
        zone: Where, when the event is bound to a named region.
        confidence: Detector posterior in [0, 1].
        observed: ``True`` if seen, ``False`` if inferred. See the module
            docstring — this flag is load-bearing.
        importance: Triage weight in [0, 1].
        clip: The footage backing this event.
        manifest_sha: Which run produced it. An event without one cannot be
            attributed to a model version or config, so it cannot be defended.
        schema_version: The contract this record was written against.
    """

    event_id: uuid.UUID
    site_id: str
    ts_ns: int
    subject: EntityRef
    verb: Verb
    confidence: float
    observed: bool
    importance: float
    manifest_sha: str
    object: EntityRef | None = None
    zone: EntityRef | None = None
    clip: ClipRef | None = None
    schema_version: Literal["1.0"] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version!r}; this build "
                f"writes and reads {SCHEMA_VERSION!r}"
            )
        if not self.site_id:
            raise SchemaError("Event.site_id must not be empty")
        if not self.manifest_sha:
            raise SchemaError(
                "Event.manifest_sha is required: an event that cannot be traced "
                "to the run that produced it cannot be defended in a review"
            )
        if not isinstance(self.verb, Verb):
            raise SchemaError(
                f"Event.verb must be a Verb from the closed v1 vocabulary, "
                f"got {self.verb!r}"
            )

        for name in ("confidence", "importance"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SchemaError(
                    f"Event.{name} must lie in [0, 1], got {value}. A "
                    "probability outside the unit interval is not a "
                    "probability."
                )

        if self.verb in INTERACTION_VERBS and self.object is None:
            raise SchemaError(
                f"{self.verb.value!r} is an INTERACTION verb and requires an "
                "object. 'picked_up' is not a fact until it says what was "
                "picked up."
            )
        if self.verb in POSE_VERBS and self.object is not None:
            raise SchemaError(
                f"{self.verb.value!r} is a POSE verb and must not carry an "
                f"object, got {self.object}. A pose is a claim about the "
                "subject's body, not about another entity."
            )
        if self.zone is not None and self.zone.kind != "zone":
            raise SchemaError(
                f"Event.zone must reference a zone entity, got kind "
                f"{self.zone.kind!r}"
            )

    @property
    def is_inferred(self) -> bool:
        """True when this record is a prediction rather than an observation."""
        return not self.observed

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain JSON-compatible mapping."""

        def entity(ref: EntityRef | None) -> dict[str, str] | None:
            return None if ref is None else {"kind": ref.kind, "id": ref.id}

        return {
            "event_id": str(self.event_id),
            "site_id": self.site_id,
            "ts_ns": self.ts_ns,
            "subject": entity(self.subject),
            "verb": self.verb.value,
            "object": entity(self.object),
            "zone": entity(self.zone),
            "confidence": self.confidence,
            "observed": self.observed,
            "importance": self.importance,
            "clip": (
                None
                if self.clip is None
                else {
                    "video_id": self.clip.video_id,
                    "start_ts_ns": self.clip.start_ts_ns,
                    "end_ts_ns": self.clip.end_ts_ns,
                    "content_sha": self.clip.content_sha,
                }
            ),
            "manifest_sha": self.manifest_sha,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Event":
        """Rebuild an event from :meth:`to_dict` output.

        Raises:
            SchemaError: if the payload is malformed, carries an unknown verb,
                or was written against a different schema version.
        """
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaError(
                f"cannot read a record written against schema_version "
                f"{version!r}; this build supports {SCHEMA_VERSION!r}. Migrate "
                "the file rather than reinterpreting its fields."
            )

        def entity(raw: dict[str, str] | None) -> EntityRef | None:
            if raw is None:
                return None
            return EntityRef(kind=raw["kind"], id=raw["id"])  # type: ignore[arg-type]

        subject = entity(payload["subject"])
        if subject is None:
            raise SchemaError("Event.subject is required")

        raw_verb = payload["verb"]
        try:
            verb = Verb(raw_verb)
        except ValueError as exc:
            raise SchemaError(
                f"unknown verb {raw_verb!r}. The v1 vocabulary is closed; a new "
                "verb requires a schema version bump and a published "
                "precision/recall floor."
            ) from exc

        raw_clip = payload.get("clip")
        clip = (
            None
            if raw_clip is None
            else ClipRef(
                video_id=raw_clip["video_id"],
                start_ts_ns=raw_clip["start_ts_ns"],
                end_ts_ns=raw_clip["end_ts_ns"],
                content_sha=raw_clip.get("content_sha"),
            )
        )

        try:
            return cls(
                event_id=uuid.UUID(str(payload["event_id"])),
                site_id=payload["site_id"],
                ts_ns=int(payload["ts_ns"]),
                subject=subject,
                verb=verb,
                confidence=float(payload["confidence"]),
                observed=bool(payload["observed"]),
                importance=float(payload["importance"]),
                manifest_sha=payload["manifest_sha"],
                object=entity(payload.get("object")),
                zone=entity(payload.get("zone")),
                clip=clip,
                schema_version=SCHEMA_VERSION,
            )
        except KeyError as exc:
            raise SchemaError(f"event record is missing required field {exc}") from exc

    def with_importance(self, importance: float) -> "Event":
        """Return a copy with a new importance score.

        Importance is a triage policy applied after detection, so rescoring is
        expected. Returning a copy keeps the record immutable: the log is
        append-only, and an event that can be edited in place is not evidence.
        """
        return replace(self, importance=importance)


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

_ENTITY_STRUCT = pa.struct([("kind", pa.string()), ("id", pa.string())])

_CLIP_STRUCT = pa.struct(
    [
        ("video_id", pa.string()),
        ("start_ts_ns", pa.int64()),
        ("end_ts_ns", pa.int64()),
        ("content_sha", pa.string()),
    ]
)


def events_arrow_schema() -> pa.Schema:
    """The Arrow schema for a v1 event table.

    Timestamps are ``int64`` nanoseconds rather than Arrow timestamps so the
    stored value is exactly the integer the detector produced, with no timezone
    normalisation or microsecond truncation between writer and reader.

    ``confidence`` and ``importance`` are float64 rather than float32: they are
    compared against thresholds, and a value that reads back as 0.7999999523
    against a 0.8 alert threshold is a support ticket.
    """
    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("ts_ns", pa.int64(), nullable=False),
            pa.field("subject", _ENTITY_STRUCT, nullable=False),
            pa.field("verb", pa.string(), nullable=False),
            pa.field("object", _ENTITY_STRUCT, nullable=True),
            pa.field("zone", _ENTITY_STRUCT, nullable=True),
            pa.field("confidence", pa.float64(), nullable=False),
            pa.field("observed", pa.bool_(), nullable=False),
            pa.field("importance", pa.float64(), nullable=False),
            pa.field("clip", _CLIP_STRUCT, nullable=True),
            pa.field("manifest_sha", pa.string(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
        ],
        metadata={b"iron_schema_version": SCHEMA_VERSION.encode()},
    )


def events_to_table(events: Sequence[Event]) -> pa.Table:
    """Build an Arrow table from events, in the canonical v1 schema."""
    return pa.Table.from_pylist(
        [event.to_dict() for event in events], schema=events_arrow_schema()
    )


def write_events_parquet(events: Iterable[Event], path: Path | str) -> Path:
    """Write events to a Parquet file.

    Args:
        events: Events to write. An empty sequence still produces a valid file
            with the schema attached, so a reader can tell "no events" from
            "no file".
        path: Destination. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(events_to_table(list(events)), destination, compression="snappy")
    return destination


def read_events_parquet(path: Path | str) -> list[Event]:
    """Read events back, enforcing the schema version on the way in.

    Raises:
        SchemaError: if the file was written against a different schema
            version, or a row violates the v1 contract. Reinterpreting a
            foreign version's fields under v1 names is how a log silently
            starts meaning something else.
    """
    table = pq.read_table(Path(path))

    metadata = table.schema.metadata or {}
    stored = metadata.get(b"iron_schema_version")
    if stored is not None and stored.decode() != SCHEMA_VERSION:
        raise SchemaError(
            f"{path} was written against schema_version {stored.decode()!r}; "
            f"this build supports {SCHEMA_VERSION!r}"
        )

    return [Event.from_dict(row) for row in table.to_pylist()]
