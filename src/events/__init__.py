"""Typed event records — the product's core data contract.

Facts originate here and nowhere else. The language layer renders these
records and compiles questions into queries over them; it never authors a
fact. See :mod:`src.events.schema` for the reasoning and the v1 vocabulary.
"""

from src.events.schema import (
    ENTITY_KINDS,
    GEOMETRIC_VERBS,
    INTERACTION_VERBS,
    POSE_VERBS,
    SCHEMA_VERSION,
    ClipRef,
    EntityKind,
    EntityRef,
    Event,
    SchemaError,
    Verb,
    deterministic_event_id,
    events_arrow_schema,
    events_to_table,
    read_events_parquet,
    write_events_parquet,
)

__all__ = [
    "ENTITY_KINDS",
    "GEOMETRIC_VERBS",
    "INTERACTION_VERBS",
    "POSE_VERBS",
    "SCHEMA_VERSION",
    "ClipRef",
    "EntityKind",
    "EntityRef",
    "Event",
    "SchemaError",
    "Verb",
    "deterministic_event_id",
    "events_arrow_schema",
    "events_to_table",
    "read_events_parquet",
    "write_events_parquet",
]
