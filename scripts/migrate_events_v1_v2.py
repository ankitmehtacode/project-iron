"""Migrate schema-v1 events (Parquet, ``observed: bool``) into the v2 class
hierarchy (:mod:`src.model.events`).

The mapping is deterministic and total over well-formed v1 records:

    observed=True  -> ObservedEvent
    observed=False -> InferredEvent(basis=<migration placeholder>)

v1 never represented a prediction or a hypothesis — those are new
epistemic states in v2 with no v1 counterpart — so every v1 record maps to
exactly one of the two classes above; there is no ambiguity in *which*
class a record becomes. What can go wrong is a record that is valid under
v1's rules but fails v2 construction (the two rule sets are checked
independently rather than assumed identical, so a future divergence is
caught here instead of silently producing a malformed v2 record). Those
records are reported as unmappable, not dropped silently and not forced
through.

Usage:
    python scripts/migrate_events_v1_v2.py INPUT.parquet [--out OUT.json]
        [--report REPORT.json] [--dry-run]

``--dry-run`` prints the report and writes nothing. Exit code is 1 if any
record was unmappable, 0 otherwise, in both modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.events.schema import Event as EventV1
from src.events.schema import SchemaError, read_events_parquet
from src.model.events import (
    EventError,
    InferredEvent,
    ObservedEvent,
    event_v2_to_dict,
)

MIGRATED_BASIS = (
    "migrated from schema v1 (observed=False); v1 recorded no inference "
    "basis, so the original reason this record was inferred rather than "
    "observed cannot be recovered"
)


@dataclass(frozen=True)
class MigrationReport:
    """Summary of one migration run."""

    total: int
    mapped_observed: int
    mapped_inferred: int
    unmappable: tuple[dict[str, str], ...]

    @property
    def unmappable_count(self) -> int:
        return len(self.unmappable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "mapped_observed": self.mapped_observed,
            "mapped_inferred": self.mapped_inferred,
            "unmappable_count": self.unmappable_count,
            "unmappable": list(self.unmappable),
        }


def migrate_event(v1: EventV1) -> ObservedEvent | InferredEvent:
    """Map one v1 record to its v2 class.

    Raises:
        EventError: if the v2 contract rejects the mapped record even
            though it was valid under v1 — see the module docstring.
    """
    common: dict[str, Any] = dict(
        event_id=v1.event_id,
        site_id=v1.site_id,
        ts_ns=v1.ts_ns,
        subject=v1.subject,
        verb=v1.verb,
        confidence=v1.confidence,
        importance=v1.importance,
        manifest_sha=v1.manifest_sha,
        object=v1.object,
        zone=v1.zone,
        clip=v1.clip,
    )
    if v1.observed:
        return ObservedEvent(**common)
    return InferredEvent(**common, basis=MIGRATED_BASIS)


def migrate(
    events: list[EventV1],
) -> tuple[list[ObservedEvent | InferredEvent], MigrationReport]:
    """Migrate a full list of v1 events, collecting unmappable records."""
    mapped: list[ObservedEvent | InferredEvent] = []
    unmappable: list[dict[str, str]] = []
    n_observed = 0
    n_inferred = 0
    for v1 in events:
        try:
            v2 = migrate_event(v1)
        except (EventError, SchemaError) as exc:
            unmappable.append({"event_id": str(v1.event_id), "reason": str(exc)})
            continue
        mapped.append(v2)
        if isinstance(v2, ObservedEvent):
            n_observed += 1
        else:
            n_inferred += 1
    report = MigrationReport(
        total=len(events),
        mapped_observed=n_observed,
        mapped_inferred=n_inferred,
        unmappable=tuple(unmappable),
    )
    return mapped, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="v1 events Parquet file")
    parser.add_argument(
        "--out", type=Path, default=None, help="write mapped v2 events as JSON"
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the migration report as JSON"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the report; write nothing"
    )
    args = parser.parse_args(argv)

    events = read_events_parquet(args.input)
    mapped, report = migrate(events)

    print(json.dumps(report.as_dict(), indent=2))

    if not args.dry_run:
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps([event_v2_to_dict(e) for e in mapped], indent=2)
            )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report.as_dict(), indent=2))

    return 1 if report.unmappable_count else 0


if __name__ == "__main__":
    sys.exit(main())
