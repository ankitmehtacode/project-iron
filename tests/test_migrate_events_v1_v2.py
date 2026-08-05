"""Objective 2 — the v1 -> v2 migration script, round-tripped on the demo set.

``scripts/demo_events.py::build_sequence`` is the "existing demo parquet"
Day 13 asks the migration to be proven against: a three-event incident
with two observed events and one inferred one (the exit, lost behind a
pillar). It is written to a real Parquet file with
:func:`src.events.schema.write_events_parquet` here rather than reused
from a fixture on disk, so the test exercises the actual v1 writer, not a
hand-built payload.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.events.schema import EntityRef, Event, Verb, write_events_parquet
from src.model.events import InferredEvent, ObservedEvent, event_v2_from_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import demo_events  # noqa: E402
import migrate_events_v1_v2 as migrate_mod  # noqa: E402


@pytest.fixture()
def demo_v1_parquet(tmp_path: Path) -> Path:
    events = demo_events.build_sequence(manifest_sha="deadbeef" * 8)
    return write_events_parquet(events, tmp_path / "v1_demo.parquet")


def test_migrate_maps_the_demo_incident_2_observed_1_inferred(
    demo_v1_parquet: Path,
) -> None:
    from src.events.schema import read_events_parquet

    v1_events = read_events_parquet(demo_v1_parquet)
    mapped, report = migrate_mod.migrate(v1_events)

    assert report.total == 3
    assert report.mapped_observed == 2
    assert report.mapped_inferred == 1
    assert report.unmappable_count == 0
    assert sum(isinstance(e, ObservedEvent) for e in mapped) == 2
    assert sum(isinstance(e, InferredEvent) for e in mapped) == 1


def test_migrated_inferred_event_carries_a_stated_basis(demo_v1_parquet: Path) -> None:
    from src.events.schema import read_events_parquet

    v1_events = read_events_parquet(demo_v1_parquet)
    mapped, _ = migrate_mod.migrate(v1_events)
    inferred = [e for e in mapped if isinstance(e, InferredEvent)]
    assert len(inferred) == 1
    assert inferred[0].basis  # non-empty: STRUCTURAL rule on InferredEvent
    assert inferred[0].verb == Verb.EXITED


def test_migrated_events_preserve_identity_subject_verb_and_timestamps(
    demo_v1_parquet: Path,
) -> None:
    from src.events.schema import read_events_parquet

    v1_events = {e.event_id: e for e in read_events_parquet(demo_v1_parquet)}
    mapped, _ = migrate_mod.migrate(list(v1_events.values()))
    for v2 in mapped:
        v1 = v1_events[v2.event_id]
        assert v2.site_id == v1.site_id
        assert v2.ts_ns == v1.ts_ns
        assert v2.subject == v1.subject
        assert v2.verb == v1.verb
        assert v2.confidence == v1.confidence
        assert v2.object == v1.object
        assert v2.zone == v1.zone
        assert v2.clip == v1.clip


def test_migration_round_trips_through_json(
    demo_v1_parquet: Path, tmp_path: Path
) -> None:
    exit_code = migrate_mod.main(
        [
            str(demo_v1_parquet),
            "--out",
            str(tmp_path / "v2.json"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert exit_code == 0

    payloads = json.loads((tmp_path / "v2.json").read_text())
    assert len(payloads) == 3
    restored = [event_v2_from_dict(p) for p in payloads]
    assert sum(isinstance(e, ObservedEvent) for e in restored) == 2
    assert sum(isinstance(e, InferredEvent) for e in restored) == 1

    report = json.loads((tmp_path / "report.json").read_text())
    assert report == {
        "total": 3,
        "mapped_observed": 2,
        "mapped_inferred": 1,
        "unmappable_count": 0,
        "unmappable": [],
    }


def test_dry_run_writes_nothing(demo_v1_parquet: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "v2.json"
    report_path = tmp_path / "report.json"
    exit_code = migrate_mod.main(
        [
            str(demo_v1_parquet),
            "--out",
            str(out_path),
            "--report",
            str(report_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not out_path.exists()
    assert not report_path.exists()


def test_unmappable_record_is_reported_not_dropped_silently_or_forced() -> None:
    """A record that fails the v2 contract is caught and reported by
    event_id and reason, and does not appear in the mapped output.

    Built as a duck-typed stand-in for a v1 record rather than a real
    ``Event`` because v1's own constructor already enforces confidence in
    [0, 1] — there is currently no way to get an out-of-range value past
    the v1 schema onto disk. This exercises the safety net for the day the
    two rule sets diverge, which the module docstring says explicitly not
    to assume they won't.
    """
    bad = SimpleNamespace(
        event_id=uuid.uuid4(),
        site_id="site-hq-1",
        ts_ns=1,
        subject=EntityRef("session", "sess-1"),
        verb=Verb.ENTERED,
        confidence=1.5,  # out of [0, 1] — v2 construction must reject this
        observed=True,
        importance=0.5,
        manifest_sha="sha",
        object=None,
        zone=None,
        clip=None,
    )
    good = demo_events.build_sequence(manifest_sha="sha")[0]

    mapped, report = migrate_mod.migrate([good, bad])  # type: ignore[list-item]

    assert report.total == 2
    assert report.unmappable_count == 1
    assert report.unmappable[0]["event_id"] == str(bad.event_id)
    assert len(mapped) == 1
    assert mapped[0].event_id == good.event_id
