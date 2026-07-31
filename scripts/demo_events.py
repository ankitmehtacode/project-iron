"""Write and read back a three-event sequence, to show the schema end to end.

The staged scene: an unidentified person enters the lobby, picks up a laptop,
and leaves. It is deliberately the shape of an incident someone would later be
asked to explain, because that is the case the schema is designed for.

Three things in the output are worth looking at:

- The subject is a ``session`` entity, not an ``enrolled`` one. Nobody was
  identified. The schema has no way to blur that line, which is what stops a
  system from asserting that an unknown person *is* a named employee.
- The exit event has ``observed=False``. The person left through a blind spot,
  so their departure is inferred, not seen. Anything rendering this must show
  it differently from the other two.
- Every event carries a ``clip`` and a ``manifest_sha``: the footage that backs
  it, and the run that produced it.

Run:
    python scripts/demo_events.py
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from src.config import IronConfig
from src.events import (
    ClipRef,
    EntityRef,
    Event,
    Verb,
    read_events_parquet,
    write_events_parquet,
)
from src.provenance import RunManifest

SECOND_NS = 1_000_000_000
BASE_TS = 1_785_000_000 * SECOND_NS  # a fixed wall-clock instant


def build_sequence(manifest_sha: str) -> list[Event]:
    """Stage the three-event incident."""
    person = EntityRef("session", "sess-4f2a91")
    laptop = EntityRef("asset", "laptop-114")
    lobby = EntityRef("zone", "lobby")

    def clip(offset_s: int, length_s: int = 4) -> ClipRef:
        start = BASE_TS + offset_s * SECOND_NS
        return ClipRef(
            video_id="cam-lobby-01/2026-07-31T09-00",
            start_ts_ns=start,
            end_ts_ns=start + length_s * SECOND_NS,
            content_sha=f"{offset_s:064x}",
        )

    return [
        Event(
            event_id=uuid.uuid5(uuid.NAMESPACE_OID, "demo-entered"),
            site_id="site-hq-1",
            ts_ns=BASE_TS,
            subject=person,
            verb=Verb.ENTERED,
            zone=lobby,
            confidence=0.94,
            observed=True,
            importance=0.30,
            clip=clip(0),
            manifest_sha=manifest_sha,
        ),
        Event(
            event_id=uuid.uuid5(uuid.NAMESPACE_OID, "demo-picked-up"),
            site_id="site-hq-1",
            ts_ns=BASE_TS + 12 * SECOND_NS,
            subject=person,
            verb=Verb.PICKED_UP,
            # Required: an INTERACTION verb without an object is incoherent.
            object=laptop,
            zone=lobby,
            confidence=0.81,
            observed=True,
            # High: an unidentified person handling a tracked asset is the
            # combination that should reach a human.
            importance=0.88,
            clip=clip(10, length_s=6),
            manifest_sha=manifest_sha,
        ),
        Event(
            event_id=uuid.uuid5(uuid.NAMESPACE_OID, "demo-exited"),
            site_id="site-hq-1",
            ts_ns=BASE_TS + 31 * SECOND_NS,
            subject=person,
            verb=Verb.EXITED,
            zone=lobby,
            confidence=0.62,
            # Inferred: the track was lost behind a pillar and the departure
            # reconstructed. Must never render as though it were seen.
            observed=False,
            importance=0.75,
            clip=clip(29),
            manifest_sha=manifest_sha,
        ),
    ]


def render(event: Event) -> str:
    """Format one event for a human.

    Note what this does not do: it never adds a fact. Every word comes from a
    field. "Likely" appears only because ``observed`` is False, and the subject
    is called "an unidentified person" only because its kind is ``session``.
    That is the whole grounding rule in one function.
    """
    who = (
        "an unidentified person"
        if event.subject.kind == "session"
        else f"enrolled identity {event.subject.id}"
    )
    what = event.verb.value.replace("_", " ")
    target = f" {event.object.id}" if event.object is not None else ""
    where = f" in {event.zone.id}" if event.zone is not None else ""
    hedge = "likely " if event.is_inferred else ""
    return (
        f"{hedge}{who} {what}{target}{where} "
        f"(confidence {event.confidence:.2f}, importance {event.importance:.2f}, "
        f"{'observed' if event.observed else 'INFERRED'}, event {event.event_id})"
    )


def main() -> int:
    config = IronConfig.load()
    manifest = RunManifest.capture(config, model_paths=[])
    events = build_sequence(manifest.manifest_sha)

    with tempfile.TemporaryDirectory() as tmp:
        path = write_events_parquet(events, Path(tmp) / "demo_events.parquet")
        print(
            f"Wrote {len(events)} events to {path.name} "
            f"({path.stat().st_size} bytes)"
        )

        restored = read_events_parquet(path)
        print(
            f"Read back {len(restored)} events; "
            f"identical to what was written: {restored == events}"
        )
        print()

        print("Rendered timeline (every word traces to a field):")
        for event in restored:
            print(f"  - {render(event)}")
        print()

        inferred = [e for e in restored if e.is_inferred]
        print(
            f"{len(inferred)} of {len(restored)} events are inferred rather than "
            "observed and must be rendered distinctly downstream."
        )
        print(f"All events carry manifest_sha {manifest.manifest_sha[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
