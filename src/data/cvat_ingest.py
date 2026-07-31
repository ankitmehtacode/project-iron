"""CVAT export -> GT event parquet + track parquet, validated on the way in.

Validation reuses the event schema's own coherence rules rather than
re-implementing them. An annotation that would be an illegal ``Event`` is
rejected here, at ingest, with the CVAT track id in the message so the
annotator can find and fix it — instead of becoming a GT row that silently
poisons a metric.

Two outputs, deliberately separate:

- **Events**: the verb-level GT, in the production event schema.
- **Tracks**: per-frame boxes, which are a different kind of claim. Merging
  them would force one row shape to describe both "this person is here in this
  frame" and "this person picked something up", and the join key between them
  is the track id.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.events import (
    ENTITY_KINDS,
    INTERACTION_VERBS,
    POSE_VERBS,
    ClipRef,
    EntityRef,
    Event,
    SchemaError,
    Verb,
    deterministic_event_id,
    write_events_parquet,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_FPS = 30.0


class CvatIngestError(ValueError):
    """Raised when a CVAT export cannot be turned into valid GT."""


TRACK_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("ts_ns", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("xtl", pa.float64(), nullable=False),
        pa.field("ytl", pa.float64(), nullable=False),
        pa.field("xbr", pa.float64(), nullable=False),
        pa.field("ybr", pa.float64(), nullable=False),
        pa.field("outside", pa.bool_(), nullable=False),
        pa.field("occluded", pa.bool_(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=True),
        pa.field("video_id", pa.string(), nullable=False),
    ]
)


@dataclass
class IngestResult:
    """What an ingest produced, and what it refused."""

    events: list[Event] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    """Human-readable reasons, each naming the CVAT track id."""

    def write(self, events_path: Path, tracks_path: Path) -> tuple[Path, Path]:
        write_events_parquet(self.events, events_path)
        tracks_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(self.tracks, schema=TRACK_SCHEMA),
            tracks_path,
            compression="snappy",
        )
        return events_path, tracks_path


def _attributes(element: ElementTree.Element) -> dict[str, str]:
    return {
        attribute.get("name", ""): (attribute.text or "").strip()
        for attribute in element.findall("attribute")
    }


def _entity(kind: str, identifier: str, field_name: str, track: str) -> EntityRef:
    if kind not in ENTITY_KINDS:
        raise CvatIngestError(
            f"track {track}: {field_name}_kind {kind!r} is not one of "
            f"{ENTITY_KINDS}"
        )
    if not identifier:
        raise CvatIngestError(f"track {track}: {field_name}_id is empty")
    return EntityRef(kind=kind, id=identifier)


def ingest_cvat_xml(
    xml_path: Path,
    *,
    site_id: str,
    manifest_sha: str,
    fps: float = DEFAULT_FPS,
) -> IngestResult:
    """Parse a CVAT ``annotations.xml`` export into events and tracks.

    Args:
        xml_path: CVAT XML export (``CVAT for video 1.1`` format).
        site_id: Site the footage came from. Part of the timeline key with
            ``ts_ns``; timestamps from two sites are not comparable alone.
        manifest_sha: Manifest of this ingest run, carried by every GT event —
            ingests have bugs too.
        fps: Frame rate used to synthesize timestamps from frame indices.

    Raises:
        CvatIngestError: on malformed XML, an unknown verb label, or an
            annotation that could not form a valid Event. Rejections that are
            *policy* (a label outside the vocabulary) accumulate in
            ``rejected``; rejections that are *structural* raise.
    """
    if not manifest_sha:
        raise CvatIngestError("an ingest run needs a manifest_sha")

    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as exc:
        raise CvatIngestError(f"{xml_path} is not parseable XML: {exc}") from exc

    root = tree.getroot()
    meta_name = root.findtext("./meta/task/name") or xml_path.stem
    video_id = meta_name

    result = IngestResult()
    known_verbs = {verb.value: verb for verb in Verb}

    for track in root.findall("track"):
        track_id = track.get("id", "?")
        label = track.get("label", "")
        boxes = track.findall("box")
        if not boxes:
            result.rejected.append(f"track {track_id} ({label!r}): no boxes")
            continue

        attributes = _attributes(boxes[0])
        subject_id = attributes.get("subject_id", "") or f"cvat-track-{track_id}"
        subject = _entity(
            attributes.get("subject_kind", "session"), subject_id, "subject", track_id
        )

        # Tracks are emitted for every label, verb or not: a person box is
        # track GT even when no verb was annotated on it.
        for box in boxes:
            frame_idx = int(box.get("frame", "0"))
            result.tracks.append(
                {
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "ts_ns": int(frame_idx / fps * NANOSECONDS_PER_SECOND),
                    "label": label,
                    "xtl": float(box.get("xtl", "0")),
                    "ytl": float(box.get("ytl", "0")),
                    "xbr": float(box.get("xbr", "0")),
                    "ybr": float(box.get("ybr", "0")),
                    "outside": box.get("outside", "0") == "1",
                    "occluded": box.get("occluded", "0") == "1",
                    "subject_id": subject.id,
                    "video_id": video_id,
                }
            )

        verb = known_verbs.get(label)
        if verb is None:
            result.rejected.append(
                f"track {track_id}: label {label!r} is not in the v1 verb "
                "vocabulary. The vocabulary is closed — regenerate the CVAT "
                "label config with scripts/cvat_project.py rather than adding "
                "labels by hand."
            )
            continue

        # Coherence, decided by the same rule Event.__post_init__ enforces.
        object_ref: EntityRef | None = None
        object_id = attributes.get("object_id", "")
        if verb in INTERACTION_VERBS:
            if not object_id:
                result.rejected.append(
                    f"track {track_id}: {verb.value!r} is an INTERACTION verb "
                    "and requires an object, but object_id is empty. "
                    "'picked_up' is not a fact until it says what was picked up."
                )
                continue
            object_ref = _entity(
                attributes.get("object_kind", "asset"), object_id, "object", track_id
            )
        elif verb in POSE_VERBS and object_id:
            result.rejected.append(
                f"track {track_id}: {verb.value!r} is a POSE verb and must not "
                f"carry an object, but object_id={object_id!r} was annotated. "
                "A pose is a claim about the subject's body."
            )
            continue

        zone_id = attributes.get("zone_id", "")
        zone = EntityRef("zone", zone_id) if zone_id else None

        start_frame = min(int(b.get("frame", "0")) for b in boxes)
        end_frame = max(int(b.get("frame", "0")) for b in boxes)
        start_ns = int(start_frame / fps * NANOSECONDS_PER_SECOND)
        # +1 keeps the interval positive when a track occupies a single frame,
        # which ClipRef requires and which a single-frame annotation would
        # otherwise violate.
        end_ns = int(end_frame / fps * NANOSECONDS_PER_SECOND) + 1

        try:
            confidence = float(attributes.get("confidence", "1.0"))
        except ValueError:
            result.rejected.append(
                f"track {track_id}: confidence "
                f"{attributes.get('confidence')!r} is not a number"
            )
            continue

        try:
            result.events.append(
                Event(
                    event_id=deterministic_event_id(
                        site_id, video_id, track_id, verb.value
                    ),
                    site_id=site_id,
                    ts_ns=(start_ns + end_ns) // 2,
                    subject=subject,
                    verb=verb,
                    object=object_ref,
                    zone=zone,
                    confidence=confidence,
                    observed=attributes.get("observed", "true").lower() != "false",
                    # Importance is a production triage score, not a property
                    # of GT; scoring GT with the model under evaluation would
                    # be circular.
                    importance=0.0,
                    clip=ClipRef(
                        video_id=video_id,
                        start_ts_ns=start_ns,
                        end_ts_ns=end_ns,
                    ),
                    manifest_sha=manifest_sha,
                )
            )
        except SchemaError as exc:
            # The schema is the authority. Anything it rejects is recorded
            # against its CVAT track id so the annotator can find it.
            result.rejected.append(f"track {track_id}: {exc}")

    return result
