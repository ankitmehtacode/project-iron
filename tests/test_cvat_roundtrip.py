"""Tests for the CVAT round trip: generated label config, validated ingest.

The property under test is that annotation and production cannot drift. The
label config is generated from the `Verb` enum, so these assert the generation
tracks the enum rather than a copy of it, and that the ingest applies the same
coherence rules the event schema does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cvat_project  # noqa: E402

from src.data.cvat_ingest import CvatIngestError, ingest_cvat_xml  # noqa: E402
from src.events import (  # noqa: E402
    INTERACTION_VERBS,
    POSE_VERBS,
    SCHEMA_VERSION,
    Verb,
    read_events_parquet,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cvat_miniature.xml"
MANIFEST = "e" * 64
SITE = "site-zero"


# ---------------------------------------------------------------------------
# The label config is generated, not written
# ---------------------------------------------------------------------------


def test_every_verb_gets_exactly_one_label() -> None:
    """Adding a verb to the enum must change what annotators see."""
    labels = cvat_project.build_label_config()
    assert [entry["name"] for entry in labels] == [verb.value for verb in Verb]
    assert len(labels) == 19


def test_interaction_labels_require_an_object_attribute() -> None:
    """Coherence baked into the form: an incoherent annotation is unenterable.

    Rejecting it at ingest is a fallback. Making it impossible to type is the
    actual fix, and it saves the annotator a round trip.
    """
    labels = {entry["name"]: entry for entry in cvat_project.build_label_config()}
    for verb in INTERACTION_VERBS:
        names = {a["name"] for a in labels[verb.value]["attributes"]}
        assert "object_id" in names, f"{verb.value} has no object attribute"


def test_pose_labels_have_no_object_attribute() -> None:
    labels = {entry["name"]: entry for entry in cvat_project.build_label_config()}
    for verb in POSE_VERBS:
        names = {a["name"] for a in labels[verb.value]["attributes"]}
        assert "object_id" not in names, f"{verb.value} offers an object field"


def test_every_label_carries_the_observed_flag() -> None:
    """The blind-spot traversals in the capture protocol need it."""
    for entry in cvat_project.build_label_config():
        names = {a["name"] for a in entry["attributes"]}
        assert "observed" in names


def test_spec_records_the_schema_version() -> None:
    """So an export made under one vocabulary cannot be ingested under another."""
    assert cvat_project.build_project_spec()["schema_version"] == SCHEMA_VERSION


def test_generator_writes_a_file(tmp_path: Path) -> None:
    output = tmp_path / "labels.json"
    assert cvat_project.main(["-o", str(output), "--labels-only"]) == 0
    import json

    assert len(json.loads(output.read_text())) == 19


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_valid_annotations_become_events() -> None:
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)

    verbs = sorted(e.verb.value for e in result.events)
    assert verbs == ["entered", "exited", "picked_up"]

    for event in result.events:
        assert event.site_id == SITE
        assert event.manifest_sha == MANIFEST
        assert event.clip is not None
        assert event.clip.end_ts_ns > event.clip.start_ts_ns
        assert event.importance == 0.0, "GT is not scored by the model under eval"


def test_the_inferred_exit_keeps_observed_false() -> None:
    """The flag that separates what was seen from what was reconstructed."""
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    exited = next(e for e in result.events if e.verb is Verb.EXITED)
    assert exited.observed is False
    assert exited.is_inferred


def test_interaction_carries_its_object() -> None:
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    picked = next(e for e in result.events if e.verb is Verb.PICKED_UP)
    assert picked.object is not None
    assert picked.object.kind == "asset"
    assert picked.object.id == "laptop-114"


def test_incoherent_annotations_are_rejected_with_their_track_id() -> None:
    """Three deliberate violations in the fixture, each named so it is fixable."""
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    assert len(result.rejected) == 3

    joined = " | ".join(result.rejected)
    assert "track 3" in joined and "requires an object" in joined
    assert "track 4" in joined and "must not carry an object" in joined
    assert "track 5" in joined and "closed" in joined


def test_rejected_annotations_produce_no_events() -> None:
    """A rejection must not leak a partial event into the GT."""
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    assert len(result.events) == 3
    assert all(e.verb.value != "sat" for e in result.events)


def test_tracks_are_emitted_for_every_label_including_rejected_verbs() -> None:
    """A box is track GT even when its verb annotation was rejected.

    The two claims are independent: "this person is here in this frame" stays
    true whether or not the verb attached to it was coherent.
    """
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    track_ids = {row["track_id"] for row in result.tracks}
    assert track_ids == {"0", "1", "2", "3", "4", "5"}
    assert len(result.tracks) == 7  # track 0 has two boxes, the rest one each


def test_round_trip_through_parquet(tmp_path: Path) -> None:
    result = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    events_path, tracks_path = result.write(
        tmp_path / "gt_events.parquet", tmp_path / "gt_tracks.parquet"
    )

    assert read_events_parquet(events_path) == result.events

    tracks = pq.read_table(tracks_path)
    assert tracks.num_rows == len(result.tracks)
    assert "occluded" in tracks.column_names


def test_ingest_is_idempotent() -> None:
    """Same export in, same event ids out — GT joins must survive re-ingest."""
    first = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    second = ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha=MANIFEST)
    assert first.events == second.events


def test_ingest_requires_a_manifest_sha() -> None:
    with pytest.raises(CvatIngestError, match="manifest_sha"):
        ingest_cvat_xml(FIXTURE, site_id=SITE, manifest_sha="")


def test_malformed_xml_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xml"
    broken.write_text("<annotations><track>")
    with pytest.raises(CvatIngestError, match="not parseable"):
        ingest_cvat_xml(broken, site_id=SITE, manifest_sha=MANIFEST)


def test_labels_in_the_fixture_exist_in_the_generated_config() -> None:
    """The fixture must not test a vocabulary the generator cannot produce."""
    generated = {entry["name"] for entry in cvat_project.build_label_config()}
    for label in ("entered", "picked_up", "sat", "exited"):
        assert label in generated
    assert "loitered_suspiciously" not in generated
