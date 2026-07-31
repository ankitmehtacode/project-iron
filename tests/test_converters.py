"""Tests for the canonical converter interface and the NTU skeleton converter.

The fixture is synthesized in the test, per the day's rules: nothing is
downloaded. What matters is that the synthesized files follow the real NTU
format — filename convention and frame-count header — so the converter is
exercised against the actual dialect, just tiny.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.converters import (
    ACTION_TO_VERB,
    CanonicalClips,
    Converter,
    NtuParseError,
    NtuSkeletonConverter,
)
from src.events import (
    POSE_VERBS,
    Verb,
    deterministic_event_id,
    read_events_parquet,
)

MANIFEST = "d" * 64


def write_skeleton(directory: Path, stem: str, frames: int = 90) -> Path:
    """Write a minimal but format-correct .skeleton file.

    Real NTU files carry per-frame joint blocks after the frame-count header;
    one body with a single joint line is enough to be structurally honest
    without committing kilobytes.
    """
    lines = [str(frames)]
    for _ in range(frames):
        lines.append("1")  # one body in frame
        lines.append("0 0 0 0 0 0 0 0 0 2")  # body info
        lines.append("1")  # one joint (real files have 25)
        lines.append("0.1 0.2 0.3 0 0 0 0 0 0 0 0 2")
    path = directory / f"{stem}.skeleton"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def ntu_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "ntu"
    raw.mkdir()
    write_skeleton(raw, "S001C001P001R001A008", frames=90)  # sitting down
    write_skeleton(raw, "S001C002P002R001A009", frames=60)  # standing up
    write_skeleton(raw, "S002C001P001R002A043", frames=120)  # falling
    write_skeleton(raw, "S001C001P003R001A027", frames=45)  # jump up: unmapped
    return raw


def test_converter_satisfies_the_protocol() -> None:
    assert isinstance(NtuSkeletonConverter(MANIFEST), Converter)


def test_conversion_produces_typed_gt(ntu_dir: Path) -> None:
    result = NtuSkeletonConverter(MANIFEST).convert(ntu_dir)

    assert isinstance(result, CanonicalClips)
    assert len(result.clips) == 4, "every sample becomes a clip, mapped or not"
    assert len(result.events) == 3

    by_clip = {e.event_id: e for e in result.events}
    assert len(by_clip) == 3, "event ids must be unique"

    verbs = sorted(e.verb for e in result.events)
    assert verbs == sorted([Verb.SAT, Verb.STOOD, Verb.FELL])

    for event in result.events:
        assert event.confidence == 1.0, "GT is annotated, not detected"
        assert event.observed is True
        assert event.manifest_sha == MANIFEST
        assert event.subject.kind == "session", (
            "NTU performers are numbered, not identified; they belong in the "
            "anonymous keyspace"
        )
        assert event.ts_ns > 0


def test_unmapped_actions_are_recorded_not_dropped(ntu_dir: Path) -> None:
    """The gap in the mapping must be visible in the output.

    A converter that silently drops unmapped classes produces GT whose holes
    are invisible, and a detector evaluated against it gets credit for not
    firing on activities the GT never mentions.
    """
    result = NtuSkeletonConverter(MANIFEST).convert(ntu_dir)
    assert "S001C001P003R001A027" in result.skipped
    assert "no exact verb mapping" in result.skipped["S001C001P003R001A027"]


def test_conversion_is_idempotent(ntu_dir: Path) -> None:
    """Re-converting the same input yields identical events, ids included.

    Random event ids would make every re-conversion look like a fresh set of
    facts, and any join against a previous conversion would silently break.
    """
    first = NtuSkeletonConverter(MANIFEST).convert(ntu_dir)
    second = NtuSkeletonConverter(MANIFEST).convert(ntu_dir)
    assert first.events == second.events


def test_round_trip_through_the_production_parquet_path(
    ntu_dir: Path, tmp_path: Path
) -> None:
    """GT written by the converter reads back through the production reader.

    One language: if GT needed a special reader it would be a second dialect,
    which is the disease the canonical form exists to cure.
    """
    result = NtuSkeletonConverter(MANIFEST).convert(ntu_dir)
    path = result.write_events(tmp_path / "ntu_gt.parquet")
    assert read_events_parquet(path) == result.events


def test_mapping_only_targets_object_free_verbs() -> None:
    """The NTU mapping can never produce an incoherent event.

    NTU annotations name no objects, so any INTERACTION verb in this mapping
    would produce events the schema itself rejects. Guarding the mapping keeps
    that a review-time fact rather than a runtime surprise.
    """
    for verb in ACTION_TO_VERB.values():
        assert verb in POSE_VERBS or verb is Verb.FELL


def test_corrupt_file_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """Corruption must surface, not hide among legitimately unmapped samples."""
    raw = tmp_path / "ntu"
    raw.mkdir()
    (raw / "S001C001P001R001A008.skeleton").write_text("not-a-number\n")
    with pytest.raises(NtuParseError, match="frame count"):
        NtuSkeletonConverter(MANIFEST).convert(raw)


def test_bad_filename_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "ntu"
    raw.mkdir()
    write_skeleton(raw, "freeform_name")
    with pytest.raises(NtuParseError, match="refusing to guess"):
        NtuSkeletonConverter(MANIFEST).convert(raw)


def test_converter_requires_a_manifest_sha() -> None:
    with pytest.raises(ValueError, match="manifest_sha"):
        NtuSkeletonConverter("")


def test_deterministic_event_id_rejects_empty_parts() -> None:
    with pytest.raises(Exception):
        deterministic_event_id("a", "")
