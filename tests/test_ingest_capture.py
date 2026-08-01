"""Ingest refuses footage it has no basis to hold.

Lane C is the only lane admissible for calibration and the only one built on
people who can walk up to us and withdraw. That property is worth exactly as
much as the check that enforces it, so the refusal path is tested harder than
the happy path.

The capture itself is not here yet. These tests run against a synthesized
stand-in, which is enough to prove the *path* works and is explicitly not
enough to produce a parity number — see the day-8 report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ingest_capture as ing  # noqa: E402


def _consent(tmp_path: Path, **overrides) -> Path:
    payload = {"subjects": 3, "captured_on": "2026-08-08", "basis": "written"}
    payload.update(overrides)
    path = tmp_path / "consent.json"
    path.write_text(json.dumps(payload))
    return path


def _video(tmp_path: Path, name: str, width: int, height: int, frames: int = 8) -> Path:
    """A tiny real video file, written with the pinned OpenCV."""
    import cv2

    path = tmp_path / name
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("no encoder available in this OpenCV build")
    rng = np.random.default_rng(4)
    for index in range(frames):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        x = 5 + index * 3
        frame[10 : 10 + 20, x : x + 20] = 200
        frame += rng.integers(0, 3, frame.shape, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# --- the refusals ---------------------------------------------------------


def test_ingest_without_a_consent_record_is_refused() -> None:
    with pytest.raises(ing.IngestRefused, match="--consent is required"):
        ing.ConsentRecord.load(None)


def test_ingest_with_a_missing_consent_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ing.IngestRefused, match="does not exist"):
        ing.ConsentRecord.load(tmp_path / "nope.json")


def test_consent_record_must_say_who_and_when(tmp_path: Path) -> None:
    """A record that cannot support a withdrawal request is not a record."""
    path = tmp_path / "thin.json"
    path.write_text(json.dumps({"basis": "verbal"}))
    with pytest.raises(ing.IngestRefused, match="subjects, captured_on"):
        ing.ConsentRecord.load(path)


def test_cli_refuses_and_exits_nonzero_without_consent(tmp_path: Path) -> None:
    """The refusal must reach the exit code, not just the log."""
    source = tmp_path / "raw"
    source.mkdir()
    code = ing.main(["--source", str(source)])
    assert code == 1


def test_unreadable_source_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ing.IngestRefused, match="no videos under"):
        ing.ingest(
            empty, tmp_path / "store", ing.ConsentRecord.load(_consent(tmp_path)), ()
        )


# --- the path itself ------------------------------------------------------


def test_ingest_decodes_once_and_stores_frames(tmp_path: Path) -> None:
    """Decoded frames are the fixture; the video is provenance only."""
    source = tmp_path / "raw"
    source.mkdir()
    _video(source, "corridor.mp4", 640, 480)

    store = tmp_path / "store"
    manifest = ing.ingest(source, store, ing.ConsentRecord.load(_consent(tmp_path)), ())

    assert len(manifest["clips"]) == 1
    record = manifest["clips"][0]
    stored = store / f"{record['clip_id']}.npz"
    assert stored.exists()

    with np.load(stored) as data:
        frames = np.asarray(data["rgb"])
    assert frames.shape[0] == record["frames"]
    assert frames.dtype == np.uint8
    assert ing.content_sha(frames) == record["content_sha"]


def test_the_store_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    """Re-ingesting the same footage must not make a second copy."""
    source = tmp_path / "raw"
    source.mkdir()
    _video(source, "corridor.mp4", 640, 480)
    store = tmp_path / "store"
    consent = ing.ConsentRecord.load(_consent(tmp_path))

    first = ing.ingest(source, store, consent, ())
    written = sorted(p.name for p in store.glob("*.npz"))
    second = ing.ingest(source, store, consent, ())

    assert first["clips"][0]["content_sha"] == second["clips"][0]["content_sha"]
    assert sorted(p.name for p in store.glob("*.npz")) == written


def test_manifest_records_the_consent_basis(tmp_path: Path) -> None:
    """The consent record travels with the data, by hash."""
    source = tmp_path / "raw"
    source.mkdir()
    _video(source, "corridor.mp4", 640, 480)
    consent_path = _consent(tmp_path, subjects=7, captured_on="2026-08-09")

    manifest = ing.ingest(
        source, tmp_path / "store", ing.ConsentRecord.load(consent_path), ()
    )
    assert manifest["lane"] == "C"
    assert manifest["consent"]["subjects"] == 7
    assert manifest["consent"]["captured_on"] == "2026-08-09"
    assert len(manifest["consent"]["record_sha"]) == 64


def test_condition_tags_travel_with_the_clip(tmp_path: Path) -> None:
    from src.data.golden import Condition

    source = tmp_path / "raw"
    source.mkdir()
    _video(source, "corridor.mp4", 640, 480)
    manifest = ing.ingest(
        source,
        tmp_path / "store",
        ing.ConsentRecord.load(_consent(tmp_path)),
        (Condition.DAYLIGHT, Condition.CROWDED),
    )
    assert manifest["clips"][0]["conditions"] == ["daylight", "crowded"]


def test_a_clip_below_the_gate_raster_is_flagged_as_not_exercising_downscale(
    tmp_path: Path,
) -> None:
    """The day-2 defect, caught at ingest instead of in a report.

    A 320x176 clip never touches the downscale path, so both the full-res and
    gated paths see a bit-identical raster and any parity claim from it is
    vacuous. Ingest records this per clip so nobody has to rediscover it.
    """
    source = tmp_path / "raw"
    source.mkdir()
    _video(source, "small.mp4", 320, 176)
    _video(source, "big.mp4", 1280, 720)

    manifest = ing.ingest(
        source, tmp_path / "store", ing.ConsentRecord.load(_consent(tmp_path)), ()
    )
    flags = {c["source_file"]: c["exercises_downscale"] for c in manifest["clips"]}
    assert flags["small.mp4"] is False
    assert flags["big.mp4"] is True
