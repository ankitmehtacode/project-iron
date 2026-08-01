"""Ingest captured office footage into a content-addressed, consented store.

The path exists so that real footage is usable the moment it lands, rather than
becoming a week of ad-hoc scripts written under pressure with a customer
waiting.

Four things it does, and why each is not optional:

**Decoding is the ingest, not a runtime step.** Day 6 established that decoded
frames are the fixture, not the video: OpenCV builds disagree on decode, so a
pipeline that decodes at test time is not reproducible across environments.
Ingest decodes once and stores frames.

**The store is content-addressed.** A clip is named by the hash of its decoded
frames, so re-ingesting the same footage is idempotent and a clip swapped
underneath a manifest changes its own identity. This is the same property the
golden sets rely on.

**Consent is a required field, and ingest refuses without it.** Lane C is the
only lane admissible for calibration and the only one built on people who can
walk up to us and withdraw. A capture with no resolvable consent record is not
lane C — it is footage of people with no basis for holding it — so it is
refused at the door rather than quarantined later. The refusal is the feature.

**Condition tags travel with the clip.** An eval set stratified by condition is
the difference between coverage and thirty clips of the same bright corridor.

    python scripts/ingest_capture.py --source data/raw/office_capture_v1 \\
        --consent docs/consent/office_capture_v1.signed.json \\
        --conditions daylight single_person

Exit codes:
    0  every clip ingested
    1  refused: no consent record, no source, or an unreadable video
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IronConfig  # noqa: E402
from src.data.golden import Condition  # noqa: E402

DATASET_NAME = "office-capture-v1"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


class IngestRefused(RuntimeError):
    """Raised when footage may not be ingested. Never downgraded to a warning."""


@dataclass(frozen=True)
class ConsentRecord:
    """The consent basis for a capture, resolved from a file on disk."""

    path: Path
    sha: str
    subjects: int
    captured_on: str

    @classmethod
    def load(cls, path: Path | None) -> "ConsentRecord":
        if path is None:
            raise IngestRefused(
                "--consent is required. Lane C is the only lane admissible for "
                "calibration, and it is defined by consent records held with "
                "the data. Footage with no consent record is not lane C; it is "
                "footage of people with no basis for holding it."
            )
        if not path.exists():
            raise IngestRefused(
                f"consent record {path} does not exist. Ingest will not proceed "
                "on a promise that the paperwork is somewhere else."
            )
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IngestRefused(f"consent record {path} is not readable JSON: {exc}")

        missing = [k for k in ("subjects", "captured_on") if k not in payload]
        if missing:
            raise IngestRefused(
                f"consent record {path} is missing {', '.join(missing)}. A "
                "record that does not say who consented and when cannot "
                "support a withdrawal request."
            )
        return cls(
            path=path,
            sha=hashlib.sha256(raw).hexdigest(),
            subjects=int(payload["subjects"]),
            captured_on=str(payload["captured_on"]),
        )


def decode_video(path: Path) -> np.ndarray:
    """Decode a video to an ``[N, H, W, 3]`` uint8 array, RGB.

    Decoding happens exactly once, here. The decoded array is what everything
    downstream sees, because OpenCV builds do not agree frame-for-frame and a
    fixture that is re-decoded per run is not a fixture.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IngestRefused(f"cannot open {path}; it is not a readable video")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise IngestRefused(f"{path} decoded to zero frames")
    return np.stack(frames)


def content_sha(frames: np.ndarray) -> str:
    """Hash of the decoded frames — the clip's identity in the store."""
    digest = hashlib.sha256()
    digest.update(str(frames.shape).encode())
    digest.update(str(frames.dtype).encode())
    digest.update(np.ascontiguousarray(frames).tobytes())
    return digest.hexdigest()


def ingest(
    source: Path,
    store_root: Path,
    consent: ConsentRecord,
    conditions: tuple[Condition, ...],
) -> dict[str, Any]:
    """Decode every video under ``source`` into the content-addressed store."""
    if not source.exists():
        raise IngestRefused(f"source {source} does not exist")

    videos = sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise IngestRefused(
            f"no videos under {source} (looked for {sorted(VIDEO_SUFFIXES)})"
        )

    store_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for video in videos:
        frames = decode_video(video)
        sha = content_sha(frames)
        clip_id = f"{video.stem}__{sha[:12]}"
        target = store_root / f"{clip_id}.npz"

        # Content-addressed, so re-ingesting the same footage is idempotent
        # rather than a second copy under a second name.
        if not target.exists():
            np.savez_compressed(target, rgb=frames)

        height, width = int(frames.shape[1]), int(frames.shape[2])
        records.append(
            {
                "clip_id": clip_id,
                "content_sha": sha,
                "source_file": video.name,
                "frames": int(frames.shape[0]),
                "resolution": [width, height],
                "conditions": [c.value for c in conditions],
                # Recorded per clip because the gate's downscale only runs when
                # the source is larger than the gate raster. Day 2's parity
                # claim was vacuous for exactly this reason, and a 320x176
                # clip ingested here would be vacuous the same way.
                "exercises_downscale": width > 320 and height > 180,
                "size_bytes": target.stat().st_size,
            }
        )

    return {
        "schema_version": "1.0",
        "dataset": DATASET_NAME,
        "lane": "C",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "ingested_by": "scripts/ingest_capture.py",
        "consent": {
            "record": str(consent.path),
            "record_sha": consent.sha,
            "subjects": consent.subjects,
            "captured_on": consent.captured_on,
        },
        "decode_note": (
            "Frames were decoded once at ingest and stored. The decoded array "
            "is the fixture; the source video is provenance only. OpenCV "
            "builds do not agree frame-for-frame."
        ),
        "clips": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--consent",
        type=Path,
        default=None,
        help="signed consent record for this capture. REQUIRED — lane C is "
        "defined by it and ingest refuses without it.",
    )
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--store", type=Path, default=None)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    store = args.store or (config.paths.resolved_data_dir / "captures" / DATASET_NAME)

    try:
        consent = ConsentRecord.load(args.consent)
        conditions = tuple(Condition(c) for c in args.conditions)
        manifest = ingest(args.source, store, consent, conditions)
    except IngestRefused as exc:
        print(f"INGEST REFUSED: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"INGEST REFUSED: unknown condition tag — {exc}", file=sys.stderr)
        return 1

    manifest_path = store / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Ingested {len(manifest['clips'])} clip(s) -> {store}")
    for record in manifest["clips"]:
        flag = "" if record["exercises_downscale"] else "   NO DOWNSCALE"
        print(
            f"  {record['clip_id']:44} "
            f"{record['resolution'][0]}x{record['resolution'][1]} "
            f"{record['frames']}f{flag}"
        )
    if not any(r["exercises_downscale"] for r in manifest["clips"]):
        print(
            "\nWARNING: no clip is larger than the 320x180 gate raster, so none "
            "of them exercises the downscale path. A parity claim measured on "
            "this capture would be vacuous — this is exactly how the day-2 "
            "claim went wrong.",
            file=sys.stderr,
        )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
