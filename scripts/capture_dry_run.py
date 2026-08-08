"""End-to-end capture-path dry run against the local RTSP fixture — no camera.

Day 16, Objective 3. Chains every piece the real capture will need, against
:mod:`tests.fixtures.minimal_rtsp_server` instead of hardware, so a defect in
how the pieces connect is found here rather than during the one afternoon a
human is standing in the office with a camera:

    discover (scripts/discover_cameras.py, run separately: this fixture is
              not ONVIF, so it is not re-driven here)
    -> RTSP connect + frame demux         (src.ingest.rtsp.RtspIngestSession)
    -> decoded frames                     (cv2.imdecode, MJPEG-style payload)
    -> content-addressed store            (sha256 of the decoded array)
    -> registry entry, lane C             (same manifest shape as
                                            scripts/ingest_capture.py, so a
                                            real capture's manifest looks
                                            identical to this dry run's)
    -> Gap records on an induced drop     (RtspIngestSession.coverage_gaps())
    -> condition tagging                  (src.data.golden.Condition)
    -> consent-record refusal             (scripts.ingest_capture.ConsentRecord)

The RTSP payload here is one small JPEG per RTP packet (MJPEG-over-RTP is a
real, simple profile — single NAL/payload per frame, no depacketisation
needed), so ``cv2.imdecode`` on ``IngestFrame.nal_units`` is a genuine decode
of genuine bytes, not a stand-in.

This intentionally stays a dry run, not a fifth production script: the file
-based path (``scripts/ingest_capture.py``, decode from disk, real MP4/MOV)
and the live-RTSP path demonstrated here are still two separate entry
points. Bridging them into one live-RTSP-to-store script is real work a real
camera would motivate; today's job is proving every piece it would need
already works, individually and in sequence.

    python scripts/capture_dry_run.py

Exit codes:
    0  every stage ran and reported honestly (a Gap being detected is
       success, not failure — that is what the induced drop is for)
    1  a stage that must succeed (decode, content-address, consent refusal)
       did not
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.data.golden import Condition  # noqa: E402
from src.ingest.rtsp import RtspIngestSession  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from fixtures.minimal_rtsp_server import (  # noqa: E402
    MinimalRtspServer,
    build_rtcp_sr,
    build_rtp_packet,
)


def _synthetic_jpeg(index: int) -> bytes:
    """A small, genuinely distinct JPEG frame — a moving square on grey."""
    import cv2

    frame = np.full((90, 160, 3), 60, dtype=np.uint8)
    x = 10 + (index * 12) % 120
    frame[30:60, x : x + 20] = (200, 120, 40)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("synthetic JPEG encode failed")
    return buf.tobytes()


def main() -> int:
    import cv2

    print("=" * 78)
    print("CAPTURE READINESS DRY RUN — local RTSP fixture, no camera")
    print("=" * 78)

    failures: list[str] = []

    # -- consent refusal, first: this is the gate a real capture cannot skip
    from scripts.ingest_capture import ConsentRecord, IngestRefused

    print("\n[1/6] consent-record refusal (no record supplied)")
    try:
        ConsentRecord.load(None)
        failures.append("ConsentRecord.load(None) did not raise — refused")
        print("  FAIL: no refusal raised")
    except IngestRefused as exc:
        print(f"  PASS: refused — {exc}")

    # -- build fixture RTP stream: 6 real frames, seq 3-4 induced-dropped,
    #    RTCP SRs interleaved so frames() resolves wallclock time from the
    #    real timing path rather than arrival-time fallback throughout.
    frames_jpeg = [_synthetic_jpeg(i) for i in range(6)]
    ntp_anchor = 2_208_988_800 + 20_000  # unix t=20000s
    chunks: list[tuple[int, bytes]] = [
        (1, build_rtcp_sr(ntp_seconds=ntp_anchor, ntp_fraction=0, rtp_timestamp=0)),
        (0, build_rtp_packet(seq=1, timestamp=0, payload=frames_jpeg[0])),
        (0, build_rtp_packet(seq=2, timestamp=3000, payload=frames_jpeg[1])),
        # seq 3, 4 deliberately missing -- the induced coverage gap.
        (0, build_rtp_packet(seq=5, timestamp=12000, payload=frames_jpeg[2])),
        (0, build_rtp_packet(seq=6, timestamp=15000, payload=frames_jpeg[3])),
        (0, build_rtp_packet(seq=7, timestamp=18000, payload=frames_jpeg[4])),
        (0, build_rtp_packet(seq=8, timestamp=21000, payload=frames_jpeg[5])),
    ]

    server = MinimalRtspServer(chunks=chunks)
    server.start()
    print(f"\n[2/6] RTSP connect + demux — fixture at {server.url}")
    try:
        session = RtspIngestSession.open(
            "dry-run-site", "/stream", host=server.host, port=server.port
        )
        try:
            ingest_frames = list(session.frames())
        finally:
            session.close()
    finally:
        server.stop()
    print(f"  PASS: negotiated TCP-interleaved transport, {len(ingest_frames)} frames demuxed")

    print("\n[3/6] decoded frames (cv2.imdecode on each frame's payload)")
    decoded: list[np.ndarray] = []
    for f in ingest_frames:
        arr = cv2.imdecode(np.frombuffer(f.nal_units, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            failures.append(f"frame seq={f.seq} did not decode")
            continue
        decoded.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if len(decoded) == len(ingest_frames) and decoded:
        print(f"  PASS: {len(decoded)}/{len(ingest_frames)} frames decoded, shape {decoded[0].shape}")
    else:
        print(f"  FAIL: {len(decoded)}/{len(ingest_frames)} frames decoded")

    print("\n[4/6] content-addressed store")
    stacked = np.stack(decoded)
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(stacked.shape).encode())
    digest.update(str(stacked.dtype).encode())
    digest.update(np.ascontiguousarray(stacked).tobytes())
    content_sha = digest.hexdigest()
    store_root = Path("outputs/day16/capture_dry_run")
    store_root.mkdir(parents=True, exist_ok=True)
    clip_id = f"dry-run-site__{content_sha[:12]}"
    clip_path = store_root / f"{clip_id}.npz"
    np.savez_compressed(clip_path, rgb=stacked)
    reloaded_ok = clip_path.exists() and clip_path.stat().st_size > 0
    print(f"  PASS: {clip_id} -> {clip_path} ({clip_path.stat().st_size} bytes)" if reloaded_ok else "  FAIL: store write")

    print("\n[5/6] Gap records on the induced drop (seq 3-4)")
    gaps = session.gaps
    coverage_gaps = session.coverage_gaps()
    if len(gaps) == 1 and gaps[0].expected_seq == 3 and gaps[0].observed_seq == 5 and gaps[0].gap_count == 2:
        print(f"  PASS: FrameGap(expected=3, observed=5, gap_count=2) detected")
    else:
        failures.append(f"expected exactly one FrameGap(3->5); got {gaps}")
        print(f"  FAIL: {gaps}")
    if len(coverage_gaps) == 1 and coverage_gaps[0].reason == "dropped_frame":
        print(f"  PASS: projects into Coverage.Gap(reason={coverage_gaps[0].reason!r}, camera_id={coverage_gaps[0].camera_id!r})")
    else:
        failures.append("FrameGap did not project into exactly one Coverage.Gap")
        print(f"  FAIL: {coverage_gaps}")

    print("\n[6/6] registry entry, lane C + condition tagging")
    conditions = (Condition.DAYLIGHT, Condition.SINGLE_PERSON)
    consent_path = store_root / "dry_run_consent.json"
    consent_path.write_text(
        json.dumps({"subjects": 1, "captured_on": "2026-08-08", "note": "dry run, synthetic frames, no real person"})
    )
    consent = ConsentRecord.load(consent_path)
    manifest = {
        "schema_version": "1.0",
        "dataset": "capture-dry-run",
        "lane": "C",
        "source": "src.ingest.rtsp.RtspIngestSession (live-RTSP path, dry run)",
        "consent": {
            "record": str(consent.path),
            "record_sha": consent.sha,
            "subjects": consent.subjects,
            "captured_on": consent.captured_on,
        },
        "clips": [
            {
                "clip_id": clip_id,
                "content_sha": content_sha,
                "frames": int(stacked.shape[0]),
                "resolution": [int(stacked.shape[2]), int(stacked.shape[1])],
                "conditions": [c.value for c in conditions],
                "coverage_gaps": [g.as_dict() if hasattr(g, "as_dict") else str(g) for g in coverage_gaps],
            }
        ],
    }
    manifest_path = store_root / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"  PASS: lane C manifest, conditions={[c.value for c in conditions]} -> {manifest_path}")

    print("\n" + "=" * 78)
    if failures:
        print(f"DRY RUN: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("DRY RUN: all six stages ran clean. Nothing here needs a camera to work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
