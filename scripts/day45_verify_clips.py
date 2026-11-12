"""Day 45, objective 1: verify the site-zero-solo clips decode; strip audio.

For every video in ``--dir`` (clips A, B, C):

- refuses unless the dataset clears the consent gate
  (src/data/footage.py: a signed, unedited consent record);
- reads the container with ffprobe (codec, resolution, frame rate, stream
  frame count, audio streams);
- decodes EVERY frame with the pinned OpenCV (opencv-python-headless
  4.8.1.78) and hashes the decoded frames: the decode count, resolution and
  fps reported are OpenCV's, not the container's claims;
- with ``--strip-audio``: remuxes the video stream with no re-encode
  (``ffmpeg -map 0:v:0 -c copy -an``), checks the remux decodes to the SAME
  frame hash, and only then replaces the original. The owner's voice is
  personal data and has no use here, so the audio-bearing original is not
  kept. A mismatch leaves the original untouched and fails.

Also validates ``marks.csv`` (name, x_m, y_m, role in {calib, test}).

Writes a JSON manifest with no frames in it. Exit 1 if any clip fails to
open or decode (the Day-45 instruction: stop and report).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.footage import require_consented  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


def ffprobe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data: dict[str, Any] = json.loads(out)
    return data


def decode(path: Path) -> dict[str, Any]:
    """Decode every frame with the pinned OpenCV; hash them in order."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": False}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    digest = hashlib.sha256()
    n = 0
    shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            cap.release()
            return {
                "opened": True,
                "error": f"frame {n} shape {frame.shape} != {shape}",
            }
        digest.update(frame.tobytes())
        n += 1
    cap.release()
    if n == 0 or shape is None:
        return {"opened": True, "error": "decoded zero frames"}
    return {
        "opened": True,
        "frames_decoded": n,
        "width": int(shape[1]),
        "height": int(shape[0]),
        "fps": fps,
        "decoded_frames_sha256": digest.hexdigest(),
        "opencv": cv2.__version__,
    }


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def strip_audio(path: Path, decoded_sha: str) -> dict[str, Any]:
    tmp = path.with_name(path.stem + ".noaudio" + path.suffix)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-an",
            str(tmp),
        ],
        check=True,
    )
    after = decode(tmp)
    if after.get("decoded_frames_sha256") != decoded_sha:
        tmp.unlink(missing_ok=True)
        return {
            "stripped": False,
            "error": "remux changed the decoded frames; kept original",
        }
    before_sha = file_sha(path)
    tmp.replace(path)
    return {
        "stripped": True,
        "file_sha256_before": before_sha,
        "file_sha256_after": file_sha(path),
        "decoded_frames_identical": True,
    }


def check_marks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    problems = []
    need = {"name", "x_m", "y_m", "role"}
    if not rows or not need <= set(rows[0]):
        return {"present": True, "problems": [f"columns must include {sorted(need)}"]}
    for r in rows:
        if r["role"].strip() not in ("calib", "test"):
            problems.append(f"{r['name']}: role {r['role']!r}")
        try:
            float(r["x_m"]), float(r["y_m"])
        except ValueError:
            problems.append(f"{r['name']}: non-numeric position")
    calib = [r["name"] for r in rows if r["role"].strip() == "calib"]
    test = [r["name"] for r in rows if r["role"].strip() == "test"]
    if len(calib) < 4:
        problems.append(f"{len(calib)} calib marks; a homography needs at least 4")
    if len(test) != 6:
        problems.append(f"{len(test)} test marks; the protocol says 6")
    return {"present": True, "calib": calib, "test": test, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, default=ROOT / "data/raw/site_zero_solo")
    p.add_argument("--strip-audio", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    videos = sorted(v for v in args.dir.iterdir() if v.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        print(f"no video in {args.dir}", file=sys.stderr)
        return 1
    require_consented(videos)
    result: dict[str, Any] = {"dir": str(args.dir), "clips": {}}
    failed = []
    for v in videos:
        probe = ffprobe(v)
        streams = probe.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        vs = video_streams[0] if video_streams else {}
        info: dict[str, Any] = {
            "file_sha256": file_sha(v),
            "container": probe.get("format", {}).get("format_name"),
            "codec": vs.get("codec_name"),
            "profile": vs.get("profile"),
            "pix_fmt": vs.get("pix_fmt"),
            "container_frames": vs.get("nb_frames"),
            "container_fps": vs.get("avg_frame_rate"),
            "audio_streams": len(audio),
            "rotation_side_data": [
                d for d in vs.get("side_data_list", []) if "rotation" in d
            ],
        }
        info.update(decode(v))
        if not info.get("opened") or "error" in info:
            failed.append(v.name)
        elif args.strip_audio and audio:
            info["audio_strip"] = strip_audio(v, info["decoded_frames_sha256"])
            if not info["audio_strip"]["stripped"]:
                failed.append(v.name)
        result["clips"][v.name] = info
        print(
            v.name,
            {
                k: info.get(k)
                for k in (
                    "codec",
                    "width",
                    "height",
                    "fps",
                    "frames_decoded",
                    "container_frames",
                    "audio_streams",
                    "error",
                )
            },
        )
    result["marks"] = check_marks(args.dir / "marks.csv")
    print("marks.csv", result["marks"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
