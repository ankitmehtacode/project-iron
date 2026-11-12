"""Calibration page: click the calib marks on one EMPTY frame, write the
floor-points file (Day 45).

A separate local server, not a route of the Inspector: the Inspector is
read-only by design (it opens artifacts and never writes them), and this
page's only purpose is to write one file. Same standard-library pattern,
localhost only.

    python -m src.inspector.calibrate --clip data/raw/site_zero_solo/A.mp4 \\
        --marks data/raw/site_zero_solo/marks.csv --camera-id site_zero_cam \\
        --out configs/calibration/site_zero_solo/site_zero_cam.yaml

The frame is decoded in memory from the empty-room clip (frame ``--frame``,
default 0) and served as a PNG; it is never written to disk. The clip must
clear the consent gate (src/data/footage.py). The file written is the
existing floor-points format (src/pipeline/floor.py): image points clicked
here, floor points from marks.csv (role ``calib``), and the camera's floor
position and height typed in from a tape measure. Its content hash is the
twin_rev. Lens distortion is not corrected. The response reports, per calib
mark, how far the fitted homography puts the clicked pixel from the measured
mark: with more than four marks, the only evidence of click error and lens
bending.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from src.data.footage import require_consented
from src.pipeline.floor import CalibrationError, write_floor_points

STATIC = Path(__file__).resolve().parent / "static" / "calibrate.html"


def calib_marks(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {"name": r["name"], "x_m": float(r["x_m"]), "y_m": float(r["y_m"])}
        for r in rows
        if r["role"].strip() == "calib"
    ]


def frame_png(clip: Path, index: int) -> bytes:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {clip}")
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf.tobytes())


def save(
    body: dict[str, Any],
    marks: list[dict[str, Any]],
    camera_id: str,
    out: Path,
    clip: Path,
    index: int,
) -> dict[str, Any]:
    clicked = body.get("points", {})
    names = [m["name"] for m in marks if m["name"] in clicked]
    if len(names) < 4:
        raise CalibrationError(f"{len(names)} marks clicked; a homography needs 4")
    by_name = {m["name"]: m for m in marks}
    image = np.array([clicked[n] for n in names], dtype=np.float64)
    floor = np.array([[by_name[n]["x_m"], by_name[n]["y_m"]] for n in names])
    xyz = tuple(float(v) for v in body["camera_xyz_m"])
    if len(xyz) != 3:
        raise CalibrationError("camera_xyz_m must be [x, y, z]")
    note = (
        f"site calibration, {date.today().isoformat()}: {len(names)} calib marks "
        f"clicked on frame {index} of {clip.name} (empty room).\n"
        "Lens distortion NOT corrected. camera_xyz_m typed in from a tape measure."
    )
    projector = write_floor_points(
        out, camera_id, image, floor, xyz, note  # type: ignore[arg-type]
    )
    back, ok = projector.project(image)
    residual = np.linalg.norm(back - floor, axis=1)
    return {
        "written": str(out),
        "twin_rev": projector.twin_rev,
        "marks_used": names,
        "residual_m": {n: float(r) for n, r in zip(names, residual)},
        "all_in_front": bool(ok.all()),
    }


def make_server(
    port: int, clip: Path, marks_csv: Path, camera_id: str, out: Path, index: int
) -> ThreadingHTTPServer:
    require_consented([clip])
    marks = calib_marks(marks_csv)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/marks":
                self._json({"camera_id": camera_id, "marks": marks, "frame": index})
            elif self.path == "/api/frame.png":
                self._send(200, frame_png(clip, index), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/calibration":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length))
                self._json(save(body, marks, camera_id, out, clip, index))
            except (CalibrationError, KeyError, ValueError, TypeError) as exc:
                # Surfaced, never swallowed: the page shows the refusal.
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:
            """Quiet."""

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--port", type=int, default=8898)
    args = p.parse_args(argv)
    server = make_server(
        args.port, args.clip, args.marks, args.camera_id, args.out, args.frame
    )
    host, port = server.server_address[:2]
    print(f"calibration page -> http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
