"""Local, read-only HTTP server for the Iron Inspector.

Standard library only, deliberately. FastAPI would be pleasant and it would be
a new runtime dependency on a tool whose entire job is to be available: this
has to run on an air-gapped Tier-3 rack, and ``http.server`` is already there.
The routing below is small enough that a framework would be carrying its
maintenance cost for the sake of decorators.

Read-only, also deliberately. The Inspector is a verification instrument. It
opens artifacts, it never writes them, and it binds to localhost because the
artifacts it reads include footage-derived ground truth.

    make inspect
    python -m src.inspector.server --port 8899
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from src.inspector import artifacts as art

STATIC_DIR = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


def build_routes(store: art.Artifacts) -> list[tuple[re.Pattern[str], Callable]]:
    """URL patterns to handlers. Each handler returns ``(status, body, type)``."""

    def json_response(payload: Any, status: int = 200) -> tuple[int, bytes, str]:
        if isinstance(payload, art.Absent):
            payload = payload.as_dict()
            status = 404
        body = json.dumps(payload, default=_fallback).encode("utf-8")
        return status, body, "application/json; charset=utf-8"

    def scorecards(_: re.Match[str], __: dict[str, list[str]]):
        return json_response({"scorecards": art.list_scorecards(store)})

    def scorecard(match: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.read_scorecard(store, match.group("name")))

    def compare(_: re.Match[str], query: dict[str, list[str]]):
        left = query.get("left", [""])[0]
        right = query.get("right", [""])[0]
        if not left or not right:
            return json_response(
                {"refused": True, "reasons": ["two scorecards are required"]}, 400
            )
        return json_response(art.comparison(store, left, right))

    def envelope(_: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.read_envelope(store))

    def golden(_: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.active_golden(store))

    def events(_: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.read_events(store))

    def provenance(_: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.provenance(store))

    def clip(match: re.Match[str], __: dict[str, list[str]]):
        return json_response(art.clip_analysis(store, match.group("clip")))

    def frame(match: re.Match[str], __: dict[str, list[str]]):
        result = art.clip_frame_png(store, match.group("clip"), int(match.group("i")))
        if isinstance(result, art.Absent):
            return json_response(result)
        return 200, result, "image/png"

    return [
        (re.compile(r"^/api/scorecards$"), scorecards),
        (re.compile(r"^/api/scorecard/(?P<name>[\w.\-]+)$"), scorecard),
        (re.compile(r"^/api/compare$"), compare),
        (re.compile(r"^/api/envelope$"), envelope),
        (re.compile(r"^/api/golden$"), golden),
        (re.compile(r"^/api/events$"), events),
        (re.compile(r"^/api/provenance$"), provenance),
        (re.compile(r"^/api/clip/(?P<clip>[\w.\-]+)$"), clip),
        (re.compile(r"^/api/clip/(?P<clip>[\w.\-]+)/frame/(?P<i>\d+)$"), frame),
    ]


def _fallback(value: Any) -> Any:
    """Serialise the few numpy scalars that reach the JSON encoder."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


class InspectorHandler(BaseHTTPRequestHandler):
    routes: list[tuple[re.Pattern[str], Callable]] = []
    server_version = "IronInspector"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        for pattern, handler in self.routes:
            match = pattern.match(path)
            if match:
                try:
                    status, body, content_type = handler(match, query)
                except Exception as exc:  # noqa: BLE001
                    # Surfaced, never swallowed: a viewer that silently shows
                    # nothing on error is indistinguishable from one showing a
                    # real empty result.
                    status, content_type = 500, "application/json; charset=utf-8"
                    body = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}", "path": path}
                    ).encode("utf-8")
                self._respond(status, body, content_type)
                return

        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
            self._respond(404, b"not found", "text/plain; charset=utf-8")
            return
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._respond(200, target.read_bytes(), content_type)

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default; the terminal is for the eval output, not requests."""


def make_server(port: int, store: art.Artifacts | None = None) -> ThreadingHTTPServer:
    store = store or art.Artifacts.from_config()
    handler = type(
        "BoundInspectorHandler", (InspectorHandler,), {"routes": build_routes(store)}
    )
    # Localhost only. The artifacts include ground truth derived from footage;
    # this is not a service, it is a local instrument.
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)

    server = make_server(args.port)
    host, port = server.server_address[:2]
    print(f"Iron Inspector  ->  http://{host}:{port}")
    print("read-only; Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
