"""A minimal RTSP server, test-only, for driving the ingest client without hardware.

Speaks just enough RTSP to satisfy :func:`src.ingest.rtsp.negotiate_transport`
(OPTIONS/DESCRIBE/SETUP/PLAY), then streams a caller-supplied sequence of
interleaved ``$``-framed chunks — RTP and RTCP alike — over the same TCP
connection. This is the "looped file served over RTSP" fixture the Day-12
objective asked for, specialised to what the client under test actually
needs: real RTSP framing and real interleaved chunk delivery, with the
"video" itself being whatever synthetic RTP/RTCP bytes the test constructs.

Not a general-purpose RTSP server — no auth, no multiple streams, no real
SDP. It exists to prove the CLIENT's protocol handling correct against a
real socket, not to be a camera replacement.
"""

from __future__ import annotations

import re
import socket
import threading
from dataclasses import dataclass


@dataclass
class MinimalRtspServer:
    """Runs in a background thread; serves one client connection.

    ``chunks`` is the sequence of ``(channel, payload)`` pairs written as
    interleaved frames after PLAY — the caller builds these from
    :mod:`tests.test_rtcp`-style fixtures for RTP/RTCP payloads.
    """

    chunks: list[tuple[int, bytes]]
    host: str = "127.0.0.1"
    port: int = 0  # 0 = OS-assigned free port

    def __post_init__(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self.port = self._server_sock.getsockname()[1]
        self._server_sock.listen(1)
        self._thread = threading.Thread(target=self._serve_one, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self._server_sock.close()
        except OSError:
            pass

    @property
    def url(self) -> str:
        return f"rtsp://{self.host}:{self.port}/stream"

    def _serve_one(self) -> None:
        try:
            conn, _ = self._server_sock.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5.0)
            try:
                self._handle_session(conn)
            except (OSError, ConnectionError):
                pass

    def _handle_session(self, conn: socket.socket) -> None:
        buffer = b""
        played = False
        while not played:
            while b"\r\n\r\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            header_blob, _, buffer = buffer.partition(b"\r\n\r\n")
            lines = header_blob.decode("iso-8859-1").split("\r\n")
            method = lines[0].split(" ")[0]
            cseq_line = next((l for l in lines if l.lower().startswith("cseq")), "CSeq: 1")
            cseq = cseq_line.split(":", 1)[1].strip()

            if method == "OPTIONS":
                self._respond(conn, cseq, extra="Public: OPTIONS, DESCRIBE, SETUP, PLAY\r\n")
            elif method == "DESCRIBE":
                sdp = (
                    "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=stream\r\nt=0 0\r\n"
                    "m=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n"
                )
                self._respond(
                    conn,
                    cseq,
                    extra=f"Content-Type: application/sdp\r\nContent-Length: {len(sdp)}\r\n",
                    body=sdp,
                )
            elif method == "SETUP":
                transport_line = next(l for l in lines if l.lower().startswith("transport"))
                requested = transport_line.split(":", 1)[1].strip()
                match = re.search(r"interleaved=(\d+)-(\d+)", requested)
                channels = match.group(0) if match else "interleaved=0-1"
                self._respond(
                    conn,
                    cseq,
                    extra=f"Transport: RTP/AVP/TCP;unicast;{channels}\r\nSession: TESTSESSION\r\n",
                )
            elif method == "PLAY":
                self._respond(conn, cseq, extra="Session: TESTSESSION\r\nRange: npt=0.000-\r\n")
                played = True
            else:
                self._respond(conn, cseq, status="551 Option not supported")

        for channel, payload in self.chunks:
            frame = bytes([0x24, channel]) + len(payload).to_bytes(2, "big") + payload
            conn.sendall(frame)

    @staticmethod
    def _respond(
        conn: socket.socket, cseq: str, *, status: str = "200 OK", extra: str = "", body: str = ""
    ) -> None:
        response = f"RTSP/1.0 {status}\r\nCSeq: {cseq}\r\n{extra}\r\n{body}"
        conn.sendall(response.encode("iso-8859-1"))


def build_rtp_packet(*, seq: int, timestamp: int, payload: bytes, ssrc: int = 0x1) -> bytes:
    """Minimal RTP header (RFC 3550 §5.1) + payload, no CSRC list."""
    byte0 = (2 << 6)  # V=2, P=0, X=0, CC=0
    byte1 = 96  # M=0, PT=96 (dynamic, matches the fixture SDP)
    header = bytes([byte0, byte1]) + seq.to_bytes(2, "big") + timestamp.to_bytes(4, "big") + ssrc.to_bytes(4, "big")
    return header + payload


def build_rtcp_sr(*, ntp_seconds: int, ntp_fraction: int, rtp_timestamp: int, ssrc: int = 0x1) -> bytes:
    """Minimal RTCP SR packet (RFC 3550 §6.4.1), matching tests/test_rtcp.py."""
    import struct

    byte0 = (2 << 6)
    header = struct.pack(">BBH", byte0, 200, 6)
    ssrc_bytes = struct.pack(">I", ssrc)
    sender_info = struct.pack(">IIIII", ntp_seconds, ntp_fraction, rtp_timestamp, 1, 1)
    return header + ssrc_bytes + sender_info
