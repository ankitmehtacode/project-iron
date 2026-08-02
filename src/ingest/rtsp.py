"""RTSP session negotiation, TCP-interleaved transport, gap detection.

Scope, stated plainly
----------------------
This module owns the INGEST layer: opening an RTSP session, forcing TCP
transport, demultiplexing the interleaved binary stream into its RTP and
RTCP channels, deriving wall-clock timestamps from RTCP (via
:mod:`src.ingest.rtcp`), and detecting dropped packets. It does NOT decode
video — turning RTP-carried H.264/H.265 NAL units into pixels is a separate,
already-solved problem (``cv2.VideoCapture`` with an ffmpeg backend, used
elsewhere in this repo, e.g. ``scripts/ingest_capture.py``). Reassembled NAL
payloads are handed to the caller; what this module gets right is the part
those tools do not expose: forced TCP, RTCP-derived time, and honest gap
records.

Why TCP, not UDP
-----------------
UDP drops silently. A dropped RTP packet over UDP is just a gap in the
sequence numbers with no signal that anything went wrong — it surfaces
downstream as an unexplained gate miss with no way to distinguish "the
camera saw nothing" from "the network lost a packet." TCP transport
(RTP/AVP/TCP, RFC 2326 §10.12, the "interleaved" mode where RTP and RTCP
share the RTSP control connection using ``$``-framed binary chunks) makes
loss a connection-level event instead of a silent one.
:func:`negotiate_transport` refuses a server that will not offer TCP.

Why every dropped frame gets a gap record
-------------------------------------------
Same "never silently omit" rule this project applies everywhere else — the
observability partition never lets a frame disappear from its accounting,
the provenance manifest records absence rather than omitting a field. An
RTP sequence-number discontinuity is recorded as a :class:`FrameGap`, never
just skipped over.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from src.ingest.rtcp import RtcpParseError, RtcpTimeMapper, parse_sender_report

DEFAULT_RTSP_PORT = 554
_INTERLEAVED_MARKER = 0x24  # ASCII '$', RFC 2326 §10.12 framing byte
_RTSP_VERSION = "RTSP/1.0"
_USER_AGENT = "project-iron-ingest/1"


class RtspError(RuntimeError):
    """Base class for RTSP session failures."""


class TransportRefused(RtspError):
    """Raised when a server will not offer RTP/AVP/TCP interleaved transport.

    Never silently falls back to UDP: a caller depending on this module for
    frame timing has already decided TCP is required, and downgrading
    behind its back would reintroduce the silent-drop failure mode this
    module exists to close.
    """


class TimestampSource(str, Enum):
    """How a frame's ``ts_ns`` was derived. Always explicit, never assumed."""

    RTCP_SENDER_REPORT = "rtcp_sender_report"
    """Wall-clock time, mapped from an observed RTCP SR. The target."""

    ARRIVAL_TIME = "arrival_time"
    """Local receipt time — used only before any SR has been observed."""


@dataclass(frozen=True)
class FrameGap:
    """A detected discontinuity in RTP sequence numbers: dropped packet(s).

    ``gap_count`` is the number of missing sequence numbers, not the number
    of times a gap was observed — a single detection can span more than one
    lost packet.
    """

    site_id: str
    expected_seq: int
    observed_seq: int
    gap_count: int
    detected_at_ts_ns: int
    ts_source: TimestampSource


@dataclass(frozen=True)
class IngestFrame:
    """One reassembled RTP payload group (a frame's worth of NAL units).

    ``nal_units`` is the raw, still-encoded Annex-B-style payload — decoding
    to pixels is out of scope here (see module docstring); this is what a
    downstream decoder (``cv2.VideoCapture``/ffmpeg) consumes.
    """

    site_id: str
    seq: int
    rtp_timestamp: int
    ts_ns: int
    ts_source: TimestampSource
    nal_units: bytes


@dataclass
class RtspTransport:
    """Negotiated transport parameters for one media stream."""

    rtp_channel: int
    rtcp_channel: int
    clock_rate_hz: int


def _read_response(sock: socket.socket) -> tuple[int, dict[str, str], bytes]:
    """Read one RTSP response: status code, headers, and any body prefix.

    RTSP responses are HTTP-like: a status line, headers terminated by a
    blank line, then an optional body whose length the ``Content-Length``
    header states. Returns any bytes read past the header terminator so the
    caller does not lose data that arrived in the same TCP read.
    """
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            raise RtspError("connection closed while reading RTSP response headers")
        buffer += chunk
    header_blob, _, rest = buffer.partition(b"\r\n\r\n")
    lines = header_blob.decode("iso-8859-1").split("\r\n")
    status_line = lines[0]
    match = re.match(r"RTSP/1\.0 (\d+) ", status_line)
    if not match:
        raise RtspError(f"malformed RTSP status line: {status_line!r}")
    status = int(match.group(1))
    headers = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    return status, headers, rest


def _send_request(
    sock: socket.socket, method: str, url: str, cseq: int, extra_headers: dict[str, str]
) -> None:
    header_lines = [f"{method} {url} {_RTSP_VERSION}", f"CSeq: {cseq}", f"User-Agent: {_USER_AGENT}"]
    for key, value in extra_headers.items():
        header_lines.append(f"{key}: {value}")
    request = "\r\n".join(header_lines) + "\r\n\r\n"
    sock.sendall(request.encode("iso-8859-1"))


def negotiate_transport(
    sock: socket.socket, url: str, *, cseq_start: int = 1
) -> tuple[RtspTransport, bytes]:
    """OPTIONS / DESCRIBE / SETUP / PLAY, forcing RTP/AVP/TCP interleaved.

    Returns the negotiated transport plus any leftover bytes read past the
    PLAY response's headers (which may already contain the start of
    interleaved media data on a fast connection).

    Raises:
        TransportRefused: the server's SETUP response does not confirm TCP
            interleaved transport. This function never retries with UDP.
    """
    cseq = cseq_start

    _send_request(sock, "OPTIONS", url, cseq, {})
    status, _, _ = _read_response(sock)
    if status != 200:
        raise RtspError(f"OPTIONS failed: status {status}")
    cseq += 1

    _send_request(sock, "DESCRIBE", url, cseq, {"Accept": "application/sdp"})
    status, headers, _ = _read_response(sock)
    if status != 200:
        raise RtspError(f"DESCRIBE failed: status {status}")
    cseq += 1

    # A real SDP parse would extract the clock rate from the media
    # attribute (e.g. "a=rtpmap:96 H264/90000"); 90000 Hz is the mandated
    # rate for H.264 video (RFC 6184) and the near-universal default for
    # RTSP video generally, used as a documented fallback when SDP parsing
    # is not available. A future SDP parser should replace this.
    clock_rate_hz = 90_000

    _send_request(
        sock,
        "SETUP",
        url,
        cseq,
        {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"},
    )
    status, headers, rest = _read_response(sock)
    if status != 200:
        raise RtspError(f"SETUP failed: status {status}")
    transport_header = headers.get("transport", "")
    if "RTP/AVP/TCP" not in transport_header:
        raise TransportRefused(
            f"server did not confirm TCP interleaved transport; SETUP "
            f"response Transport header was {transport_header!r}. UDP "
            "drops silently, so this module refuses rather than falling "
            "back to it."
        )
    channels_match = re.search(r"interleaved=(\d+)-(\d+)", transport_header)
    if not channels_match:
        raise RtspError(
            f"SETUP confirmed TCP but did not state interleaved channel "
            f"numbers: {transport_header!r}"
        )
    rtp_channel, rtcp_channel = int(channels_match.group(1)), int(channels_match.group(2))
    session_id = headers.get("session", "").split(";")[0]
    cseq += 1

    play_headers = {"Range": "npt=0.000-"}
    if session_id:
        play_headers["Session"] = session_id
    _send_request(sock, "PLAY", url, cseq, play_headers)
    status, _, rest2 = _read_response(sock)
    if status != 200:
        raise RtspError(f"PLAY failed: status {status}")

    return (
        RtspTransport(rtp_channel=rtp_channel, rtcp_channel=rtcp_channel, clock_rate_hz=clock_rate_hz),
        rest + rest2,
    )


def iter_interleaved_frames(
    sock: socket.socket, leftover: bytes
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(channel, payload)`` for each ``$``-framed interleaved chunk.

    RFC 2326 §10.12: each chunk is ``$`` + 1-byte channel + 2-byte
    big-endian length + that many payload bytes. A malformed marker byte
    (anything other than ``$``) ends iteration rather than guessing where
    the next chunk starts — resynchronising on a corrupted stream by
    scanning for the next ``$`` risks treating payload bytes that happen to
    equal 0x24 as a new frame marker, producing a wrong-length read that
    corrupts every subsequent frame silently.
    """
    buffer = bytearray(leftover)
    while True:
        while len(buffer) < 4:
            chunk = sock.recv(65536)
            if not chunk:
                return
            buffer.extend(chunk)
        if buffer[0] != _INTERLEAVED_MARKER:
            raise RtspError(
                f"expected interleaved frame marker 0x24, got 0x{buffer[0]:02x}; "
                "refusing to resynchronise by scanning, which risks treating "
                "payload bytes as a new frame marker"
            )
        channel = buffer[1]
        length = (buffer[2] << 8) | buffer[3]
        while len(buffer) < 4 + length:
            chunk = sock.recv(65536)
            if not chunk:
                return
            buffer.extend(chunk)
        payload = bytes(buffer[4 : 4 + length])
        del buffer[: 4 + length]
        yield channel, payload


@dataclass
class RtspIngestSession:
    """Ties negotiation, demux, RTCP timing, and gap detection together.

    Construct via :meth:`open`, then iterate :meth:`frames` for
    :class:`IngestFrame` and consult :attr:`gaps` for anything dropped.
    """

    site_id: str
    transport: RtspTransport
    _sock: socket.socket
    _leftover: bytes
    _time_mapper: RtcpTimeMapper = field(init=False)
    _last_seq: int | None = field(default=None, init=False)
    gaps: list[FrameGap] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._time_mapper = RtcpTimeMapper(clock_rate_hz=self.transport.clock_rate_hz)

    @classmethod
    def open(cls, site_id: str, url: str, *, host: str, port: int = DEFAULT_RTSP_PORT, timeout_s: float = 5.0) -> "RtspIngestSession":
        sock = socket.create_connection((host, port), timeout=timeout_s)
        transport, leftover = negotiate_transport(sock, url)
        return cls(site_id=site_id, transport=transport, _sock=sock, _leftover=leftover)

    def close(self) -> None:
        self._sock.close()

    def _handle_rtcp(self, payload: bytes) -> None:
        try:
            report = parse_sender_report(payload)
        except RtcpParseError:
            # Compound RTCP packets often lead with SR but may not always;
            # a non-SR first packet in this position is not itself an
            # ingest error — SDES/BYE packets are valid RTCP too.
            return
        self._time_mapper.observe(report)

    def _timestamp_for(self, rtp_timestamp: int) -> tuple[int, TimestampSource]:
        if self._time_mapper.has_anchor:
            return self._time_mapper.map_to_wallclock_ns(rtp_timestamp), TimestampSource.RTCP_SENDER_REPORT
        import time

        return time.time_ns(), TimestampSource.ARRIVAL_TIME

    def _check_sequence_gap(self, seq: int, ts_ns: int, ts_source: TimestampSource) -> None:
        if self._last_seq is not None:
            expected = (self._last_seq + 1) & 0xFFFF
            if seq != expected:
                gap_count = (seq - expected) & 0xFFFF
                self.gaps.append(
                    FrameGap(
                        site_id=self.site_id,
                        expected_seq=expected,
                        observed_seq=seq,
                        gap_count=gap_count,
                        detected_at_ts_ns=ts_ns,
                        ts_source=ts_source,
                    )
                )
        self._last_seq = seq

    def frames(self) -> Iterator[IngestFrame]:
        """Yield reassembled frames, updating :attr:`gaps` as drops are found.

        A minimal RTP header parse (RFC 3550 §5.1) extracts the sequence
        number, timestamp, and payload for gap detection and time mapping.
        Payload reassembly across multiple RTP packets sharing one RTP
        timestamp (the standard signal that they belong to the same frame)
        is intentionally simple — one payload per yielded frame — because
        depacketisation format (H.264 FU-A vs. single NAL vs. STAP-A) is
        codec-specific and belongs in a decoder-facing layer, not here.
        """
        for channel, payload in iter_interleaved_frames(self._sock, self._leftover):
            self._leftover = b""
            if channel == self.transport.rtcp_channel:
                self._handle_rtcp(payload)
                continue
            if channel != self.transport.rtp_channel:
                continue
            if len(payload) < 12:
                continue  # shorter than a minimal RTP header; not a frame
            seq = (payload[2] << 8) | payload[3]
            rtp_ts = int.from_bytes(payload[4:8], "big")
            csrc_count = payload[0] & 0x0F
            header_len = 12 + 4 * csrc_count
            nal_units = payload[header_len:]

            ts_ns, ts_source = self._timestamp_for(rtp_ts)
            self._check_sequence_gap(seq, ts_ns, ts_source)

            yield IngestFrame(
                site_id=self.site_id,
                seq=seq,
                rtp_timestamp=rtp_ts,
                ts_ns=ts_ns,
                ts_source=ts_source,
                nal_units=nal_units,
            )
