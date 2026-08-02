"""RTSP session negotiation, interleaved demux, and gap detection.

Runs the real client (:mod:`src.ingest.rtsp`) against a real TCP socket
connected to :mod:`tests.fixtures.minimal_rtsp_server` — no hardware, no
mocked sockets, the actual RTSP handshake and ``$``-framed interleaved
binary protocol on the wire. This is the "local RTSP server fixture" the
Day-12 objective asked the ingest path be verified against.
"""

from __future__ import annotations

import socket

import pytest

from src.ingest.rtsp import (
    RtspIngestSession,
    TimestampSource,
    TransportRefused,
    negotiate_transport,
)
from fixtures.minimal_rtsp_server import (
    MinimalRtspServer,
    build_rtcp_sr,
    build_rtp_packet,
)


def _connect(server: MinimalRtspServer) -> socket.socket:
    return socket.create_connection((server.host, server.port), timeout=5.0)


def test_negotiate_transport_confirms_tcp_interleaved() -> None:
    server = MinimalRtspServer(chunks=[])
    server.start()
    try:
        sock = _connect(server)
        transport, _ = negotiate_transport(sock, server.url)
        assert transport.rtp_channel == 0
        assert transport.rtcp_channel == 1
        assert transport.clock_rate_hz == 90_000
    finally:
        server.stop()


def test_full_session_yields_frames_with_rtcp_derived_time() -> None:
    ntp_anchor_seconds = 2_208_988_800 + 5_000  # unix t=5000s
    chunks = [
        (1, build_rtcp_sr(ntp_seconds=ntp_anchor_seconds, ntp_fraction=0, rtp_timestamp=0)),
        (0, build_rtp_packet(seq=1, timestamp=0, payload=b"NALU-frame-1")),
        (0, build_rtp_packet(seq=2, timestamp=90_000, payload=b"NALU-frame-2")),  # +1s
    ]
    server = MinimalRtspServer(chunks=chunks)
    server.start()
    try:
        session = RtspIngestSession.open("site-a", "/stream", host=server.host, port=server.port)
        try:
            frames = list(session.frames())
        finally:
            session.close()
    finally:
        server.stop()

    assert len(frames) == 2
    assert frames[0].nal_units == b"NALU-frame-1"
    assert frames[1].nal_units == b"NALU-frame-2"
    # First frame: RTCP SR (channel 1) arrived before it, so this should be
    # RTCP-derived, not arrival time.
    assert frames[0].ts_source == TimestampSource.RTCP_SENDER_REPORT
    assert frames[0].ts_ns == 5_000 * 1_000_000_000
    # Second frame is +90000 RTP ticks at 90kHz = +1 second.
    assert frames[1].ts_source == TimestampSource.RTCP_SENDER_REPORT
    assert frames[1].ts_ns == (5_000 + 1) * 1_000_000_000
    assert session.gaps == []


def test_frames_before_any_rtcp_sr_use_arrival_time() -> None:
    chunks = [
        (0, build_rtp_packet(seq=1, timestamp=0, payload=b"NALU-frame-1")),
    ]
    server = MinimalRtspServer(chunks=chunks)
    server.start()
    try:
        session = RtspIngestSession.open("site-a", "/stream", host=server.host, port=server.port)
        try:
            frames = list(session.frames())
        finally:
            session.close()
    finally:
        server.stop()

    assert len(frames) == 1
    assert frames[0].ts_source == TimestampSource.ARRIVAL_TIME
    # Sanity: arrival time is a real, recent wall-clock ns value, not zero.
    assert frames[0].ts_ns > 0


def test_sequence_gap_is_recorded_not_silently_dropped() -> None:
    chunks = [
        (0, build_rtp_packet(seq=1, timestamp=0, payload=b"frame-1")),
        (0, build_rtp_packet(seq=2, timestamp=3000, payload=b"frame-2")),
        # seq 3 and 4 are missing — simulates two dropped RTP packets.
        (0, build_rtp_packet(seq=5, timestamp=15000, payload=b"frame-5")),
    ]
    server = MinimalRtspServer(chunks=chunks)
    server.start()
    try:
        session = RtspIngestSession.open("site-a", "/stream", host=server.host, port=server.port)
        try:
            frames = list(session.frames())
        finally:
            session.close()
    finally:
        server.stop()

    assert len(frames) == 3  # every packet still yields a frame
    assert len(session.gaps) == 1
    gap = session.gaps[0]
    assert gap.expected_seq == 3
    assert gap.observed_seq == 5
    assert gap.gap_count == 2
    assert gap.site_id == "site-a"


def test_sequence_number_wraparound_is_not_a_false_gap() -> None:
    chunks = [
        (0, build_rtp_packet(seq=0xFFFE, timestamp=0, payload=b"a")),
        (0, build_rtp_packet(seq=0xFFFF, timestamp=100, payload=b"b")),
        (0, build_rtp_packet(seq=0x0000, timestamp=200, payload=b"c")),  # wraps
    ]
    server = MinimalRtspServer(chunks=chunks)
    server.start()
    try:
        session = RtspIngestSession.open("site-a", "/stream", host=server.host, port=server.port)
        try:
            list(session.frames())
        finally:
            session.close()
    finally:
        server.stop()

    assert session.gaps == []


def test_server_refusing_tcp_is_not_silently_downgraded_to_udp() -> None:
    # A server offering only UDP transport must cause a refusal, never a
    # silent fallback — build a tiny variant server that only ever offers
    # RTP/AVP;unicast (no /TCP, no interleaved) in its SETUP response.
    import threading

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    def serve() -> None:
        conn, _ = server_sock.accept()
        with conn:
            conn.settimeout(5.0)
            buffer = b""
            for _ in range(3):  # OPTIONS, DESCRIBE, SETUP
                while b"\r\n\r\n" not in buffer:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buffer += chunk
                header_blob, _, buffer = buffer.partition(b"\r\n\r\n")
                lines = header_blob.decode("iso-8859-1").split("\r\n")
                method = lines[0].split(" ")[0]
                cseq = next(l for l in lines if l.lower().startswith("cseq")).split(":", 1)[1].strip()
                if method == "SETUP":
                    conn.sendall(
                        f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n"
                        f"Transport: RTP/AVP;unicast;client_port=5000-5001\r\n\r\n".encode()
                    )
                else:
                    conn.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n\r\n".encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        with pytest.raises(TransportRefused):
            negotiate_transport(sock, "/stream")
    finally:
        server_sock.close()
