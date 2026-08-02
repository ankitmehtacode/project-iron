"""RTCP Sender Report parsing and RTP-to-wallclock mapping.

Pure binary parsing, no network or camera required — every fixture here is a
hand-built byte sequence matching RFC 3550 §6.4.1, so this suite proves the
parser correct independent of any RTSP session ever being opened.
"""

from __future__ import annotations

import struct

import pytest

from src.ingest.rtcp import (
    NTP_UNIX_EPOCH_OFFSET_SECONDS,
    RtcpParseError,
    RtcpTimeMapper,
    SenderReport,
    parse_sender_report,
)


def build_sr_packet(
    *,
    ssrc: int = 0x1234ABCD,
    ntp_seconds_since_1900: int,
    ntp_fraction: int = 0,
    rtp_timestamp: int,
    packet_count: int = 100,
    octet_count: int = 150_000,
    version: int = 2,
    packet_type: int = 200,
) -> bytes:
    """Hand-assemble one RTCP SR packet per RFC 3550 §6.4.1."""
    byte0 = (version << 6) | (0 << 5) | 0  # V=2, P=0, RC=0
    length_words = 6  # SSRC + 5 sender-info words, minus 1, per RFC — kept simple
    header = struct.pack(">BBH", byte0, packet_type, length_words)
    ssrc_bytes = struct.pack(">I", ssrc)
    sender_info = struct.pack(
        ">IIIII",
        ntp_seconds_since_1900,
        ntp_fraction,
        rtp_timestamp,
        packet_count,
        octet_count,
    )
    return header + ssrc_bytes + sender_info


def test_parse_sender_report_basic_fields() -> None:
    packet = build_sr_packet(
        ssrc=0xDEADBEEF,
        ntp_seconds_since_1900=NTP_UNIX_EPOCH_OFFSET_SECONDS + 1000,  # unix t=1000s
        ntp_fraction=0,
        rtp_timestamp=90_000,
        packet_count=42,
        octet_count=9000,
    )
    report = parse_sender_report(packet)
    assert report.ssrc == 0xDEADBEEF
    assert report.ntp_wallclock_unix_ns == 1000 * 1_000_000_000
    assert report.rtp_timestamp == 90_000
    assert report.packet_count == 42
    assert report.octet_count == 9000


def test_ntp_fraction_converts_to_sub_second_precision() -> None:
    # Fraction = 0.5 * 2^32 should be exactly 500,000,000 ns.
    half = 1 << 31
    packet = build_sr_packet(
        ntp_seconds_since_1900=NTP_UNIX_EPOCH_OFFSET_SECONDS,
        ntp_fraction=half,
        rtp_timestamp=0,
    )
    report = parse_sender_report(packet)
    assert report.ntp_wallclock_unix_ns == 500_000_000


def test_parse_refuses_truncated_packet() -> None:
    packet = build_sr_packet(ntp_seconds_since_1900=NTP_UNIX_EPOCH_OFFSET_SECONDS, rtp_timestamp=0)
    with pytest.raises(RtcpParseError, match="too short"):
        parse_sender_report(packet[:20])


def test_parse_refuses_wrong_packet_type() -> None:
    # PT=201 is Receiver Report, not Sender Report.
    packet = build_sr_packet(
        ntp_seconds_since_1900=NTP_UNIX_EPOCH_OFFSET_SECONDS, rtp_timestamp=0, packet_type=201
    )
    with pytest.raises(RtcpParseError, match="not SR"):
        parse_sender_report(packet)


def test_parse_refuses_wrong_rtp_version() -> None:
    packet = build_sr_packet(
        ntp_seconds_since_1900=NTP_UNIX_EPOCH_OFFSET_SECONDS, rtp_timestamp=0, version=1
    )
    with pytest.raises(RtcpParseError, match="version"):
        parse_sender_report(packet)


def test_mapper_raises_before_any_report_observed() -> None:
    mapper = RtcpTimeMapper(clock_rate_hz=90_000)
    assert not mapper.has_anchor
    with pytest.raises(RuntimeError, match="no RTCP Sender Report"):
        mapper.map_to_wallclock_ns(12345)


def test_mapper_interpolates_forward_from_anchor() -> None:
    mapper = RtcpTimeMapper(clock_rate_hz=90_000)
    anchor = SenderReport(
        ssrc=1,
        ntp_wallclock_unix_ns=1_000_000_000_000,  # some arbitrary unix ns
        rtp_timestamp=0,
        packet_count=1,
        octet_count=1,
    )
    mapper.observe(anchor)
    assert mapper.has_anchor

    # One second later at 90kHz is +90000 RTP ticks -> +1e9 ns.
    result = mapper.map_to_wallclock_ns(90_000)
    assert result == 1_000_000_000_000 + 1_000_000_000


def test_mapper_interpolates_backward_from_anchor() -> None:
    mapper = RtcpTimeMapper(clock_rate_hz=90_000)
    mapper.observe(
        SenderReport(ssrc=1, ntp_wallclock_unix_ns=10_000_000_000, rtp_timestamp=90_000, packet_count=1, octet_count=1)
    )
    # Half a second BEFORE the anchor.
    result = mapper.map_to_wallclock_ns(45_000)
    assert result == 10_000_000_000 - 500_000_000


def test_mapper_handles_rtp_timestamp_wraparound() -> None:
    # Anchor near the top of the 32-bit range; query just past wraparound.
    mapper = RtcpTimeMapper(clock_rate_hz=90_000)
    near_max = 0xFFFFFFFF - 1000
    mapper.observe(
        SenderReport(ssrc=1, ntp_wallclock_unix_ns=0, rtp_timestamp=near_max, packet_count=1, octet_count=1)
    )
    # 2000 ticks forward wraps around 32-bit: near_max + 2000 mod 2^32.
    wrapped = (near_max + 2000) & 0xFFFFFFFF
    result = mapper.map_to_wallclock_ns(wrapped)
    expected_ns = round((2000 / 90_000) * 1_000_000_000)
    assert result == expected_ns


def test_mapper_uses_latest_observation_only() -> None:
    mapper = RtcpTimeMapper(clock_rate_hz=90_000)
    mapper.observe(
        SenderReport(ssrc=1, ntp_wallclock_unix_ns=0, rtp_timestamp=0, packet_count=1, octet_count=1)
    )
    mapper.observe(
        SenderReport(ssrc=1, ntp_wallclock_unix_ns=5_000_000_000, rtp_timestamp=90_000, packet_count=2, octet_count=2)
    )
    # Mapping should be relative to the SECOND (latest) anchor.
    result = mapper.map_to_wallclock_ns(90_000)
    assert result == 5_000_000_000


def test_mapper_rejects_non_positive_clock_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        RtcpTimeMapper(clock_rate_hz=0)
