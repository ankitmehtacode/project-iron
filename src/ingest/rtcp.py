"""RTCP Sender Report parsing, and RTP-to-wall-clock time mapping (RFC 3550).

Why this module exists on its own
----------------------------------
The `(site_id, ts_ns)` key everything downstream joins on, and the future
<50 ms multi-camera sync budget, both need frame timestamps anchored to a
shared wall clock — not to when this process happened to receive the bytes.
Two cameras on the same network deliver frames with different, varying
network latency; timestamping by arrival time bakes that jitter into every
"did these two cameras see the same moment" comparison.

RTCP Sender Reports solve this by construction: an RTP sender periodically
emits an SR packet pairing its own NTP wall-clock time with the RTP media
timestamp at that instant. Given one SR and the stream's known clock rate,
every RTP timestamp on that stream converts to a wall-clock time by simple
linear interpolation — no clock synchronisation protocol between us and the
camera is needed beyond what the camera already sends.

This module is deliberately self-contained and independently testable: RTCP
SR is a small, fully-specified binary format (RFC 3550 §6.4.1), so every
function here is exercised with synthetic byte sequences in
``tests/test_rtcp.py`` — no camera or network required to prove it correct.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

NTP_UNIX_EPOCH_OFFSET_SECONDS = 2_208_988_800
"""Seconds between the NTP epoch (1900-01-01) and the Unix epoch (1970-01-01).

The fixed constant every NTP-to-Unix conversion in the world uses; this is
not a measured value; it's a calendar fact.
"""

RTCP_SR_PACKET_TYPE = 200
"""RFC 3550 §12.1: PT=200 identifies a Sender Report packet."""

_SR_FIXED_HEADER_LEN = 4  # V/P/RC byte, PT byte, 2-byte length
_SR_SENDER_INFO_LEN = 20  # SSRC(4) + NTP MSW(4) + NTP LSW(4) + RTP ts(4) + pkt count(4) + octet count(4) - SSRC counted separately below
_SR_MIN_LEN = 28  # header(4) + SSRC(4) + sender info(20)


class RtcpParseError(ValueError):
    """Raised when bytes claimed to be an RTCP SR packet are not one.

    Never guessed past — camera input is hostile per this project's privacy
    and security law, and a malformed or truncated RTCP packet must refuse
    rather than silently produce a wrong timestamp.
    """


@dataclass(frozen=True)
class SenderReport:
    """One parsed RTCP SR: the wall-clock <-> RTP-media-time anchor it carries."""

    ssrc: int
    ntp_wallclock_unix_ns: int
    """The sender's NTP time at report generation, converted to Unix ns."""

    rtp_timestamp: int
    """The RTP media timestamp corresponding to ``ntp_wallclock_unix_ns``."""

    packet_count: int
    octet_count: int


def parse_sender_report(payload: bytes) -> SenderReport:
    """Parse one RTCP SR packet (RFC 3550 §6.4.1).

    Args:
        payload: the RTCP packet bytes, header included. A compound RTCP
            packet (SR followed by SDES, etc.) is accepted — only the first
            packet is parsed; the caller slices further packets using the
            ``length`` field if it needs them, mirroring how RTCP framing
            already requires walking packets one at a time.

    Raises:
        RtcpParseError: the payload is too short, is not PT=200, or the
            version field is not RTP version 2. Refuses rather than
            returning a report built from garbage bytes.
    """
    if len(payload) < _SR_MIN_LEN:
        raise RtcpParseError(
            f"RTCP SR packet too short: {len(payload)} bytes, need at least "
            f"{_SR_MIN_LEN}"
        )
    version = (payload[0] >> 6) & 0b11
    if version != 2:
        raise RtcpParseError(f"RTCP version {version} != 2 (RFC 3550)")
    packet_type = payload[1]
    if packet_type != RTCP_SR_PACKET_TYPE:
        raise RtcpParseError(
            f"packet type {packet_type} is not SR ({RTCP_SR_PACKET_TYPE})"
        )

    ssrc, ntp_msw, ntp_lsw, rtp_ts, pkt_count, octet_count = struct.unpack(
        ">IIIIII", payload[4:28]
    )
    ntp_unix_ns = _ntp_to_unix_ns(ntp_msw, ntp_lsw)
    return SenderReport(
        ssrc=ssrc,
        ntp_wallclock_unix_ns=ntp_unix_ns,
        rtp_timestamp=rtp_ts,
        packet_count=pkt_count,
        octet_count=octet_count,
    )


def _ntp_to_unix_ns(msw: int, lsw: int) -> int:
    """64-bit NTP timestamp (32-bit seconds + 32-bit fraction) -> Unix ns."""
    unix_seconds = msw - NTP_UNIX_EPOCH_OFFSET_SECONDS
    fraction_ns = round((lsw / (1 << 32)) * 1_000_000_000)
    return unix_seconds * 1_000_000_000 + fraction_ns


class RtcpTimeMapper:
    """Maps RTP timestamps to wall-clock time using the most recent SR pair.

    Needs at least one :class:`SenderReport` before it can map anything —
    ``map_to_wallclock_ns`` raises rather than falling back to arrival time
    silently, because a caller that cannot tell "no SR yet" from "SR says
    this" would report an arrival-time stamp as if it were RTCP-derived,
    which is exactly the confusion this whole module exists to prevent.
    """

    def __init__(self, clock_rate_hz: int) -> None:
        if clock_rate_hz <= 0:
            raise ValueError(f"clock_rate_hz must be positive, got {clock_rate_hz}")
        self._clock_rate_hz = clock_rate_hz
        self._latest: SenderReport | None = None

    @property
    def has_anchor(self) -> bool:
        return self._latest is not None

    def observe(self, report: SenderReport) -> None:
        """Record a new SR as the anchor for future mappings.

        Only the most recent SR is kept. A two-point interpolation (using
        the two most recent SRs to correct for clock-rate drift) is a
        reasonable future refinement, but a single anchor plus the
        nominal clock rate is what the RFC's basic mapping already
        provides, and is enough for a first camera-ingest deliverable.
        """
        self._latest = report

    def map_to_wallclock_ns(self, rtp_timestamp: int) -> int:
        """Wall-clock Unix ns for an RTP timestamp on this stream.

        Handles 32-bit RTP timestamp wraparound by treating the difference
        from the anchor as a signed 32-bit value — correct as long as the
        RTP timestamp being mapped is within about half the wraparound
        period of the anchor, which for a 90 kHz clock is roughly 6.6
        hours. A session that runs longer than that between SR reports has
        a bigger problem than this edge case.

        Raises:
            RuntimeError: no SR has been observed yet.
        """
        if self._latest is None:
            raise RuntimeError(
                "no RTCP Sender Report observed yet; cannot map RTP time to "
                "wall-clock time. Use ARRIVAL_TIME explicitly and label it "
                "as such rather than guessing."
            )
        delta = _signed_wrap_diff32(rtp_timestamp, self._latest.rtp_timestamp)
        delta_ns = round((delta / self._clock_rate_hz) * 1_000_000_000)
        return self._latest.ntp_wallclock_unix_ns + delta_ns


def _signed_wrap_diff32(a: int, b: int) -> int:
    """``a - b`` for 32-bit values, interpreted as the shorter signed distance."""
    diff = (a - b) & 0xFFFFFFFF
    if diff >= 0x80000000:
        diff -= 0x100000000
    return diff
