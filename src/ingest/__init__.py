"""Camera ingest: RTSP session handling, RTCP-derived wall-clock time.

Built ahead of hardware existing (Day 12) so that plugging in a camera is the
only remaining step. See :mod:`src.ingest.rtcp` for RTCP Sender Report
parsing and :mod:`src.ingest.rtsp` for the RTSP/RTP session layer.
"""
