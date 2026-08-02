"""Credential masking and WS-Discovery response parsing.

No network or camera needed — these are pure functions over strings/XML
bytes. The parts of discover_cameras.py that DO need a network (the
multicast probe itself, the ONVIF Media service queries) are exercised
manually once a camera exists; what is unit-tested here is the part that
handles untrusted bytes off the wire and the part that must never leak a
credential, which are exactly the two places a mistake would be silent.
"""

from __future__ import annotations

from discover_cameras import DiscoveredDevice, _parse_probe_match, mask_credentials


def test_mask_credentials_hides_user_and_password() -> None:
    url = "rtsp://admin:s3cr3t@192.168.1.50:554/live/main"
    masked = mask_credentials(url)
    assert "admin" not in masked
    assert "s3cr3t" not in masked
    assert masked == "rtsp://***@192.168.1.50:554/live/main"


def test_mask_credentials_no_op_when_no_credentials_present() -> None:
    url = "rtsp://192.168.1.50:554/live/main"
    assert mask_credentials(url) == url


def test_mask_credentials_handles_special_characters_in_password() -> None:
    # Passwords with @ or : are percent-encoded in a real URL, but a
    # malformed one should still not leak the raw credential blob.
    url = "rtsp://user:p@ss@192.168.1.50/stream"
    masked = mask_credentials(url)
    assert "p@ss" not in masked or masked.count("@") <= 1


def test_parse_probe_match_extracts_xaddr_and_host() -> None:
    xml = b"""<?xml version="1.0"?>
    <e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
                xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <e:Body>
        <d:ProbeMatches>
          <d:ProbeMatch>
            <d:XAddrs>http://192.168.1.77/onvif/device_service</d:XAddrs>
            <d:Scopes>onvif://www.onvif.org/type/NetworkVideoTransmitter</d:Scopes>
          </d:ProbeMatch>
        </d:ProbeMatches>
      </e:Body>
    </e:Envelope>"""
    device = _parse_probe_match(xml)
    assert device is not None
    assert device.host == "192.168.1.77"
    assert device.xaddr == "http://192.168.1.77/onvif/device_service"
    assert "NetworkVideoTransmitter" in device.scopes[0]


def test_parse_probe_match_returns_none_for_malformed_xml() -> None:
    assert _parse_probe_match(b"not xml at all") is None


def test_parse_probe_match_returns_none_without_xaddrs() -> None:
    xml = b"""<?xml version="1.0"?>
    <e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope">
      <e:Body><e:Empty/></e:Body>
    </e:Envelope>"""
    assert _parse_probe_match(xml) is None


def test_parse_probe_match_tolerates_alternate_namespace_prefix() -> None:
    # Some vendors use a different prefix (e.g. wsdd: instead of d:) — the
    # parser matches on local tag name, not prefix, so this must still work.
    xml = b"""<?xml version="1.0"?>
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:wsdd="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <soap:Body>
        <wsdd:ProbeMatches>
          <wsdd:ProbeMatch>
            <wsdd:XAddrs>http://10.0.0.5/onvif/device_service</wsdd:XAddrs>
          </wsdd:ProbeMatch>
        </wsdd:ProbeMatches>
      </soap:Body>
    </soap:Envelope>"""
    device = _parse_probe_match(xml)
    assert device is not None
    assert device.host == "10.0.0.5"
