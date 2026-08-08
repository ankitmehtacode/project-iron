"""Discover ONVIF cameras on a subnet and enumerate their stream profiles.

Two phases, matching how ONVIF actually works:

1. **WS-Discovery** (RFC-adjacent, ONVIF Core Spec §7): a UDP multicast
   Probe to 239.255.255.250:3702, answered by every ONVIF device on the
   local network segment with its service address (``XAddrs``). Hand-rolled
   here rather than adding a dependency for it — it is a small,
   fully-specified, connectionless protocol, and ``onvif-zeep-async`` only
   covers device CONTROL once an address is already known (see the pin
   comment in ``locking-requirements.txt`` for why that specific package,
   and why not the similarly-named one).
2. **Media service queries**, via ``onvif-zeep-async``, against each
   discovered (or manually supplied) device: ``GetProfiles`` then
   ``GetStreamUri`` per profile, to read resolution, codec, fps, and the
   RTSP URI — never guessed, never assembled from a hardcoded vendor
   pattern. Many camera-integration tutorials hardcode a per-vendor RTSP
   path template; this script deliberately does not, because a template
   that is wrong for one firmware revision fails silently at the point
   nobody is looking (3am, a camera got replaced under warranty).

Credentials never appear in output. An RTSP URI with an embedded
``user:pass@`` is masked before printing or writing to the manifest — see
:func:`mask_credentials`. This follows the "no person data or secrets in
logs/filenames" platform-security law: an RTSP credential leaking into a
log file is the same class of mistake as a name leaking into one.

    python scripts/discover_cameras.py --subnet 192.168.1.0/24
    python scripts/discover_cameras.py --manual --host 192.168.1.50 \\
        --rtsp-url rtsp://192.168.1.50/live/main --user admin

Exit codes:
    0  discovery ran (zero devices found is not a failure — it is reported)
    1  a manually-specified device could not be queried
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import re
import socket
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WS_DISCOVERY_MULTICAST_ADDR = "239.255.255.250"
WS_DISCOVERY_PORT = 3702
_WS_DISCOVERY_TIMEOUT_S = 4.0

_PROBE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{message_id}</w:MessageID>
    <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>"""

_CREDENTIAL_PATTERN = re.compile(r"(rtsp://)([^/@]+)@")


def mask_credentials(url: str) -> str:
    """Replace an embedded ``user:pass@`` with ``***@`` before it is ever printed.

    Applied everywhere this script emits an RTSP URI — stdout, the JSON
    manifest, error messages. A camera credential in a log line is a secret
    leak the same way a database password would be; there is no
    "internal tooling, so it is fine" exception for it.
    """
    return _CREDENTIAL_PATTERN.sub(r"\1***@", url)


@dataclass(frozen=True)
class DiscoveredDevice:
    """One device that answered a WS-Discovery probe."""

    xaddr: str
    """Service address from the probe response — where to send SOAP requests."""

    host: str
    scopes: tuple[str, ...] = field(default_factory=tuple)


def _parse_probe_match(xml_bytes: bytes) -> DiscoveredDevice | None:
    """Extract the XAddrs and host from a WS-Discovery ProbeMatch response.

    Deliberately tolerant of namespace-prefix variation between vendors
    (some use ``d:``, some ``wsdd:``, and so on) by matching on local tag
    name rather than requiring an exact prefix.
    """
    import xml.etree.ElementTree as ElementTree

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None

    xaddr_text = None
    scopes: list[str] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "XAddrs" and elem.text:
            xaddr_text = elem.text.strip().split()[0]
        elif tag == "Scopes" and elem.text:
            scopes = elem.text.strip().split()

    if not xaddr_text:
        return None
    host = xaddr_text.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    return DiscoveredDevice(xaddr=xaddr_text, host=host, scopes=tuple(scopes))


def ws_discover(timeout_s: float = _WS_DISCOVERY_TIMEOUT_S) -> list[DiscoveredDevice]:
    """Send one WS-Discovery Probe and collect ProbeMatch responses.

    Multicast, so this reaches every ONVIF device on the local network
    segment (not routed across subnets — WS-Discovery is link-local by
    design, which is also why there is no ``--subnet`` restriction to
    enforce here beyond "whatever segment this host is on").
    """
    message = _PROBE_TEMPLATE.format(message_id=uuid.uuid4())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
    sock.settimeout(timeout_s)

    devices: dict[str, DiscoveredDevice] = {}
    try:
        sock.sendto(message.encode("utf-8"), (WS_DISCOVERY_MULTICAST_ADDR, WS_DISCOVERY_PORT))
        while True:
            try:
                data, _addr = sock.recvfrom(65536)
            except socket.timeout:
                break
            device = _parse_probe_match(data)
            if device is not None:
                devices[device.xaddr] = device
    finally:
        sock.close()
    return list(devices.values())


@dataclass(frozen=True)
class StreamProfile:
    """One ONVIF media profile's stream characteristics."""

    token: str
    name: str
    resolution: tuple[int, int] | None
    codec: str | None
    fps: float | None
    rtsp_uri_masked: str
    rtsp_uri: str = field(repr=False)
    """Unmasked URI, kept only in memory for the caller that needs it to
    actually connect. ``rtsp_uri_masked`` is what gets printed or written
    to disk."""

    def as_public_dict(self) -> dict[str, Any]:
        """Manifest-safe view: NEVER includes the unmasked URI."""
        return {
            "token": self.token,
            "name": self.name,
            "resolution": list(self.resolution) if self.resolution else None,
            "codec": self.codec,
            "fps": self.fps,
            "rtsp_uri": self.rtsp_uri_masked,
        }


async def query_stream_profiles(
    host: str, port: int, user: str | None, password: str | None
) -> list[StreamProfile]:
    """ONVIF Media service: GetProfiles then GetStreamUri per profile."""
    from onvif import ONVIFCamera

    camera = ONVIFCamera(host, port, user, password)
    await camera.update_xaddrs()
    media = await camera.create_media_service()
    profiles = await media.GetProfiles()

    results = []
    for profile in profiles:
        stream_setup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"},
        }
        uri_response = await media.GetStreamUri(
            {"StreamSetup": stream_setup, "ProfileToken": profile.token}
        )
        uri = uri_response.Uri

        video_encoder = getattr(profile, "VideoEncoderConfiguration", None)
        resolution = None
        codec = None
        fps = None
        if video_encoder is not None:
            res = getattr(video_encoder, "Resolution", None)
            if res is not None:
                resolution = (int(res.Width), int(res.Height))
            codec = getattr(video_encoder, "Encoding", None)
            rate_control = getattr(video_encoder, "RateControl", None)
            if rate_control is not None:
                fps = float(getattr(rate_control, "FrameRateLimit", 0)) or None

        results.append(
            StreamProfile(
                token=profile.token,
                name=str(getattr(profile, "Name", profile.token)),
                resolution=resolution,
                codec=str(codec) if codec else None,
                fps=fps,
                rtsp_uri=uri,
                rtsp_uri_masked=mask_credentials(uri),
            )
        )
    await camera.close()
    return results


class ManualEntryUnavailable(RuntimeError):
    """Raised when the interactive fallback has no terminal to prompt on."""


def prompt_manual_entry() -> tuple[str, str | None]:
    """Interactive fallback when a device does not answer ONVIF at all.

    Returns ``(rtsp_url, name)``. Used when WS-Discovery finds nothing, or
    when a found device's Media service query fails (some consumer cameras
    advertise ONVIF discovery but implement only a subset of the spec).

    Raises:
        ManualEntryUnavailable: stdin has no line to read — a dry run or a
            CI/cron invocation with no camera on the segment and no
            ``--manual --rtsp-url`` given. This used to surface as a bare
            ``EOFError`` traceback, which is not "reporting honestly" by
            this module's own exit-code contract (0 devices found is not a
            failure); it is a crash standing in for one.
    """
    print(
        "\nNo ONVIF response, or ONVIF query failed. Enter the RTSP URL by "
        "hand (check the camera's own admin page or manual for the exact "
        "path — this script does not guess a vendor's URL pattern).",
        file=sys.stderr,
    )
    try:
        url = input("RTSP URL: ").strip()
        name = input("Label for this stream (optional): ").strip() or None
    except EOFError:
        raise ManualEntryUnavailable(
            "stdin has no RTSP URL to read (not a terminal, and neither "
            "--manual nor --rtsp-url was given). Treating this the same as "
            "zero devices found, not as a failure."
        )
    return url, name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subnet",
        default=None,
        help="informational only — WS-Discovery is link-local multicast and "
        "always covers the whole local segment; this is not a filter.",
    )
    parser.add_argument("--onvif-port", type=int, default=80)
    parser.add_argument("--user", default=None, help="ONVIF device credentials")
    parser.add_argument("--password", default=None)
    parser.add_argument("--manual", action="store_true", help="skip discovery, prompt for RTSP URL directly")
    parser.add_argument("--host", default=None, help="with --manual: device host, for the manifest only")
    parser.add_argument("--rtsp-url", default=None, help="with --manual: skip the interactive prompt")
    parser.add_argument("--out", type=Path, default=Path("outputs/day12/discovered_cameras.json"))
    args = parser.parse_args(argv)

    if args.subnet:
        try:
            ipaddress.ip_network(args.subnet, strict=False)
        except ValueError as exc:
            print(f"--subnet {args.subnet!r} is not a valid CIDR: {exc}", file=sys.stderr)
            return 1

    manifest: dict[str, Any] = {"devices": []}

    if args.manual:
        url = args.rtsp_url
        name = None
        if url is None:
            try:
                url, name = prompt_manual_entry()
            except ManualEntryUnavailable as exc:
                print(f"{exc}", file=sys.stderr)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
                print(f"\nwritten to {args.out} (0 devices)")
                return 0
        print(f"manual entry: {mask_credentials(url)}")
        manifest["devices"].append(
            {
                "host": args.host,
                "discovery": "manual",
                "profiles": [{"name": name, "rtsp_uri": mask_credentials(url)}],
            }
        )
    else:
        print(f"WS-Discovery probe (multicast {WS_DISCOVERY_MULTICAST_ADDR}:{WS_DISCOVERY_PORT})...")
        found = ws_discover()
        print(f"{len(found)} device(s) responded")
        if not found:
            print(
                "No devices responded to WS-Discovery. This can mean: no "
                "ONVIF camera on this network segment, WS-Discovery blocked "
                "by a switch/VLAN (multicast is commonly filtered), or the "
                "camera requires manual configuration. Falling back to "
                "manual entry.",
                file=sys.stderr,
            )
            try:
                url, name = prompt_manual_entry()
                manifest["devices"].append(
                    {"host": None, "discovery": "manual-fallback", "profiles": [{"name": name, "rtsp_uri": mask_credentials(url)}]}
                )
            except ManualEntryUnavailable as exc:
                # 0 devices found and no terminal to prompt on: still "ran
                # clean and reported honestly" per this module's own exit
                # contract, not a failure to raise out of.
                print(f"{exc}", file=sys.stderr)
        for device in found:
            print(f"  {device.host}  xaddr={device.xaddr}")
            try:
                profiles = asyncio.run(
                    query_stream_profiles(device.host, args.onvif_port, args.user, args.password)
                )
            except Exception as exc:  # noqa: BLE001 — any ONVIF/network failure falls back
                print(f"    ONVIF query failed ({exc}); falling back to manual entry", file=sys.stderr)
                try:
                    url, name = prompt_manual_entry()
                except ManualEntryUnavailable as exc2:
                    print(f"    {exc2}", file=sys.stderr)
                    continue
                manifest["devices"].append(
                    {
                        "host": device.host,
                        "discovery": "ws-discovery+manual-fallback",
                        "profiles": [{"name": name, "rtsp_uri": mask_credentials(url)}],
                    }
                )
                continue

            for profile in profiles:
                res = f"{profile.resolution[0]}x{profile.resolution[1]}" if profile.resolution else "?"
                print(f"    {profile.name:20} {res:12} {profile.codec or '?':8} {profile.fps or '?'}fps  {profile.rtsp_uri_masked}")
            manifest["devices"].append(
                {
                    "host": device.host,
                    "discovery": "ws-discovery",
                    "profiles": [p.as_public_dict() for p in profiles],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwritten to {args.out} (credentials masked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
