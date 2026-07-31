"""Fetch a registered dataset, or record the license verification it needs.

Two subcommands, in the order they must happen:

    python scripts/fetch_dataset.py --verify-license MEVA \\
        --license-url https://... --i-have-read-it
    python scripts/fetch_dataset.py MEVA --url https://.../meva.tar \\
        --expected-sha <sha256>

The order is enforced, not suggested: fetching consults the registry, and the
registry refuses any entry without a human-recorded license snapshot. The
``--i-have-read-it`` flag is deliberately awkward — it is a signature, and the
recorded snapshot (URL, date, text hash, who) is the diff-able evidence of what
was signed.

Downloads are content-addressed under ``data/raw/<name>/<sha256>/``:

- **Idempotent**: a payload whose sha directory already exists is not fetched
  again, and re-running after success is a no-op.
- **Resumable**: partial downloads land in a ``.partial`` file and resume with
  a Range request where the server supports it.
- **Refuses on mismatch**: a payload whose hash differs from ``--expected-sha``
  is deleted, not installed. A dataset that does not match its published hash
  is not "close enough" — it is a different dataset, possibly a tampered one.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.config import IronConfig
from src.data.registry import (
    DatasetRegistry,
    LicenseSnapshot,
    RegistryError,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"

CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_registry_yaml(name: str, updates: dict[str, Any]) -> None:
    """Apply field updates to one entry in configs/datasets.yaml.

    The YAML stays the single at-rest source; this rewrites the one entry
    rather than regenerating the file, so hand-written comments survive.
    """
    payload = yaml.safe_load(REGISTRY_PATH.read_text())
    for raw in payload.get("datasets", []):
        if normalise(str(raw.get("name", ""))) == normalise(name):
            raw.update(updates)
            break
    else:
        raise RegistryError(f"{name!r} not found in {REGISTRY_PATH}")
    REGISTRY_PATH.write_text(
        "# Regenerated in place by scripts/fetch_dataset.py — comments above\n"
        "# individual entries may have been altered; the registry semantics\n"
        "# live in src/data/registry.py.\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    )


def verify_license(args: argparse.Namespace) -> int:
    """Record a license snapshot for one dataset.

    Fetches the license text from the URL the human provides, hashes it, and
    stores URL + date + hash + identity in the registry. The human attests
    with ``--i-have-read-it``; the tool records, it does not judge — deciding
    what the text permits is the human's job, recorded in
    ``--verified-class``.
    """
    registry = DatasetRegistry.load(REGISTRY_PATH)
    entry = registry.get(args.name)  # raises for blocked/unknown

    if not args.i_have_read_it:
        print(
            "Refusing: pass --i-have-read-it after actually reading the "
            "license text at the URL. This flag is a recorded signature, not "
            "a formality — the snapshot will carry your username and today's "
            "date.",
            file=sys.stderr,
        )
        return 2

    print(f"Fetching license text from {args.license_url} ...")
    try:
        with urllib.request.urlopen(args.license_url, timeout=60) as response:
            text = response.read()
    except (urllib.error.URLError, OSError) as exc:
        print(f"Could not fetch the license text: {exc}", file=sys.stderr)
        return 1

    snapshot = LicenseSnapshot(
        url=args.license_url,
        verified_date=date.today(),
        text_sha256=hashlib.sha256(text).hexdigest(),
        verified_by=args.verified_by or getpass.getuser(),
        verified_class=args.verified_class,
    )

    previous = entry.license_snapshot
    if previous is not None and previous.text_sha256 != snapshot.text_sha256:
        print(
            "NOTE: the license text hash CHANGED since the previous snapshot "
            f"({previous.verified_date}: {previous.text_sha256[:12]}... -> "
            f"{snapshot.text_sha256[:12]}...). The upstream license was "
            "edited; re-read it with that in mind."
        )

    _update_registry_yaml(
        entry.name,
        {"license_snapshot": snapshot.model_dump(mode="json")},
    )
    print(
        f"Snapshot recorded for {entry.name}: {snapshot.text_sha256[:16]}... "
        f"verified {snapshot.verified_date} by {snapshot.verified_by}"
        + (f" as {snapshot.verified_class!r}" if snapshot.verified_class else "")
    )
    if entry.hypothesis_class and args.verified_class:
        if args.verified_class.strip().lower() not in entry.hypothesis_class.lower():
            print(
                f"  hypothesis was: {entry.hypothesis_class!r} — the verified "
                "class wins; update the lane in configs/datasets.yaml if the "
                "verified terms change what is permitted."
            )
    return 0


def _download(url: str, destination: Path) -> None:
    """Stream a URL to ``destination``, resuming a partial file if present."""
    existing = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        print(f"  resuming at byte {existing:,}")

    with urllib.request.urlopen(request, timeout=120) as response:
        # 206 means the server honoured the Range; anything else on a resume
        # means it did not, and appending would corrupt the payload.
        mode = "ab" if existing and response.status == 206 else "wb"
        if existing and response.status != 206:
            print("  server ignored the Range request; restarting from zero")
        with open(destination, mode) as handle:
            shutil.copyfileobj(response, handle, CHUNK_BYTES)


def fetch(args: argparse.Namespace) -> int:
    """Registry check -> snapshot check -> download -> sha verify -> store."""
    registry = DatasetRegistry.load(REGISTRY_PATH)
    try:
        entry = registry.require_fetchable(args.name)
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    config = IronConfig.load()
    dataset_root = config.paths.resolved_data_dir / "raw" / entry.name

    final_dir = dataset_root / args.expected_sha
    if final_dir.exists():
        print(
            f"{entry.name} already present at {final_dir} — content-addressed "
            "store makes re-fetching a no-op."
        )
        return 0

    dataset_root.mkdir(parents=True, exist_ok=True)
    partial = dataset_root / f"{args.expected_sha}.partial"
    print(f"Downloading {args.url}")
    try:
        _download(args.url, partial)
    except (urllib.error.URLError, OSError) as exc:
        print(
            f"Download failed: {exc}. The partial file is kept at {partial} "
            "and the next run will resume it.",
            file=sys.stderr,
        )
        return 1

    print("Verifying sha256 ...")
    actual = sha256_file(partial)
    if actual != args.expected_sha:
        partial.unlink()
        print(
            f"SHA MISMATCH: expected {args.expected_sha}, got {actual}. The "
            "payload was deleted, not installed — a dataset that does not "
            "match its published hash is a different dataset, possibly a "
            "tampered one. Re-check the hash source before retrying.",
            file=sys.stderr,
        )
        return 1

    final_dir.mkdir(parents=True, exist_ok=True)
    payload_name = Path(urllib.parse.urlparse(args.url).path).name or "payload"
    shutil.move(str(partial), final_dir / payload_name)

    _update_registry_yaml(entry.name, {"content_sha": actual})
    print(f"Stored at {final_dir / payload_name}")
    print(f"Registry updated: content_sha = {actual[:16]}...")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="dataset name as registered")
    parser.add_argument("--url", help="payload URL (fetch mode)")
    parser.add_argument("--expected-sha", help="published sha256 of the payload")
    parser.add_argument(
        "--verify-license",
        action="store_true",
        help="record a license snapshot instead of fetching",
    )
    parser.add_argument("--license-url", help="URL of the license text")
    parser.add_argument(
        "--i-have-read-it",
        action="store_true",
        help="attestation that a human read the license text at --license-url",
    )
    parser.add_argument(
        "--verified-class",
        default="",
        help="license class as determined from the text you read",
    )
    parser.add_argument("--verified-by", default="", help="who read it")
    args = parser.parse_args(argv)

    try:
        if args.verify_license:
            if not args.license_url:
                parser.error("--verify-license requires --license-url")
            return verify_license(args)
        if not args.url or not args.expected_sha:
            parser.error("fetch mode requires --url and --expected-sha")
        return fetch(args)
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
