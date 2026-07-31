"""Tests for the fetch/verify tooling.

No network: payloads and license texts are local files served via ``file://``
URLs. What is under test is the gate order (verify before fetch), the
content-addressed store's idempotence, and the sha-mismatch refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_dataset  # noqa: E402

from src.data.registry import DatasetRegistry  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary registry plus a temporary data root."""
    registry = tmp_path / "datasets.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {"name": "unverified-set", "lane": "R"},
                    {
                        "name": "verified-set",
                        "lane": "R",
                        "license_snapshot": {
                            "url": "https://example.org/license",
                            "verified_date": "2026-07-31",
                            "text_sha256": "b" * 64,
                            "verified_by": "test",
                        },
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(fetch_dataset, "REGISTRY_PATH", registry)
    monkeypatch.setenv("IRON_PATHS__DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def payload_for(tmp_path: Path, content: bytes = b"dataset payload") -> tuple[str, str]:
    """Write a local payload and return its file:// URL and sha256."""
    import hashlib

    payload = tmp_path / "payload.tar"
    payload.write_bytes(content)
    return payload.as_uri(), hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Gate order
# ---------------------------------------------------------------------------


def test_fetch_refuses_without_a_license_snapshot(sandbox: Path) -> None:
    url, sha = payload_for(sandbox)
    code = fetch_dataset.main(["unverified-set", "--url", url, "--expected-sha", sha])
    assert code == 2
    assert not (sandbox / "data" / "raw" / "unverified-set").exists()


def test_fetch_refuses_blocked_names(sandbox: Path) -> None:
    url, sha = payload_for(sandbox)
    code = fetch_dataset.main(["DukeMTMC", "--url", url, "--expected-sha", sha])
    assert code == 2


def test_verify_license_requires_the_attestation(sandbox: Path) -> None:
    """Without --i-have-read-it nothing is recorded; the flag is a signature."""
    license_text = sandbox / "LICENSE.txt"
    license_text.write_text("research use only")
    code = fetch_dataset.main(
        [
            "unverified-set",
            "--verify-license",
            "--license-url",
            license_text.as_uri(),
        ]
    )
    assert code == 2
    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    assert registry.get("unverified-set").license_snapshot is None


def test_verify_then_fetch_end_to_end(sandbox: Path) -> None:
    """The whole intended path: read license -> snapshot -> fetch -> store."""
    import hashlib

    license_text = sandbox / "LICENSE.txt"
    license_text.write_text("research use only, v1")

    code = fetch_dataset.main(
        [
            "unverified-set",
            "--verify-license",
            "--license-url",
            license_text.as_uri(),
            "--i-have-read-it",
            "--verified-class",
            "research-only",
            "--verified-by",
            "test-human",
        ]
    )
    assert code == 0

    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    snapshot = registry.get("unverified-set").license_snapshot
    assert snapshot is not None
    assert snapshot.text_sha256 == hashlib.sha256(b"research use only, v1").hexdigest()
    assert snapshot.verified_by == "test-human"

    url, sha = payload_for(sandbox)
    code = fetch_dataset.main(["unverified-set", "--url", url, "--expected-sha", sha])
    assert code == 0
    stored = sandbox / "data" / "raw" / "unverified-set" / sha / "payload.tar"
    assert stored.exists()

    # And the registry recorded what landed.
    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    assert registry.get("unverified-set").content_sha == sha


# ---------------------------------------------------------------------------
# Content-addressed store behaviour
# ---------------------------------------------------------------------------


def test_fetch_is_idempotent(sandbox: Path) -> None:
    """A second fetch of the same sha is a no-op, not a re-download."""
    url, sha = payload_for(sandbox)
    assert (
        fetch_dataset.main(["verified-set", "--url", url, "--expected-sha", sha]) == 0
    )

    stored = sandbox / "data" / "raw" / "verified-set" / sha / "payload.tar"
    marker = stored.read_bytes()
    stored.write_bytes(marker + b" locally modified")

    # Second run must not touch the stored copy — the sha directory existing is
    # the whole check, because the name IS the content hash.
    assert (
        fetch_dataset.main(["verified-set", "--url", url, "--expected-sha", sha]) == 0
    )
    assert stored.read_bytes().endswith(b" locally modified")


def test_sha_mismatch_deletes_and_refuses(sandbox: Path) -> None:
    """A payload that does not match its published hash must not be installed."""
    url, _ = payload_for(sandbox, b"actual bytes")
    wrong_sha = "c" * 64

    code = fetch_dataset.main(
        ["verified-set", "--url", url, "--expected-sha", wrong_sha]
    )
    assert code == 1

    dataset_root = sandbox / "data" / "raw" / "verified-set"
    assert not (dataset_root / wrong_sha).exists(), "mismatched payload was installed"
    assert not list(dataset_root.glob("*.partial")), "mismatched payload was kept"


def test_relicense_is_surfaced_not_silent(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-verifying against changed license text must call out the change."""
    license_text = sandbox / "LICENSE.txt"

    license_text.write_text("terms v1")
    fetch_dataset.main(
        [
            "unverified-set",
            "--verify-license",
            "--license-url",
            license_text.as_uri(),
            "--i-have-read-it",
        ]
    )

    license_text.write_text("terms v2 — now with a commercial restriction")
    fetch_dataset.main(
        [
            "unverified-set",
            "--verify-license",
            "--license-url",
            license_text.as_uri(),
            "--i-have-read-it",
        ]
    )
    output = capsys.readouterr().out
    assert "CHANGED" in output
