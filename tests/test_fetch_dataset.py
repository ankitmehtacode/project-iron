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


HF_COMMIT_SHA = "a" * 40
HF_RAW_URL = f"https://huggingface.co/datasets/org/repo/raw/{HF_COMMIT_SHA}/README.md"


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
                    {
                        "name": "hf-unverified",
                        "lane": "R",
                        "hosting": "huggingface",
                    },
                    {
                        "name": "hf-verified",
                        "lane": "R",
                        "hosting": "huggingface",
                        "license_snapshot": {
                            "url": HF_RAW_URL,
                            "verified_date": "2026-09-08",
                            "text_sha256": "c" * 64,
                            "verified_by": "test",
                            "resolved_commit_sha": HF_COMMIT_SHA,
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


# ---------------------------------------------------------------------------
# Day 39: hosting: huggingface — the SSR-volatility bug, reproduced and fixed
# ---------------------------------------------------------------------------


def test_hashing_a_rendered_page_whole_is_volatile_by_construction() -> None:
    """Reproduces the exact mechanism found live on CHIRLA, 2026-09-08,
    without needing a real HuggingFace fetch: a rendered dataset-card page
    embeds a per-request-volatile JSON blob (a live counter, a timestamp)
    around otherwise-stable license text. The OLD approach — hash whatever
    the URL returns, whole — produces two different hashes for the SAME
    license text on two different "requests". This is the regression that
    would have caught last night's bug before it reached a terminal.
    """
    import hashlib

    license_text = b'"license": "CC-BY-4.0", "outOfScopeUse": "surveillance..."'
    page_fetch_one = (
        b'{"lastModified": "2026-09-08T21:03:00Z", "likes": 41, ' + license_text + b"}"
    )
    page_fetch_two = (
        b'{"lastModified": "2026-09-08T21:07:00Z", "likes": 42, ' + license_text + b"}"
    )

    assert license_text in page_fetch_one and license_text in page_fetch_two
    old_hash_one = hashlib.sha256(page_fetch_one).hexdigest()
    old_hash_two = hashlib.sha256(page_fetch_two).hexdigest()
    assert old_hash_one != old_hash_two, "the bug this test pins down did not reproduce"


def test_verify_license_refuses_a_rendered_hf_page(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """STRUCTURAL: hosting: huggingface entries refuse the rendered
    dataset-card URL shape outright, rather than hashing SSR noise. The
    resolved-sha hint in the refusal message must not require live network —
    it degrades gracefully when resolution itself fails."""
    monkeypatch.setattr(
        fetch_dataset,
        "_resolve_hf_commit_sha",
        lambda repo_id, revision, token: (_ for _ in ()).throw(
            RuntimeError("no network in sandbox")
        ),
    )
    code = fetch_dataset.main(
        [
            "hf-unverified",
            "--verify-license",
            "--license-url",
            "https://huggingface.co/datasets/org/repo",
            "--i-have-read-it",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "volatile" in err.lower()
    assert "/raw/" in err

    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    assert registry.get("hf-unverified").license_snapshot is None


def test_verify_license_pins_main_to_a_resolved_commit_and_is_stable(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NEW raw-endpoint approach: resolve "main" to a real commit sha
    once, hash the pinned raw file — re-verifying against the same commit
    reproduces the identical hash, unlike hashing the rendered page."""
    stable_text = b'"license": "CC-BY-4.0", "outOfScopeUse": "surveillance..."'
    resolve_calls = []

    def fake_resolve(repo_id: str, revision: str, token: str | None) -> str:
        resolve_calls.append(revision)
        return HF_COMMIT_SHA

    monkeypatch.setattr(fetch_dataset, "_resolve_hf_commit_sha", fake_resolve)
    monkeypatch.setattr(
        fetch_dataset,
        "_fetch_hf_raw_file",
        lambda repo_id, commit_sha, path, token: stable_text,
    )

    url = "https://huggingface.co/datasets/org/repo/raw/main/README.md"
    for _ in range(2):
        code = fetch_dataset.main(
            [
                "hf-unverified",
                "--verify-license",
                "--license-url",
                url,
                "--i-have-read-it",
                "--verified-class",
                "CC-BY-4.0",
            ]
        )
        assert code == 0

    assert resolve_calls == ["main", "main"]
    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    snapshot = registry.get("hf-unverified").license_snapshot
    assert snapshot is not None
    assert snapshot.resolved_commit_sha == HF_COMMIT_SHA
    assert snapshot.text_sha256 == __import__("hashlib").sha256(stable_text).hexdigest()


def test_verify_license_respects_an_already_pinned_commit_sha(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A URL that already names a full 40-hex commit sha is a deliberate
    historical pin — it must not be silently re-pointed at whatever "main"
    resolves to right now."""

    def fail_if_called(repo_id: str, revision: str, token: str | None) -> str:
        raise AssertionError("must not resolve when the URL already pins a commit")

    monkeypatch.setattr(fetch_dataset, "_resolve_hf_commit_sha", fail_if_called)
    monkeypatch.setattr(
        fetch_dataset, "_fetch_hf_raw_file", lambda *a, **k: b"pinned content"
    )

    code = fetch_dataset.main(
        [
            "hf-unverified",
            "--verify-license",
            "--license-url",
            HF_RAW_URL,
            "--i-have-read-it",
        ]
    )
    assert code == 0
    registry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH)
    assert (
        registry.get("hf-unverified").license_snapshot.resolved_commit_sha
        == HF_COMMIT_SHA
    )


# ---------------------------------------------------------------------------
# Day 39: hosting: huggingface — the HuggingFace-native fetch path
# ---------------------------------------------------------------------------


def _stub_hf_config(
    monkeypatch: pytest.MonkeyPatch, content_by_path: dict[str, bytes]
) -> None:
    """Wire the four seams fetch_huggingface calls, with no live network:
    commit resolution, card metadata, matched files (with authoritative
    hashes precomputed from content_by_path), and raw-file bytes."""
    import hashlib

    monkeypatch.setattr(
        fetch_dataset,
        "_resolve_hf_commit_sha",
        lambda repo_id, revision, token: HF_COMMIT_SHA,
    )
    monkeypatch.setattr(
        fetch_dataset,
        "_hf_card_data",
        lambda repo_id, commit_sha, token: {
            "configs": [
                {
                    "config_name": "scenario_a",
                    "data_files": [{"split": "train", "path": "scenario_a/*"}],
                }
            ],
            "dataset_info": [
                {
                    "config_name": "scenario_a",
                    "splits": [{"name": "train", "num_examples": 2, "num_bytes": 42}],
                }
            ],
        },
    )
    monkeypatch.setattr(
        fetch_dataset,
        "_hf_matching_files",
        lambda repo_id, commit_sha, patterns, token: [
            {
                "path": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_sha1": None,
            }
            for path, content in content_by_path.items()
        ],
    )
    monkeypatch.setattr(
        fetch_dataset,
        "_fetch_hf_raw_file",
        lambda repo_id, commit_sha, path, token: content_by_path[path],
    )


def test_hf_fetch_requires_a_config_name(sandbox: Path) -> None:
    code = fetch_dataset.main(["hf-verified"])
    assert code == 2


def test_hf_fetch_downloads_verifies_and_writes_a_manifest(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_hf_config(monkeypatch, {"scenario_a/train.jsonl": b"row one\nrow two\n"})

    code = fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"])
    assert code == 0

    dest = sandbox / "data" / "raw" / "hf-verified" / HF_COMMIT_SHA / "scenario_a"
    assert (dest / "scenario_a" / "train.jsonl").read_bytes() == b"row one\nrow two\n"

    import json

    manifest = json.loads((dest / "_manifest.json").read_text())
    assert manifest["method"] == "huggingface_configs_v1"
    assert manifest["commit_sha"] == HF_COMMIT_SHA
    assert manifest["config"] == "scenario_a"
    assert manifest["splits"] == [{"name": "train", "num_examples": 2, "num_bytes": 42}]


def test_hf_fetch_is_idempotent(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    _stub_hf_config(monkeypatch, {"scenario_a/train.jsonl": b"row one\n"})
    real_fetch = fetch_dataset._fetch_hf_raw_file

    def counting_fetch(repo_id, commit_sha, path, token):
        calls.append(path)
        return real_fetch(repo_id, commit_sha, path, token)

    monkeypatch.setattr(fetch_dataset, "_fetch_hf_raw_file", counting_fetch)

    assert fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"]) == 0
    assert len(calls) == 1

    assert fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"]) == 0
    assert len(calls) == 1, "a second run must not re-download an existing config"


def test_hf_fetch_hash_mismatch_deletes_and_refuses(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest's authoritative hash comes from HuggingFace's own API
    (via _hf_matching_files); if what actually lands on disk hashes
    differently, refuse and delete — never install a payload that doesn't
    match, exactly like the tarball path."""
    _stub_hf_config(monkeypatch, {"scenario_a/train.jsonl": b"expected content"})
    monkeypatch.setattr(
        fetch_dataset,
        "_fetch_hf_raw_file",
        lambda repo_id, commit_sha, path, token: b"corrupted in transit",
    )

    code = fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"])
    assert code == 1

    dataset_root = sandbox / "data" / "raw" / "hf-verified" / HF_COMMIT_SHA
    assert not (dataset_root / "scenario_a").exists()
    assert not list(
        dataset_root.glob("scenario_a.partial*")
    ), "mismatched payload was kept"


def test_hf_fetch_unknown_config_name_fails_clearly(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_hf_config(monkeypatch, {"scenario_a/train.jsonl": b"row one\n"})
    code = fetch_dataset.main(["hf-verified", "--hf-config", "not_a_real_config"])
    assert code == 2
    dataset_root = sandbox / "data" / "raw" / "hf-verified" / HF_COMMIT_SHA
    assert not (dataset_root / "not_a_real_config").exists()


def test_hf_fetch_refuses_without_a_license_snapshot(sandbox: Path) -> None:
    """The existing gate order (verify before fetch) is untouched by
    routing on hosting — require_fetchable runs before hosting is even
    consulted."""
    code = fetch_dataset.main(["hf-unverified", "--hf-config", "scenario_a"])
    assert code == 2


def test_git_blob_sha1_matches_gits_own_algorithm() -> None:
    """Pins the non-LFS authoritative-hash algorithm against a known git
    blob hash (`git hash-object` for the empty string is
    e69de29bb2d1d6434b8b29ae775ad8c2e48c5391)."""
    assert (
        fetch_dataset._git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    )
