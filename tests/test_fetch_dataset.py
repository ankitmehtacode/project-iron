"""Tests for the fetch/verify tooling.

No network: payloads and license texts are local files served via ``file://``
URLs. What is under test is the gate order (verify before fetch), the
content-addressed store's idempotence, and the sha-mismatch refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_dataset  # noqa: E402

from src.data.registry import DatasetRegistry, RegistryError  # noqa: E402


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
                    {
                        "name": "hf-mixed-lanes",
                        "lane": "S",
                        "hosting": "huggingface",
                        "license_snapshot": {
                            "url": HF_RAW_URL,
                            "verified_date": "2026-10-14",
                            "text_sha256": "d" * 64,
                            "verified_by": "test",
                            "resolved_commit_sha": HF_COMMIT_SHA,
                        },
                        "fetch_deny_patterns": ["test/real_*/*"],
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
        "_fetch_hf_file",
        lambda repo_id, commit_sha, path, token, destination, on_progress=None: (
            destination.write_bytes(content_by_path[path])
        ),
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
    real_fetch = fetch_dataset._fetch_hf_file

    def counting_fetch(repo_id, commit_sha, path, token, destination, on_progress=None):
        calls.append(path)
        return real_fetch(repo_id, commit_sha, path, token, destination)

    monkeypatch.setattr(fetch_dataset, "_fetch_hf_file", counting_fetch)

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
        "_fetch_hf_file",
        lambda repo_id, commit_sha, path, token, destination, on_progress=None: (
            destination.write_bytes(b"corrupted in transit")
        ),
    )

    code = fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"])
    assert code == 1

    dataset_root = sandbox / "data" / "raw" / "hf-verified" / HF_COMMIT_SHA
    assert not (dataset_root / "scenario_a").exists()
    # The staging directory is kept for resume, but holds no file at all:
    # the mismatched bytes were deleted, not parked.
    assert not [
        f for f in dataset_root.rglob("*") if f.is_file()
    ], "mismatched payload was kept"


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


# ---------------------------------------------------------------------------
# Day 40: the Day-39 HF fetch path against CHIRLA's real repo shape
# ---------------------------------------------------------------------------
#
# Found live on bdager/CHIRLA@f657183: all 22 benchmark parquet files are
# Git-LFS. /raw/<sha>/<path> returns the ~130-byte LFS *pointer* ("version
# https://git-lfs.github.com/spec/v1 ..."), /resolve/<sha>/<path> returns the
# parquet ("PAR1..."). The Day-39 tests stubbed the download seam, so the
# endpoint choice was never exercised. Separately, listing recursed over the
# whole 10.9GB repo (annotations/, benchmark/, videos/) when every config's
# files sit under data/.


# The real download functions, captured before any test stubs them.
_REAL_DOWNLOADERS = {
    name: getattr(fetch_dataset, name)
    for name in ("_fetch_hf_raw_file", "_fetch_hf_file")
    if hasattr(fetch_dataset, name)
}


def test_hf_data_files_download_via_resolve_not_raw(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"PAR1 pretend parquet"
    _stub_hf_config(monkeypatch, {"scenario_a/train.parquet": content})
    for name, real in _REAL_DOWNLOADERS.items():
        monkeypatch.setattr(fetch_dataset, name, real)

    requested: list[str] = []

    class _Response:
        status = 200

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, *size: int) -> bytes:
            data, self._done = (b"" if getattr(self, "_done", False) else content), True
            return data

    def fake_urlopen(request: object, timeout: float = 0) -> _Response:
        requested.append(getattr(request, "full_url", str(request)))
        return _Response()

    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", fake_urlopen)

    assert fetch_dataset.main(["hf-verified", "--hf-config", "scenario_a"]) == 0
    assert requested == [
        "https://huggingface.co/datasets/org/repo/resolve/"
        f"{HF_COMMIT_SHA}/scenario_a/train.parquet"
    ]


def test_hf_listing_is_scoped_to_the_patterns_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_list_repo_tree(repo_id: str, **kwargs: object) -> list[object]:
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        fetch_dataset.huggingface_hub, "list_repo_tree", fake_list_repo_tree
    )
    fetch_dataset._hf_matching_files(
        "bdager/CHIRLA",
        HF_COMMIT_SHA,
        [
            "data/reid_reappearance_gallery.parquet/gallery-*",
            "data/reid_reappearance_query.parquet/query-*",
        ],
        None,
    )
    assert seen.get("path_in_repo") == "data"


def test_hf_fetch_pins_an_explicit_revision_and_records_main_too(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned, head = "c" * 40, "d" * 40
    _stub_hf_config(monkeypatch, {"scenario_a/train.jsonl": b"row\n"})
    monkeypatch.setattr(
        fetch_dataset,
        "_resolve_hf_commit_sha",
        lambda repo_id, revision, token: pinned if revision == pinned else head,
    )

    code = fetch_dataset.main(
        ["hf-verified", "--hf-config", "scenario_a", "--revision", pinned]
    )
    assert code == 0

    import json

    dest = sandbox / "data" / "raw" / "hf-verified" / pinned / "scenario_a"
    manifest = json.loads((dest / "_manifest.json").read_text())
    assert manifest["commit_sha"] == pinned
    assert manifest["requested_revision"] == pinned
    assert manifest["main_head_sha"] == head


# ---------------------------------------------------------------------------
# 2026-10-14: --hf-path for repos with no declared configs, and
# fetch_deny_patterns (PhysicalAI-SmartSpaces: synthetic repo, two real scenes)
# ---------------------------------------------------------------------------


def _stub_hf_paths(
    monkeypatch: pytest.MonkeyPatch,
    content_by_path: dict[str, bytes],
    card_data: object = None,
) -> list[str]:
    """Like _stub_hf_config, for a config-less card. The listing stub
    honours the requested patterns (fnmatch), so deny tests exercise the
    real interaction between what was asked for and what is denied.
    Returns the list every download-seam call is appended to."""
    import fnmatch
    import hashlib

    downloaded: list[str] = []
    monkeypatch.setattr(
        fetch_dataset,
        "_resolve_hf_commit_sha",
        lambda repo_id, revision, token: HF_COMMIT_SHA,
    )
    monkeypatch.setattr(
        fetch_dataset,
        "_hf_card_data",
        lambda repo_id, commit_sha, token: card_data or {"license": "cc-by-4.0"},
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
            if any(fnmatch.fnmatch(path, p) for p in patterns)
        ],
    )

    def fake_fetch(
        repo_id: str,
        commit_sha: str,
        path: str,
        token: object,
        destination: Path,
        on_progress: object = None,
    ) -> None:
        downloaded.append(path)
        destination.write_bytes(content_by_path[path])

    monkeypatch.setattr(fetch_dataset, "_fetch_hf_file", fake_fetch)
    return downloaded


_MIXED_REPO = {
    "train/scene_000/videos/cam_0.mp4": b"synthetic video",
    "train/scene_000/ground_truth.json": b"{}",
    "test/real_026/videos/cam_0.mp4": b"real people",
}


def _paths_dirs(sandbox: Path, name: str) -> list[Path]:
    root = sandbox / "data" / "raw" / name / HF_COMMIT_SHA
    return sorted(root.glob("paths-*")) if root.exists() else []


def test_hf_path_fetch_downloads_verifies_and_writes_a_manifest(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    _stub_hf_paths(monkeypatch, _MIXED_REPO)
    code = fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"])
    assert code == 0

    (dest,) = _paths_dirs(sandbox, "hf-mixed-lanes")
    assert not dest.name.endswith(".partial")
    assert (
        dest / "train/scene_000/videos/cam_0.mp4"
    ).read_bytes() == b"synthetic video"
    manifest = json.loads((dest / "_manifest.json").read_text())
    assert manifest["method"] == "huggingface_paths_v1"
    assert manifest["patterns"] == ["train/*"]
    assert manifest["commit_sha"] == HF_COMMIT_SHA
    assert sorted(f["path"] for f in manifest["files"]) == [
        "train/scene_000/ground_truth.json",
        "train/scene_000/videos/cam_0.mp4",
    ]


def test_hf_path_fetch_is_idempotent(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = _stub_hf_paths(monkeypatch, _MIXED_REPO)
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0
    first = len(downloaded)
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0
    assert len(downloaded) == first, "same patterns must not re-download"


def test_hf_path_fetch_hash_mismatch_deletes_and_refuses(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_hf_paths(monkeypatch, _MIXED_REPO)
    monkeypatch.setattr(
        fetch_dataset,
        "_fetch_hf_file",
        lambda *a, **k: a[4].write_bytes(b"corrupted in transit"),
    )
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 1
    root = sandbox / "data" / "raw" / "hf-mixed-lanes" / HF_COMMIT_SHA
    assert not [p for p in root.glob("paths-*") if not p.name.endswith(".partial")]
    assert not [f for f in root.rglob("*") if f.is_file()], "mismatched payload kept"


def test_hf_path_fetch_refuses_when_the_card_declares_configs(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = _stub_hf_paths(
        monkeypatch,
        _MIXED_REPO,
        card_data={"configs": [{"config_name": "a", "data_files": []}]},
    )
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 2
    assert downloaded == []


@pytest.mark.parametrize(
    "flags",
    [[], ["--hf-config", "scenario_a", "--hf-path", "train/*"]],
    ids=["neither", "both"],
)
def test_hf_fetch_requires_exactly_one_of_config_or_path(
    sandbox: Path, flags: list[str]
) -> None:
    assert fetch_dataset.main(["hf-mixed-lanes", *flags]) == 2


def test_fetching_a_denied_path_raises_before_any_download(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structural guarantee: a request that reaches a denied path
    raises, and nothing — not even the allowed files it also matched — is
    downloaded or installed."""
    downloaded = _stub_hf_paths(monkeypatch, _MIXED_REPO)
    args = argparse.Namespace(
        name="hf-mixed-lanes",
        hf_config=[],
        hf_path=["*"],
        revision="main",
        hf_token="",
    )
    with pytest.raises(RegistryError, match="denies"):
        fetch_dataset.fetch(args)
    assert downloaded == []
    assert _paths_dirs(sandbox, "hf-mixed-lanes") == []

    # Through the CLI the same refusal is exit code 2, still nothing fetched.
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "test/*"]) == 2
    assert downloaded == []


def test_fetching_a_denied_path_via_a_config_raises_too(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = _stub_hf_paths(
        monkeypatch,
        _MIXED_REPO,
        card_data={"configs": [{"config_name": "all", "data_files": [{"path": "*"}]}]},
    )
    args = argparse.Namespace(
        name="hf-mixed-lanes",
        hf_config=["all"],
        hf_path=[],
        revision="main",
        hf_token="",
    )
    with pytest.raises(RegistryError, match="denies"):
        fetch_dataset.fetch(args)
    assert downloaded == []


def test_hf_path_fetch_refuses_a_listing_that_escapes_staging(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = _stub_hf_paths(monkeypatch, {"../../escaped.bin": b"x"})
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "*"]) == 1
    assert downloaded == []
    assert not (sandbox / "data" / "raw" / "escaped.bin").exists()


def test_smartspaces_deny_patterns_cover_exactly_the_real_world_scenes() -> None:
    """Against the REAL registry entry: both real-world scenes are denied,
    the synthetic test scenes beside them are not."""
    entry = DatasetRegistry.load(fetch_dataset.REGISTRY_PATH).get(
        "PhysicalAI-SmartSpaces"
    )
    files = lambda *paths: [{"path": p} for p in paths]  # noqa: E731
    for real in (
        "MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_0000.mp4",
        "MTMC_Tracking_2026/test/Warehouse_027/calibration.json",
    ):
        with pytest.raises(RegistryError):
            fetch_dataset._refuse_denied_paths(files(real), entry.fetch_deny_patterns)
    fetch_dataset._refuse_denied_paths(
        files(
            "MTMC_Tracking_2026/test/Warehouse_025/videos/Camera_0000.mp4",
            "MTMC_Tracking_2026/train/Warehouse_000/ground_truth.json",
        ),
        entry.fetch_deny_patterns,
    )


@pytest.mark.parametrize(
    "sneaky",
    [
        "./test/real_026/videos/cam_0.mp4",
        "train//../test/real_026/videos/cam_0.mp4",
        "train/../test/real_026/videos/cam_0.mp4",
        "/test/real_026/videos/cam_0.mp4",
    ],
)
def test_non_normal_listing_paths_are_refused_before_any_download(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, sneaky: str
) -> None:
    """Each of these dodges a string match on the deny pattern but would be
    written into the denied directory; the listing is refused instead."""
    downloaded = _stub_hf_paths(monkeypatch, {sneaky: b"real people"})
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "*"]) == 2
    assert downloaded == []
    assert _paths_dirs(sandbox, "hf-mixed-lanes") == []


def test_deny_patterns_refuse_a_commit_other_than_the_verified_one(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deny list describes the verified commit; a moved main may hold
    real-world paths it never saw."""
    downloaded = _stub_hf_paths(monkeypatch, _MIXED_REPO)
    monkeypatch.setattr(
        fetch_dataset,
        "_resolve_hf_commit_sha",
        lambda repo_id, revision, token: "e" * 40,
    )
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 2
    assert downloaded == []


# ---------------------------------------------------------------------------
# 2026-10-14: resumable downloads. The first SmartSpaces fetch died on
# http.client.IncompleteRead 82MB into a 186MB video, after ~11GB had landed.
# ---------------------------------------------------------------------------


class _FakeHttp:
    """A urlopen stand-in serving ``content_by_path`` (matched by URL suffix),
    honouring Range with a 206, and failing according to a script of
    faults consumed one per request: "cut" drops the stream halfway with
    IncompleteRead, "503"/"404" raise that HTTPError, None serves cleanly."""

    def __init__(
        self, content_by_path: dict[str, bytes], faults: list[str | None] = ()
    ) -> None:
        self.content_by_path = content_by_path
        self.faults = list(faults)
        self.requests: list[tuple[str, str | None]] = []

    def __call__(self, request: object, timeout: float = 0) -> "_FakeResponse":
        import urllib.error

        url = request.full_url
        rng = request.get_header("Range")
        self.requests.append((url, rng))
        fault = self.faults.pop(0) if self.faults else None
        if fault in ("503", "404"):
            raise urllib.error.HTTPError(url, int(fault), "fault", {}, None)
        (content,) = [c for p, c in self.content_by_path.items() if url.endswith(p)]
        start = int(rng.removeprefix("bytes=").rstrip("-")) if rng else 0
        body = content[start:]
        return _FakeResponse(
            206 if rng else 200, body, len(body) // 2 if fault == "cut" else None
        )


class _FakeResponse:
    def __init__(self, status: int, body: bytes, cut_at: int | None) -> None:
        self.status, self._body, self._cut_at, self._pos = status, body, cut_at, 0

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        import http.client

        end = len(self._body) if size < 0 else self._pos + size
        if self._cut_at is not None:
            if self._pos >= self._cut_at:
                raise http.client.IncompleteRead(b"", len(self._body) - self._pos)
            end = min(end, self._cut_at)
        chunk, self._pos = self._body[self._pos : end], min(end, len(self._body))
        return chunk


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of sleeping through them."""
    recorded: list[float] = []
    monkeypatch.setattr(fetch_dataset, "_sleep", recorded.append)
    return recorded


_BLOB = bytes(range(256)) * 4  # 1024 bytes, every offset distinguishable


def test_download_resumes_with_range_after_the_stream_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _FakeHttp({"blob.bin": _BLOB}, faults=["cut"])
    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", server)

    dest = tmp_path / "blob.bin.partial"
    fetch_dataset._download("https://example.test/blob.bin", dest)

    assert dest.read_bytes() == _BLOB
    assert [rng for _, rng in server.requests] == [None, "bytes=512-"]
    assert sleeps == [1.0]


def test_download_retries_a_5xx_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _FakeHttp({"blob.bin": _BLOB}, faults=["503", "503"])
    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", server)

    dest = tmp_path / "blob.bin.partial"
    fetch_dataset._download("https://example.test/blob.bin", dest)

    assert dest.read_bytes() == _BLOB
    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0], "backoff must grow while nothing lands"


def test_download_does_not_retry_a_4xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    import urllib.error

    server = _FakeHttp({"blob.bin": _BLOB}, faults=["404"])
    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", server)

    with pytest.raises(urllib.error.HTTPError):
        fetch_dataset._download("https://example.test/blob.bin", tmp_path / "x")
    assert len(server.requests) == 1 and sleeps == []


def test_download_gives_up_when_nothing_lands_for_the_whole_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    import urllib.error

    budget = fetch_dataset.RETRY_ATTEMPTS
    server = _FakeHttp({"blob.bin": _BLOB}, faults=["503"] * (budget + 1))
    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", server)

    with pytest.raises(urllib.error.HTTPError):
        fetch_dataset._download("https://example.test/blob.bin", tmp_path / "x")
    assert len(server.requests) == budget + 1
    assert max(sleeps) <= fetch_dataset.RETRY_BACKOFF_CAP_SECONDS


def _real_hf_download(monkeypatch: pytest.MonkeyPatch, server: _FakeHttp) -> None:
    """Undo _stub_hf_paths' download stub so the real streaming path runs
    against ``server``."""
    monkeypatch.setattr(
        fetch_dataset, "_fetch_hf_file", _REAL_DOWNLOADERS["_fetch_hf_file"]
    )
    monkeypatch.setattr(fetch_dataset.urllib.request, "urlopen", server)


_VIDEO = "train/scene_000/videos/cam_0.mp4"


def test_a_killed_fetch_resumes_from_the_byte_offset_on_rerun(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """Run 1: the stream dies halfway, then the server stays down for the
    whole retry budget — exit 1, partial kept. Run 2 resumes at that byte
    with a Range request and installs the verified file."""
    _stub_hf_paths(monkeypatch, {_VIDEO: _BLOB})
    budget = fetch_dataset.RETRY_ATTEMPTS
    server = _FakeHttp({_VIDEO: _BLOB}, faults=["cut"] + ["503"] * budget)
    _real_hf_download(monkeypatch, server)

    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 1
    (staging,) = _paths_dirs(sandbox, "hf-mixed-lanes")
    assert staging.name.endswith(".partial")
    partial = staging / (_VIDEO + ".partial")
    assert partial.stat().st_size == 512

    server.requests.clear()
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0
    assert [rng for _, rng in server.requests] == ["bytes=512-"]
    (installed,) = _paths_dirs(sandbox, "hf-mixed-lanes")
    assert not installed.name.endswith(".partial")
    assert (installed / _VIDEO).read_bytes() == _BLOB
    assert not list(installed.rglob("*.partial"))


def test_a_hash_mismatch_after_resume_is_rejected(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """The resumed tail comes from a server now serving different bytes:
    the stitched file fails HuggingFace's hash, is deleted, and nothing is
    installed — resume must never launder a corrupt prefix or suffix."""
    _stub_hf_paths(monkeypatch, {_VIDEO: _BLOB})
    budget = fetch_dataset.RETRY_ATTEMPTS
    _real_hf_download(
        monkeypatch, _FakeHttp({_VIDEO: _BLOB}, faults=["cut"] + ["503"] * budget)
    )
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 1

    tampered = _BLOB[:512] + bytes(512)
    _real_hf_download(monkeypatch, _FakeHttp({_VIDEO: tampered}))
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 1

    (staging,) = _paths_dirs(sandbox, "hf-mixed-lanes")
    assert staging.name.endswith(".partial"), "a mismatched file was installed"
    assert not [f for f in staging.rglob("*") if f.is_file()]


def test_an_already_installed_file_is_skipped_on_rerun(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run 1 verifies ground_truth.json, then the video 404s. Run 2 asks
    the server for the video only, and says the other file was present."""
    repo = {"train/scene_000/ground_truth.json": b"{}", _VIDEO: _BLOB}
    _stub_hf_paths(monkeypatch, repo)
    _real_hf_download(monkeypatch, _FakeHttp(repo, faults=[None, "404"]))
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 1
    capsys.readouterr()

    server = _FakeHttp(repo)
    _real_hf_download(monkeypatch, server)
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0

    assert [url.rsplit("/", 3)[-3:] for url, _ in server.requests] == [
        ["scene_000", "videos", "cam_0.mp4"]
    ]
    out = capsys.readouterr().out
    assert "[1/2] present     train/scene_000/ground_truth.json" in out
    assert "[2/2] downloaded  train/scene_000/videos/cam_0.mp4" in out
    assert "0.00 / 0.00 GB" in out  # running byte count on every result line


def test_a_staged_file_that_fails_its_hash_is_downloaded_again(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _stub_hf_paths(monkeypatch, {_VIDEO: _BLOB})
    downloaded_once = _FakeHttp({_VIDEO: _BLOB}, faults=[None])
    _real_hf_download(monkeypatch, downloaded_once)
    root = sandbox / "data" / "raw" / "hf-mixed-lanes" / HF_COMMIT_SHA
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0
    # Turn the install back into a staging dir from an earlier run whose
    # file has since rotted.
    (installed,) = _paths_dirs(sandbox, "hf-mixed-lanes")
    staging = root / (installed.name + ".partial")
    installed.rename(staging)
    (staging / "_manifest.json").unlink()
    (staging / _VIDEO).write_bytes(b"bit rot")

    server = _FakeHttp({_VIDEO: _BLOB})
    _real_hf_download(monkeypatch, server)
    assert fetch_dataset.main(["hf-mixed-lanes", "--hf-path", "train/*"]) == 0
    assert len(server.requests) == 1
    assert (installed / _VIDEO).read_bytes() == _BLOB
