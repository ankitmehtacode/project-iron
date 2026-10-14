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
  a Range request where the server supports it; dropped connections,
  timeouts and 5xx are retried with backoff (:func:`_download`, shared by
  the tarball and HuggingFace paths).
- **Refuses on mismatch**: a payload whose hash differs from ``--expected-sha``
  is deleted, not installed. A dataset that does not match its published hash
  is not "close enough" — it is a different dataset, possibly a tampered one.

``hosting: huggingface`` entries (Day 39) route through a second, entirely
separate mechanism for both subcommands — see :func:`_verify_license_
huggingface` and :func:`fetch_huggingface`. This exists because a rendered
HuggingFace dataset-card page is not a static file: it embeds per-request-
volatile state (live download/like counts, a ``lastModified`` timestamp)
around otherwise-stable license text, so the naive "hash whatever the URL
returns" approach that works for a GitHub ``LICENSE`` file produces a
different hash on every fetch of the same unchanged license — confirmed live
on CHIRLA, 2026-09-08. The fix is structural: ``DatasetEntry.hosting`` routes
to the right strategy before any URL is touched, never a domain string
sniffed out of ``--license-url`` deep inside a function body.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import http.client
import json
import posixpath
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import huggingface_hub
import yaml

from src.config import IronConfig
from src.data.registry import (
    DatasetEntry,
    DatasetRegistry,
    LicenseSnapshot,
    RegistryError,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"

CHUNK_BYTES = 1 << 20

# Retry budget for one download. The counter resets whenever an attempt made
# progress, so a long file on a flaky link keeps going as long as bytes keep
# landing; only a server that repeatedly fails without delivering anything
# exhausts it. Found live 2026-10-14: http.client.IncompleteRead 82MB into a
# 186MB SmartSpaces video, which is not an OSError and crashed the run.
RETRY_ATTEMPTS = 6
RETRY_BACKOFF_CAP_SECONDS = 60.0
_TRANSIENT_ERRORS = (
    http.client.HTTPException,  # IncompleteRead, RemoteDisconnected, ...
    urllib.error.URLError,  # includes HTTPError; 4xx is re-raised below
    TimeoutError,
    ConnectionError,
)
# Everything a download can raise once retries are exhausted. IncompleteRead
# is an http.client.HTTPException, not an OSError — catching OSError alone is
# exactly how the 2026-10-14 fetch crashed instead of reporting.
DOWNLOAD_ERRORS = (OSError, http.client.HTTPException)
_sleep = time.sleep

# A rendered dataset page (huggingface.co/datasets/<repo>) optionally followed
# by /raw/<revision>/<path> or /blob/<revision>/<path> — the raw/blob group is
# absent for exactly the volatile page shape verify_license must refuse.
_HF_DATASET_URL_RE = re.compile(
    r"^https://huggingface\.co/datasets/(?P<repo_id>[^/]+/[^/]+?)"
    r"(?:/(?:raw|blob)/(?P<revision>[^/]+)/(?P<path>.+))?/?$"
)
_FULL_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_registry_yaml(name: str, updates: dict[str, Any]) -> None:
    """Apply field updates to one entry in configs/datasets.yaml, in place.

    Only the lines of each updated field change: they are spliced out and
    replaced by a fresh dump of the new value, or appended to the end of the
    entry for a field it did not have. A field already holding the new value
    is left alone, so a no-op write is byte-identical. Every other line --
    including every hand-written ``#`` comment -- is copied through untouched.

    Until Day 40 this round-tripped the whole file through ``safe_dump``,
    which drops comments; the a2c1837 write removed all 36 comment lines that
    way. So the result is checked before it is written: it must parse to
    exactly the old registry with ``updates`` applied, and keep every comment
    line, in order. Either check failing raises and leaves the file untouched.
    """
    text = REGISTRY_PATH.read_text()
    before = yaml.safe_load(text)
    entry = next(
        (
            raw
            for raw in before.get("datasets", [])
            if normalise(str(raw.get("name", ""))) == normalise(name)
        ),
        None,
    )
    if entry is None:
        raise RegistryError(f"{name!r} not found in {REGISTRY_PATH}")
    expected = yaml.safe_load(text)
    for raw in expected["datasets"]:
        if raw["name"] == entry["name"]:
            raw.update(updates)

    new_text = text
    for key, value in updates.items():
        if key in entry and entry[key] == value:
            continue
        new_text = _splice_field(new_text, entry["name"], key, value)

    if yaml.safe_load(new_text) != expected:
        raise RegistryError(
            f"refusing to write {REGISTRY_PATH}: the in-place edit of {name!r} "
            "did not parse back to the intended registry"
        )
    if _comment_lines(new_text) != _comment_lines(text):
        raise RegistryError(
            f"refusing to write {REGISTRY_PATH}: the in-place edit of {name!r} "
            "would have changed or dropped a comment line"
        )
    if new_text != text:
        REGISTRY_PATH.write_text(new_text)


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


def _splice_field(text: str, name: str, key: str, value: Any) -> str:
    """``text`` with field ``key`` of entry ``name`` replaced by ``value``
    (or appended to the entry), every other line copied through verbatim."""
    root = yaml.compose(text)
    datasets = next(v for k, v in root.value if k.value == "datasets")
    entry = next(
        node
        for node in datasets.value
        if any(k.value == "name" and v.value == name for k, v in node.value)
    )
    fields = entry.value
    indent = fields[0][0].start_mark.column
    lines = text.splitlines(keepends=True)
    replacement = _dump_field(key, value, indent)

    for i, (key_node, _) in enumerate(fields):
        if key_node.value == key:
            start = key_node.start_mark.line
            end = _field_end_line(lines, fields, i, indent)
            return "".join(lines[:start] + replacement + lines[end:])

    end = _field_end_line(lines, fields, len(fields) - 1, indent)
    if end and not lines[end - 1].endswith("\n"):
        lines[end - 1] += "\n"
    return "".join(lines[:end] + replacement + lines[end:])


def _field_end_line(lines: list[str], fields: list[Any], i: int, indent: int) -> int:
    """Exclusive end line of field ``i``: up to the next field, or the end of
    its value, minus any trailing blank or comment lines -- a comment at the
    entry's own indent or shallower belongs to whatever follows it."""
    start = fields[i][0].start_mark.line
    if i + 1 < len(fields):
        end = fields[i + 1][0].start_mark.line
    else:
        mark = fields[i][1].end_mark
        end = min(mark.line + (1 if mark.column > 0 else 0), len(lines))
    while end - 1 > start:
        line = lines[end - 1]
        stripped = line.lstrip()
        depth = len(line) - len(stripped)
        if stripped.strip() and not (stripped.startswith("#") and depth <= indent):
            break
        end -= 1
    return end


def _dump_field(key: str, value: Any, indent: int) -> list[str]:
    """``key: value`` dumped exactly as ``safe_dump`` would inside a list
    item, re-indented to ``indent`` columns."""
    dumped = yaml.safe_dump([{key: value}], sort_keys=False, allow_unicode=True)
    return [
        (" " * indent + line[2:]) if line.strip() else line
        for line in dumped.splitlines(keepends=True)
    ]


def verify_license(args: argparse.Namespace) -> int:
    """Record a license snapshot for one dataset.

    Reads ``entry.hosting`` and dispatches to the matching verification
    strategy BEFORE any URL is touched — the routing decision is this typed
    field, never a guess at ``--license-url``'s domain string. The human
    attests with ``--i-have-read-it``; the tool records, it does not judge —
    deciding what the text permits is the human's job, recorded in
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

    if entry.hosting == "huggingface":
        return _verify_license_huggingface(args, entry)
    return _verify_license_direct(args, entry)


def _verify_license_direct(args: argparse.Namespace, entry: DatasetEntry) -> int:
    """The original, unchanged path: hash whatever ``--license-url`` returns.

    Correct for a static file (a GitHub ``LICENSE``, MEVA's tarball terms) —
    the URL's bytes ARE the license text, with no per-request-volatile state
    wrapped around them. Never used for ``hosting: huggingface`` entries; see
    :func:`_verify_license_huggingface` for why that host needs a different
    strategy.
    """
    print(f"Fetching license text from {args.license_url} ...")
    try:
        with urllib.request.urlopen(args.license_url, timeout=60) as response:
            text = response.read()
    except DOWNLOAD_ERRORS as exc:
        print(f"Could not fetch the license text: {exc}", file=sys.stderr)
        return 1

    snapshot = LicenseSnapshot(
        url=args.license_url,
        verified_date=date.today(),
        text_sha256=hashlib.sha256(text).hexdigest(),
        verified_by=args.verified_by or getpass.getuser(),
        verified_class=args.verified_class,
    )
    return _record_snapshot(entry, snapshot)


def _hf_token(explicit: str) -> str | None:
    """The token to authenticate a HuggingFace request with.

    Never a second credential store: an explicit ``--hf-token`` wins,
    otherwise whatever ``huggingface-cli login`` (or ``huggingface_hub``'s own
    login flow) already cached on disk — the same token last night's manual
    session actually authenticated with.
    """
    return explicit or huggingface_hub.get_token()


def _parse_hf_dataset_url(url: str) -> tuple[str, str | None, str | None]:
    """``(repo_id, revision, path_in_repo)`` from a huggingface.co dataset URL.

    ``revision``/``path`` are ``None`` for a rendered dataset-card URL (no
    ``/raw/`` or ``/blob/`` segment) — exactly the shape
    :func:`_verify_license_huggingface` must refuse.

    Raises:
        ValueError: if ``url`` is not a recognisable huggingface.co dataset
            URL at all.
    """
    match = _HF_DATASET_URL_RE.match(url)
    if match is None:
        raise ValueError(
            f"{url!r} does not look like a huggingface.co dataset URL "
            "(expected https://huggingface.co/datasets/<owner>/<name>"
            "[/raw/<revision>/<path>])"
        )
    return match.group("repo_id"), match.group("revision"), match.group("path")


def _resolve_hf_commit_sha(repo_id: str, revision: str, token: str | None) -> str:
    """The exact commit ``revision`` (typically ``"main"``) resolves to right
    now — the pinning half of Day 39's fix: hashing ``/raw/main/<path>``
    trusts whatever ``main`` happens to point at when someone re-reads it
    later, which is exactly the kind of drift a pinned commit is supposed to
    make detectable instead of silent."""
    info = huggingface_hub.HfApi().dataset_info(repo_id, revision=revision, token=token)
    sha = info.sha
    if not sha:
        raise RegistryError(
            f"HuggingFace returned no commit sha for {repo_id}@{revision}"
        )
    return sha


def _fetch_hf_raw_file(
    repo_id: str, commit_sha: str, path: str, token: str | None
) -> bytes:
    """Bytes of one file at one pinned commit, via the raw (non-rendered)
    endpoint — the stable content the volatile dataset-card page wraps in
    per-request state. For license text only: on a Git-LFS file /raw/
    returns the LFS pointer, not the content — see :func:`_fetch_hf_file`."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/{commit_sha}/{path}"
    return _hf_get(url, token, timeout=60)


def _fetch_hf_file(
    repo_id: str,
    commit_sha: str,
    path: str,
    token: str | None,
    destination: Path,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """Stream one data file at one pinned commit to ``destination`` via
    /resolve/ — the endpoint that follows Git-LFS to the real content. Found
    live on CHIRLA (Day 40): every benchmark parquet is LFS, and /raw/
    returned its ~130-byte pointer ("version https://git-lfs.github.com/spec/
    v1 ..."). Resumes a partial ``destination`` and retries transient
    failures — see :func:`_download`."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{commit_sha}/{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # The timeout bounds each socket read, not the whole file, so it is the
    # stall detector: 2026-10-14 a SmartSpaces stream went silent at exactly
    # 128 MiB, and 600s meant ten idle minutes before the Range retry fired.
    _download(url, destination, headers=headers, timeout=60, on_progress=on_progress)


def _hf_get(url: str, token: str | None, timeout: float) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _verify_license_huggingface(args: argparse.Namespace, entry: DatasetEntry) -> int:
    """``hosting: huggingface`` verification: refuse a rendered page,
    resolve a pinned commit, hash the raw file at that commit.

    Found and confirmed live on CHIRLA, 2026-09-08: the rendered dataset-card
    page (``huggingface.co/datasets/<repo>``, no ``/raw/`` segment) embeds a
    ``data-props`` JSON payload carrying ``lastModified``, live download/like
    counts, and discussion stats alongside the actual license and
    Out-of-Scope-Use text — two fetches minutes apart hashed differently even
    though the license text itself was independently confirmed unchanged.
    That page shape is refused here, structurally, rather than silently
    hashed.
    """
    try:
        repo_id, revision, path = _parse_hf_dataset_url(args.license_url)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    token = _hf_token(args.hf_token)

    if revision is None or path is None:
        message = (
            f"REFUSED: {args.license_url!r} is a rendered HuggingFace "
            "dataset-card page, not a static file. It embeds per-request-"
            "volatile state (lastModified, live download/like counts, "
            "discussion stats) around the license text, so hashing it "
            "produces a different hash on every fetch even when the license "
            "itself is unchanged — confirmed live on CHIRLA, 2026-09-08: two "
            "fetches minutes apart, two different hashes, identical license "
            "text underneath. Point --license-url at the raw file instead: "
            f"https://huggingface.co/datasets/{repo_id}/raw/<commit_sha>/README.md"
        )
        try:
            resolved = _resolve_hf_commit_sha(repo_id, "main", token)
        except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
            message += (
                f"\n(Could not resolve <commit_sha> automatically here: {exc}. "
                "Re-run with network access to huggingface.co — this tool "
                "resolves it for you once it has one.)"
            )
        else:
            message += (
                "\nResolved just now: "
                f"https://huggingface.co/datasets/{repo_id}/raw/{resolved}/README.md"
            )
        print(message, file=sys.stderr)
        return 2

    if _FULL_COMMIT_SHA_RE.match(revision):
        resolved_sha = revision
    else:
        print(f"Resolving {repo_id}@{revision} to a commit sha ...")
        try:
            resolved_sha = _resolve_hf_commit_sha(repo_id, revision, token)
        except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
            print(f"Could not resolve {revision!r} to a commit: {exc}", file=sys.stderr)
            return 1

    print(f"Fetching {path} from {repo_id}@{resolved_sha} ...")
    try:
        text = _fetch_hf_raw_file(repo_id, resolved_sha, path, token)
    except DOWNLOAD_ERRORS as exc:
        print(f"Could not fetch the raw file: {exc}", file=sys.stderr)
        return 1

    snapshot = LicenseSnapshot(
        url=f"https://huggingface.co/datasets/{repo_id}/raw/{resolved_sha}/{path}",
        verified_date=date.today(),
        text_sha256=hashlib.sha256(text).hexdigest(),
        verified_by=args.verified_by or getpass.getuser(),
        verified_class=args.verified_class,
        resolved_commit_sha=resolved_sha,
    )
    return _record_snapshot(entry, snapshot)


def _record_snapshot(entry: DatasetEntry, snapshot: LicenseSnapshot) -> int:
    """The tail shared by both verification strategies: compare against
    whatever was recorded before, write the registry, report what happened.
    """
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
    if entry.hypothesis_class and snapshot.verified_class:
        if (
            snapshot.verified_class.strip().lower()
            not in entry.hypothesis_class.lower()
        ):
            print(
                f"  hypothesis was: {entry.hypothesis_class!r} — the verified "
                "class wins; update the lane in configs/datasets.yaml if the "
                "verified terms change what is permitted."
            )
    return 0


def _download(
    url: str,
    destination: Path,
    headers: dict[str, str] | None = None,
    timeout: float = 120,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """Stream ``url`` to ``destination``, resuming a partial file with a
    Range request and retrying transient failures (dropped connections,
    timeouts, 5xx) with capped exponential backoff. Each retry resumes from
    whatever reached disk. A 4xx is not transient and is raised at once.
    ``on_progress`` receives the destination's size after every chunk.

    No hash is checked here — the caller verifies the finished file, which is
    also the backstop for a server that mishandles Range."""
    failures = 0
    while True:
        before = _size_or_zero(destination)
        try:
            _download_once(url, destination, headers or {}, timeout, on_progress)
            return
        except _TRANSIENT_ERRORS as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise
            if _size_or_zero(destination) > before:
                failures = 0
            failures += 1
            if failures > RETRY_ATTEMPTS:
                raise
            delay = min(RETRY_BACKOFF_CAP_SECONDS, 2.0 ** (failures - 1))
            print(
                f"  {type(exc).__name__}: {exc} — retry {failures}/"
                f"{RETRY_ATTEMPTS} in {delay:.0f}s, resuming at byte "
                f"{_size_or_zero(destination):,}"
            )
            _sleep(delay)


def _download_once(
    url: str,
    destination: Path,
    headers: dict[str, str],
    timeout: float,
    on_progress: Callable[[int], None] | None,
) -> None:
    existing = _size_or_zero(destination)
    request = urllib.request.Request(url, headers=headers)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        print(f"  resuming at byte {existing:,}")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        # 206 means the server honoured the Range; anything else on a resume
        # means it did not, and appending would corrupt the payload.
        mode = "ab" if existing and response.status == 206 else "wb"
        if existing and response.status != 206:
            print("  server ignored the Range request; restarting from zero")
        written = existing if mode == "ab" else 0
        with open(destination, mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                written += len(chunk)
                if on_progress:
                    on_progress(written)


def _size_or_zero(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _git_blob_sha1(content: bytes) -> str:
    """Git's own blob hash: sha1("blob {len}\\0" + content) — the
    authoritative per-file hash HuggingFace's repo-tree API reports for a
    file NOT tracked with Git LFS (``sibling.blob_id``). LFS-tracked files
    report a real sha256 instead (``sibling.lfs.sha256``); this exists so
    the non-LFS case gets the identical refuse-on-mismatch discipline rather
    than a silent skip."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _hf_file_hashes(path: Path, meta: dict[str, Any]) -> tuple[str, str | None]:
    """(actual, expected) for one downloaded file, streamed from disk:
    sha256 for an LFS file, git's blob sha1 for a non-LFS one."""
    if meta["sha256"] is not None:
        return sha256_file(path), meta["sha256"]
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest(), meta["git_sha1"]


def _hf_repo_id(entry: DatasetEntry) -> str:
    """The HuggingFace repo id, derived from the entry's own recorded
    ``license_snapshot.url`` rather than a second, separately-maintained
    field — the URL already names the repo whichever verification path
    produced it (rendered-page or raw-file shape both parse)."""
    if entry.license_snapshot is None:
        raise RegistryError(
            f"{entry.name!r} has no license_snapshot to derive a "
            "HuggingFace repo id from."
        )
    repo_id, _, _ = _parse_hf_dataset_url(entry.license_snapshot.url)
    return repo_id


def _hf_card_data(repo_id: str, commit_sha: str, token: str | None) -> Any:
    return (
        huggingface_hub.HfApi()
        .dataset_info(repo_id, revision=commit_sha, token=token)
        .card_data
    )


def _hf_config_patterns(card_data: Any, config_name: str) -> list[str]:
    """Glob patterns for one named config, from the dataset card's own
    ``configs:`` YAML frontmatter — the authors' declared scenario
    boundaries, used as-is rather than re-derived by guessing at a directory
    layout.

    Raises:
        KeyError: ``config_name`` is not one of the configs this repo
            declares.
    """
    for cfg in (card_data.get("configs") if card_data is not None else None) or []:
        if cfg.get("config_name") != config_name:
            continue
        patterns: list[str] = []
        for data_file in cfg.get("data_files", []):
            path = data_file.get("path")
            if isinstance(path, list):
                patterns.extend(path)
            elif path:
                patterns.append(path)
        return patterns
    raise KeyError(config_name)


def _hf_config_split_stats(card_data: Any, config_name: str) -> list[dict[str, Any]]:
    """Per-split example counts and byte sizes for one named config, from
    the card's own embedded ``dataset_info:`` YAML — recorded into the
    manifest as a cross-check, never used to decide whether a fetch
    succeeded (the per-file hash comparison alone decides that)."""
    raw = (card_data.get("dataset_info") if card_data is not None else None) or []
    candidates = raw if isinstance(raw, list) else [raw]
    for candidate in candidates:
        if candidate.get("config_name", "default") == config_name:
            splits = candidate.get("splits", [])
            return [
                {
                    "name": split.get("name"),
                    "num_examples": split.get("num_examples"),
                    "num_bytes": split.get("num_bytes"),
                }
                for split in splits
            ]
    return []


def _hf_matching_files(
    repo_id: str, commit_sha: str, patterns: list[str], token: str | None
) -> list[dict[str, Any]]:
    """Every repo file at ``commit_sha`` matching any of ``patterns``,
    normalised to ``{path, size, sha256, git_sha1}`` — ``sha256`` set for
    Git-LFS files (the authoritative hash HuggingFace itself reports via
    ``sibling.lfs.sha256``), ``git_sha1`` set otherwise (``sibling.blob_id``,
    verified via :func:`_git_blob_sha1` after download).

    The listing is scoped to the deepest directory every pattern shares:
    recursing over CHIRLA's whole 10.9GB repo did not return in two minutes
    (Day 40); its configs all live under ``data/``, listed in under one."""
    matches: list[dict[str, Any]] = []
    for item in huggingface_hub.list_repo_tree(
        repo_id,
        path_in_repo=_patterns_root(patterns),
        recursive=True,
        expand=True,
        revision=commit_sha,
        repo_type="dataset",
        token=token,
    ):
        path = getattr(item, "path", None)
        blob_id = getattr(item, "blob_id", None)
        if path is None or blob_id is None:  # a RepoFolder, not a RepoFile
            continue
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            continue
        lfs = getattr(item, "lfs", None)
        matches.append(
            {
                "path": path,
                "size": getattr(item, "size", None),
                "sha256": lfs.sha256 if lfs is not None else None,
                "git_sha1": blob_id if lfs is None else None,
            }
        )
    return matches


def _patterns_root(patterns: list[str]) -> str | None:
    """The deepest directory every glob pattern's literal prefix shares, or
    ``None`` (the repo root) when they share none."""
    roots = [posixpath.dirname(re.split(r"[*?\[]", p, maxsplit=1)[0]) for p in patterns]
    if not roots:
        return None
    return posixpath.commonpath(roots) or None


def fetch_huggingface(args: argparse.Namespace, entry: DatasetEntry) -> int:
    """The HuggingFace-native fetch path (Day 39): named configs — or, for a
    repo that declares none, explicit path patterns (2026-10-14) — with
    resolved per-file hashes from HuggingFace's own API, content-addressed
    by commit.

    Refuses and deletes on any per-file hash mismatch — never installs a
    payload that does not match, exactly like the tarball path's sha256
    check. Both modes install through :func:`_install_hf_files`, so neither
    can drift into weaker checks than the other.
    """
    if bool(args.hf_config) == bool(args.hf_path):
        print(
            "REFUSED: hosting: huggingface entries need EITHER at least one "
            "--hf-config <name> (repeatable) naming one of the dataset's own "
            "declared configs, OR — only for a repo that declares no configs "
            "— at least one --hf-path <pattern> (repeatable). Not both, not "
            "neither.",
            file=sys.stderr,
        )
        return 2

    token = _hf_token(args.hf_token)
    try:
        repo_id = _hf_repo_id(entry)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"Resolving {repo_id}@{args.revision} (and main) to commit shas ...")
    try:
        commit_sha = _resolve_hf_commit_sha(repo_id, args.revision, token)
        main_head_sha = (
            commit_sha
            if args.revision == "main"
            else _resolve_hf_commit_sha(repo_id, "main", token)
        )
        card_data = _hf_card_data(repo_id, commit_sha, token)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
        print(f"Could not reach {repo_id} on huggingface.co: {exc}", file=sys.stderr)
        return 1

    iron_config = IronConfig.load()
    dataset_root = iron_config.paths.resolved_data_dir / "raw" / entry.name / commit_sha

    pinned = (
        entry.license_snapshot.resolved_commit_sha if entry.license_snapshot else None
    )
    if entry.fetch_deny_patterns and commit_sha != pinned:
        print(
            f"REFUSED: {entry.name!r} has fetch_deny_patterns, which were "
            f"written against the verified commit {pinned!r}; this request "
            f"resolved to {commit_sha!r}. A later commit can add paths the "
            "deny list never saw (e.g. new real-world scenes). Pass "
            f"--revision {pinned} or re-verify and re-review the deny list.",
            file=sys.stderr,
        )
        return 2

    print(f"Fetching {repo_id}@{commit_sha} (main is at {main_head_sha})")
    provenance = {"requested_revision": args.revision, "main_head_sha": main_head_sha}
    if args.hf_path:
        return _fetch_hf_paths(
            repo_id,
            commit_sha,
            args.hf_path,
            card_data,
            dataset_root,
            token,
            provenance,
            entry.fetch_deny_patterns,
        )
    exit_code = 0
    for config_name in args.hf_config:
        code = _fetch_hf_config(
            repo_id,
            commit_sha,
            config_name,
            card_data,
            dataset_root,
            token,
            provenance,
            entry.fetch_deny_patterns,
        )
        exit_code = code if code != 0 else exit_code
    return exit_code


def _refuse_denied_paths(files: list[dict[str, Any]], deny_patterns: list[str]) -> None:
    """Refuse the whole fetch if any file it would install matches one of
    the entry's ``fetch_deny_patterns`` — checked before the first byte is
    downloaded, and never by silently dropping the file: a request that
    reached a denied path is the error, not something to route around.

    A listing path not already in plain normal form (``./x``, ``a//b``,
    ``a/../b``, absolute) is refused outright rather than normalised: the
    deny match runs on the string, the write lands on the resolved path,
    and the two must never be allowed to disagree.

    Raises:
        RegistryError: naming every denied or non-normal path the request
            matched.
    """
    abnormal = sorted(
        meta["path"]
        for meta in files
        if meta["path"].startswith("/")
        or posixpath.normpath(meta["path"]) != meta["path"]
    )
    if abnormal:
        raise RegistryError(
            f"REFUSED: the listing returned {len(abnormal)} path(s) not in "
            f"normal form, e.g. {abnormal[:3]!r}. A listing from the network "
            "is untrusted input; a non-normal path could match a deny "
            "pattern differently from where it would be written."
        )
    denied = sorted(
        meta["path"]
        for meta in files
        if any(fnmatch.fnmatch(meta["path"], pattern) for pattern in deny_patterns)
    )
    if denied:
        raise RegistryError(
            f"REFUSED: this fetch would install {len(denied)} path(s) the "
            f"registry entry denies (fetch_deny_patterns={deny_patterns!r}), "
            f"e.g. {denied[:3]!r}. These are registered under a different "
            "lane; narrow the request so it cannot reach them."
        )


def _audit_installed_tree(dest_dir: Path, deny_patterns: list[str]) -> None:
    """Refuse when an already-installed folder holds a denied path (Day 41).

    The "already present — idempotent" return used to skip the deny check
    entirely in config mode: whatever sat in the folder, put there by an
    earlier version of the deny list, a different tool, or by hand, was
    reported as a clean no-op. Now every file on disk is checked against
    the same patterns a fresh listing would be.

    Raises:
        RegistryError: naming the denied paths found on disk.
    """
    if not deny_patterns:
        return
    on_disk = sorted(
        p.relative_to(dest_dir).as_posix()
        for p in dest_dir.rglob("*")
        if p.is_file() and p.name != "_manifest.json"
    )
    denied = sorted(
        path
        for path in on_disk
        if any(fnmatch.fnmatch(path, pattern) for pattern in deny_patterns)
    )
    if denied:
        raise RegistryError(
            f"REFUSED: {dest_dir} is already on disk and holds {len(denied)} "
            f"path(s) the registry entry denies (fetch_deny_patterns="
            f"{deny_patterns!r}), e.g. {denied[:3]!r}. Nothing was deleted: "
            "remove them deliberately, after checking how they got there."
        )


def _fetch_hf_paths(
    repo_id: str,
    commit_sha: str,
    patterns: list[str],
    card_data: Any,
    dataset_root: Path,
    token: str | None,
    provenance: dict[str, str],
    deny_patterns: list[str],
) -> int:
    """Fetch every file matching explicit ``patterns`` into
    ``dataset_root/paths-<digest>/`` — for a repo whose card declares no
    configs (PhysicalAI-SmartSpaces), so there are no author-declared
    scenario boundaries to name.

    Refused when the card DOES declare configs: there, the authors' own
    boundaries are the interface, and a hand-written glob would be exactly
    the "re-derived by guessing at a directory layout" the config path
    exists to avoid. The label is a digest of the sorted patterns, so the
    same request is idempotent and a different request can never land in,
    or be mistaken for, an earlier one's directory.
    """
    if card_data.get("configs") if card_data is not None else None:
        print(
            f"REFUSED: {repo_id}@{commit_sha} declares configs in its own "
            "dataset card — use --hf-config with one of them, not --hf-path.",
            file=sys.stderr,
        )
        return 2

    normalised = sorted(set(patterns))
    digest = hashlib.sha256("\n".join(normalised).encode()).hexdigest()[:16]
    label = f"paths-{digest}"
    try:
        files = _hf_matching_files(repo_id, commit_sha, normalised, token)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
        print(f"Could not list files for {normalised!r}: {exc}", file=sys.stderr)
        return 1
    _refuse_denied_paths(files, deny_patterns)
    if (dataset_root / label).exists():
        _audit_installed_tree(dataset_root / label, deny_patterns)
    return _install_hf_files(
        repo_id,
        commit_sha,
        label,
        files,
        dataset_root,
        token,
        {"patterns": normalised, "method": "huggingface_paths_v1", **provenance},
    )


def _fetch_hf_config(
    repo_id: str,
    commit_sha: str,
    config_name: str,
    card_data: Any,
    dataset_root: Path,
    token: str | None,
    provenance: dict[str, str],
    deny_patterns: list[str],
) -> int:
    """Fetch one named config into ``dataset_root/<config_name>/``."""
    if (dataset_root / config_name).exists():
        _audit_installed_tree(dataset_root / config_name, deny_patterns)
        print(
            f"{config_name} already present at {dataset_root / config_name} "
            f"(commit {commit_sha}) — idempotent, not re-fetched."
        )
        return 0

    try:
        patterns = _hf_config_patterns(card_data, config_name)
    except KeyError:
        print(
            f"REFUSED: {config_name!r} is not a config {repo_id}@{commit_sha} "
            "declares in its own dataset card.",
            file=sys.stderr,
        )
        return 2

    try:
        files = _hf_matching_files(repo_id, commit_sha, patterns, token)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
        print(f"Could not list files for {config_name!r}: {exc}", file=sys.stderr)
        return 1
    _refuse_denied_paths(files, deny_patterns)
    return _install_hf_files(
        repo_id,
        commit_sha,
        config_name,
        files,
        dataset_root,
        token,
        {
            "config": config_name,
            "method": "huggingface_configs_v1",
            **provenance,
            "splits": _hf_config_split_stats(card_data, config_name),
        },
    )


def _install_hf_files(
    repo_id: str,
    commit_sha: str,
    label: str,
    files: list[dict[str, Any]],
    dataset_root: Path,
    token: str | None,
    manifest_fields: dict[str, Any],
) -> int:
    """Download ``files`` into ``dataset_root/<label>/`` with the checks both
    HuggingFace modes share: idempotent on an existing label, refuse an
    empty match, refuse a repo path that would escape the staging directory,
    download via /resolve/ (LFS content, never the pointer), verify each file
    against HuggingFace's own reported hash, and install atomically by
    renaming a fully-verified staging directory into place.

    Resumable across runs (2026-10-14): the staging directory survives a
    failure, every file in it got there only by passing its hash check, and
    a rerun re-verifies and skips those. The file in flight streams to
    ``<path>.partial`` and resumes from its byte offset."""
    dest_dir = dataset_root / label
    if dest_dir.exists():
        print(
            f"{label} already present at {dest_dir} (commit {commit_sha}) — "
            "idempotent, not re-fetched."
        )
        return 0

    if not files:
        print(
            f"REFUSED: {label!r} matched zero files under {repo_id}@{commit_sha} "
            "— refusing to record an empty fetch as success.",
            file=sys.stderr,
        )
        return 1

    staging = dataset_root / f"{label}.partial"
    staging_root = staging.resolve()
    for meta in files:
        if not (staging_root / meta["path"]).resolve().is_relative_to(staging_root):
            print(
                f"REFUSED: repo path {meta['path']!r} resolves outside the "
                "staging directory — a listing from the network is untrusted "
                "input, and this one tried to escape.",
                file=sys.stderr,
            )
            return 1
    staging.mkdir(parents=True, exist_ok=True)

    total = sum(m["size"] for m in files)
    done = 0
    show_live = sys.stdout.isatty()
    for index, meta in enumerate(files, start=1):
        target = staging / meta["path"]
        tag = f"[{index:>{len(str(len(files)))}}/{len(files)}]"

        if target.exists():
            actual, expected = _hf_file_hashes(target, meta)
            if expected is not None and actual == expected:
                done += meta["size"]
                print(f"  {tag} present     {meta['path']}  {_progress(done, total)}")
                continue
            print(
                f"  {tag} {meta['path']} is in staging but hashes to {actual}, "
                f"not {expected} — deleting it and downloading again."
            )
            target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        if partial.exists() and partial.stat().st_size > meta["size"]:
            partial.unlink()  # longer than the file can be: not a prefix of it

        def live(file_bytes: int, base: int = done) -> None:
            if show_live:
                print(f"\r  {_progress(base + file_bytes, total)}", end="", flush=True)

        try:
            if _size_or_zero(partial) < meta["size"] or meta["size"] == 0:
                _fetch_hf_file(repo_id, commit_sha, meta["path"], token, partial, live)
        except DOWNLOAD_ERRORS as exc:
            if show_live:
                print()
            print(
                f"Download failed on {meta['path']} after retries: "
                f"{type(exc).__name__}: {exc}. Kept {_size_or_zero(partial):,} "
                f"bytes at {partial} and every verified file in {staging}; "
                "rerun the same command to resume.",
                file=sys.stderr,
            )
            return 1
        if show_live:
            print()

        actual, expected = _hf_file_hashes(partial, meta)
        if expected is None or actual != expected:
            partial.unlink()
            print(
                f"HASH MISMATCH on {meta['path']}: HuggingFace reports "
                f"{expected}, downloaded content hashes to {actual}. Deleted, "
                "not installed — a file that does not match HuggingFace's own "
                "reported hash is not close enough. Files already verified "
                f"stay in {staging}.",
                file=sys.stderr,
            )
            return 1
        partial.rename(target)
        done += meta["size"]
        print(f"  {tag} downloaded  {meta['path']}  {_progress(done, total)}")

    manifest = {
        **manifest_fields,
        "commit_sha": commit_sha,
        "repo_id": repo_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": m["path"], "size": m["size"]} for m in files],
    }
    (staging / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    staging.rename(dest_dir)
    print(f"Stored {label} at {dest_dir} ({len(files)} files).")
    return 0


def _progress(done: int, total: int) -> str:
    return f"{done / 1e9:,.2f} / {total / 1e9:,.2f} GB"


def fetch(args: argparse.Namespace) -> int:
    """Registry check -> hosting routing -> download -> hash verify -> store."""
    registry = DatasetRegistry.load(REGISTRY_PATH)
    try:
        entry = registry.require_fetchable(args.name)
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    if entry.hosting == "huggingface":
        return fetch_huggingface(args, entry)

    if not args.url or not args.expected_sha:
        print(
            "REFUSED: this entry is not hosting: huggingface, so fetch mode "
            "requires --url and --expected-sha.",
            file=sys.stderr,
        )
        return 2
    return _fetch_tarball(args, entry)


def _fetch_tarball(args: argparse.Namespace, entry: DatasetEntry) -> int:
    """The original ``--url``/``--expected-sha`` path, unchanged (Day 39:
    ``fetch`` now routes here only for non-``huggingface`` entries)."""
    iron_config = IronConfig.load()
    dataset_root = iron_config.paths.resolved_data_dir / "raw" / entry.name

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
    except DOWNLOAD_ERRORS as exc:
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
    parser.add_argument("--url", help="payload URL (fetch mode, non-HuggingFace)")
    parser.add_argument(
        "--expected-sha", help="published sha256 of the payload (non-HuggingFace)"
    )
    parser.add_argument(
        "--hf-config",
        action="append",
        default=[],
        help=(
            "named HuggingFace dataset config/scenario to fetch (hosting: "
            "huggingface entries only; repeatable)"
        ),
    )
    parser.add_argument(
        "--hf-path",
        action="append",
        default=[],
        help=(
            "fnmatch pattern over repo paths to fetch (hosting: huggingface "
            "entries whose card declares NO configs; repeatable)"
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "branch, tag or commit to fetch (hosting: huggingface only); "
            "resolved to a commit sha, and main's sha is recorded alongside"
        ),
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="HuggingFace token (default: whatever huggingface-cli login cached)",
    )
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
        return fetch(args)
    except RegistryError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
