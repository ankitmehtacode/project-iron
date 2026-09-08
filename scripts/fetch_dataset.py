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
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

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
    per-request state."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/{commit_sha}/{path}"
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
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
    except (urllib.error.URLError, OSError) as exc:
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


def _git_blob_sha1(content: bytes) -> str:
    """Git's own blob hash: sha1("blob {len}\\0" + content) — the
    authoritative per-file hash HuggingFace's repo-tree API reports for a
    file NOT tracked with Git LFS (``sibling.blob_id``). LFS-tracked files
    report a real sha256 instead (``sibling.lfs.sha256``); this exists so
    the non-LFS case gets the identical refuse-on-mismatch discipline rather
    than a silent skip."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


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
    verified via :func:`_git_blob_sha1` after download)."""
    matches: list[dict[str, Any]] = []
    for item in huggingface_hub.list_repo_tree(
        repo_id,
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


def fetch_huggingface(args: argparse.Namespace, entry: DatasetEntry) -> int:
    """The HuggingFace-native fetch path (Day 39): named configs, resolved
    per-file hashes from HuggingFace's own API, content-addressed by commit.

    Refuses and deletes on any per-file hash mismatch — never installs a
    payload that does not match, exactly like the tarball path's sha256
    check.
    """
    if not args.hf_config:
        print(
            "REFUSED: hosting: huggingface entries need at least one "
            "--hf-config <name> (repeatable) naming one of the dataset's own "
            "declared configs — not a raw directory glob.",
            file=sys.stderr,
        )
        return 2

    token = _hf_token(args.hf_token)
    try:
        repo_id = _hf_repo_id(entry)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(f"Resolving {repo_id}@main to a commit sha ...")
    try:
        commit_sha = _resolve_hf_commit_sha(repo_id, "main", token)
        card_data = _hf_card_data(repo_id, commit_sha, token)
    except Exception as exc:  # noqa: BLE001 -- CLI boundary, see module note
        print(f"Could not reach {repo_id} on huggingface.co: {exc}", file=sys.stderr)
        return 1

    iron_config = IronConfig.load()
    dataset_root = iron_config.paths.resolved_data_dir / "raw" / entry.name / commit_sha

    exit_code = 0
    for config_name in args.hf_config:
        code = _fetch_hf_config(
            repo_id, commit_sha, config_name, card_data, dataset_root, token
        )
        exit_code = code if code != 0 else exit_code
    return exit_code


def _fetch_hf_config(
    repo_id: str,
    commit_sha: str,
    config_name: str,
    card_data: Any,
    dataset_root: Path,
    token: str | None,
) -> int:
    """Fetch one named config into ``dataset_root/<config_name>/``."""
    dest_dir = dataset_root / config_name
    if dest_dir.exists():
        print(
            f"{config_name} already present at {dest_dir} (commit "
            f"{commit_sha}) — idempotent, not re-fetched."
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

    if not files:
        print(
            f"REFUSED: config {config_name!r} matched zero files under "
            f"{repo_id}@{commit_sha} (patterns={patterns!r}) — refusing to "
            "record an empty fetch as success.",
            file=sys.stderr,
        )
        return 1

    staging = dataset_root / f"{config_name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for meta in files:
        target = staging / meta["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {meta['path']} ...")
        try:
            content = _fetch_hf_raw_file(repo_id, commit_sha, meta["path"], token)
        except (urllib.error.URLError, OSError) as exc:
            shutil.rmtree(staging)
            print(f"Download failed on {meta['path']}: {exc}", file=sys.stderr)
            return 1

        if meta["sha256"] is not None:
            actual, expected = hashlib.sha256(content).hexdigest(), meta["sha256"]
        else:
            actual, expected = _git_blob_sha1(content), meta["git_sha1"]

        if expected is not None and actual != expected:
            shutil.rmtree(staging)
            print(
                f"HASH MISMATCH on {meta['path']}: HuggingFace reports "
                f"{expected}, downloaded content hashes to {actual}. Deleted, "
                "not installed — a file that does not match HuggingFace's own "
                "reported hash is not close enough.",
                file=sys.stderr,
            )
            return 1
        target.write_bytes(content)

    manifest = {
        "config": config_name,
        "commit_sha": commit_sha,
        "repo_id": repo_id,
        "method": "huggingface_configs_v1",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "splits": _hf_config_split_stats(card_data, config_name),
        "files": [{"path": m["path"], "size": m["size"]} for m in files],
    }
    (staging / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    staging.rename(dest_dir)
    print(f"Stored {config_name} at {dest_dir} ({len(files)} files).")
    return 0


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
