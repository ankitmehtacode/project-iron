"""Run manifests: what produced this result, recorded before it is produced.

A result without a manifest is inadmissible. Not as a matter of style — you
genuinely cannot answer the only questions that matter about it. Which model
weights made these embeddings? Was the checkout clean? Is this index still
valid for the encoder now on disk? Without a manifest every answer is "probably
the same as last time", and re-quantizing a model silently invalidates every
vector in every index built from it with nothing to detect the mismatch.

The manifest is written **before** processing starts, and a run that cannot
write one refuses to start. Writing it afterwards would mean the runs that
crash — the ones you most need to explain — are exactly the ones with no
record.

What ``manifest_sha`` covers
----------------------------
Everything except the start timestamp and the hash field itself. It answers
"was this the same setup?", not "was this the same moment". Two runs of
identical code, config, weights, and hardware share a ``manifest_sha`` and are
therefore comparable; that is the property the whole idea is for. Including the
clock would give every run a unique hash and make the field useless for
grouping.

OpenVINO IR is hashed as a pair
-------------------------------
An IR is an ``.xml`` topology plus a ``.bin`` of weights. Hashing only the
``.xml`` would miss a re-quantization entirely, since that rewrites the weights
and often leaves the topology byte-identical. Both files are hashed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import IronConfig

SCHEMA_VERSION: Literal["1.0"] = "1.0"
_HASH_CHUNK_BYTES = 1024 * 1024

# Recorded because each one can change numerical output. Absence is recorded
# as "not installed" rather than omitted, so a manifest never leaves ambiguity
# about whether a library was missing or merely unexamined.
_TRACKED_LIBRARIES = ("numpy", "torch", "openvino", "pyarrow", "scipy", "pydantic")


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be produced or written."""


class GitState(BaseModel):
    """Repository state, or a recorded absence of one."""

    model_config = ConfigDict(frozen=True)

    sha: str | None = None
    branch: str | None = None
    dirty: bool = False
    available: bool = True
    detail: str = ""


def git_state(repo_root: Path) -> GitState:
    """Read the git sha, branch, and dirty flag for ``repo_root``.

    Degrades gracefully: a source tarball, a container without git, or a
    directory that is simply not a checkout all yield ``available=False`` with
    an explanation, rather than raising. The run is still valid — it just
    cannot be tied to a commit, and the manifest says so plainly instead of
    recording a fake sha.
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return GitState(
            available=False,
            detail=f"{repo_root} is not a git checkout, or git is unavailable",
        )

    status = run("status", "--porcelain")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return GitState(
        sha=sha,
        branch=branch or None,
        # `git status --porcelain` prints one line per modified/untracked path,
        # so any output at all means the tree differs from HEAD.
        dirty=bool(status),
        available=True,
        detail="",
    )


def sha256_file(path: Path) -> str:
    """Stream a file through sha256.

    Chunked because model weights run to hundreds of megabytes and reading one
    into memory to hash it would double the harness's own footprint in the
    middle of a memory-stability test.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_files(paths: list[Path]) -> dict[str, str]:
    """Hash each model file, following OpenVINO IR to its weights.

    Args:
        paths: Model files the run will load.

    Returns:
        Mapping of path string to sha256, or to a ``"MISSING: ..."`` marker.
        A missing file is recorded rather than skipped: silently omitting it
        would make the manifest of a broken run look like the manifest of a
        run that simply used fewer models.
    """
    hashes: dict[str, str] = {}
    for path in paths:
        candidates = [path]
        if path.suffix == ".xml":
            weights = path.with_suffix(".bin")
            if weights.exists():
                candidates.append(weights)
        for candidate in candidates:
            key = str(candidate)
            if not candidate.exists():
                hashes[key] = "MISSING: file not found at manifest time"
                continue
            try:
                hashes[key] = sha256_file(candidate)
            except OSError as exc:
                hashes[key] = f"UNREADABLE: {exc}"
    return hashes


def library_versions() -> dict[str, str]:
    """Version of Python and of every library that can change results."""
    versions = {"python": platform.python_version()}
    for name in _TRACKED_LIBRARIES:
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "not installed"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def cpu_description() -> str:
    """Best available CPU model string for this machine.

    Thread scheduling and INT8 kernel selection both depend on the CPU, so a
    latency or numerical comparison across different models is not a
    comparison. ``platform.processor()`` is often empty on Linux, hence the
    ``/proc/cpuinfo`` fallback.
    """
    described = platform.processor()
    if described:
        return described

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine() or "unknown"


class RunManifest(BaseModel):
    """Immutable record of everything that determines a run's output."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    started_at_utc: str
    git: GitState
    config_sha: str
    model_hashes: dict[str, str] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    hostname: str
    cpu_model: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    thread_settings: dict[str, int] = Field(default_factory=dict)
    manifest_sha: str = ""

    @classmethod
    def capture(
        cls,
        config: IronConfig,
        model_paths: list[Path] | None = None,
    ) -> "RunManifest":
        """Build a manifest describing the run that is about to start.

        Args:
            config: The loaded configuration.
            model_paths: Model files the run will load. Defaults to the V-JEPA2
                IR and the CoTracker3 checkpoint from ``config``.
        """
        import psutil

        if model_paths is None:
            model_paths = [
                config.paths.resolved_vjepa_xml,
                config.paths.resolved_cotracker_checkpoint,
            ]

        manifest = cls(
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            git=git_state(config.paths.project_root),
            config_sha=config.config_sha(),
            model_hashes=hash_model_files(model_paths),
            library_versions=library_versions(),
            hostname=socket.gethostname(),
            cpu_model=cpu_description(),
            cpu_count_logical=psutil.cpu_count(logical=True) or 0,
            cpu_count_physical=psutil.cpu_count(logical=False),
            thread_settings={
                "ov_num_threads": config.runtime.ov_num_threads,
                "torch_num_threads": config.runtime.torch_num_threads,
                "seed": config.runtime.seed,
            },
        )
        return manifest.model_copy(update={"manifest_sha": manifest.compute_sha()})

    def compute_sha(self) -> str:
        """sha256 over every field except the timestamp and the hash itself.

        See the module docstring: this identifies the setup, not the moment.
        """
        payload = self.model_dump(mode="json")
        payload.pop("started_at_utc", None)
        payload.pop("manifest_sha", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write(self, path: Path) -> Path:
        """Write the manifest as JSON, creating parent directories.

        Args:
            path: Destination file, conventionally ``<output_dir>/manifest.json``.

        Returns:
            The path written.

        Raises:
            ManifestError: if the manifest cannot be written. Callers must let
                this stop the run. A run that proceeds without a manifest
                produces results nobody can later attribute, which is the exact
                failure this module exists to prevent.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        except OSError as exc:
            raise ManifestError(
                f"cannot write the run manifest to {path}: {exc}. "
                "Refusing to start: results without a manifest cannot be "
                "attributed to a model version, config, or commit."
            ) from exc
        return path

    @classmethod
    def read(cls, path: Path) -> "RunManifest":
        """Load a manifest and verify its own hash still matches its contents.

        Raises:
            ManifestError: if the file is unreadable, malformed, or has been
                edited since it was written. An append-only record that can be
                silently altered is not evidence.
        """
        try:
            payload: dict[str, Any] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"cannot read manifest at {path}: {exc}") from exc

        manifest = cls.model_validate(payload)
        expected = manifest.compute_sha()
        if manifest.manifest_sha != expected:
            raise ManifestError(
                f"manifest at {path} has been modified since it was written: "
                f"recorded sha {manifest.manifest_sha[:12]}... but its contents "
                f"hash to {expected[:12]}..."
            )
        return manifest

    def summary(self) -> str:
        """Compact multi-line description for the run log."""
        if self.git.available:
            git_line = f"{self.git.sha} ({self.git.branch})"
            if self.git.dirty:
                git_line += "  [DIRTY WORKING TREE]"
        else:
            git_line = f"unavailable — {self.git.detail}"

        lines = [
            f"manifest_sha : {self.manifest_sha}",
            f"config_sha   : {self.config_sha}",
            f"git          : {git_line}",
            f"host         : {self.hostname} ({self.cpu_model}, "
            f"{self.cpu_count_logical} logical cores)",
            f"started      : {self.started_at_utc}",
        ]
        for name, digest in sorted(self.model_hashes.items()):
            lines.append(f"model        : {digest[:16]}...  {name}")
        return "\n".join(lines)
