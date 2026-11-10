"""Run-stamped, never-overwriting output paths.

Scorecards, run manifests and run logs used to be written to fixed names
(``outputs/manifest.json``, ``logs/endurance_run.log``, one scorecard per
golden set), so every run silently replaced the record of the one before
it. That is not a hypothetical: the Day-43 audit's own read-only pass
overwrote a scorecard that git could not restore.

Every such write now goes to ``<stem>.<stamp><suffix>`` opened in exclusive
mode (``"x"``): a second writer with the same name raises
:class:`FileExistsError` instead of truncating. The stamp is UTC time to the
second plus six random hex digits, so names sort by time and two runs in
the same second still differ.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def run_stamp() -> str:
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def stamped(path: Path, stamp: str) -> Path:
    """``dir/name.ext`` -> ``dir/name.<stamp>.ext``."""
    if not stamp:
        raise ValueError("a run stamp must not be empty")
    return path.with_name(f"{path.stem}.{stamp}{path.suffix}")


def open_exclusive(path: Path) -> TextIO:
    """Create ``path`` for writing; raise if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "x")


def write_exclusive(path: Path, text: str) -> Path:
    with open_exclusive(path) as handle:
        handle.write(text)
    return path


def write_run_stamped(directory: Path, stem: str, suffix: str, text: str) -> Path:
    """Write ``directory/<stem>.<stamp><suffix>`` and return its path."""
    return write_exclusive(stamped(directory / f"{stem}{suffix}", run_stamp()), text)


def latest(path: Path) -> Path:
    """The newest stamped sibling of ``path`` (``name.*.ext``), by stamp."""
    matches = sorted(path.parent.glob(f"{path.stem}.*{path.suffix}"))
    if not matches:
        raise FileNotFoundError(f"no run-stamped {path.name} under {path.parent}")
    return matches[-1]
