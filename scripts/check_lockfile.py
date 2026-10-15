"""Fail when the running environment differs from locking-requirements.txt.

Day 42: ``.venv-pinned`` — the environment every report's numbers came
from — had drifted from the lockfile it is named after (tokenizers 0.22.2
against 0.21.0, huggingface-hub 0.36.2 against 0.26.0, transformers 4.57.6
with no pin at all), and nothing noticed. A lockfile that is never compared
with what is installed records an intention, not an environment.

Every ``name==version`` and ``name @ git+...@sha`` line of the lockfile must
appear, identically, in ``pip freeze`` of THIS interpreter. Packages the
lockfile does not name are not judged (they are transitive; pin them in the
lockfile to bring them under this check).

    .venv-pinned/bin/python scripts/check_lockfile.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LOCKFILE = Path(__file__).resolve().parent.parent / "locking-requirements.txt"

_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)==(?P<version>[^\s#;]+)")
_GIT = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)\s*@\s*(?P<url>git\+\S+)")


def canonical(name: str) -> str:
    """PEP 503 normalisation: case, ``_`` and ``.`` do not distinguish names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> dict[str, str]:
    """``{canonical name: exact spec}``. Any requirement line that is neither
    an exact pin nor a git URL is an error: a range is what this replaces."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(" --")[0].strip()  # --index-url options
        if m := _PIN.match(line):
            out[canonical(m["name"])] = m["version"]
        elif m := _GIT.match(line):
            out[canonical(m["name"])] = m["url"]
        else:
            raise ValueError(f"not an exact pin: {raw!r}")
    return out


def parse_freeze(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if m := _PIN.match(line.strip()):
            out[canonical(m["name"])] = m["version"]
        elif m := _GIT.match(line.strip()):
            out[canonical(m["name"])] = m["url"]
    return out


def differences(locked: dict[str, str], installed: dict[str, str]) -> list[str]:
    problems = []
    for name, spec in sorted(locked.items()):
        have = installed.get(name)
        if have is None:
            problems.append(f"{name}: locked {spec}, NOT INSTALLED")
        elif have.split("+")[0] != spec.split("+")[0] and have != spec:
            problems.append(f"{name}: locked {spec}, installed {have}")
    return problems


def main() -> int:
    locked = parse_requirements(LOCKFILE.read_text())
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    problems = differences(locked, parse_freeze(freeze))
    if problems:
        print(f"{len(problems)} difference(s) from {LOCKFILE.name}:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"environment matches {LOCKFILE.name}: {len(locked)} pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
