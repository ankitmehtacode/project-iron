"""Regenerate the README's directory tree from the actual repository contents.

This script exists because the hand-maintained tree in the README drifted far
from reality: it listed directories that never existed (``docs/``, ``configs/``
at the time), omitted half of ``src/``, and named a ``scripts/pipeline_example.py``
that is not in the tree. A tree that lies is worse than no tree, because readers
trust it and then waste time looking for files.

The source of truth is ``git ls-files``: what the tree shows is exactly what is
tracked, so ignored build artifacts and model weights never leak into the docs.
Outside a git checkout it falls back to walking the filesystem with the same
exclusions, so the script still works in a source tarball.

Usage:
    python scripts/gen_tree.py            # rewrite README.md in place
    python scripts/gen_tree.py --check    # exit 1 if README is stale (for CI)
    python scripts/gen_tree.py --stdout   # print the tree, touch nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

BEGIN_MARKER = "<!-- BEGIN TREE -->"
END_MARKER = "<!-- END TREE -->"
REGEN_HINT = "<!-- regenerate with: python scripts/gen_tree.py -->"

# Used only by the non-git fallback. The git path gets these for free
# from .gitignore.
FALLBACK_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".hypothesis",
        "outputs",
        "output",
        "logs",
        "models",
        "build",
        "dist",
    }
)


def _tracked_paths() -> list[Path]:
    """Return every tracked file path, relative to the repository root.

    Falls back to a filesystem walk when git is unavailable or this is not a
    checkout, so the script never hard-fails on a source tarball.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return _walked_paths()

    paths = [Path(line) for line in completed.stdout.splitlines() if line]
    return paths or _walked_paths()


def _walked_paths() -> list[Path]:
    """Filesystem fallback enumeration, honouring FALLBACK_EXCLUDES."""
    found: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in FALLBACK_EXCLUDES for part in relative.parts):
            continue
        if (
            relative.parts
            and relative.parts[0].startswith(".")
            and len(relative.parts) > 1
        ):
            continue
        found.append(relative)
    return found


def _build_nested(paths: list[Path]) -> dict[str, dict]:
    """Fold a flat path list into a nested {name: children} mapping.

    Files map to an empty dict, which is how :func:`_render` distinguishes them
    from empty directories — a distinction git cannot express anyway, since git
    does not track empty directories.
    """
    root: dict[str, dict] = {}
    for path in sorted(paths):
        cursor = root
        for part in path.parts:
            cursor = cursor.setdefault(part, {})
    return root


def _render(node: dict[str, dict], prefix: str = "") -> list[str]:
    """Render a nested mapping as box-drawing tree lines.

    Directories sort before files at each level so the shape of the project is
    readable at a glance rather than interleaved with loose files.
    """
    entries = sorted(node.items(), key=lambda kv: (not bool(kv[1]), kv[0].lower()))
    lines: list[str] = []
    for index, (name, children) in enumerate(entries):
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if children else ""
        lines.append(f"{prefix}{connector}{name}{suffix}")
        if children:
            extension = "    " if is_last else "│   "
            lines.extend(_render(children, prefix + extension))
    return lines


def build_tree_block() -> str:
    """Return the full README section, markers and fence included."""
    tree_lines = _render(_build_nested(_tracked_paths()))
    body = "\n".join(["project-iron/", *tree_lines])
    return "\n".join([BEGIN_MARKER, REGEN_HINT, "", "```", body, "```", END_MARKER])


def _splice(readme_text: str, block: str) -> str:
    """Replace the marked tree block in ``readme_text``.

    Raises:
        ValueError: if the markers are missing or inverted, rather than silently
            appending a second tree and leaving two contradictory ones behind.
    """
    start = readme_text.find(BEGIN_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"README.md must contain {BEGIN_MARKER} ... {END_MARKER} markers "
            "around the directory tree; refusing to guess where it goes."
        )
    return readme_text[:start] + block + readme_text[end + len(END_MARKER) :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the README tree is out of date; write nothing",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the generated block instead of editing the README",
    )
    args = parser.parse_args(argv)

    block = build_tree_block()

    if args.stdout:
        print(block)
        return 0

    current = README.read_text()
    updated = _splice(current, block)

    if args.check:
        if current != updated:
            print(
                "README.md directory tree is stale. " "Run: python scripts/gen_tree.py",
                file=sys.stderr,
            )
            return 1
        print("README.md directory tree is up to date.")
        return 0

    if current == updated:
        print("README.md directory tree already up to date.")
        return 0

    README.write_text(updated)
    print(f"README.md directory tree regenerated ({block.count(chr(10))} lines).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
