"""Rebuild derived artifacts after the encoder or preprocessing changed.

Named by every :class:`~src.artifacts.ArtifactMismatch` error, so it exists and
is runnable rather than being a suggestion the error makes up.

Why rebuilding is a scripted operation
--------------------------------------
Re-exporting or re-quantizing an encoder, or changing a preprocessing
constant, invalidates every vector derived from the old one. Recovering from
that in an ad-hoc notebook is how half-rebuilt indexes happen: some vectors
from the new encoder, some from the old, one searchable structure, no way to
tell which is which afterwards. So the rebuild is a single command that writes
the metadata sidecar in the same operation that writes the artifact.

Status: the audit and inventory paths are implemented and useful now. The
actual vector regeneration is not, because it depends on the encoder path whose
preprocessing defect is still open — rebuilding today would produce a second
generation of void vectors. ``--audit`` tells you what is stale; the rebuild
itself refuses with an explanation until that lands.

    python scripts/rebuild_index.py --audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.artifacts import (
    METADATA_SUFFIX,
    ArtifactMismatch,
    read_metadata,
)
from src.config import IronConfig

# Extensions treated as derived artifacts worth auditing.
ARTIFACT_GLOBS = ("*.faiss", "*.index", "*.parquet", "*.npy", "*.npz")


def find_artifacts(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ARTIFACT_GLOBS:
            for path in sorted(root.rglob(pattern)):
                if path.name.endswith(METADATA_SUFFIX):
                    continue
                found.append(path)
    return found


def audit(roots: list[Path]) -> int:
    """Report which derived artifacts can name their producers and which cannot.

    Returns:
        The number of artifacts that are void or unverifiable.
    """
    artifacts = find_artifacts(roots)
    if not artifacts:
        print("No derived artifacts found under:")
        for root in roots:
            print(f"  {root}")
        print(
            "\nNothing to rebuild. Any embeddings produced from here on will "
            "carry a metadata sidecar."
        )
        return 0

    void: list[tuple[Path, str]] = []
    described: list[Path] = []
    for path in artifacts:
        try:
            metadata = read_metadata(path)
        except ArtifactMismatch:
            void.append((path, "no metadata sidecar — predates provenance coupling"))
        else:
            described.append(path)
            print(
                f"OK    {path}\n"
                f"        encoder {metadata.encoder_sha[:12]}...  "
                f"preprocess {metadata.preprocess_sha[:12]}...  "
                f"run {metadata.manifest_sha[:12]}..."
            )

    for path, reason in void:
        print(f"VOID  {path}\n        {reason}")

    print(
        f"\n{len(described)} artifact(s) carry provenance, "
        f"{len(void)} are void or unverifiable."
    )
    if void:
        print(
            "\nVoid artifacts were written before provenance coupling existed. "
            "They came from the pipeline that applied no channel "
            "standardisation, so their vectors cannot be compared with "
            "anything produced now. Delete them or regenerate them once the "
            "encoder path is fixed."
        )
    return len(void)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="report artifact provenance status; write nothing",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="directories to scan; defaults to the configured output and cache dirs",
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    roots = (
        [Path(r) for r in args.roots]
        if args.roots
        else [config.paths.resolved_output_dir, config.paths.resolved_cache_dir]
    )

    if args.audit:
        return 0 if audit(roots) == 0 else 1

    print(
        "Rebuilding vectors is not implemented yet, deliberately.\n"
        "\n"
        "The encoder path still applies no channel standardisation (audit\n"
        "finding 3), so a rebuild today would write a second generation of\n"
        "void vectors and make the situation harder to reason about, not\n"
        "easier. Fix the preprocessing first, with its golden-vector\n"
        "measurement, then implement regeneration here.\n"
        "\n"
        "Run with --audit to see which artifacts are currently void.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
