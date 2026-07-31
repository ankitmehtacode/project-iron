"""Report which golden set ``make eval`` would measure against, and its gaps.

The metric computation itself does not exist yet — there is nothing to measure
until Site Zero produces annotated clips. What exists now is the part that
decides *what* gets measured, and that is worth having first: it is the piece
that determines whether a number means anything.

    make eval
    python scripts/eval_report.py --version v1-driving

Exit codes:
    0  the selected set is active and may back a product metric
    1  the set is empty, or legacy and selected without an explicit opt-in
"""

from __future__ import annotations

import argparse
import sys

from src.config import IronConfig
from src.data.golden import (
    GoldenSetError,
    SiteZeroPlan,
    Status,
    available_versions,
    load_golden_set,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", default=None, help="golden-set version; defaults to config"
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = config.paths.resolve(config.eval.golden_sets_dir)
    version = args.version or config.eval.golden_set_version

    try:
        golden = load_golden_set(root, version)
    except GoldenSetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Available: {available_versions(root) or 'none'}", file=sys.stderr)
        return 1

    print("=" * 72)
    print(f"GOLDEN SET: {golden.version}")
    print("=" * 72)
    print(f"domain      : {golden.domain.value}")
    print(f"status      : {golden.status.value}")
    print(f"clips       : {len(golden.clips)}")
    print(f"set_sha     : {golden.set_sha}")
    if golden.supersedes:
        print(f"supersedes  : {golden.supersedes}")
    print(f"description : {golden.description}")
    print()

    if golden.status is Status.LEGACY:
        print(
            "This set is LEGACY. It is retained for pipeline regression and\n"
            "must not back a product metric — it measures a different domain\n"
            "from the one the product runs in."
        )
        if not config.eval.allow_legacy_golden_set and args.version is None:
            print(
                "\nRefusing: config selects a legacy set without "
                "allow_legacy_golden_set.",
                file=sys.stderr,
            )
            return 1

    gap = SiteZeroPlan().gap(golden)
    print(
        f"Composition against the Site Zero plan: {gap['clips_present']}/"
        f"{gap['clips_target']} clips"
    )
    if gap["clips_short"]:
        print(f"  {gap['clips_short']} more clips needed")
    if gap["conditions_short"]:
        print(f"  {len(gap['conditions_short'])} conditions uncovered:")
        for condition, short in sorted(gap["conditions_short"].items()):
            print(f"    {condition:26} needs {short} more")

    if not golden.clips:
        print()
        print(
            "This set is EMPTY, so it can measure nothing. That is the honest\n"
            "state until Site Zero capture and annotation land — an empty set\n"
            "reporting no failures is not a passing grade."
        )
        return 1

    print()
    print("Ready to evaluate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
