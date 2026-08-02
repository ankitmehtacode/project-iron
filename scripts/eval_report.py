"""Report which golden set ``make eval`` would measure against, and its gaps.

Selection comes first, deliberately: the part that decides *what* gets measured
determines whether a number means anything, so it was built before any metric.
Scoring now runs on top of it against the synthetic indoor set, whose ground
truth is analytic and therefore exact.

Those scores are GEOMETRY and MOTION numbers on a renderer with no appearance
model. They are a real baseline for regression, and they are not a product
claim; Site Zero footage supersedes them.

    make eval
    python scripts/eval_report.py --version v1-driving
    python scripts/eval_report.py --root /tmp/candidate --version v3-indoor

Exit codes:
    0  the set was scored and a scorecard written
    1  the set is empty, unscorable, or legacy without an explicit opt-in
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "directory of golden-set manifests; defaults to config. Lets a "
            "candidate set be scored before it is promoted into the repo."
        ),
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
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

    from src.data import scorecard as scoring

    # The clip root follows the set, not a hardcoded name. A golden set records
    # which dataset its clips came from, so scoring v3 against v1's directory —
    # which the hardcoded path did the moment a second set existed — would
    # silently measure the wrong bytes under the right sha.
    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    if len(datasets) > 1:
        print(
            f"{golden.version} draws clips from {len(datasets)} datasets "
            f"({', '.join(sorted(datasets))}); scoring needs one clip root.",
            file=sys.stderr,
        )
        return 1
    dataset = datasets.pop() if datasets else "synthetic-indoor-v1"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset
    # The gate runs BEFORE the metric. Day 9's depth number existed, was
    # reproducible, and described its alignment fit; the only thing that would
    # have stopped it being quoted is a refusal computed first.
    from src.data import validity

    import numpy as _np

    gate_frames = None
    for clip in golden.clips:
        candidate = clip_root / f"{clip.clip_id}.npz"
        if candidate.exists():
            with _np.load(candidate) as sample:
                gate_frames = _np.asarray(sample["rgb"][:4])
            break

    verdicts = []
    if gate_frames is not None:
        verdict = validity.evaluate("motion_geometry", version, frames=gate_frames)
        verdicts.append(verdict.as_dict())
        if not verdict.passed:
            print()
            print(f"REFUSED: {version} cannot score motion_geometry.")
            print(f"  {verdict.reason}")
            print("  No motion metric is emitted. The refusal IS the result.")
            return 1

    print()
    card = scoring.compute(golden, clip_root, config.cascade.motion_gate_config())
    card.capability_gates = verdicts
    if not card.clips_scored:
        print("No clips could be scored; see caveats above.", file=sys.stderr)
        return 1

    card.caveats.insert(
        0,
        "SYNTHETIC-ONLY. Scored against analytic primitives with no global "
        "illumination, material response or lens model. These are GEOMETRY and "
        "MOTION numbers, not appearance numbers, and they must not be quoted "
        "externally. Site Zero footage supersedes them.",
    )
    print(card.render())

    out = scoring.write(
        card,
        config.paths.resolved_output_dir
        / "scorecards"
        / f"{golden.version}_{golden.set_sha[:12]}.json",
    )
    print(f"\nScorecard written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
