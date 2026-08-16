"""Does config B's velocity uncertainty ever fall below walking pace?
(Day 30, Objective 3.)

HARD SCOPE RULE, unchanged: no timing, throughput, CPU, or latency claim
is made anywhere in this script. Accuracy and consistency only.

The question, and why it is not the one Day 25 asked
------------------------------------------------------
Day 25's `velocity_floor_binding_audit.py` asked whether the floor is
REACHED — whether the clamp fires at all, or whether a constraint that
passes its unit tests is silently never hit by a real run. It answered
yes, and reported per regime that config B's sigma_v has
`min == p50 == max == 1.5000 m/s`.

Three order statistics coinciding is compatible with two very different
filters, and Day 25 did not have to distinguish them because it was
asking a wiring question:

  (i) the clamp fires often, and the filter's own uncertainty still
      varies underneath it; or
  (ii) the clamp fires on essentially EVERY frame, so config B's velocity
      uncertainty is a CONSTANT and the filter has stopped estimating it.

Day 29 established what that constant is:
`PERSON_SIGMA_A_MPS2 * PEDESTRIAN_STOP_DURATION_S = 1.5 m/s` is a
comfortable adult WALKING PACE, not a bound — the same mis-derivation
`PEDESTRIAN_MAX_SPEED_MPS`'s docstring rejects for the max-speed
constraint. Under (ii), config B reports a velocity uncertainty of
"roughly one walking pace" on every frame of every regime, forever. A
filter like that passes a calibration criterion the cheapest way
available: by never being confident.

That matters because ADR 0010 adopted config B on cessation coverage
moving 0.5013 -> 0.9946. A constant-uncertainty filter would produce
exactly that improvement and would deserve none of the credit.

The discriminating statistic is `fraction_at_floor`, added to
`_sigma_v_distribution` in `scripts/eval_estimator.py` so every report
path carries it, not just this script.

What this cannot settle on its own
-----------------------------------
"The floor binds everywhere" is necessary for (ii), not sufficient to
condemn config B: a bound that is always active is still a legitimate
estimator if the POINT ESTIMATES it produces are better. Day 26 measured
that they differ (RMSE moves bidirectionally, up to -62% at cessation),
so this script also reports the per-regime RMSE side by side — the
uncertainty verdict and the accuracy verdict are separate findings and
are printed separately.

    python scripts/velocity_floor_pinning_audit.py
    python scripts/velocity_floor_pinning_audit.py --version v6-motion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_estimator as ee  # noqa: E402
from src.config import IronConfig  # noqa: E402
from src.estimator.motion_model import (  # noqa: E402
    PEDESTRIAN_STOP_DURATION_S,
    PERSON_SIGMA_A_MPS2,
)

REGIMES = ("static", "onset", "sustained", "cessation", "maneuver")

PINNED_FRACTION_THRESHOLD = 0.99
"""Above this share of frames sitting exactly at the floor, config B's
velocity uncertainty is a constant in every sense that matters to a
downstream consumer. Declared here, before the measurement, so the
verdict is not chosen after seeing the number. 0.99 rather than 1.00
because `onset` legitimately has frames where natural uncertainty already
exceeds the floor (velocity really is changing fast there) and a single
such frame should not rescue a filter that is pinned everywhere else."""


def _score(version: str, root: Path, config: IronConfig, floor: bool) -> Any:
    return ee._score_golden_set(
        version,
        root,
        config,
        filter_kind="single_model",
        velocity_covariance_floor=floor,
    )


def _audit_version(version: str, root: Path, config: IronConfig) -> dict[str, Any] | None:
    reports = {}
    for label, floor in (("A", False), ("B", True)):
        report = _score(version, root, config, floor)
        if report is None or report.get("refused") or not report.get("tracks_scored"):
            print(f"Could not score {version} config {label}.")
            return None
        reports[label] = report

    rows = []
    total_frames = 0
    total_pinned = 0
    for regime in REGIMES:
        block_a = reports["A"]["by_regime"].get(regime, {})
        block_b = reports["B"]["by_regime"].get(regime, {})
        sigma_b = block_b.get("sigma_v_mps", {"n": 0})
        sigma_a = block_a.get("sigma_v_mps", {"n": 0})
        n = sigma_b.get("n", 0)
        if not n:
            rows.append({"regime": regime, "n": 0})
            continue
        total_frames += n
        total_pinned += sigma_b.get("frames_at_floor", 0)
        rows.append(
            {
                "regime": regime,
                "n": n,
                "b_min_mps": sigma_b["min_mps"],
                "b_p50_mps": sigma_b["p50_mps"],
                "b_max_mps": sigma_b["max_mps"],
                "b_frames_at_floor": sigma_b.get("frames_at_floor", 0),
                "b_fraction_at_floor": sigma_b.get("fraction_at_floor", float("nan")),
                "a_p50_mps": sigma_a.get("p50_mps", float("nan")),
                "a_rmse_m": block_a.get("filter_rmse_m", float("nan")),
                "b_rmse_m": block_b.get("filter_rmse_m", float("nan")),
                "a_coverage": block_a.get("consistency", {}).get("nees_pass_rate_within_95"),
                "b_coverage": block_b.get("consistency", {}).get("nees_pass_rate_within_95"),
            }
        )

    overall = total_pinned / total_frames if total_frames else float("nan")
    return {
        "version": version,
        "floor_sigma_v_mps": ee._floor_sigma_v_mps(),
        "rows": rows,
        "frames_scored": total_frames,
        "frames_at_floor": total_pinned,
        "fraction_at_floor": overall,
        "pinned": bool(total_frames and overall >= PINNED_FRACTION_THRESHOLD),
    }


def _print_audit(result: dict[str, Any]) -> None:
    print()
    print("=" * 92)
    print(f"{result['version']}  —  config B velocity uncertainty, per regime")
    print("=" * 92)
    print(
        f"{'regime':<11} {'n':>5}  {'B min':>8} {'B p50':>8} {'B max':>8}  "
        f"{'at floor':>14}  {'A p50':>8}  {'A RMSE':>8} {'B RMSE':>8}"
    )
    for row in result["rows"]:
        if not row["n"]:
            print(f"{row['regime']:<11} {0:>5}   (regime absent from this set)")
            continue
        print(
            f"{row['regime']:<11} {row['n']:>5}  "
            f"{row['b_min_mps']:>8.4f} {row['b_p50_mps']:>8.4f} "
            f"{row['b_max_mps']:>8.4f}  "
            f"{row['b_frames_at_floor']:>5}/{row['n']:<5} "
            f"{row['b_fraction_at_floor']*100:>5.1f}%  "
            f"{row['a_p50_mps']:>8.4f}  "
            f"{row['a_rmse_m']:>8.4f} {row['b_rmse_m']:>8.4f}"
        )
    print(
        f"\n  floor = PERSON_SIGMA_A_MPS2 ({PERSON_SIGMA_A_MPS2}) * "
        f"PEDESTRIAN_STOP_DURATION_S ({PEDESTRIAN_STOP_DURATION_S}) = "
        f"{result['floor_sigma_v_mps']:.4f} m/s — a comfortable walking pace"
    )
    print(
        f"  overall: {result['frames_at_floor']}/{result['frames_scored']} "
        f"scored frames pinned at the floor "
        f"({result['fraction_at_floor']*100:.2f}%)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = config.paths.resolve(config.eval.golden_sets_dir)
    versions = args.version or ["v5-cessation", "v3-indoor"]

    results = []
    for version in versions:
        result = _audit_version(version, root, config)
        if result is None:
            return 1
        results.append(result)
        _print_audit(result)

    print()
    print("=" * 92)
    print("VERDICT — is config B an estimator, or a constant?")
    print("=" * 92)
    for result in results:
        state = "PINNED" if result["pinned"] else "varies"
        print(
            f"  {result['version']:<14} {state:<7} "
            f"{result['fraction_at_floor']*100:.2f}% of scored frames at the "
            f"floor (threshold {PINNED_FRACTION_THRESHOLD*100:.0f}%)"
        )
    if all(r["pinned"] for r in results):
        print(
            "\n  Config B's velocity uncertainty is a CONSTANT on every set "
            "measured. It does not estimate velocity uncertainty; it reports "
            f"{results[0]['floor_sigma_v_mps']:.2f} m/s. Its calibration "
            "result is therefore not evidence that it models cessation "
            "better — a filter that is never confident cannot be caught "
            "being overconfident. Whether it is nonetheless a better "
            "ESTIMATOR is a separate question, answered by the RMSE columns "
            "above, and must be argued on those numbers alone."
        )
    else:
        print(
            "\n  The floor does NOT bind on every frame: config B's velocity "
            "uncertainty still varies with the data in at least one set, so "
            "its calibration result reflects a filter that is still "
            "estimating."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
