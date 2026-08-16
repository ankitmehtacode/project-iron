"""Measure the hard constraints against the golden sets' ground truth
(Day 29, Objective 3).

HARD SCOPE RULE, unchanged: no timing, throughput, CPU, or latency claim
is made anywhere in this script. Counts and accuracy only.

What this measures, and what a nonzero count would mean
---------------------------------------------------------
Every hard constraint claims to be true of any PHYSICAL body, regardless
of the twin, the filter, or the sensor. GT tracks are physical bodies by
construction (they are the trajectories the synthetic generator drew, not
estimates of them), so the expected violation count is exactly ZERO. A
nonzero count is a finding either way:

  - the constraint is mis-derived (threshold too tight, wrong sign, wrong
    physical claim), or
  - the generator produces unphysical motion, which would invalidate GT
    for every other measurement built on these sets.

Only the number distinguishes them, which is why this runs against GT and
not against filter output: an estimator's track can violate a hard
constraint honestly (that is what pruning is FOR), so measuring against
estimates would confound "the constraint is wrong" with "the estimator
was wrong," and the confounded number could not close either question.

What a zero count does NOT establish
---------------------------------------
Zero violations means no constraint in the registry refutes real physical
motion in these sets. Three limits on reading it as more than that, all
reported below as numbers rather than left as caveats:

1. **Margin.** A bound far above the data would record the same zero as
   a correctly-derived one, so each constraint reports the worst GT value
   seen alongside its threshold. The gap between them is how much of the
   zero is derivation and how much is slack. (The counterfactual is
   measured too, and it is not slack here: the TYPICAL-scale 1.5 m/s
   threshold rejected in `PEDESTRIAN_MAX_SPEED_MPS`'s docstring violates
   on 46.0% of v5-cessation frames, so this measurement does discriminate.)

2. **Untested constraints.** `gravity_floor_transition` is evaluated on
   every frame and exercised by none: `agent_xyz`'s vertical component is
   a CONSTANT 0.86 m in every golden track (the generator places each
   agent at `height_m / 2.0` and never moves it vertically), so GT
   vertical acceleration is identically zero. Its zero is a bounded null
   -- no vertical motion exists in this data to violate it -- not a pass.
   Same for `one_body_one_place` on v5-cessation, whose clips are all
   single-agent: 0 pairs evaluated, so 0 violations is vacuous there. It
   IS exercised on v3-indoor (2,680 pairs, closest approach 0.0185 m).

3. **Unbounded quantities.** A zero is only ever a zero over the
   constraints that EXIST. Day 29 reported one here and said so: v5-
   cessation GT reaches 36.58 m/s^2 (3.7g) of horizontal acceleration
   with 3.4% of frames above 1g, and NO constraint in the registry
   bounded it -- a walking SPEED, arrived at impossibly fast, is invisible
   to a speed bound, and horizontal motion is invisible to a free-fall
   bound. Day 29 declined to close that inside this measurement, on the
   grounds that inventing a threshold here is how a threshold gets fitted
   to the data it is supposed to judge.

   **Day 30 closed it from outside.** `max_pedestrian_acceleration` was
   derived from the biomechanics literature in
   `src/estimator/motion_model.py` and stated before being run, then
   measured here. v5-cessation now records **23 violations in 1021
   evaluated frames** -- so this script's headline for that set is no
   longer zero, and the nonzero is the generator, exactly as the second
   bullet at the top of this docstring predicts. v5-cessation is not
   re-rendered (its bytes are cited by a frozen manifest and it remains
   the record of what Days 21-29 measured); v6-motion is the physical
   replacement, and it mints only because its GT clears all five.

Constraints measured, and how each is fed from GT
----------------------------------------------------
The list is checked against `HARD_CONSTRAINTS` at runtime rather than
kept in sync by hand -- see `_score_version`'s registry assertion, and
Day 24's registry/consumer drift for why a hand-maintained list beside a
registry is a promise rather than a mechanism.

  - `one_body_one_place`: every distinct agent PAIR at every frame.
  - `max_pedestrian_velocity`: per-frame GT speed, finite-differenced
    from GT position (same convention as `src.estimator.regime`).
  - `max_pedestrian_acceleration`: per-frame |a|, with the braking flag
    taken from the sign of the speed change (the accelerating and braking
    bounds differ). Frames 0-1 are skipped: the frame-0 velocity mirror
    forces a[1] == 0 identically, so checking them would test the
    differencing convention rather than the data.
  - `mass_conservation`: consecutive OBSERVED frames of one track, with
    the elapsed gap -- on gapless synthetic GT this is one frame step,
    and the check is then "displacement reachable in dt", which is the
    same physical claim as the speed bound approached from continuity of
    existence rather than from a velocity bound. Reported separately
    anyway: they are different constraints and a future set with real
    occlusion gaps will separate them.
  - `gravity_floor_transition`: per-frame change in GT vertical velocity,
    against free fall over dt.

    python scripts/measure_gt_constraint_violations.py
    python scripts/measure_gt_constraint_violations.py --version v3-indoor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import IronConfig  # noqa: E402
from src.contracts.ground_truth import (  # noqa: E402
    GENERATOR_AXES,
    gt_position_track,
)
from src.data.golden import GoldenSetError, load_golden_set  # noqa: E402
from src.estimator.constraints import (  # noqa: E402
    GRAVITY_FLOOR_TRANSITION,
    HARD_CONSTRAINTS,
    MASS_CONSERVATION,
    MAX_PEDESTRIAN_ACCELERATION,
    MAX_PEDESTRIAN_VELOCITY,
    ONE_BODY_ONE_PLACE,
    STANDARD_GRAVITY_MPS2,
)
from src.estimator.motion_model import (  # noqa: E402
    PEDESTRIAN_MAX_ACCELERATION_MPS2,
    PEDESTRIAN_MAX_DECELERATION_MPS2,
    PEDESTRIAN_MAX_SPEED_MPS,
)
from src.model.constraint import (  # noqa: E402
    HardConstraintViolation,
    Satisfied,
    Unevaluable,
    evaluate_constraint,
)

FloatArray = npt.NDArray[np.float64]

VERTICAL_AXIS_NOTE = """Why this file no longer declares a vertical axis.

Day 29: `VERTICAL_AXIS = 1` was a hand-maintained integer sitting beside
a raw `agent_xyz` array, and the first version of this script had it set
to 2 -- citing `src/model/world.py`'s +z-up world frame, a real
convention, just not the one this array is in. It reported 16
`gravity_floor_transition` violations on v5-cessation, worst case
-36.58 m/s^2, 3.7x free fall. Every one was the agent's DEPTH motion read
as vertical motion: v5-cessation's walkers change speed abruptly by
construction, and an abrupt speed change along the depth axis looks
exactly like an impossible fall when the axis is mislabelled. The nonzero
count was a defect in the measuring apparatus, not in either thing this
script exists to measure -- the twelfth time this project found the
instrument rather than the subject at fault.

Day 30, Objective 4: the correct integer was never the fix. The fix is
that the ARRAY carries its convention. This script now reads GT through
`src.contracts.ground_truth.GtPositionTrack`, whose `axes` field is
required and undefaulted, and asks it for `vertical_component()` instead
of indexing. There is no integer here to set wrongly."""


class _Counter:
    """Violations, evaluations, and the extremal GT value seen.

    The extremum is what makes the margin reportable: for a bound of the
    form `x <= threshold`, the largest `x` in GT is the smallest
    threshold that would still have recorded zero violations.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.evaluated = 0
        self.violations = 0
        self.unevaluable = 0
        self.extreme_value = 0.0
        self.extreme_where = ""

    def record(
        self, outcome: object, value: float, where: str, *, larger_is_worse: bool = True
    ) -> None:
        self.evaluated += 1
        if isinstance(outcome, HardConstraintViolation):
            self.violations += 1
        elif isinstance(outcome, Unevaluable):
            # A hard constraint should never report this; counted rather
            # than assumed away, because "should never" is what Day 28's
            # vacuous-pass finding was made of.
            self.unevaluable += 1
        elif not isinstance(outcome, Satisfied):
            raise AssertionError(f"unhandled outcome {outcome!r}")
        worse = (
            value > self.extreme_value
            if larger_is_worse
            else value < self.extreme_value
        )
        if self.evaluated == 1 or worse:
            self.extreme_value = value
            self.extreme_where = where

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.name,
            "evaluated": self.evaluated,
            "violations": self.violations,
            "unevaluable": self.unevaluable,
            "extreme_gt_value": self.extreme_value,
            "extreme_at": self.extreme_where,
        }


# Day 30, Objective 4: `_gt_velocity` was deleted here. It was this file's
# own copy of the forward-difference convention already implemented in
# `src.estimator.regime.classify_track` and `scripts/eval_estimator.py` --
# a third copy, kept in sync by comment. `GtPositionTrack.differentiate`
# is now the single implementation, and the convention travels with the
# array rather than beside it.


def _score_version(
    version: str, root: Path, config: IronConfig
) -> dict[str, Any] | None:
    try:
        golden = load_golden_set(root, version)
    except GoldenSetError as exc:
        print(f"ERROR loading {version}: {exc}")
        return None

    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else f"synthetic-indoor-{version}"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    fps_by_clip: dict[str, float] = {}
    manifest_path = clip_root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        fps_by_clip = {c["clip_id"]: float(c["fps"]) for c in manifest.get("clips", [])}

    speed = _Counter("max_pedestrian_velocity")
    accel = _Counter("max_pedestrian_acceleration")
    mass = _Counter("mass_conservation")
    gravity = _Counter("gravity_floor_transition")
    one_body = _Counter("one_body_one_place")

    tracks = 0
    frames_total = 0
    peak_gt_accel_mps2 = 0.0
    frames_above_1g = 0
    frames_above_typical_scale_threshold = 0
    TYPICAL_SCALE_THRESHOLD_MPS = 1.5
    """The rejected mis-derivation (PERSON_SIGMA_A_MPS2 *
    PEDESTRIAN_STOP_DURATION_S). Counted, not argued about: see limit 1."""

    for clip in golden.clips:
        clip_path = clip_root / f"{clip.clip_id}.npz"
        if not clip_path.exists():
            continue
        with np.load(clip_path) as data:
            if "agent_xyz" not in data:
                continue
            agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
        fps = fps_by_clip.get(clip.clip_id, 12.0)
        dt_s = 1.0 / fps
        n_frames, n_agents, _ = agent_xyz.shape

        for agent_index in range(n_agents):
            positions = gt_position_track(agent_xyz[:, agent_index, :], GENERATOR_AXES)
            track = positions.values
            if positions.frames < 2:
                continue
            tracks += 1
            frames_total += positions.frames
            gt_velocity = positions.differentiate(dt_s)
            velocity = gt_velocity.values
            vertical_velocity = gt_velocity.vertical_component()

            for t in range(positions.frames):
                where = f"{clip.clip_id}/agent{agent_index}/frame{t}"

                speed_mps = float(np.linalg.norm(velocity[t]))
                speed.record(
                    evaluate_constraint(MAX_PEDESTRIAN_VELOCITY, speed_mps),
                    speed_mps,
                    where,
                )
                if speed_mps > TYPICAL_SCALE_THRESHOLD_MPS:
                    frames_above_typical_scale_threshold += 1

                if t >= 1:
                    accel_mps2 = float(
                        np.linalg.norm(velocity[t] - velocity[t - 1]) / dt_s
                    )
                    peak_gt_accel_mps2 = max(peak_gt_accel_mps2, accel_mps2)
                    if accel_mps2 > STANDARD_GRAVITY_MPS2:
                        frames_above_1g += 1
                    # Day 30's fifth constraint. Frames 0-1 are skipped for
                    # the same reason enforce_gt_physicality skips them: the
                    # frame-0 velocity mirror forces a[1] == 0 identically,
                    # so checking it would test the convention rather than
                    # the data. Braking direction from the speed change --
                    # the accelerating and braking bounds differ.
                    if t >= 2:
                        braking = bool(
                            np.linalg.norm(velocity[t])
                            < np.linalg.norm(velocity[t - 1])
                        )
                        accel.record(
                            evaluate_constraint(
                                MAX_PEDESTRIAN_ACCELERATION, accel_mps2, braking
                            ),
                            accel_mps2,
                            where,
                        )
                    displacement_m = float(np.linalg.norm(track[t] - track[t - 1]))
                    mass.record(
                        evaluate_constraint(MASS_CONSERVATION, displacement_m, dt_s),
                        displacement_m / dt_s,
                        where,
                    )
                    delta_vz = float(
                        vertical_velocity[t] - vertical_velocity[t - 1]
                    )
                    # Extremum tracked as the most-negative vertical
                    # acceleration -- the direction this bound binds in.
                    gravity.record(
                        evaluate_constraint(GRAVITY_FLOOR_TRANSITION, delta_vz, dt_s),
                        delta_vz / dt_s,
                        where,
                        larger_is_worse=False,
                    )

        # one_body_one_place: every distinct agent pair, every frame.
        for t in range(n_frames):
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    a = tuple(float(v) for v in agent_xyz[t, i])
                    b = tuple(float(v) for v in agent_xyz[t, j])
                    separation_m = float(
                        np.linalg.norm(agent_xyz[t, i] - agent_xyz[t, j])
                    )
                    one_body.record(
                        evaluate_constraint(ONE_BODY_ONE_PLACE, a, b),
                        separation_m,
                        f"{clip.clip_id}/agents{i}-{j}/frame{t}",
                        larger_is_worse=False,
                    )

    counters = [one_body, speed, accel, mass, gravity]
    measured_names = {counter.name for counter in counters}
    registry_names = {constraint.name for constraint in HARD_CONSTRAINTS}
    if measured_names != registry_names:
        # Day 24's registry/consumer drift, pre-empted. This script is the
        # instrument for HARD_CONSTRAINTS, and a constraint added to the
        # registry that never reaches here would be a bound nobody ever
        # measured against GT -- which is exactly how v5-cessation's 3.7g
        # went unrefuted for eight days.
        raise AssertionError(
            "this script does not measure every registered hard constraint: "
            f"missing {sorted(registry_names - measured_names)}, "
            f"unknown {sorted(measured_names - registry_names)}"
        )
    return {
        "version": version,
        "tracks": tracks,
        "frames": frames_total,
        "total_violations": sum(c.violations for c in counters),
        "total_unevaluable": sum(c.unevaluable for c in counters),
        "peak_gt_accel_mps2": peak_gt_accel_mps2,
        "frames_above_1g": frames_above_1g,
        "typical_scale_threshold_mps": TYPICAL_SCALE_THRESHOLD_MPS,
        "frames_above_typical_scale_threshold": frames_above_typical_scale_threshold,
        "constraints": [c.as_dict() for c in counters],
    }


def _print_version(result: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"GT HARD-CONSTRAINT VIOLATIONS: {result['version']}")
    print("=" * 78)
    print(f"tracks: {result['tracks']}   frames: {result['frames']}")
    print()
    header = f"{'constraint':<28}{'evaluated':>11}{'violations':>12}{'unevaluable':>13}"
    print(header)
    for row in result["constraints"]:
        print(
            f"{row['constraint']:<28}{row['evaluated']:>11}"
            f"{row['violations']:>12}{row['unevaluable']:>13}"
        )
    print()
    print("Margin between the declared bound and the worst GT value seen")
    print("(what the zero count above does and does not establish):")
    by_name = {r["constraint"]: r for r in result["constraints"]}
    speed_row = by_name["max_pedestrian_velocity"]
    print(
        f"  max_pedestrian_velocity : fastest GT speed "
        f"{speed_row['extreme_gt_value']:.4f} m/s vs bound "
        f"{PEDESTRIAN_MAX_SPEED_MPS} m/s "
        f"({PEDESTRIAN_MAX_SPEED_MPS / max(speed_row['extreme_gt_value'], 1e-9):.1f}x)"
    )
    counterfactual = result["frames_above_typical_scale_threshold"]
    print(
        f"      counterfactual: the rejected TYPICAL-scale threshold "
        f"({result['typical_scale_threshold_mps']} m/s) would violate on "
        f"{counterfactual}/{result['frames']} GT frames "
        f"({100.0 * counterfactual / max(result['frames'], 1):.1f}%)"
    )
    accel_row = by_name["max_pedestrian_acceleration"]
    print(
        f"  max_pedestrian_acceleration: worst GT |a| "
        f"{accel_row['extreme_gt_value']:.4f} m/s^2 vs bounds "
        f"{PEDESTRIAN_MAX_ACCELERATION_MPS2} (accelerating) / "
        f"{PEDESTRIAN_MAX_DECELERATION_MPS2} (braking) m/s^2 -- "
        f"{accel_row['violations']}/{accel_row['evaluated']} frames violate"
    )
    mass_row = by_name["mass_conservation"]
    print(
        f"  mass_conservation       : fastest GT gap crossing "
        f"{mass_row['extreme_gt_value']:.4f} m/s vs bound "
        f"{PEDESTRIAN_MAX_SPEED_MPS} m/s"
    )
    gravity_row = by_name["gravity_floor_transition"]
    print(
        f"  gravity_floor_transition: most-negative GT vertical accel "
        f"{gravity_row['extreme_gt_value']:.4f} m/s^2 vs bound "
        f"{-STANDARD_GRAVITY_MPS2:.4f} m/s^2"
    )
    one_body_row = by_name["one_body_one_place"]
    if one_body_row["evaluated"] == 0:
        print(
            "  one_body_one_place      : 0 pairs evaluated (single-agent "
            "clips) -- its zero is vacuous on this set"
        )
    else:
        print(
            f"  one_body_one_place      : closest GT agent approach "
            f"{one_body_row['extreme_gt_value']:.4f} m (bound is a "
            f"floating-point epsilon, not a body radius)"
        )
    if gravity_row["extreme_gt_value"] == 0.0:
        print(
            "  NOTE: GT vertical acceleration is identically zero in this "
            "set (agent height is a constant 0.86 m), so "
            "gravity_floor_transition's zero is a bounded null, not a pass."
        )
    peak_accel = result["peak_gt_accel_mps2"]
    if result["frames_above_1g"] > 0:
        print(
            f"  NOTE: peak GT |acceleration| {peak_accel:.2f} m/s^2 "
            f"({result['frames_above_1g']} frames above 1g) -- unphysical "
            "for a body on foot. Bounded, since Day 30, by "
            "max_pedestrian_acceleration; see its violation count above "
            "and this script's docstring, limit 3."
        )
    elif peak_accel == 0.0:
        print(
            "  NOTE: GT acceleration is identically zero in this set -- "
            "every track is exactly constant-velocity. Nothing here can "
            "test an acceleration-conditioned model in either direction."
        )
    else:
        print(f"  NOTE: peak GT |acceleration| {peak_accel:.2f} m/s^2.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append", help="golden set version")
    parser.add_argument("--json", type=Path, help="write results as JSON")
    args = parser.parse_args()

    config = IronConfig.load()
    root = config.paths.resolve(config.eval.golden_sets_dir)
    versions = args.version or ["v5-cessation", "v3-indoor"]

    results = []
    for version in versions:
        result = _score_version(version, root, config)
        if result is None:
            return 1
        results.append(result)
        _print_version(result)

    total = sum(r["total_violations"] for r in results)
    total_unevaluable = sum(r["total_unevaluable"] for r in results)
    print()
    print("=" * 78)
    print(f"TOTAL GT HARD-CONSTRAINT VIOLATIONS: {total}")
    print(f"TOTAL UNEVALUABLE (hard constraints): {total_unevaluable}")
    if total == 0:
        print(
            "Zero, as a correctly-derived set of hard constraints requires. "
            "See this script's docstring for what that does NOT establish."
        )
    else:
        print(
            "NONZERO -- either a constraint is mis-derived or the generator "
            "produces unphysical motion. Both are findings; see the "
            "extremal values above for which."
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
