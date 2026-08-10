"""At what frame rate does the velocity-covariance floor begin to bind?

Day 22 derived `pedestrian_velocity_covariance_floor_mps2` and found it
numerically inert at this project's 12fps: the filter's own natural
steady-state velocity variance (~0.065-0.09 (m/s)^2) sits well above the
floor (~0.0156 (m/s)^2) everywhere it was checked. Day 23 Objective 3
asks the question ADR 0010 left open: is that a 12fps-specific accident,
or does the floor never bind at any plausible frame rate?

The floor is ANALYTIC by construction (dt_s * PERSON_SIGMA_A_MPS2,
squared) -- deliberately never fitted, see motion_model.py. The natural
convergence side has no equally simple closed form (it is the steady
state of a discrete Riccati recursion whose behaviour as dt shrinks or
grows is not obvious by inspection -- a quick first-order finite-
difference argument suggests natural convergence could even MOVE THE
WRONG WAY as fps increases, which is reason enough not to guess). So,
consistent with how the 12fps number itself was established (measured
against real per-frame covariance, not derived), this script MEASURES
natural convergence directly: run the real single-model filter (config
A, no floor applied) over a synthetic constant-velocity walk at each of
a swept range of frame rates, read off the converged posterior velocity
variance, and compare it against the floor's own closed-form value at
that same dt.

No timing, throughput, or latency claim -- "frame rate" here is a
KINEMATIC/SAMPLING parameter of the filter (how often it receives an
update), never a claim about what this CPU can process per second.

    python scripts/velocity_floor_frame_rate_sweep.py
    python scripts/velocity_floor_frame_rate_sweep.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.estimator.filter import run_single_entity_filter
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import (
    motion_model_for,
    pedestrian_velocity_covariance_floor_mps2,
)
from src.model.episode import StateGraph, StateQuery, solve_state
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid

SEED = 20260811
WALK_SPEED_MPS = 0.6
"""This project's own synthetic walkers move ~0.5 m/s (see
motion_model.py's module docstring); 0.6 m/s is the same order,
deliberately not tuned to any single golden-set clip."""
SENSOR_ORIGIN_M = (0.0, 0.0, 5.0)
"""A fixed point 5m off the walking line -- keeps GT distance in the
~5.0-5.6m band (this project's dominant 3-8m bucket) across the whole
walk, without needing a real camera/extrinsics setup: the filter only
ever needs a scalar range for R, see filter.py's own sensor_origin_m
docstring."""
N_STEPS = 60
"""Fixed step count (not a fixed wall-clock duration) at every frame
rate tested, so "how many updates the filter has seen" is held constant
across the sweep and only dt_s varies -- ADR 0010 measured natural
convergence within single-digit updates, so 60 is comfortably past
steady state at every rate tested here."""
FPS_SWEEP = (
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    24.0,
    30.0,
    48.0,
    60.0,
    90.0,
    120.0,
    200.0,
    300.0,
    500.0,
    1000.0,
)
BASE_TS_NS = 1_700_000_000_000_000_000


def _natural_velocity_variance_mps2(fps: float) -> float:
    """Run config A (single model, floor DISABLED) over a synthetic
    constant-velocity walk at ``fps`` and return the converged posterior
    velocity variance (mean of the 3 velocity-diagonal covariance
    entries at the final step)."""
    dt_s = 1.0 / fps
    motion_model = motion_model_for("person", velocity_covariance_floor=False)
    measurement_model = measurement_model_for(
        sensor="velocity_floor_frame_rate_sweep", capability="state_estimation"
    )
    rng = np.random.default_rng(SEED)

    dt_ns = int(round(1e9 / fps))
    observations = []
    for t in range(N_STEPS):
        gt = np.array([WALK_SPEED_MPS * t * dt_s, 1.0, 0.0])
        distance = float(np.linalg.norm(gt - np.array(SENSOR_ORIGIN_M)))
        sigma = measurement_model.sigma_m(distance)
        noisy = gt + rng.normal(0.0, sigma, size=3)
        ts_ns = BASE_TS_NS + t * dt_ns
        observations.append(
            Observation(
                observation_id=generate_ulid(now_ns=ts_ns),
                sensor_id="sweep",
                ts_ns=ts_ns,
                frame_ref=f"sweep/frame-{t}",
                measurement=WorldPositionMeasurement(
                    x_m=float(noisy[0]), y_m=float(noisy[1]), z_m=float(noisy[2])
                ),
                uncertainty=Uncertainty(
                    kind="gaussian_3d", params=(("sigma_m", float(sigma)),)
                ),
                frame_of_reference=None,
                producer_shas=("scripts/velocity_floor_frame_rate_sweep.py",),
                envelope_status=(
                    "within_envelope"
                    if measurement_model.is_within_envelope(distance)
                    else "outside_envelope"
                ),
            )
        )

    graph = StateGraph()
    run_single_entity_filter(
        graph,
        observations,
        motion_model,
        measurement_model,
        manifest_sha="scripts/velocity_floor_frame_rate_sweep.py",
        sensor_origin_m=SENSOR_ORIGIN_M,
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    cov = estimate.cov_array()
    return float(np.mean(np.diag(cov)[3:6]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    for fps in FPS_SWEEP:
        dt_s = 1.0 / fps
        natural = _natural_velocity_variance_mps2(fps)
        floor = pedestrian_velocity_covariance_floor_mps2(dt_s)
        rows.append(
            {
                "fps": fps,
                "dt_s": dt_s,
                "natural_velocity_variance_mps2": natural,
                "floor_mps2": floor,
                "floor_binds": floor >= natural,
                "ratio_floor_over_natural": floor / natural if natural > 0 else None,
            }
        )

    print("=" * 88)
    print("VELOCITY-COVARIANCE FLOOR vs NATURAL KALMAN CONVERGENCE, by frame rate")
    print("=" * 88)
    print(
        f"{'fps':>8}  {'dt_s':>10}  {'natural (m/s)^2':>18}  {'floor (m/s)^2':>15}  "
        f"{'floor/natural':>14}  binds?"
    )
    print("-" * 88)
    for row in rows:
        ratio = row["ratio_floor_over_natural"]
        print(
            f"{row['fps']:>8.1f}  {row['dt_s']:>10.5f}  "
            f"{row['natural_velocity_variance_mps2']:>18.6f}  "
            f"{row['floor_mps2']:>15.6f}  "
            f"{ratio:>14.4f}  {'YES' if row['floor_binds'] else 'no'}"
        )

    binding = [row for row in rows if row["floor_binds"]]
    print()
    if binding:
        lo = min(row["fps"] for row in binding)
        hi = max(row["fps"] for row in binding)
        print(
            f"Floor binds (floor >= natural) for fps in the tested range: "
            f"[{lo}, {hi}]."
        )
    else:
        print(
            "Floor NEVER binds anywhere in the tested range "
            f"[{FPS_SWEEP[0]}, {FPS_SWEEP[-1]}] fps."
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(f"Written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
