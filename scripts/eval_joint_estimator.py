"""Joint (coupled) vs independent estimation for a carried asset (Day 26, Objective 4).

No timing, throughput, CPU-percentage, or latency claim is made anywhere in
this script -- same hard scope rule as ``scripts/eval_estimator.py``.

No golden set this project owns carries a real carried-object entity in its
GT (Day 26 Objective 2's finding: no "laptop 7" exists in any clip's GT or
manifest). This script therefore SYNTHESIZES one: for every scored agent
track, a carried asset's GT position is the agent's own GT position plus a
fixed, declared, RIGID offset (:data:`CARRY_OFFSET_M`) -- no synthetic
slip added to the ground truth itself, so this evaluates whether coupling
correctly exploits a genuinely rigid attachment, not how it degrades under
real-world slip, which is unmeasured (no real carried-object GT exists to
measure it against). The asset's observations are synthesized the same way
the carrier's are -- :func:`~scripts.eval_estimator._make_observations`,
reused directly, same measurement-noise model, same pinned seed stream.

Trivial baseline (Day 12 rule): INDEPENDENT per-entity filtering is the
baseline here, not a strawman. The carrier is filtered with Day 25's
adopted config B (:func:`~src.estimator.motion_model.motion_model_for`,
``"person"``, floor enabled); the asset, independently, with the
``asset_carried`` motion model that has existed, unused, since Day 20
(inflated CV process noise, no coupling to any carrier). Joint estimation
must beat this on the asset to justify existing at all -- if it does not,
that is the headline, and it would mean the coupling built Day 26
Objective 3 is not actually wired through to the posterior.

    python scripts/eval_joint_estimator.py
    python scripts/eval_joint_estimator.py --version v5-cessation
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_estimator as ee  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.data import validity  # noqa: E402
from src.data.golden import GoldenSetError, load_golden_set  # noqa: E402
from src.estimator.consistency import compute_nees, fraction_outside_bound  # noqa: E402
from src.estimator.joint import (  # noqa: E402
    Component,
    JointObservation,
    JointStateEstimate,
    resolve_joint_state,
    run_joint_filter,
)
from src.estimator.measurement_model import measurement_model_for  # noqa: E402
from src.estimator.motion_model import motion_model_for  # noqa: E402
from src.estimator.regime import classify_track  # noqa: E402
from src.estimator.filter import run_single_entity_filter  # noqa: E402
from src.model.episode import StateGraph, StateQuery  # noqa: E402

FloatArray = npt.NDArray[np.float64]

EVAL_SEED_CARRIER = ee.EVAL_SEED
EVAL_SEED_ASSET = 20260813
"""Separate stream from the carrier's own observation noise -- same
convention as EVAL_SEED_MIXTURE_SAMPLING (Day 22): the asset's noise draws
must never perturb the carrier's, so a shared filter's output does not
depend on evaluation order."""

CARRY_OFFSET_M: FloatArray = np.array([0.25, 0.0, -0.30], dtype=np.float64)
"""Declared, not fitted: a held object at roughly hip height and slightly
to the side of a person's own GT reference point. Held perfectly rigid in
GT (no synthetic slip) -- see module docstring for why."""

MIN_FRAMES_FOR_A_CONCLUSION = ee.MIN_REGIME_FRAMES_FOR_A_CONCLUSION


def _rmse(sq_errors: list[float]) -> float:
    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


def _coverage(nees_values: list[Any]) -> dict[str, float]:
    within = [r.within_bound for r in nees_values if r.within_bound is not None]
    fraction_outside = fraction_outside_bound(nees_values, "nees")
    coverage = (
        1.0 - fraction_outside if not np.isnan(fraction_outside) else float("nan")
    )
    return {
        "n": len(nees_values),
        "nees_pass_rate_within_95": float(np.mean(within)) if within else float("nan"),
        "empirical_coverage_95": coverage,
    }


def _evaluate_one_track(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    clip_id: str,
    fps: float,
    measurement_model: Any,
) -> dict[str, Any] | None:
    """One agent's track, scored both ways (joint, independent). Returns
    per-frame records for carrier and asset under each method, plus the GT
    regime label per frame (carrier's own -- the asset has no independent
    kinematics to classify)."""
    frames = track_xyz.shape[0]
    if frames <= ee.FIRST_COMPARABLE_INDEX:
        return None

    asset_xyz = track_xyz + CARRY_OFFSET_M[np.newaxis, :]
    dt_s = 1.0 / fps
    regimes = classify_track(track_xyz, dt_s)

    carrier_rng = np.random.default_rng(EVAL_SEED_CARRIER)
    asset_rng = np.random.default_rng(EVAL_SEED_ASSET)
    carrier_obs = ee._make_observations(
        track_xyz, extrinsics, measurement_model, fps, f"{clip_id}/carrier", carrier_rng
    )
    asset_obs = ee._make_observations(
        asset_xyz, extrinsics, measurement_model, fps, f"{clip_id}/asset", asset_rng
    )

    carrier_motion_model = motion_model_for("person", velocity_covariance_floor=True)

    # -- joint --
    component = Component(carrier_entity_id="carrier", carried_entity_ids=("asset",))
    joint_obs = [JointObservation("carrier", o) for o in carrier_obs] + [
        JointObservation("asset", o) for o in asset_obs
    ]
    joint_graph = StateGraph()
    run_joint_filter(
        joint_graph,
        component,
        joint_obs,
        carrier_motion_model,
        measurement_model,
        manifest_sha="scripts/eval_joint_estimator.py",
    )

    # -- independent --
    carrier_graph = StateGraph()
    run_single_entity_filter(
        carrier_graph,
        carrier_obs,
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model,
        manifest_sha="scripts/eval_joint_estimator.py",
    )
    asset_graph = StateGraph()
    run_single_entity_filter(
        asset_graph,
        asset_obs,
        motion_model_for("asset_carried", velocity_covariance_floor=True),
        measurement_model,
        manifest_sha="scripts/eval_joint_estimator.py",
    )

    records: dict[str, list[dict[str, Any]]] = {
        "carrier_joint": [],
        "carrier_independent": [],
        "asset_joint": [],
        "asset_independent": [],
    }
    for t in range(ee.FIRST_COMPARABLE_INDEX, frames):
        ts_ns = carrier_obs[t].ts_ns
        gt_carrier = track_xyz[t]
        gt_asset = asset_xyz[t]
        regime = regimes[t]

        joint_query = StateQuery(
            at_ts_ns=ts_ns, horizon_ns=0, graph_rev=joint_graph.graph_rev
        )
        resolved = resolve_joint_state(joint_query, joint_graph)
        assert isinstance(resolved, JointStateEstimate)  # component.size == 2 always
        joint_estimate = resolved
        carrier_joint_pos = joint_estimate.carrier_position_m()
        asset_joint_pos = joint_estimate.carried_position_m("asset")
        joint_full_gt_carrier = np.concatenate([gt_carrier, np.zeros(3)])
        # NEES against the carrier sub-block only (position+velocity) --
        # the asset's offset dims are not scored against a GT velocity
        # that does not exist for a rigidly-coupled offset.
        carrier_error = joint_estimate.mean_array()[:6] - joint_full_gt_carrier
        carrier_cov6 = joint_estimate.cov_array()[:6, :6]
        asset_error = asset_joint_pos - gt_asset
        asset_cov3 = joint_estimate.carried_position_cov_m2("asset")

        records["carrier_joint"].append(
            {
                "regime": regime,
                "sq_error": float(
                    np.dot(
                        carrier_joint_pos - gt_carrier, carrier_joint_pos - gt_carrier
                    )
                ),
                "nees": compute_nees(carrier_error, carrier_cov6),
            }
        )
        records["asset_joint"].append(
            {
                "regime": regime,
                "sq_error": float(np.dot(asset_error, asset_error)),
                "nees": compute_nees(asset_error, asset_cov3),
            }
        )

        carrier_query = StateQuery(
            at_ts_ns=ts_ns, horizon_ns=0, graph_rev=carrier_graph.graph_rev
        )
        carrier_estimate = ee.solve_state(carrier_query, carrier_graph)
        carrier_ind_err6 = carrier_estimate.mean_array() - joint_full_gt_carrier
        records["carrier_independent"].append(
            {
                "regime": regime,
                "sq_error": float(
                    np.dot(
                        carrier_estimate.position_m() - gt_carrier,
                        carrier_estimate.position_m() - gt_carrier,
                    )
                ),
                "nees": compute_nees(carrier_ind_err6, carrier_estimate.cov_array()),
            }
        )

        asset_query = StateQuery(
            at_ts_ns=ts_ns, horizon_ns=0, graph_rev=asset_graph.graph_rev
        )
        asset_estimate = ee.solve_state(asset_query, asset_graph)
        asset_ind_full_gt = np.concatenate([gt_asset, np.zeros(3)])
        asset_ind_err6 = asset_estimate.mean_array() - asset_ind_full_gt
        records["asset_independent"].append(
            {
                "regime": regime,
                "sq_error": float(
                    np.dot(
                        asset_estimate.position_m() - gt_asset,
                        asset_estimate.position_m() - gt_asset,
                    )
                ),
                "nees": compute_nees(asset_ind_err6, asset_estimate.cov_array()),
            }
        )

    return records


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "rmse_m": _rmse([r["sq_error"] for r in records]),
        **_coverage([r["nees"] for r in records]),
    }


def _aggregate_by_regime(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from src.estimator.regime import MOTION_REGIMES

    return {
        regime: _aggregate([r for r in records if r["regime"] == regime])
        for regime in MOTION_REGIMES
    }


def _margin(candidate_rmse: float, baseline_rmse: float) -> float:
    """Positive = candidate (joint) beats baseline (independent) -- same
    sign convention as src.eval.baselines.margin."""
    if np.isnan(candidate_rmse) or np.isnan(baseline_rmse):
        return float("nan")
    return baseline_rmse - candidate_rmse


@dataclass
class DirectionalCheckResult:
    """Day 25's directional no-trade rule, applied to THIS evaluation's
    actual comparison (joint vs independent, per entity) rather than
    Day 22-25's cessation-vs-steady-regime comparison -- the rule is the
    same (overconfidence fails outright, underconfidence is a cost); the
    axis it is checked across is different, so this is a purpose-built
    type rather than a forced reuse of NoTradePass/NoTradePassWithCost's
    cessation-shaped fields."""

    entity: str
    baseline_coverage: float
    candidate_coverage: float
    delta_toward_nominal: float
    verdict: str


def _directional_check(
    entity: str, baseline_coverage: float, candidate_coverage: float
) -> DirectionalCheckResult | None:
    if np.isnan(baseline_coverage) or np.isnan(candidate_coverage):
        return None
    delta = abs(baseline_coverage - ee.NOMINAL_COVERAGE) - abs(
        candidate_coverage - ee.NOMINAL_COVERAGE
    )
    if delta >= -ee.NO_TRADE_DEGRADATION_TOLERANCE:
        verdict = "PASS"
    else:
        overconfident = (candidate_coverage - ee.NOMINAL_COVERAGE) < 0
        verdict = "FAIL_OVERCONFIDENT" if overconfident else "PASS_WITH_COST"
    return DirectionalCheckResult(
        entity, baseline_coverage, candidate_coverage, delta, verdict
    )


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
        import json

        manifest = json.loads(manifest_path.read_text())
        fps_by_clip = {c["clip_id"]: float(c["fps"]) for c in manifest.get("clips", [])}

    gate_frames = None
    for clip in golden.clips:
        candidate = clip_root / f"{clip.clip_id}.npz"
        if candidate.exists():
            with np.load(candidate) as sample:
                gate_frames = np.asarray(sample["rgb"][:4])
            break
    if gate_frames is not None:
        verdict = validity.evaluate("state_estimation", version, frames=gate_frames)
        if not verdict.passed:
            print(f"\nREFUSED: {version} cannot score state_estimation.")
            print(f"  {verdict.reason}")
            return {"version": version, "refused": True}
        print(f"[Day-10 validity gate] {version}: PASS state_estimation")

    measurement_model = measurement_model_for(
        sensor=version, capability="state_estimation"
    )

    all_records: dict[str, list[dict[str, Any]]] = {
        "carrier_joint": [],
        "carrier_independent": [],
        "asset_joint": [],
        "asset_independent": [],
    }
    tracks_scored = 0
    for clip in golden.clips:
        clip_path = clip_root / f"{clip.clip_id}.npz"
        if not clip_path.exists():
            continue
        with np.load(clip_path) as data:
            if "agent_xyz" not in data:
                continue
            agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
            extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        fps = fps_by_clip.get(clip.clip_id, 12.0)
        n_agents = agent_xyz.shape[1]
        for agent_index in range(n_agents):
            track = agent_xyz[:, agent_index, :]
            result = _evaluate_one_track(
                track, extrinsics, clip.clip_id, fps, measurement_model
            )
            if result is None:
                continue
            tracks_scored += 1
            for key in all_records:
                all_records[key].extend(result[key])

    if tracks_scored == 0:
        return {"version": version, "refused": False, "tracks_scored": 0}

    return {
        "version": version,
        "refused": False,
        "tracks_scored": tracks_scored,
        "carrier_joint": _aggregate(all_records["carrier_joint"]),
        "carrier_independent": _aggregate(all_records["carrier_independent"]),
        "asset_joint": _aggregate(all_records["asset_joint"]),
        "asset_independent": _aggregate(all_records["asset_independent"]),
        "asset_joint_by_regime": _aggregate_by_regime(all_records["asset_joint"]),
        "asset_independent_by_regime": _aggregate_by_regime(
            all_records["asset_independent"]
        ),
        "carrier_joint_by_regime": _aggregate_by_regime(all_records["carrier_joint"]),
        "carrier_independent_by_regime": _aggregate_by_regime(
            all_records["carrier_independent"]
        ),
    }


def _print_report(report: dict[str, Any]) -> None:
    version = report["version"]
    print("=" * 78)
    print(f"JOINT vs INDEPENDENT: {version}")
    print("=" * 78)
    if report.get("refused") or report.get("tracks_scored", 0) == 0:
        print("No trajectory scored (refused or empty).")
        return

    print(f"tracks scored: {report['tracks_scored']}")
    print()

    for label, entity in (
        ("CARRIER (should be neutral)", "carrier"),
        ("CARRIED ASSET (coupling should help most)", "asset"),
    ):
        joint = report[f"{entity}_joint"]
        indep = report[f"{entity}_independent"]
        margin = _margin(joint["rmse_m"], indep["rmse_m"])
        print(f"-- {label} --")
        print(
            f"  independent (baseline): n={indep['n']:5}  "
            f"RMSE {indep['rmse_m']:.4f} m  "
            f"coverage {indep['empirical_coverage_95']:.4f}"
        )
        print(
            f"  joint (candidate)     : n={joint['n']:5}  "
            f"RMSE {joint['rmse_m']:.4f} m  "
            f"coverage {joint['empirical_coverage_95']:.4f}"
        )
        print(f"  margin (positive = joint beats independent): {margin:+.4f} m")
        check = _directional_check(
            entity, indep["empirical_coverage_95"], joint["empirical_coverage_95"]
        )
        if check is not None:
            print(
                f"  directional no-trade check: {check.verdict} "
                f"(delta-toward-nominal {check.delta_toward_nominal:+.4f})"
            )
        print()

    _print_by_regime(
        "Carried asset",
        report["asset_joint_by_regime"],
        report["asset_independent_by_regime"],
    )
    _print_by_regime(
        "Carrier",
        report["carrier_joint_by_regime"],
        report["carrier_independent_by_regime"],
    )


def _print_by_regime(
    label: str,
    joint_by_regime: dict[str, dict[str, Any]],
    independent_by_regime: dict[str, dict[str, Any]],
) -> None:
    print(f"{label}, by GT regime:")
    print(
        f"{'regime':<12} {'n':>6}  {'indep RMSE':>11}  {'joint RMSE':>11}  "
        f"{'margin':>9}  {'indep cov':>10}  {'joint cov':>10}"
    )
    for regime, joint_block in joint_by_regime.items():
        indep_block = independent_by_regime[regime]
        if joint_block["n"] == 0:
            print(f"{regime:<12} {0:>6}  (empty)")
            continue
        margin = _margin(joint_block["rmse_m"], indep_block["rmse_m"])
        print(
            f"{regime:<12} {joint_block['n']:>6}  {indep_block['rmse_m']:>11.4f}  "
            f"{joint_block['rmse_m']:>11.4f}  {margin:>+9.4f}  "
            f"{indep_block['empirical_coverage_95']:>10.4f}  "
            f"{joint_block['empirical_coverage_95']:>10.4f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
    versions = args.version or ["v5-cessation", "v3-indoor"]

    any_failed = False
    for version in versions:
        report = _score_version(version, root, config)
        if report is None:
            any_failed = True
            continue
        _print_report(report)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
