"""What does the component-size cap cost, when it actually triggers? (Day 27)

No timing, throughput, CPU-percentage, or latency claim is made anywhere in
this script -- same hard scope rule as every other eval script in this
project. Component counts/sizes remain explicitly in scope as structural
scene properties.

The only implemented joint-inference topology is carrier+carried (Day 26
Objective 3, a star rooted at one carrier) -- general person-to-person
proximity coupling, the kind Day 26 Objective 2 measured on
``crowded_6agents``, is NOT implemented (see that objective's own report:
proximity was a MEASUREMENT to sanity-check the sparsity assumption before
building anything, never a solver's own grouping key). So there is no
genuinely-6-PERSON joint solve to cap in this codebase today.

To still measure the cap's real cost on real data rather than a synthetic
toy, this script builds an HONESTLY-LABELED mechanism test: v3-indoor's
``crowded_6agents`` scene (the one Day 26 found merges at 1.10m) supplies
6 real GT tracks; one is designated "carrier" and the other five "carried"
purely to exercise a genuinely 6-entity joint Component using the topology
that exists. This is NOT a claim that five people are rigidly attached to
a sixth -- it is the only honest way to put a real, data-derived 6-entity
joint solve in front of the cap mechanism without inventing synthetic
data. The comparison that matters (capped vs uncapped on the SAME 6-entity
inputs) is unaffected by this labeling choice, since both runs see
identical observations either way.

    python scripts/measure_component_cap_cost.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_estimator as ee  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.estimator.consistency import compute_nees  # noqa: E402
from src.estimator.joint import (  # noqa: E402
    Component,
    ComponentCapConfig,
    DegradedComponentEstimate,
    JointObservation,
    JointStateEstimate,
    resolve_joint_state,
    run_joint_filter,
)
from src.estimator.measurement_model import measurement_model_for  # noqa: E402
from src.estimator.motion_model import motion_model_for  # noqa: E402
from src.model.episode import StateGraph, StateQuery  # noqa: E402

FloatArray = npt.NDArray[np.float64]

CLIP_ID = "crowded_6agents__cam_a"
DATASET = "synthetic-indoor-v3"
EVAL_SEED = 20260814
UNCAPPED = ComponentCapConfig(
    max_component_size=10, degradation_action="independent_fallback"
)
CAPPED = ComponentCapConfig(
    max_component_size=2, degradation_action="independent_fallback"
)


def _load_clip() -> tuple[FloatArray, FloatArray, float]:
    config = IronConfig.load()
    clip_root = config.paths.resolved_data_dir / "synthetic" / DATASET
    with np.load(clip_root / f"{CLIP_ID}.npz") as data:
        agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    manifest = json.loads((clip_root / "dataset_manifest.json").read_text())
    fps = next(c["fps"] for c in manifest["clips"] if c["clip_id"] == CLIP_ID)
    return agent_xyz, extrinsics, float(fps)


def _run(cap_config: ComponentCapConfig) -> dict[str, Any]:
    agent_xyz, extrinsics, fps = _load_clip()
    n_agents = agent_xyz.shape[1]
    assert n_agents == 6, f"expected 6 agents, got {n_agents}"

    measurement_model = measurement_model_for(
        sensor="v3-indoor", capability="state_estimation"
    )
    entity_ids = [f"agent-{i}" for i in range(n_agents)]
    component = Component(
        carrier_entity_id=entity_ids[0], carried_entity_ids=tuple(entity_ids[1:])
    )

    rng = np.random.default_rng(EVAL_SEED)
    obs_by_entity: dict[str, list[Any]] = {}
    for i, eid in enumerate(entity_ids):
        track = agent_xyz[:, i, :]
        obs_by_entity[eid] = ee._make_observations(
            track, extrinsics, measurement_model, fps, eid, rng
        )

    joint_obs = [
        JointObservation(eid, o) for eid in entity_ids for o in obs_by_entity[eid]
    ]

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model,
        manifest_sha="scripts/measure_component_cap_cost.py",
        cap_config=cap_config,
    )

    frames = agent_xyz.shape[0]
    per_entity: dict[str, list[dict[str, Any]]] = {eid: [] for eid in entity_ids}
    for t in range(ee.FIRST_COMPARABLE_INDEX, frames):
        ts_ns = obs_by_entity[entity_ids[0]][t].ts_ns
        query = StateQuery(at_ts_ns=ts_ns, horizon_ns=0, graph_rev=graph.graph_rev)
        resolved = resolve_joint_state(query, graph)

        for i, eid in enumerate(entity_ids):
            gt_pos = agent_xyz[t, i, :]
            if isinstance(resolved, DegradedComponentEstimate):
                single = resolved.estimate_for(eid)
                pos = single.position_m()
                full_gt = np.concatenate([gt_pos, np.zeros(3)])
                nees = compute_nees(single.mean_array() - full_gt, single.cov_array())
            elif isinstance(resolved, JointStateEstimate):
                if eid == component.carrier_entity_id:
                    pos = resolved.carrier_position_m()
                    full_gt = np.concatenate([gt_pos, np.zeros(3)])
                    nees = compute_nees(
                        resolved.mean_array()[:6] - full_gt,
                        resolved.cov_array()[:6, :6],
                    )
                else:
                    pos = resolved.carried_position_m(eid)
                    nees = compute_nees(
                        pos - gt_pos, resolved.carried_position_cov_m2(eid)
                    )
            else:
                raise AssertionError(f"unexpected resolved type: {type(resolved)}")
            per_entity[eid].append(
                {"sq_error": float(np.dot(pos - gt_pos, pos - gt_pos)), "nees": nees}
            )

    return {eid: _aggregate(records) for eid, records in per_entity.items()}


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    from src.estimator.consistency import fraction_outside_bound

    sq_errors = [r["sq_error"] for r in records]
    nees_list = [r["nees"] for r in records]
    within = [r.within_bound for r in nees_list if r.within_bound is not None]
    fraction_outside = fraction_outside_bound(nees_list, "nees")
    coverage = (
        1.0 - fraction_outside if not np.isnan(fraction_outside) else float("nan")
    )
    return {
        "n": len(records),
        "rmse_m": float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan"),
        "coverage_95": coverage,
        "nees_pass_rate": float(np.mean(within)) if within else float("nan"),
    }


def main() -> int:
    print("=" * 78)
    print(
        f"COMPONENT-SIZE CAP COST: {CLIP_ID} (6 real GT tracks, 1 carrier + 5 carried)"
    )
    print("=" * 78)
    print(
        f"UNCAPPED: max_component_size={UNCAPPED.max_component_size} "
        f"(component size 6 fits -- genuine joint solve)"
    )
    print(
        f"CAPPED:   max_component_size={CAPPED.max_component_size} "
        f"(component size 6 exceeds -- degrades to independent_fallback)"
    )
    print()

    uncapped = _run(UNCAPPED)
    capped = _run(CAPPED)

    print(
        f"{'entity':<10} {'uncapped RMSE':>14}  {'capped RMSE':>12}  {'cost (m)':>9}  "
        f"{'uncapped cov':>13}  {'capped cov':>11}"
    )
    for eid in uncapped:
        u, c = uncapped[eid], capped[eid]
        cost = c["rmse_m"] - u["rmse_m"]
        print(
            f"{eid:<10} {u['rmse_m']:>14.4f}  {c['rmse_m']:>12.4f}  {cost:>+9.4f}  "
            f"{u['coverage_95']:>13.4f}  {c['coverage_95']:>11.4f}"
        )

    mean_cost = float(
        np.mean([capped[e]["rmse_m"] - uncapped[e]["rmse_m"] for e in uncapped])
    )
    print()
    print(
        "Mean RMSE cost of capping (capped - uncapped), across all 6 entities: "
        f"{mean_cost:+.4f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
