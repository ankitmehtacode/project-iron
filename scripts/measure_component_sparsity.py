"""Measure the multi-entity factor graph's founding assumption before any solver exists.

The multi-entity work this project has deferred since Day 20 (see
``src/model/episode.py``'s module docstring, and ``asset_carried``'s
``carrier_entity_id`` interface in ``src/estimator/motion_model.py``) rests on an
ANALYTICAL claim, recalled going into Day 26 as living in "Data model v0.3 §3":
that entity coupling is sparse enough that joint-inference components stay
small (informally, "typically 1-6 entities"). Grep-verified before writing a
line of measurement code: **no such document, section, or sentence exists
anywhere in this repository** (`docs/`, `src/model/`, `FOUNDATION_REPORT.md`).
`src/model/relationship.py`'s ``Relationship.predicate`` is a free string with
only illustrative examples (``"same_identity_as"``, ``"contains"``,
``"assigned_to"``) -- there is no closed vocabulary for proximity, carried-object,
or shared-zone coupling anywhere in the schema, and no dataset in this project
tags any of the three. The claim is unwritten, not merely unmeasured -- which
makes it a stronger candidate for the Day-25/26 rule (a closed-form or
undocumented analytical claim about a system's behaviour is a hypothesis, not
a measurement) than a claim that was at least written down carefully once.
This script measures what CAN be measured on data this project owns, and
states plainly what cannot be, rather than assuming the claim either way.

Of the three relationship types Day 26's objective names:

- **Proximity within a declared metric threshold** -- MEASURABLE directly from
  GT ``agent_xyz`` (every golden set already carries exact synthetic position
  for every agent, every frame).
- **Carried-object coupling** -- NOT MEASURABLE on any golden set this project
  owns. No carried-asset entity (a "laptop 7" carried by a person) exists in
  any clip's GT or manifest; ``BAG_CARRIED`` is a clip-level ``Condition`` tag
  (metadata about the SCENE), not a per-frame entity relationship.
- **Shared-zone occupancy** -- NOT MEASURABLE. No golden set tags per-agent
  zone/room membership; ``gen_synthetic_indoor.py`` records one fixed
  ``room_size`` per scene, not a subdivided zone graph.

So this script's entire measurable signal is proximity-based connectivity
between agents, on the one golden set that has multi-agent clips at all
(v3-indoor: 1/2/3/6 agents per clip, 30 clips total). v4.1-gate (0-1 agents/
clip) and v5-cessation (always exactly 1 agent/clip) are included for
completeness and are trivially component-size 1 everywhere by construction --
not because sparsity held on them, but because there is nothing in their GT
to couple.

**Component size and count are structural properties of the scene GT, not a
performance claim** -- this project's hard scope rule (no timing/throughput/
CPU/latency claim) explicitly carves these out. Component LIFETIME is
reported in frames (a GT/scene property, like Day 21's "10-14 frame recovery
tail"), with each clip's own declared fps noted for context only.

    python scripts/measure_component_sparsity.py
    python scripts/measure_component_sparsity.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.config import IronConfig
from src.data.golden import GoldenSetError, load_golden_set

FloatArray = npt.NDArray[np.float64]

GOLDEN_VERSIONS: tuple[str, ...] = ("v3-indoor", "v4.1-gate", "v5-cessation")

PROXIMITY_THRESHOLD_SWEEP_M: tuple[float, ...] = (
    0.3,
    0.5,
    0.75,
    0.9,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
    1.3,
    1.4,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    14.0,
)
"""Room shells in this project's synthetic generator are 10m x 14m on the
floor plan (``scripts/gen_synthetic_indoor.py``'s ``room = (10.0, 3.0,
14.0)``), so the sweep runs from well inside personal space (0.3m) to the
room's own diagonal, past the point where every threshold value stops
changing the answer."""

DEFAULT_PROXIMITY_THRESHOLD_M = 1.5
"""Declared, not fitted: proxemics' "close social" distance boundary
(~1.2-2.1m, Hall 1966) is the natural physical threshold for "close enough
to plausibly interact or hand off an object" -- the actual coupling
relationships (carried-object, conversation) this measurement is a proxy
for. Used only for the single-threshold component-size/lifetime report;
the full sensitivity sweep above does not depend on this choice."""

MIN_FRAMES_FOR_LIFETIME = 1


def _connected_components(n: int, adjacency: set[tuple[int, int]]) -> list[list[int]]:
    """Plain BFS connected components over ``n`` nodes -- no networkx
    dependency for a graph that never exceeds 6 nodes in this project's
    own data."""
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b in adjacency:
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen = [False] * n
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in neighbors[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(sorted(component))
    return components


def _pairwise_adjacency(
    positions: FloatArray, threshold_m: float
) -> set[tuple[int, int]]:
    n = positions.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(positions[i] - positions[j])) < threshold_m:
                edges.add((i, j))
    return edges


@dataclass
class ClipFrames:
    clip_id: str
    fps: float
    agent_xyz: FloatArray  # [T, n_agents, 3]


def _load_clip_frames(root: Path, version: str) -> list[ClipFrames]:
    golden = load_golden_set(root, version)
    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else f"synthetic-indoor-{version}"
    config = IronConfig.load()
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    fps_by_clip: dict[str, float] = {}
    manifest_path = clip_root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        fps_by_clip = {c["clip_id"]: float(c["fps"]) for c in manifest.get("clips", [])}

    out = []
    for clip in golden.clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            if "agent_xyz" not in data:
                continue
            agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
        if agent_xyz.shape[1] == 0:
            continue
        out.append(
            ClipFrames(
                clip_id=clip.clip_id,
                fps=fps_by_clip.get(clip.clip_id, 12.0),
                agent_xyz=agent_xyz,
            )
        )
    return out


@dataclass
class SetResult:
    version: str
    n_clips: int
    n_clips_multi_agent: int
    agent_count_histogram: dict[int, int]
    threshold_sweep: list[dict[str, Any]] = field(default_factory=list)
    default_threshold_component_sizes: list[int] = field(default_factory=list)
    default_threshold_per_frame_max: list[int] = field(default_factory=list)
    lifetimes_frames: list[int] = field(default_factory=list)
    per_clip_knee: list[dict[str, Any]] = field(default_factory=list)


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": len(values),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _measure_set(root: Path, version: str) -> SetResult | None:
    try:
        clips = _load_clip_frames(root, version)
    except GoldenSetError as exc:
        print(f"ERROR loading {version}: {exc}")
        return None

    agent_count_histogram: dict[int, int] = {}
    n_multi = 0
    for clip in clips:
        n_agents = clip.agent_xyz.shape[1]
        agent_count_histogram[n_agents] = agent_count_histogram.get(n_agents, 0) + 1
        if n_agents > 1:
            n_multi += 1

    result = SetResult(
        version=version,
        n_clips=len(clips),
        n_clips_multi_agent=n_multi,
        agent_count_histogram=agent_count_histogram,
    )

    # -- per-clip knee: smallest threshold at which THIS clip first merges at
    # all, and smallest threshold at which it fully merges into one component
    # -- names which specific scene drives a pooled-statistic jump, not just
    # that one exists.
    for clip in clips:
        n_agents = clip.agent_xyz.shape[1]
        if n_agents <= 1:
            continue
        first_merge: float | None = None
        full_merge: float | None = None
        for threshold in PROXIMITY_THRESHOLD_SWEEP_M:
            clip_max = 1
            for t in range(clip.agent_xyz.shape[0]):
                edges = _pairwise_adjacency(clip.agent_xyz[t], threshold)
                components = _connected_components(n_agents, edges)
                clip_max = max(clip_max, max(len(c) for c in components))
            if first_merge is None and clip_max > 1:
                first_merge = threshold
            if full_merge is None and clip_max == n_agents:
                full_merge = threshold
            if first_merge is not None and full_merge is not None:
                break
        result.per_clip_knee.append(
            {
                "clip_id": clip.clip_id,
                "n_agents": n_agents,
                "first_merge_threshold_m": first_merge,
                "full_merge_threshold_m": full_merge,
            }
        )

    # -- threshold sweep: per-frame max component size, pooled across clips --
    for threshold in PROXIMITY_THRESHOLD_SWEEP_M:
        per_frame_max: list[int] = []
        for clip in clips:
            n_agents = clip.agent_xyz.shape[1]
            if n_agents <= 1:
                per_frame_max.extend([n_agents] * clip.agent_xyz.shape[0])
                continue
            for t in range(clip.agent_xyz.shape[0]):
                positions = clip.agent_xyz[t]
                edges = _pairwise_adjacency(positions, threshold)
                components = _connected_components(n_agents, edges)
                per_frame_max.append(max(len(c) for c in components))
        dist = _distribution(per_frame_max)
        fraction_merged = (
            float(np.mean(np.array(per_frame_max) > 1))
            if per_frame_max
            else float("nan")
        )
        result.threshold_sweep.append(
            {
                "threshold_m": threshold,
                **dist,
                "fraction_frames_merged": fraction_merged,
            }
        )

    # -- default-threshold component sizes (every component, every frame) --
    # and per-frame max, and lifetime of merged partitions.
    for clip in clips:
        n_agents = clip.agent_xyz.shape[1]
        if n_agents <= 1:
            result.default_threshold_component_sizes.extend(
                [n_agents] * clip.agent_xyz.shape[0]
            )
            result.default_threshold_per_frame_max.extend(
                [n_agents] * clip.agent_xyz.shape[0]
            )
            continue

        prior_partition: frozenset[frozenset[int]] | None = None
        run_length = 0
        for t in range(clip.agent_xyz.shape[0]):
            positions = clip.agent_xyz[t]
            edges = _pairwise_adjacency(positions, DEFAULT_PROXIMITY_THRESHOLD_M)
            components = _connected_components(n_agents, edges)
            sizes = [len(c) for c in components]
            result.default_threshold_component_sizes.extend(sizes)
            result.default_threshold_per_frame_max.append(max(sizes))

            partition = frozenset(frozenset(c) for c in components)
            is_merged = max(sizes) > 1
            if is_merged and partition == prior_partition:
                run_length += 1
            else:
                if (
                    prior_partition is not None
                    and run_length >= MIN_FRAMES_FOR_LIFETIME
                ):
                    prior_was_merged = any(len(c) > 1 for c in prior_partition)
                    if prior_was_merged:
                        result.lifetimes_frames.append(run_length)
                run_length = 1 if is_merged else 0
            prior_partition = partition if is_merged else None
        if prior_partition is not None and run_length >= MIN_FRAMES_FOR_LIFETIME:
            prior_was_merged = any(len(c) > 1 for c in prior_partition)
            if prior_was_merged:
                result.lifetimes_frames.append(run_length)

    return result


def _print_result(result: SetResult) -> None:
    print("=" * 78)
    print(f"COMPONENT SPARSITY: {result.version}")
    print("=" * 78)
    print(f"clips: {result.n_clips}  (multi-agent: {result.n_clips_multi_agent})")
    histogram = dict(sorted(result.agent_count_histogram.items()))
    print(f"agent-count histogram (clips): {histogram}")
    print()
    print("Threshold sweep (per-frame MAX component size, pooled):")
    print(
        f"{'threshold_m':>12}  {'p50':>6}  {'p95':>6}  {'max':>6}  {'frac merged':>12}"
    )
    for row in result.threshold_sweep:
        print(
            f"{row['threshold_m']:>12.2f}  {row['p50']:>6.2f}  {row['p95']:>6.2f}  "
            f"{row['max']:>6.0f}  {row['fraction_frames_merged']:>12.4f}"
        )
    print()
    dist = _distribution(result.default_threshold_component_sizes)
    frame_dist = _distribution(result.default_threshold_per_frame_max)
    print(
        f"At default threshold ({DEFAULT_PROXIMITY_THRESHOLD_M}m): "
        "component-size distribution (every component, every frame): "
        f"n={dist['n']} p50={dist['p50']:.2f} p95={dist['p95']:.2f} "
        f"max={dist['max']:.0f}"
    )
    print(
        "  per-frame MAX component size: "
        f"n={frame_dist['n']} p50={frame_dist['p50']:.2f} "
        f"p95={frame_dist['p95']:.2f} max={frame_dist['max']:.0f}"
    )
    if result.lifetimes_frames:
        life_dist = _distribution(result.lifetimes_frames)
        print(
            "  merged-component lifetimes (frames), "
            f"n={len(result.lifetimes_frames)}: "
            f"p50={life_dist['p50']:.1f} p95={life_dist['p95']:.1f} "
            f"max={life_dist['max']:.0f}"
        )
    else:
        print("  no merged component ever formed at this threshold on this set")
    print()

    if result.per_clip_knee:
        print("Per-clip knee (smallest threshold that first merges / fully merges):")
        print(
            f"{'clip_id':<38} {'n':>3}  {'first merge (m)':>16}  {'full merge (m)':>15}"
        )
        for row in sorted(
            result.per_clip_knee,
            key=lambda r: (
                r["first_merge_threshold_m"] is None,
                r["first_merge_threshold_m"],
            ),
        ):
            fm = row["first_merge_threshold_m"]
            full = row["full_merge_threshold_m"]
            print(
                f"{row['clip_id']:<38} {row['n_agents']:>3}  "
                f"{('%.2f' % fm) if fm is not None else 'never':>16}  "
                f"{('%.2f' % full) if full is not None else 'never':>15}"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )

    results = []
    for version in GOLDEN_VERSIONS:
        result = _measure_set(root, version)
        if result is None:
            continue
        _print_result(result)
        results.append(result)

    if args.json:
        payload = {
            "default_threshold_m": DEFAULT_PROXIMITY_THRESHOLD_M,
            "sets": [
                {
                    "version": r.version,
                    "n_clips": r.n_clips,
                    "n_clips_multi_agent": r.n_clips_multi_agent,
                    "agent_count_histogram": r.agent_count_histogram,
                    "threshold_sweep": r.threshold_sweep,
                    "default_threshold_component_size_distribution": _distribution(
                        r.default_threshold_component_sizes
                    ),
                    "default_threshold_per_frame_max_distribution": _distribution(
                        r.default_threshold_per_frame_max
                    ),
                    "lifetimes_frames": r.lifetimes_frames,
                    "per_clip_knee": r.per_clip_knee,
                }
                for r in results
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Written: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
