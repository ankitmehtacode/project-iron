"""Component sizes on SmartSpaces crowding (Day 41, Objective 4).

PhysicalAI-SmartSpaces — lane S, synthetic. Not evidence about real crowding.

Day 26 (``scripts/measure_component_sparsity.py``) could not test the
joint solver's component cap: every owned set had 2–6 agents, so a cap of
6 (``DEFAULT_COMPONENT_CAP_CONFIG``) was never stressed. SmartSpaces ground
truth has real entity counts, so this re-runs the same measurement on it:

- the SAME coupling relationship — two entities are coupled when the 3D
  Euclidean distance between their positions is strictly below a
  threshold — and the SAME threshold sweep, 1.10 m knee included;
- the SAME declared default threshold, 1.5 m (Day 26's proxemics bound);
- positions are each object's ``3d location`` (the 3D box centre — Day 41
  Objective 2), so a forklift's centre sits higher than a person's; the
  Day-26 relationship is 3D and is kept 3D rather than redefined here.

**Entity counts are reported first.** If p95 entities per frame is at or
below the cap, the data cannot stress the cap, and the component numbers
after it are not interpreted.

Cosmos Transfer re-renders (001, 003, 005, 007, 021) share their source
scene's ground truth byte for byte (integrity report), so they are
excluded: counting them would count the same crowd twice.

Component lifetime is per component: a merged component (size > 1) lives
for as many consecutive frames as the exact same set of object ids stays
one component. Day 26's lifetime was per clip-wide partition, which is the
same thing when at most one merged component exists at a time — true of
Day 26's data, not of these scenes.

Component size and count are structural properties of the scene GT, not a
performance claim. No timing claim is made.

    python scripts/measure_smartspaces_components.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_component_sparsity import (  # noqa: E402
    DEFAULT_PROXIMITY_THRESHOLD_M,
    PROXIMITY_THRESHOLD_SWEEP_M,
)
from smartspaces_integrity import (  # noqa: E402
    CT25_RERENDER_OF,
    LANE_LABEL,
    REPO_ROOT,
    load_scene,
    scene_dirs,
)

from src.estimator.joint import DEFAULT_COMPONENT_CAP_CONFIG  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "smartspaces" / "day41_component_sizes.json"
CAP = DEFAULT_COMPONENT_CAP_CONFIG.max_component_size
KNEE_M = 1.10
REPORTED_THRESHOLDS_M = (KNEE_M, DEFAULT_PROXIMITY_THRESHOLD_M)
FPS = 30.0
STATIONARY_MPS = 0.05
"""A member of an over-cap component counts as stationary when its frame-to-
frame displacement is below 0.05 m/s at the scenes' 30 fps. Descriptive
only: it says what the over-cap components are made of."""
assert KNEE_M in PROXIMITY_THRESHOLD_SWEEP_M


def proximity_mst(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum spanning tree of the complete 3D-distance graph, as
    ``(row, col, distance)`` edge arrays.

    Two entities are in one component of the Day-26 graph at threshold t
    (edges where distance < t) exactly when every edge on their MST path is
    shorter than t — the single-linkage property — so one tree per frame
    answers every threshold in the sweep. Equivalence with Day 26's own
    ``_pairwise_adjacency`` + ``_connected_components`` is a test
    (``tests/test_smartspaces_components.py``)."""
    n = positions.shape[0]
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    dist = squareform(pdist(positions))
    # scipy reads a zero entry as "no edge", and coincident entities are at
    # distance 0. Adding one constant to every edge leaves the MST unchanged
    # (every spanning tree has n - 1 edges); the true distances are read
    # back from ``dist`` so the strict < stays exact.
    tree = minimum_spanning_tree(dist + 1.0 - np.eye(n)).tocoo()
    row = tree.row.astype(np.int64)
    col = tree.col.astype(np.int64)
    return row, col, dist[row, col]


def sweep_components(
    n: int,
    mst: tuple[np.ndarray, np.ndarray, np.ndarray],
    thresholds: tuple[float, ...],
    label_thresholds: tuple[float, ...],
) -> tuple[list[int], dict[float, np.ndarray]]:
    """For ascending ``thresholds``: the largest component size at each, and
    a component label per entity at each of ``label_thresholds``.

    One union-find pass over the MST edges in distance order, joining an
    edge once its distance is strictly below the current threshold."""
    row, col, dist = mst
    order = np.argsort(dist, kind="stable")
    parent = list(range(n))
    size = [1] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    largest = 1 if n else 0
    maxima: list[int] = []
    labels: dict[float, np.ndarray] = {}
    edge = 0
    for threshold in thresholds:
        while edge < order.size and dist[order[edge]] < threshold:
            a, b = find(int(row[order[edge]])), find(int(col[order[edge]]))
            if a != b:
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                largest = max(largest, size[a])
            edge += 1
        maxima.append(largest)
        if threshold in label_thresholds:
            labels[threshold] = np.array([find(i) for i in range(n)], dtype=np.int64)
    return maxima, labels


def _dist(values: list[int] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def measure_scene(scene: Path) -> dict[str, Any]:
    gt = load_scene(scene)
    order = np.argsort(gt.frame, kind="stable")
    frame = gt.frame[order]
    ids = gt.object_id[order]
    cls = gt.class_index[order]
    loc = gt.location[order]
    bounds = np.flatnonzero(np.diff(frame)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [frame.size]])
    class_names = list(gt.class_names)
    person = class_names.index("Person") if "Person" in class_names else -1

    entities_per_frame: list[int] = []
    per_class_per_frame: dict[str, list[int]] = defaultdict(list)
    sweep_frame_max: dict[float, list[int]] = defaultdict(list)
    sweep_cap_frames: dict[float, int] = defaultdict(int)
    sizes: dict[float, list[int]] = defaultdict(list)
    entity_component_size: dict[float, dict[str, list[int]]] = {
        t: defaultdict(list) for t in REPORTED_THRESHOLDS_M
    }
    entity_frames_over_cap: dict[float, int] = defaultdict(int)
    people_only_max: dict[float, list[int]] = defaultdict(list)
    lifetimes: dict[float, list[int]] = defaultdict(list)
    alive: dict[float, dict[frozenset[int], int]] = {
        t: {} for t in REPORTED_THRESHOLDS_M
    }
    over_cap_classes: dict[float, dict[str, int]] = {
        t: defaultdict(int) for t in REPORTED_THRESHOLDS_M
    }
    over_cap_speeds: dict[float, list[float]] = defaultdict(list)
    previous_position: dict[int, np.ndarray] = {}

    for s, e in zip(starts, ends):
        positions = loc[s:e]
        frame_ids = ids[s:e]
        frame_cls = cls[s:e]
        n = e - s
        entities_per_frame.append(int(n))
        counts = np.bincount(frame_cls, minlength=len(class_names))
        for k, name in enumerate(class_names):
            per_class_per_frame[name].append(int(counts[k]))
        mst = proximity_mst(positions)
        maxima, labelled = sweep_components(
            int(n), mst, PROXIMITY_THRESHOLD_SWEEP_M, REPORTED_THRESHOLDS_M
        )
        people_labels: dict[float, np.ndarray] = {}
        if person >= 0:
            people = positions[frame_cls == person]
            _, people_labels = sweep_components(
                people.shape[0],
                proximity_mst(people),
                REPORTED_THRESHOLDS_M,
                REPORTED_THRESHOLDS_M,
            )
        for threshold, frame_max in zip(PROXIMITY_THRESHOLD_SWEEP_M, maxima):
            sweep_frame_max[threshold].append(frame_max)
            if frame_max > CAP:
                sweep_cap_frames[threshold] += 1
        for threshold in REPORTED_THRESHOLDS_M:
            labels = labelled[threshold]
            _, inverse, counts = np.unique(
                labels, return_inverse=True, return_counts=True
            )
            comp_sizes = counts
            sizes[threshold].extend(comp_sizes.tolist())
            per_entity = counts[inverse]
            entity_frames_over_cap[threshold] += int((per_entity > CAP).sum())
            for i in np.flatnonzero(per_entity > CAP):
                over_cap_classes[threshold][class_names[frame_cls[i]]] += 1
                last = previous_position.get(int(frame_ids[i]))
                if last is not None:
                    over_cap_speeds[threshold].append(
                        float(np.linalg.norm(positions[i] - last)) * FPS
                    )
            for k, name in enumerate(class_names):
                entity_component_size[threshold][name].extend(
                    per_entity[frame_cls == k].tolist()
                )
            if person >= 0:
                p_labels = people_labels[threshold]
                people_only_max[threshold].append(
                    int(np.unique(p_labels, return_counts=True)[1].max())
                    if p_labels.size
                    else 0
                )
            labels = inverse
            current = {
                frozenset(frame_ids[labels == c].tolist())
                for c in np.flatnonzero(comp_sizes > 1)
            }
            previous = alive[threshold]
            nxt = {members: previous.get(members, 0) + 1 for members in current}
            for members, run in previous.items():
                if members not in current:
                    lifetimes[threshold].append(run)
            alive[threshold] = nxt
        for i in range(int(n)):
            previous_position[int(frame_ids[i])] = positions[i]

    for threshold in REPORTED_THRESHOLDS_M:
        lifetimes[threshold].extend(alive[threshold].values())  # open at clip end

    n_frames = len(entities_per_frame)
    return {
        "scene": scene.name,
        "split": scene.parent.name,
        "frames": n_frames,
        "entities_per_frame": _dist(entities_per_frame),
        "entities_per_frame_by_class": {
            name: _dist(v) for name, v in per_class_per_frame.items()
        },
        "unique_entities_by_class": {
            name: int(np.unique(ids[cls == k]).size)
            for k, name in enumerate(class_names)
        },
        "threshold_sweep": [
            {
                "threshold_m": t,
                "per_frame_max_component": _dist(sweep_frame_max[t]),
                "fraction_frames_merged": float(
                    np.mean(np.array(sweep_frame_max[t]) > 1)
                ),
                "fraction_frames_cap_triggered": sweep_cap_frames[t] / n_frames,
            }
            for t in PROXIMITY_THRESHOLD_SWEEP_M
        ],
        "at_threshold": {
            str(t): {
                "component_size_every_component": _dist(sizes[t]),
                "per_frame_max_component": _dist(sweep_frame_max[t]),
                "fraction_frames_cap_triggered": sweep_cap_frames[t] / n_frames,
                "fraction_entity_frames_in_over_cap_component": (
                    entity_frames_over_cap[t] / max(sum(entities_per_frame), 1)
                ),
                "component_size_containing_entity_by_class": {
                    name: _dist(v) for name, v in entity_component_size[t].items()
                },
                "people_only_graph_per_frame_max": _dist(people_only_max[t]),
                "merged_component_lifetime_frames": _dist(lifetimes[t]),
                "over_cap_member_frames_by_class": dict(over_cap_classes[t]),
                "over_cap_member_fraction_stationary": (
                    float(np.mean(np.array(over_cap_speeds[t]) < STATIONARY_MPS))
                    if over_cap_speeds[t]
                    else None
                ),
            }
            for t in REPORTED_THRESHOLDS_M
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", default=[])
    args = parser.parse_args(argv)

    results = []
    for scene in scene_dirs():
        if scene.name in CT25_RERENDER_OF:
            continue
        if args.scene and scene.name not in args.scene:
            continue
        print(f"{scene.parent.name}/{scene.name} ...", flush=True)
        r = measure_scene(scene)
        results.append(r)
        epf = r["entities_per_frame"]
        at = r["at_threshold"][str(DEFAULT_PROXIMITY_THRESHOLD_M)]
        print(
            f"  entities/frame p50 {epf['p50']:.0f} p95 {epf['p95']:.0f} "
            f"max {epf['max']:.0f}; at {DEFAULT_PROXIMITY_THRESHOLD_M} m: "
            f"component p95 {at['component_size_every_component']['p95']:.0f} "
            f"max {at['component_size_every_component']['max']:.0f}, cap triggered "
            f"on {at['fraction_frames_cap_triggered']:.4f} of frames"
        )
    payload = {
        "label": LANE_LABEL,
        "cap": CAP,
        "cap_config_sha": DEFAULT_COMPONENT_CAP_CONFIG.sha,
        "coupling": (
            "3D Euclidean distance between 3d location (box centre) strictly "
            "below the threshold — Day 26's relationship, unchanged"
        ),
        "thresholds_m": list(PROXIMITY_THRESHOLD_SWEEP_M),
        "reported_thresholds_m": list(REPORTED_THRESHOLDS_M),
        "excluded_rerenders": sorted(CT25_RERENDER_OF),
        "stationary_below_mps": STATIONARY_MPS,
        "scenes": results,
    }
    if not args.scene:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Written: {ARTIFACT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
