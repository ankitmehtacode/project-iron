"""Readers for the artifacts the Inspector displays.

Every function here reads a real file off disk. There is no fixture path, no
sample payload, and no default-shaped object standing in for a missing one —
the Inspector exists to verify claims, and a viewer that can invent data cannot
verify anything.

When an artifact is absent that is a *result*, not an error to paper over. Each
reader returns an :class:`Absent` carrying the path it looked for and the
command that produces it, and the UI renders that as an instruction. This is
the same discipline the eval harness follows: a measurement that cannot be
taken is reported as not taken, never as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Absent:
    """A named artifact that is not on disk, and how to produce it."""

    what: str
    looked_for: str
    produced_by: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "absent": True,
            "what": self.what,
            "looked_for": self.looked_for,
            "produced_by": self.produced_by,
        }


@dataclass(frozen=True)
class Artifacts:
    """Where everything lives. Constructed from the real config, never guessed."""

    project_root: Path
    scorecards_dir: Path
    golden_dir: Path
    envelope_path: Path
    data_dir: Path
    events_path: Path
    associations_dir: Path
    coverage_queries_dir: Path

    @classmethod
    def from_config(cls) -> "Artifacts":
        from src.config import IronConfig

        config = IronConfig.load()
        root = config.paths.project_root
        return cls(
            project_root=root,
            scorecards_dir=root / "outputs" / "scorecards",
            golden_dir=config.paths.resolve(config.eval.golden_sets_dir),
            envelope_path=root / "configs" / "envelope" / "gate_320x180.envelope.json",
            data_dir=config.paths.resolved_data_dir,
            events_path=root / "outputs" / "events" / "events.parquet",
            associations_dir=root / "outputs" / "associations",
            coverage_queries_dir=root / "outputs" / "coverage_queries",
        )

    def relative(self, path: Path) -> str:
        """Path as shown in the UI. Every number on screen names its source."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


def list_scorecards(artifacts: Artifacts) -> list[dict[str, Any]]:
    """Every scorecard on disk, newest first, with its identifying shas."""
    if not artifacts.scorecards_dir.exists():
        return []
    rows = []
    for path in sorted(artifacts.scorecards_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "file": artifacts.relative(path),
                    "unreadable": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "file": artifacts.relative(path),
                "name": path.stem,
                "golden_set_version": payload.get("golden_set_version"),
                "golden_set_sha": payload.get("golden_set_sha"),
                "envelope_sha": payload.get("envelope", {}).get("envelope_sha"),
                "clips_scored": payload.get("clips_scored"),
                "modified_epoch": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: r.get("modified_epoch", 0), reverse=True)
    return rows


def read_scorecard(artifacts: Artifacts, name: str) -> dict[str, Any] | Absent:
    path = artifacts.scorecards_dir / f"{name}.json"
    if not path.exists():
        return Absent(
            what=f"scorecard {name!r}",
            looked_for=artifacts.relative(path),
            produced_by="make eval",
        )
    payload = json.loads(path.read_text())
    payload["_source"] = artifacts.relative(path)
    payload["_hard_coverage"] = _hard_coverage_ids(
        artifacts, payload.get("golden_set_version", "")
    )
    return payload


def _hard_coverage_ids(artifacts: Artifacts, version: str) -> list[str]:
    """Clips the golden set marks as deliberately hard to observe.

    Read from the manifest rather than inferred from low scores, so a clip that
    merely came out badly is never quietly reclassified as intentional.
    """
    if not version:
        return []
    from src.data.golden import GoldenSetError, load_golden_set

    try:
        golden = load_golden_set(artifacts.golden_dir, version)
    except (GoldenSetError, FileNotFoundError):
        return []
    return [clip.clip_id for clip in golden.clips if clip.hard_coverage]


def comparison(artifacts: Artifacts, left_name: str, right_name: str) -> dict[str, Any]:
    """Compare two scorecards, or refuse and say why.

    The refusal is the point. Two scorecards built on different golden sets or
    different capability envelopes are answers to different questions, and a
    delta between them renders as a regression that never happened. The Python
    already refuses this; the UI must refuse it identically rather than draw a
    chart the library would not have produced.
    """
    left = read_scorecard(artifacts, left_name)
    right = read_scorecard(artifacts, right_name)
    for card in (left, right):
        if isinstance(card, Absent):
            return {"refused": True, "reason": card.what + " is not on disk"}

    reasons = []
    if left["golden_set_sha"] != right["golden_set_sha"]:
        reasons.append(
            "different golden sets "
            f"({str(left['golden_set_sha'])[:12]} vs "
            f"{str(right['golden_set_sha'])[:12]}) — a delta across them is "
            "not a delta, because the two runs measured different clips"
        )
    left_env = left.get("envelope", {}).get("envelope_sha")
    right_env = right.get("envelope", {}).get("envelope_sha")
    if left_env != right_env:
        reasons.append(
            "different capability envelopes "
            f"({str(left_env)[:12]} vs {str(right_env)[:12]}) — the envelope "
            "decides which misses count as gate defects, so the same gate "
            "scores differently under each"
        )

    if reasons:
        return {
            "refused": True,
            "left": left_name,
            "right": right_name,
            "reasons": reasons,
        }

    metrics = {}
    for card, side in ((left, "left"), (right, "right")):
        for metric in card.get("metrics", []):
            metrics.setdefault(metric["name"], {})[side] = metric["value"]
    return {
        "refused": False,
        "left": left_name,
        "right": right_name,
        "metrics": metrics,
    }


def read_envelope(artifacts: Artifacts) -> dict[str, Any] | Absent:
    if not artifacts.envelope_path.exists():
        return Absent(
            what="measured capability envelope",
            looked_for=artifacts.relative(artifacts.envelope_path),
            produced_by=(
                "python scripts/measure_envelope.py --displacements 2 3 4 6 8 "
                "12 20 30 --model-out configs/envelope/gate_320x180.envelope.json"
            ),
        )
    payload = json.loads(artifacts.envelope_path.read_text())
    payload["_source"] = artifacts.relative(artifacts.envelope_path)
    return payload


def read_events(artifacts: Artifacts) -> dict[str, Any] | Absent:
    """The event log, if one has been written.

    None has been at the time of writing, and the Inspector says so rather than
    rendering an empty table that looks like "no events happened".
    """
    if not artifacts.events_path.exists():
        return Absent(
            what="event log",
            looked_for=artifacts.relative(artifacts.events_path),
            produced_by=(
                "no command yet — the event compiler does not write this file. "
                "An empty table here would claim nothing happened; nothing has "
                "run."
            ),
        )
    import pyarrow.parquet as pq

    table = pq.read_table(artifacts.events_path)
    rows = table.to_pylist()
    return {"_source": artifacts.relative(artifacts.events_path), "rows": rows}


def list_associations(artifacts: Artifacts) -> list[dict[str, Any]]:
    """Every resolved association component on disk, newest first.

    Written by ``scripts/associate_golden_set.py`` (Day 37, Objective 3) —
    a real, systematic sweep of a golden set's multi-agent clips through
    ``resolve_data_association``, not the two hand-picked components
    ``scripts/build_association_demo.py`` (Day 35, retired) used to prove
    the Inspector could render both verdict shapes. Every row's own
    ``selection`` field says HOW it was chosen (``"systematic_sweep"`` vs
    the one ``"boundary_fixture_hand_selected"`` case kept only because the
    systematic sweep itself produced no Ambiguous verdict — see that
    script's module docstring). Absence here means the same thing it means
    everywhere else in this module: nothing has been produced, not that
    nothing was asked.
    """
    if not artifacts.associations_dir.exists():
        return []
    rows = []
    for path in sorted(artifacts.associations_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "file": artifacts.relative(path),
                "component_id": payload.get("component_id", path.stem),
                "verdict_kind": (payload.get("verdict") or {}).get("kind"),
                "source_clip": payload.get("source_clip"),
                "selection": payload.get("selection"),
                "modified_epoch": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: r.get("modified_epoch", 0), reverse=True)
    return rows


def read_association(
    artifacts: Artifacts, component_id: str
) -> dict[str, Any] | Absent:
    path = artifacts.associations_dir / f"{component_id}.json"
    if not path.exists():
        return Absent(
            what=f"association resolution {component_id!r}",
            looked_for=artifacts.relative(path),
            produced_by="python scripts/associate_golden_set.py",
        )
    payload = json.loads(path.read_text())
    payload["_source"] = artifacts.relative(path)
    return payload


def list_coverage_queries(artifacts: Artifacts) -> list[dict[str, Any]]:
    """Every `prove_absence` result on disk, newest first.

    Written by ``scripts/prove_absence_query.py`` (Day 37, Objective 4).
    Every row's own ``kind`` (``"absence"`` or ``"cannot_establish"``) and
    ``scenario_realism`` (``"actual_project_state"`` or
    ``"constructed_from_real_types"``) travel with it, so a viewer can
    never mistake a query built to exercise a rendering path for one that
    reports something that occurred. Absence here means the same thing it
    means everywhere else in this module: nothing has been produced, not
    that nothing was asked.
    """
    if not artifacts.coverage_queries_dir.exists():
        return []
    rows = []
    for path in sorted(artifacts.coverage_queries_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "file": artifacts.relative(path),
                "query_id": payload.get("query_id", path.stem),
                "kind": payload.get("kind"),
                "scenario_realism": payload.get("scenario_realism"),
                "subject_id": payload.get("subject_id"),
                "modified_epoch": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda r: r.get("modified_epoch", 0), reverse=True)
    return rows


def read_coverage_query(artifacts: Artifacts, query_id: str) -> dict[str, Any] | Absent:
    path = artifacts.coverage_queries_dir / f"{query_id}.json"
    if not path.exists():
        return Absent(
            what=f"coverage/absence query {query_id!r}",
            looked_for=artifacts.relative(path),
            produced_by="python scripts/prove_absence_query.py",
        )
    payload = json.loads(path.read_text())
    payload["_source"] = artifacts.relative(path)
    return payload


def active_golden(artifacts: Artifacts) -> dict[str, Any] | Absent:
    from src.config import IronConfig
    from src.data.golden import GoldenSetError, load_golden_set

    config = IronConfig.load()
    version = config.eval.golden_set_version
    try:
        golden = load_golden_set(artifacts.golden_dir, version)
    except (GoldenSetError, FileNotFoundError):
        return Absent(
            what=f"golden set {version!r}",
            looked_for=artifacts.relative(
                artifacts.golden_dir / f"{version}.golden.json"
            ),
            produced_by=(
                "python scripts/gen_synthetic_indoor.py --scene-set v3 "
                "--write-golden v3-indoor"
            ),
        )
    return {
        "_source": artifacts.relative(artifacts.golden_dir / f"{version}.golden.json"),
        "version": golden.version,
        "domain": golden.domain.value,
        "status": golden.status.value,
        "set_sha": golden.set_sha,
        "supersedes": golden.supersedes,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "content_sha": clip.content_sha,
                "hard_coverage": clip.hard_coverage,
                "conditions": [c.value for c in clip.conditions],
                "source_dataset": clip.source_dataset,
                "notes": clip.notes,
            }
            for clip in golden.clips
        ],
    }


def clip_path(artifacts: Artifacts, clip_id: str) -> Path | Absent:
    golden = active_golden(artifacts)
    if isinstance(golden, Absent):
        return golden
    datasets = {c["source_dataset"] for c in golden["clips"] if c["source_dataset"]}
    dataset = datasets.pop() if len(datasets) == 1 else "synthetic-indoor-v1"
    path = artifacts.data_dir / "synthetic" / dataset / f"{clip_id}.npz"
    if not path.exists():
        return Absent(
            what=f"clip {clip_id!r}",
            looked_for=artifacts.relative(path),
            produced_by=(
                "python scripts/gen_synthetic_indoor.py --scene-set v3 "
                "--frames 40 --seed 20260808 "
                f"--output data/synthetic/{dataset}"
            ),
        )
    return path


def clip_analysis(artifacts: Artifacts, clip_id: str) -> dict[str, Any] | Absent:
    """Per-frame observability, gate decisions, and silhouette area.

    This is the view that answers, for one clip, the question that took two
    days to answer in prose: is a given miss a gate defect or a physical limit?
    It runs the *same* partition and the *same* gate the scorecard runs — not a
    reimplementation — so what the screen shows and what the number says cannot
    drift apart.
    """
    path = clip_path(artifacts, clip_id)
    if isinstance(path, Absent):
        return path

    from src.cascade import MotionGate, StageContext
    from src.cascade.envelope import DEFAULT_ENVELOPE_PATH, MeasuredEnvelope
    from src.config import IronConfig
    from src.data.scorecard import (
        FIRST_AGENT_INSTANCE_ID,
        world_motion,
        Observability,
        observability_partition,
    )

    config = IronConfig.load()
    gate_config = config.cascade.motion_gate_config()
    envelope = MeasuredEnvelope.load(artifacts.project_root / DEFAULT_ENVELOPE_PATH)

    partition = observability_partition(path, gate_config, envelope)

    from src.contracts.ground_truth import GENERATOR_AXES, GtPositionClip
    from src.model.world import UNREGISTERED

    with np.load(path) as data:
        rgb = np.asarray(data["rgb"])
        instances = np.asarray(data["instances"])
        # The .npz clip records neither a twin_rev nor an axis
        # convention. GtPositionClip makes both explicit: UNREGISTERED
        # for the revision (Day-15 migration -- an absence typed rather
        # than a bare ndarray silently readable as belonging to whatever
        # revision a future caller assumes), and GENERATOR_AXES for the
        # frame (Day 30 -- this was previously wrapped in
        # WorldPositionArray, whose module documents +z-up, which
        # agent_xyz is NOT in; see GtPositionClip's docstring).
        agent_xyz = GtPositionClip(
            values=np.asarray(data["agent_xyz"]),
            axes=GENERATOR_AXES,
            twin_rev=UNREGISTERED,
        ).values
        track_uv = np.asarray(data["track_uv"])
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)

    frames, agents = agent_xyz.shape[0], agent_xyz.shape[1]
    native_px = instances.shape[1] * instances.shape[2]
    gate_px = gate_config.gate_width * gate_config.gate_height or native_px
    to_gate = gate_px / native_px
    uv_to_gate = gate_config.gate_width / instances.shape[2]

    gate = MotionGate(gate_config)
    wake = []
    for index in range(frames):
        output = gate.process(rgb[index][np.newaxis, ...], StageContext(index, index))
        wake.append(bool(output.wake_next))

    moved = np.zeros((frames, agents), dtype=bool)
    speed = np.zeros((frames, agents))
    if agents:
        moved = world_motion(agent_xyz, intrinsics, extrinsics)
        steps = np.linalg.norm(np.diff(track_uv, axis=0), axis=2) * uv_to_gate
        speed[1:] = np.nan_to_num(steps, nan=0.0, posinf=0.0)

    per_agent = []
    for agent in range(agents):
        area = (instances == FIRST_AGENT_INSTANCE_ID + agent).sum(axis=(1, 2)) * to_gate
        per_agent.append(
            {
                "agent": agent,
                "silhouette_gate_px": [round(float(a), 2) for a in area],
                "speed_gate_px": [round(float(s), 3) for s in speed[:, agent]],
                "moved": [bool(m) for m in moved[:, agent]],
                "wake_threshold_px": [
                    (
                        None
                        if not np.isfinite(t := envelope.wake_threshold_px(float(s)))
                        else round(t, 2)
                    )
                    for s in speed[:, agent]
                ],
            }
        )

    return {
        "clip_id": clip_id,
        "_source": artifacts.relative(path),
        "frames": frames,
        "resolution": [int(instances.shape[2]), int(instances.shape[1])],
        "warmup_frames": partition.warm,
        "labels": [int(x) for x in partition.labels],
        "label_names": {int(o): o.name for o in Observability},
        "wake": wake,
        "agents": per_agent,
        "gate": {
            "width": gate_config.gate_width,
            "height": gate_config.gate_height,
            "min_foreground_fraction": gate_config.min_foreground_fraction,
            "derived_threshold_px": gate_config.envelope_threshold_px(),
        },
        "envelope_sha": envelope.sha,
    }


def clip_frame_png(artifacts: Artifacts, clip_id: str, index: int) -> bytes | Absent:
    """One decoded frame as PNG, for the scrubber."""
    path = clip_path(artifacts, clip_id)
    if isinstance(path, Absent):
        return path
    import cv2

    with np.load(path) as data:
        rgb = np.asarray(data["rgb"])
    index = max(0, min(index, rgb.shape[0] - 1))
    frame = rgb[index]
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to encode frame {index} of {clip_id}")
    return bytes(buffer)


def provenance(artifacts: Artifacts) -> dict[str, Any]:
    """Git state, config sha, and what the displayed numbers rest on.

    A view showing numbers without a resolvable manifest gets a warning band,
    so this always reports what it could and could not resolve rather than
    failing closed and showing nothing.
    """
    from src.provenance import git_state

    unresolved: list[str] = []
    try:
        git = git_state(artifacts.project_root)
        git_payload = {
            "sha": getattr(git, "sha", None) or getattr(git, "commit", None),
            "branch": getattr(git, "branch", None),
            "dirty": bool(getattr(git, "dirty", False)),
        }
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        git_payload = {"error": f"{type(exc).__name__}: {exc}"}
        unresolved.append("git state")

    envelope = read_envelope(artifacts)
    golden = active_golden(artifacts)
    if isinstance(envelope, Absent):
        unresolved.append("capability envelope")
    if isinstance(golden, Absent):
        unresolved.append("active golden set")

    return {
        "git": git_payload,
        "envelope": envelope.as_dict()
        if isinstance(envelope, Absent)
        else {
            "sha": _sha_of(artifacts.envelope_path),
            "source": artifacts.relative(artifacts.envelope_path),
            "measured_stack": envelope.get("measured_stack"),
        },
        "golden": golden.as_dict()
        if isinstance(golden, Absent)
        else {
            "version": golden["version"],
            "set_sha": golden["set_sha"],
            "status": golden["status"],
            "source": golden["_source"],
            "clips": len(golden["clips"]),
        },
        "config_sha": _sha_of(artifacts.project_root / "configs" / "default.yaml"),
        "config_source": "configs/default.yaml",
        "unresolved": unresolved,
    }


def _sha_of(path: Path) -> str | None:
    import hashlib

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
