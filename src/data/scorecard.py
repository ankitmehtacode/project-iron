"""Compute a scorecard against a golden set with exact ground truth.

What this scores, and what it deliberately does not
---------------------------------------------------
The synthetic indoor dataset carries exact analytic GT, so the metrics here are
about **geometry and motion** — which frames contain movement, whether the
motion gate agrees, whether occlusion is where it should be. They are not
appearance metrics. An analytic renderer has no global illumination and no
material response, so a detection or re-identification number measured on it
would describe the renderer rather than the world.

Reporting a bad number is the job
---------------------------------
Every metric is reported whether or not it flatters the system. A scorecard
that only lists what passed is marketing; the first honest baseline is the
thing every later change is measured against, and its value comes entirely
from not having been curated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

# World displacement above which an agent counts as having moved this frame.
GT_MOTION_THRESHOLD_M = 0.01

# ``instances`` labels agents as ``100 + agent_id``; lower ids are furniture.
FIRST_AGENT_INSTANCE_ID = 100


@dataclass(frozen=True)
class Metric:
    """One measured number, with enough context to interpret it."""

    name: str
    value: float
    unit: str
    higher_is_better: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "detail": self.detail,
        }


@dataclass
class Scorecard:
    """A full evaluation run's results."""

    golden_set_version: str
    golden_set_sha: str
    domain: str
    clips_scored: int
    metrics: list[Metric] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    per_clip: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "golden_set_version": self.golden_set_version,
            "golden_set_sha": self.golden_set_sha,
            "domain": self.domain,
            "clips_scored": self.clips_scored,
            "metrics": [m.as_dict() for m in self.metrics],
            "caveats": self.caveats,
            "per_clip": self.per_clip,
        }

    def render(self) -> str:
        lines = [
            "=" * 74,
            f"SCORECARD — golden set {self.golden_set_version} ({self.domain})",
            "=" * 74,
            f"set_sha     : {self.golden_set_sha}",
            f"clips scored: {self.clips_scored}",
            "",
            f"{'metric':<38} {'value':>12}  unit",
            "-" * 74,
        ]
        for metric in self.metrics:
            arrow = "^" if metric.higher_is_better else "v"
            lines.append(
                f"{metric.name:<38} {metric.value:>12.4f}  {metric.unit} ({arrow} better)"
            )
            if metric.detail:
                lines.append(f"    {metric.detail}")
        if self.caveats:
            lines.extend(["", "CAVEATS"])
            lines.extend(f"  - {c}" for c in self.caveats)
        return "\n".join(lines)


class Observability(IntEnum):
    """What a given camera could have been expected to see in one frame.

    Ordered by how much is asked of the gate, so ``max`` over the agents in a
    frame yields the frame's label: the easiest visible mover is the one the
    gate had the best chance of catching, and it sets the bar.
    """

    NO_MOTION = 0
    """No agent moved. Scoreable, and the only source of true negatives."""

    NOT_OBSERVABLE = 1
    """Movers exist but put no pixels on this sensor — out of frustum, or
    occluded by static geometry. Excluded from every denominator."""

    BELOW_ENVELOPE = 2
    """Movers are visible and unoccluded but too small for the gate's
    foreground threshold to resolve. Excluded from recall, reported as
    ``envelope_limited_misses``."""

    ABOVE_ENVELOPE = 3
    """Visible, unoccluded, and large enough to be resolvable. The only honest
    recall denominator."""


@dataclass(frozen=True)
class Partition:
    """Per-frame observability labels for one clip and one camera."""

    labels: np.ndarray
    """``Observability`` per frame, over the whole clip including warmup."""

    warm: int
    """Warmup frames the gate is not scored on; slice with ``labels[warm:]``."""

    envelope_gate_px: float
    """Foreground area, in gate pixels, the envelope boundary sits at."""

    def after_warmup(self) -> np.ndarray:
        return self.labels[self.warm :]


def observability_partition(clip_path: Path, gate_config: Any) -> Partition:
    """Partition GT motion into what this camera could actually have seen.

    Scoring a camera against world-space motion asks it to detect things
    outside its frustum, which is how a coverage gap gets recorded as a gate
    defect. Observability is resolved per camera from the renderer's own
    instance masks rather than by reprojecting agent centroids: the mask is
    exact, it already accounts for partial occlusion and frame-edge clipping,
    and it does not require this module to re-derive a projection convention.

    The envelope boundary is the gate's own resolving power. An agent occupying
    fewer than ``min_foreground_fraction`` of the gate's pixels cannot trip the
    gate no matter how it is tuned, short of tuning it onto sensor noise, so
    such a frame measures the camera's reach and not the gate's quality.

    Args:
        clip_path: ``.npz`` clip carrying ``agent_xyz`` and ``instances``.
        gate_config: the gate being scored; supplies the envelope boundary.

    Returns:
        A :class:`Partition` labelling every frame in the clip.
    """
    with np.load(clip_path) as data:
        agent_xyz = np.asarray(data["agent_xyz"])
        instances = np.asarray(data["instances"])

    frames, agents = agent_xyz.shape[0], agent_xyz.shape[1]
    native_px = instances.shape[1] * instances.shape[2]
    gate_px = gate_config.gate_width * gate_config.gate_height or native_px
    envelope_gate_px = gate_config.min_foreground_fraction * gate_px

    # Silhouette area is counted natively and scaled into gate pixels, because
    # the gate's threshold is a fraction of the frame it actually processes.
    # A 1080p clip and a 720p clip of the same scene must land on the same side
    # of the envelope, and they only do so in gate space.
    to_gate = gate_px / native_px

    moved = np.zeros((frames, agents), dtype=bool)
    if agents:
        deltas = np.linalg.norm(np.diff(agent_xyz, axis=0), axis=2)
        moved[1:] = deltas > GT_MOTION_THRESHOLD_M

    labels = np.full(frames, Observability.NO_MOTION, dtype=np.int8)
    for agent in range(agents):
        moving = moved[:, agent]
        if not moving.any():
            continue
        area = (instances == FIRST_AGENT_INSTANCE_ID + agent).sum(axis=(1, 2)) * to_gate
        rank = np.where(
            area >= envelope_gate_px,
            Observability.ABOVE_ENVELOPE,
            np.where(
                area > 0, Observability.BELOW_ENVELOPE, Observability.NOT_OBSERVABLE
            ),
        )
        labels = np.where(moving, np.maximum(labels, rank), labels).astype(np.int8)

    return Partition(
        labels=labels,
        warm=gate_config.warmup_frames,
        envelope_gate_px=float(envelope_gate_px),
    )


def motion_gate_metrics(
    clip_path: Path, gate_config: Any
) -> tuple[dict[str, int], dict[str, float]]:
    """Score the motion gate against motion this camera could have seen.

    GT for "does this frame contain motion" is exact — agent world positions
    are analytic — but world motion is not the question a single camera can be
    asked. Frames are first partitioned by :func:`observability_partition`, and
    only above-envelope motion reaches the recall denominator. The other two
    buckets are counted and returned so that excluding them is visible rather
    than silent.

    Returns:
        ``(counts, rates)`` — raw confusion counts and derived rates. Both,
        because a rate computed from three frames looks identical to one
        computed from three hundred, and only the counts say which it was.
    """
    from src.cascade import MotionGate, StageContext

    with np.load(clip_path) as data:
        rgb = np.asarray(data["rgb"])

    partition = observability_partition(clip_path, gate_config)

    frames = rgb.shape[0]
    gate = MotionGate(gate_config)
    predicted = np.zeros(frames, dtype=bool)
    for index in range(frames):
        output = gate.process(rgb[index][np.newaxis, ...], StageContext(index, index))
        predicted[index] = output.wake_next

    labels = partition.after_warmup()
    guess = predicted[partition.warm :]

    scoreable = (labels == Observability.NO_MOTION) | (
        labels == Observability.ABOVE_ENVELOPE
    )
    truth = labels == Observability.ABOVE_ENVELOPE

    tp = int(np.sum(truth & guess))
    fp = int(np.sum(scoreable & ~truth & guess))
    fn = int(np.sum(truth & ~guess))
    tn = int(np.sum(scoreable & ~truth & ~guess))

    # Bucket 2 misses: visible motion the gate slept through that it could not
    # have resolved. Not a defect, and not silently dropped either.
    below = labels == Observability.BELOW_ENVELOPE
    unobservable = labels == Observability.NOT_OBSERVABLE

    counts = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "scored_frames": int(np.sum(scoreable)),
        # Total below-envelope frames, so scored + below + unobservable closes
        # against the frame count. An accounting that does not close is one
        # where an excluded frame can go missing without anyone noticing.
        "below_envelope_frames": int(np.sum(below)),
        "envelope_limited_misses": int(np.sum(below & ~guess)),
        "unobservable_frames": int(np.sum(unobservable)),
        # A wake outside the envelope is neither TP nor FP — there was nothing
        # resolvable to wake on — but it still costs compute, and the excluded
        # buckets must not become somewhere for wakes to hide. The gate's
        # stay-awake latch is the expected cause.
        "wakes_outside_envelope": int(np.sum((below | unobservable) & guess)),
    }
    rates = {
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "wake_rate": float(np.mean(guess)) if len(guess) else float("nan"),
        "gt_motion_rate": (
            float(np.mean(truth[scoreable])) if np.any(scoreable) else float("nan")
        ),
    }
    return counts, rates


def occlusion_metrics(clip_path: Path) -> dict[str, float]:
    """Descriptive statistics of the GT itself, not of any model.

    Included because a scorecard whose difficulty is unstated is unreadable:
    a perfect score on clips with no occlusion says nothing about a system that
    must handle occlusion.
    """
    with np.load(clip_path) as data:
        occluded = np.asarray(data["track_occluded"])
        depth = np.asarray(data["depth_m"])
    return {
        "occluded_track_fraction": float(np.mean(occluded)) if occluded.size else 0.0,
        "depth_min_m": float(np.min(depth)),
        "depth_max_m": float(np.max(depth)),
        "depth_median_m": float(np.median(depth)),
    }


def compute(golden: Any, clip_root: Path, gate_config: Any) -> Scorecard:
    """Score every clip in a golden set."""
    card = Scorecard(
        golden_set_version=golden.version,
        golden_set_sha=golden.set_sha,
        domain=golden.domain.value,
        clips_scored=0,
    )

    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "scored_frames": 0,
        "below_envelope_frames": 0,
        "envelope_limited_misses": 0,
        "unobservable_frames": 0,
        "wakes_outside_envelope": 0,
    }
    occlusion_fractions: list[float] = []

    for clip in golden.clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            card.caveats.append(f"{clip.clip_id}: clip file missing at {path}")
            continue

        counts, rates = motion_gate_metrics(path, gate_config)
        occlusion = occlusion_metrics(path)
        for key in totals:
            totals[key] += counts[key]
        occlusion_fractions.append(occlusion["occluded_track_fraction"])

        card.per_clip[clip.clip_id] = {
            **{k: float(v) for k, v in counts.items()},
            **{k: float(v) for k, v in rates.items()},
            **occlusion,
        }
        card.clips_scored += 1

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    # Every frame the set put in front of the gate, scoreable or not. The
    # three buckets must close against this.
    presented = (
        totals["scored_frames"]
        + totals["below_envelope_frames"]
        + totals["unobservable_frames"]
    )
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 and np.isfinite(precision + recall)
        else float("nan")
    )

    card.metrics = [
        Metric(
            "motion_gate.recall",
            recall,
            "fraction",
            True,
            f"{tp} of {tp + fn} moving frames woke the next stage. A missed "
            "wake loses the event outright.",
        ),
        Metric(
            "motion_gate.precision",
            precision,
            "fraction",
            True,
            f"{tp} of {tp + fp} wakes were on genuinely moving frames. Low "
            "precision costs compute, not evidence.",
        ),
        Metric("motion_gate.f1", f1, "fraction", True),
        Metric(
            "motion_gate.false_negatives",
            float(fn),
            "frames",
            False,
            "The number that matters most: frames with real motion the gate "
            "slept through.",
        ),
        Metric(
            "motion_gate.false_positives",
            float(fp),
            "frames",
            False,
            "Wakes on frames with no motion; a compute cost.",
        ),
        Metric(
            "envelope.limited_misses",
            float(totals["envelope_limited_misses"]),
            "frames",
            False,
            "Visible, unoccluded motion too small for the gate to resolve at "
            f"{gate_config.gate_width}x{gate_config.gate_height}. NOT a gate "
            "defect and NOT tunable away — this is the camera's capability "
            "envelope, and it is a mount-position and coverage input.",
        ),
        Metric(
            "envelope.unobservable_frames",
            float(totals["unobservable_frames"]),
            "frames",
            False,
            "Frames whose GT motion put no pixels on this sensor. Excluded "
            "from every denominator: scoring them would ask a camera to see "
            "through its own frustum.",
        ),
        Metric(
            "envelope.wakes_outside_envelope",
            float(totals["wakes_outside_envelope"]),
            "frames",
            False,
            "Wakes on excluded frames — neither TP nor FP, but still compute. "
            "Reported so the excluded buckets cannot hide a wake storm; the "
            "stay-awake latch is the expected cause.",
        ),
        Metric(
            "gt.occluded_track_fraction",
            float(np.mean(occlusion_fractions)) if occlusion_fractions else 0.0,
            "fraction",
            False,
            "Difficulty of the set itself, not a model result. A perfect score "
            "on unoccluded clips says nothing about occlusion handling.",
        ),
        Metric(
            "coverage.frames_scored", float(totals["scored_frames"]), "frames", True
        ),
        Metric(
            "coverage.observable_fraction",
            (totals["scored_frames"] / presented) if presented else float("nan"),
            "fraction",
            True,
            f"{totals['scored_frames']} of {presented} frames carried ground "
            "truth this camera could act on. Read every rate above against "
            "this: a recall of 1.0 over a set that is mostly unobservable is a "
            "statement about the set, not about the gate.",
        ),
    ]
    return card


def write(card: Scorecard, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.as_dict(), indent=2, sort_keys=True) + "\n")
    return path
