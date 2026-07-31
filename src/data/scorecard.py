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
from pathlib import Path
from typing import Any

import numpy as np


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


def motion_gate_metrics(
    clip_path: Path, gate_config: Any
) -> tuple[dict[str, int], dict[str, float]]:
    """Score the motion gate against exact per-frame GT motion.

    Ground truth for "does this frame contain motion" is exact here: agent
    world positions are known analytically, so a frame contains motion iff some
    agent moved more than a threshold since the previous frame.

    Returns:
        ``(counts, rates)`` — raw confusion counts and derived rates. Both,
        because a rate computed from three frames looks identical to one
        computed from three hundred, and only the counts say which it was.
    """
    from src.cascade import MotionGate, StageContext

    with np.load(clip_path) as data:
        rgb = np.asarray(data["rgb"])
        agent_xyz = np.asarray(data["agent_xyz"])

    frames = rgb.shape[0]
    # GT motion: any agent displaced more than 1 cm between consecutive frames.
    moved = np.zeros(frames, dtype=bool)
    if agent_xyz.shape[1] > 0:
        deltas = np.linalg.norm(np.diff(agent_xyz, axis=0), axis=2).max(axis=1)
        moved[1:] = deltas > 0.01

    gate = MotionGate(gate_config)
    predicted = np.zeros(frames, dtype=bool)
    for index in range(frames):
        output = gate.process(rgb[index][np.newaxis, ...], StageContext(index, index))
        predicted[index] = output.wake_next

    warm = gate_config.warmup_frames
    truth, guess = moved[warm:], predicted[warm:]

    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    tn = int(np.sum(~truth & ~guess))

    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "scored_frames": len(truth)}
    rates = {
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "wake_rate": float(np.mean(guess)) if len(guess) else float("nan"),
        "gt_motion_rate": float(np.mean(truth)) if len(truth) else float("nan"),
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

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "scored_frames": 0}
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
    ]
    return card


def write(card: Scorecard, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.as_dict(), indent=2, sort_keys=True) + "\n")
    return path
