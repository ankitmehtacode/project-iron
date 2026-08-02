"""Score CoTracker3 on the golden set, gate first.

Follows the Day-10 pattern that condemned v3 for depth and appearance:
- Run the ``point_tracking`` validity gate on every clip in the set.
- If the SET refuses (any clip below the corner-density floor), do NOT score;
  a tracker's accuracy on a starved matcher describes its extrapolator, not
  its correspondence. Report which clips refused, with evidence.
- If the set passes, run CoTracker3 at the GT track heads and evaluate with
  the TAP-Vid protocol *unmodified* (``compute_tapvid_metrics`` from
  ``cotracker.evaluation.core.eval_utils``). Report per-clip and broken out
  by observability bucket, because accuracy on frames the query point is
  off-sensor or occluded must not be pooled with accuracy on frames it is
  actually visible in.

The offline oracle vs streaming production caveat, in the same file that
scores it, so the number does not travel without it:
    ``models/weights/cotracker3/scaled_offline.pth`` is bidirectional — the
    tracker sees future frames when predicting frame ``t``. That makes it an
    upper bound on any online (streaming) tracker, not the production number.
    The offline-minus-online penalty is unmeasured; a Day-12+ item.

    python scripts/eval_tracking.py
    python scripts/eval_tracking.py --limit 3
    python scripts/eval_tracking.py --set v3-indoor

Exit codes:
    0  gates ran and (if pass) scoring completed; a refusal is a run.
    1  cotracker weights or dataset clips missing.
    2  cotracker package not importable — the pin has never been installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.data import validity  # noqa: E402
from src.data.golden import load_golden_set  # noqa: E402

DEFAULT_WEIGHTS = Path("models/weights/cotracker3/scaled_offline.pth")


def load_clip(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {k: payload[k] for k in payload.files}


def gate_clip(clip_id: str, frames: np.ndarray) -> validity.GateResult:
    return validity.evaluate("point_tracking", clip_id, frames=frames)


def observable_mask(uv: np.ndarray, occluded: np.ndarray, height: int, width: int) -> np.ndarray:
    """True where the GT track point is inside the sensor AND not occluded.

    Shape: ``[T, N]``. Off-sensor and occluded are separate concepts and this
    module treats them the same way for bucketing — a matcher cannot be
    graded on a frame where the GT point isn't visibly present regardless of
    which of those two makes it invisible.
    """
    in_frame = (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    return in_frame & (~occluded)


def score_clip_with_tracker(
    clip_id: str,
    clip: dict[str, np.ndarray],
    predictor: Any,
) -> dict[str, Any]:
    """Query CoTracker at each GT track head and grade with TAP-Vid.

    ``track_uv`` in the v3 fixture is shape ``[T, N, 2]`` (per-frame UV of
    ``N`` tracked agents), and ``track_occluded`` is ``[T, N]``. Each track's
    query point is (frame 0, GT uv at frame 0); the query is skipped if the
    point is not observable in frame 0, because the tracker cannot lock onto
    something absent from the query frame."""
    import torch
    from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics

    rgb = clip["rgb"]  # [T, H, W, 3] uint8
    gt_uv = clip["track_uv"]  # [T, N, 2]
    gt_occ = clip["track_occluded"]  # [T, N]
    height, width = rgb.shape[1], rgb.shape[2]

    observable = observable_mask(gt_uv, gt_occ, height, width)

    # Queries at frame 0 only — TAP-Vid `query_mode="first"`.
    valid_queries = np.where(observable[0])[0]
    if valid_queries.size == 0:
        return {
            "clip_id": clip_id,
            "scored": False,
            "reason": "no GT track head is observable in frame 0",
        }

    queries_np = np.stack(
        [
            np.zeros(valid_queries.shape, dtype=np.float32),  # query frame
            gt_uv[0, valid_queries, 0],  # x
            gt_uv[0, valid_queries, 1],  # y
        ],
        axis=1,
    )[None]  # [1, N', 3] (t, x, y)

    video = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).float()  # [1, T, 3, H, W]
    queries = torch.from_numpy(queries_np).float()

    with torch.no_grad():
        tracks, vis = predictor(video, queries=queries)

    pred_tracks = tracks[0].cpu().numpy()  # [T, N', 2]
    pred_vis = vis[0].cpu().numpy()  # [T, N']

    # TAP-Vid metric shape: [B, N, T] / [B, N, T, 2]
    gt_tracks_bnt2 = np.transpose(gt_uv[:, valid_queries, :], (1, 0, 2))[None]
    gt_occluded_bnt = np.transpose(gt_occ[:, valid_queries], (1, 0))[None]
    pred_tracks_bnt2 = np.transpose(pred_tracks, (1, 0, 2))[None]
    pred_occluded_bnt = np.transpose(pred_vis < 0.5, (1, 0))[None]
    query_bn3 = np.stack(
        [
            np.zeros(valid_queries.shape),
            gt_uv[0, valid_queries, 1],  # y
            gt_uv[0, valid_queries, 0],  # x — TAP-Vid query is [t, y, x]
        ],
        axis=1,
    )[None]

    metrics_all = compute_tapvid_metrics(
        query_points=query_bn3,
        gt_occluded=gt_occluded_bnt,
        gt_tracks=gt_tracks_bnt2,
        pred_occluded=pred_occluded_bnt,
        pred_tracks=pred_tracks_bnt2,
        query_mode="first",
    )

    # Observability bucket breakdown — same metric, restricted to observable
    # frames. When no frames are observable in a bucket, we report NaN and say
    # so; a per-clip zero on an empty bucket would look like a real score.
    def bucket(mask: np.ndarray) -> dict[str, float]:
        # Zero-out unobservable frames in gt_occluded so those frames are not
        # counted in the pts_within_x fraction. This is the same trick the
        # TAP-Vid paper uses for evaluation restriction.
        gt_occ_bucket = np.transpose((~mask)[:, valid_queries], (1, 0))[None]
        if not mask[:, valid_queries].any():
            return {"pts_within_avg": float("nan"), "occlusion_accuracy": float("nan"), "average_jaccard": float("nan"), "n_frames": 0}
        m = compute_tapvid_metrics(
            query_points=query_bn3,
            gt_occluded=gt_occ_bucket,
            gt_tracks=gt_tracks_bnt2,
            pred_occluded=pred_occluded_bnt,
            pred_tracks=pred_tracks_bnt2,
            query_mode="first",
        )
        return {
            "pts_within_avg": float(m["average_pts_within_thresh"].mean()),
            "occlusion_accuracy": float(m["occlusion_accuracy"].mean()),
            "average_jaccard": float(m["average_jaccard"].mean()),
            "n_frames": int(mask[:, valid_queries].any(axis=1).sum()),
        }

    observable_mask_full = observable  # [T, N]
    off_sensor_mask = ~((gt_uv[..., 0] >= 0) & (gt_uv[..., 0] < width) & (gt_uv[..., 1] >= 0) & (gt_uv[..., 1] < height))
    occluded_only_mask = gt_occ & ~off_sensor_mask

    return {
        "clip_id": clip_id,
        "scored": True,
        "n_queries": int(valid_queries.size),
        "n_frames": int(rgb.shape[0]),
        "all_frames": {
            "pts_within_avg": float(metrics_all["average_pts_within_thresh"].mean()),
            "occlusion_accuracy": float(metrics_all["occlusion_accuracy"].mean()),
            "average_jaccard": float(metrics_all["average_jaccard"].mean()),
        },
        "observable_only": bucket(observable_mask_full),
        "off_sensor_only": bucket(off_sensor_mask),
        "occluded_only": bucket(occluded_only_mask),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--set", dest="set_version", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/day11/tracking.json")
    )
    args = parser.parse_args(argv)

    try:
        import cotracker  # noqa: F401
    except ImportError:
        print(
            "cotracker package not importable. This means the pin in "
            "locking-requirements.txt has never been installed in this venv. "
            "Install with:\n"
            "    pip install -r locking-requirements.txt\n"
            "or, for just this dep:\n"
            "    pip install einops 'git+https://github.com/facebookresearch/co-tracker.git@<sha>'",
            file=sys.stderr,
        )
        return 2

    if not args.weights.exists():
        print(f"weights not at {args.weights}", file=sys.stderr)
        return 1

    config = IronConfig.load()
    version = args.set_version or config.eval.golden_set_version
    golden = load_golden_set(config.paths.resolve(config.eval.golden_sets_dir), version)

    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else "synthetic-indoor-v3"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    clips = list(golden.clips)
    if args.limit:
        clips = clips[: args.limit]

    verdicts: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    passed_gate = 0
    refused_gate = 0

    for clip in clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            print(f"MISSING {clip.clip_id}", file=sys.stderr)
            continue
        data = load_clip(path)
        gate = gate_clip(clip.clip_id, data["rgb"])
        verdicts.append(
            {"clip_id": clip.clip_id, **gate.as_dict()}
        )
        state = "PASS" if gate.passed else "REFUSE"
        density = gate.evidence.get("corner_density_per_pixel")
        print(f"{state}  {clip.clip_id}  density={density:.3e}")
        if gate.passed:
            passed_gate += 1
        else:
            refused_gate += 1

    print(
        f"\ngate summary: passed={passed_gate}, refused={refused_gate}, "
        f"total={passed_gate + refused_gate}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if refused_gate > 0:
        payload = {
            "set": version,
            "dataset": dataset,
            "verdict": "REFUSE",
            "reason": (
                f"{refused_gate} of {passed_gate + refused_gate} clips fell "
                "below the corner-density floor. A pooled tracking number "
                "across a set where the majority of clips starve the matcher "
                "reports its extrapolator; the set is not scored, per the "
                "Day-10 rule that a validity gate is checked before the "
                "metric is computed."
            ),
            "per_clip_gate": verdicts,
            "checkpoint_note": (
                "scaled_offline.pth is bidirectional; a tracking score from "
                "it would have been an offline oracle even if the set had "
                "passed. Streaming (online) penalty remains unmeasured."
            ),
        }
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nSET REFUSAL — no tracker score computed. Written to {args.out}.")
        return 0

    # All clips passed — wire the predictor and score.
    from src.models.cotracker3_wrapper import CoTracker3Wrapper

    wrapper = CoTracker3Wrapper(str(args.weights), device="CPU")
    wrapper.load()

    for clip in clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            continue
        data = load_clip(path)
        result = score_clip_with_tracker(clip.clip_id, data, wrapper.model)
        scored.append(result)
        if result.get("scored"):
            print(
                f"  {clip.clip_id}: pts<avg={result['all_frames']['pts_within_avg']:.3f} "
                f"OA={result['all_frames']['occlusion_accuracy']:.3f} "
                f"AJ={result['all_frames']['average_jaccard']:.3f} "
                f"(N={result['n_queries']} queries, T={result['n_frames']} frames)"
            )

    payload = {
        "set": version,
        "dataset": dataset,
        "verdict": "SCORED",
        "per_clip_gate": verdicts,
        "per_clip_tracking": scored,
        "checkpoint_note": (
            "scaled_offline.pth is bidirectional (attends to future frames), "
            "so every number here is an offline oracle. The streaming "
            "penalty is a separate measurement that has NOT been made yet."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
