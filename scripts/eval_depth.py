"""Score the depth stage against v3-indoor's exact ground truth.

The first depth number this project has ever produced. Stage 0 (the motion
gate) has been measured since day 6; stages 1-3 never have.

    python scripts/eval_depth.py
    python scripts/eval_depth.py --frame-stride 4 --limit 6

Exit codes:
    0  scored
    1  weights or clips missing — stated, never worked around
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IronConfig  # noqa: E402
from src.data import depth_eval  # noqa: E402
from src.data.golden import load_golden_set  # noqa: E402

DEFAULT_WEIGHTS = Path("models/weights/depth_anything_v2_small")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default="vits")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=8,
        help=(
            "score every Nth frame. Subsampling is stated rather than hidden: "
            "depth is a per-frame measurement and the flicker metric needs "
            "several frames, but every frame of every clip is not affordable "
            "on CPU."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="first N clips only")
    parser.add_argument("--out", type=Path, default=Path("outputs/perception"))
    args = parser.parse_args(argv)

    if not args.weights.exists():
        print(
            f"No depth weights at {args.weights}.\n"
            "Fetch with: python scripts/fetch_weights.py depth_anything_v2_small\n"
            "No number is reported rather than a number from an absent model.",
            file=sys.stderr,
        )
        return 1

    config = IronConfig.load()
    version = config.eval.golden_set_version
    golden = load_golden_set(config.paths.resolve(config.eval.golden_sets_dir), version)

    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else "synthetic-indoor-v1"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    from src.models.dav2_wrapper import DAv2Wrapper

    wrapper = DAv2Wrapper(str(args.weights), encoder=args.encoder)
    wrapper.load()

    clips = list(golden.clips)
    if args.limit:
        clips = clips[: args.limit]

    results = []
    for clip in clips:
        path = clip_root / f"{clip.clip_id}.npz"
        if not path.exists():
            print(f"  MISSING {clip.clip_id}", file=sys.stderr)
            continue
        result = depth_eval.score_clip(path, wrapper, args.frame_stride)
        results.append(result)
        print(
            f"  {clip.clip_id:44} AbsRel {result.aligned['absrel']:.3f}  "
            f"d<1.25 {result.aligned['delta_1_25']:.3f}  "
            f"Zstd {result.flicker.get('z_std_m', float('nan')):.3f}m"
        )

    if not results:
        print("No clips scored.", file=sys.stderr)
        return 1

    summary = depth_eval.aggregate(results)
    summary["golden_set_version"] = version
    summary["golden_set_sha"] = golden.set_sha
    summary["weights"] = str(args.weights)
    summary["encoder"] = args.encoder
    summary["frame_stride"] = args.frame_stride
    summary["caveat"] = (
        "MEASURED, NOT ASSUMED: analytic primitives are NOT the easy case for "
        "monocular depth. They are matte, untextured and perfectly Lambertian, "
        "which removes exactly the shading, texture-gradient and "
        "object-recognition cues a monocular model depends on. On the repo's "
        "one piece of real footage the same weights produce a 4.44 dynamic "
        "range with the floor correctly nearer than the ceiling; on these "
        "renders the range halves and the ordering inverts. The synthetic set "
        "is HARDER here, not easier, and cannot be used as a ceiling."
    )
    summary["per_clip"] = [
        {
            "clip_id": r.clip_id,
            "frames_scored": r.frames_scored,
            "aligned": r.aligned,
            "unaligned": r.unaligned,
            "by_bucket": r.by_bucket,
            "flicker": r.flicker,
            "alignment_drift": r.alignment_drift,
        }
        for r in results
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"depth_{version}_{golden.set_sha[:12]}.json"
    destination.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    print()
    print("=" * 74)
    print(f"DEPTH SCORECARD — {version} ({golden.set_sha[:12]})")
    print("=" * 74)
    print(f"{summary['clips']} clips, {summary['frames_scored']} frames scored")
    print()
    print(
        f"rank correlation with GT disparity : {summary['rank_correlation']:+.4f}"
        f"   (floor {summary['min_rank_correlation']:+.2f})"
    )
    if not summary["valid_for_depth_scoring"]:
        print()
        print("!! THIS SET CANNOT SCORE THE DEPTH STAGE.")
        print("   The model does not rank pixels by depth the way the world does,")
        print("   so a scale-and-shift fit is landing on the dominant background")
        print("   rather than recovering structure. Every metric below is a")
        print("   property of that fit, NOT of the model, and must not be quoted")
        print("   as a depth result. Reported so the failure is visible.")
    print()
    for label in ("unaligned", "aligned"):
        block = summary[label]
        print(
            f"{label.upper():<10} AbsRel {block['absrel']:.4f}   "
            f"RMSE {block['rmse_m']:.3f} m   "
            f"d<1.25 {block['delta_1_25']:.4f}   SILog {block['silog']:.4f}"
        )
    print()
    print("aligned, by GT distance:")
    for name, _, _ in depth_eval.DISTANCE_BUCKETS:
        block = summary["by_bucket"][name]
        print(
            f"  {name:<6} AbsRel {block['absrel']:.4f}   "
            f"RMSE {block['rmse_m']:.3f} m   d<1.25 {block['delta_1_25']:.4f}"
        )
    print()
    print(f"static-point Z std      : {summary['static_point_z_std_m']:.4f} m")
    print(f"alignment scale CV      : {summary['alignment_scale_cv']:.4f}")
    print()
    print(summary["caveat"])
    print(f"\nWritten to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
