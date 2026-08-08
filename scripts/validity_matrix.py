"""Which capabilities can this project evaluate, on what it currently owns?

Runs every registered golden set through every registered validity gate and
prints the matrix. The answer is mostly "no", and that is the point: a project
that does not know which of its claims are measurable will report the ones that
are easy and stay quiet about the rest.

    python scripts/validity_matrix.py
    python scripts/validity_matrix.py --json outputs/perception/validity_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IronConfig  # noqa: E402
from src.data import validity  # noqa: E402
from src.data.golden import available_versions, load_golden_set  # noqa: E402

CAPABILITIES = ("motion_geometry", "depth", "appearance_semantics", "point_tracking")
"""Day 16: point_tracking was registered in ``validity.GATES`` since Day 11
(``scripts/eval_tracking.py`` uses it directly) but never appeared in THE
matrix — the one script whose job is "which capabilities can this project
evaluate" was answering that question for three of the four registered
gates. Added here rather than left as a second, parallel evaluator."""
DEPTH_WEIGHTS = Path("models/weights/depth_anything_v2_small")


def sample_clip(clip_root: Path, clip_id: str, frames: int = 4):
    """Load a few frames plus GT depth from one clip, or ``None`` if absent."""
    path = clip_root / f"{clip_id}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        keys = set(data.files)
        if "rgb" not in keys:
            return None
        rgb = np.asarray(data["rgb"][:frames])
        depth = (
            np.asarray(data["depth_m"][:frames], dtype=np.float64)
            if "depth_m" in keys
            else None
        )
    return rgb, depth


def build_predictor():
    """A depth predictor, or ``None`` with the reason surfaced by the gate."""
    if not DEPTH_WEIGHTS.exists():
        return None
    from src.models.dav2_wrapper import DAv2Wrapper

    wrapper = DAv2Wrapper(str(DEPTH_WEIGHTS), encoder="vits")
    wrapper.load()
    return lambda frame: wrapper.predict({"image": frame})["depth"].data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    golden_dir = config.paths.resolve(config.eval.golden_sets_dir)
    versions = available_versions(golden_dir)

    predictor = build_predictor()
    results = []

    for version in versions:
        golden = load_golden_set(golden_dir, version)
        datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
        dataset = datasets.pop() if len(datasets) == 1 else ""
        clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

        sample = None
        for clip in golden.clips:
            sample = sample_clip(clip_root, clip.clip_id)
            if sample is not None:
                break

        for capability in CAPABILITIES:
            if sample is None:
                results.append(
                    validity.GateResult(
                        capability=capability,
                        dataset=version,
                        passed=False,
                        reason=(
                            "no clip from this set is materialised locally, so "
                            "nothing could be measured. Not a verdict on the "
                            "set — a statement that it was not present."
                        ),
                    )
                )
                continue
            rgb, depth = sample
            results.append(
                validity.evaluate(
                    capability,
                    version,
                    frames=rgb,
                    gt_depth=depth,
                    predictor=predictor,
                )
            )

    width = max(len(v) for v in versions) + 2 if versions else 12
    print("=" * 78)
    print("CAPABILITY x DATASET VALIDITY MATRIX")
    print("=" * 78)
    print("Can this dataset meaningfully score this capability?")
    print()
    header = f"{'dataset':<{width}}" + "".join(f"{c:<24}" for c in CAPABILITIES)
    print(header)
    print("-" * len(header))
    for version in versions:
        row = f"{version:<{width}}"
        for capability in CAPABILITIES:
            result = next(
                r
                for r in results
                if r.dataset == version and r.capability == capability
            )
            row += f"{'PASS' if result.passed else 'REFUSED':<24}"
        print(row)

    print()
    print("EVIDENCE")
    print("-" * 78)
    for result in results:
        if result.passed:
            continue
        print(f"{result.dataset} / {result.capability}:")
        print(f"  {result.reason}")
        if result.evidence:
            print(f"  {json.dumps(result.evidence)}")
        print()

    payload = {
        "capabilities": list(CAPABILITIES),
        "datasets": versions,
        "results": [r.as_dict() for r in results],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        print(f"Written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
