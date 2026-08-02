"""Run the full Day-10 gate battery against an Infinigen-Indoors sample.

Infinigen-Indoors-depth-candidate is registered lane S with a **binding
acceptance procedure**: generate a set, run it through the depth validity gate,
and only then may a depth number be computed on it. Day 11 executes that
procedure. Same gates that condemned v2 and v3.

Consumes a directory of Infinigen output prepared by
``scripts/infinigen_generate.py`` (which itself runs in the isolated
``.venv-infinigen`` because Infinigen requires Python 3.11 and depends on
``bpy``; the pinned measurement env is 3.10 and stays untouched). The prep
script writes one ``.npz`` per rendered frame with ``rgb`` (``[H, W, 3]``,
uint8, RGB) and ``depth_metres`` (``[H, W]``, float32) — no OpenEXR reader is
needed here.

    python scripts/run_infinigen_gate.py --sample data/infinigen_probe/scene_0001

Exit codes:
    0  gates ran, verdicts printed (a REFUSE is a run, not a failure)
    1  sample missing or malformed
    2  depth predictor missing — no faked pass, no faked refuse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.data import validity  # noqa: E402

DEFAULT_WEIGHTS = Path("models/weights/depth_anything_v2_small")


def load_sample(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames [T,H,W,3] uint8 RGB, gt_depth [T,H,W] float32 metres)``.

    Every ``.npz`` in ``root`` is one frame with ``rgb`` and ``depth_metres``.
    Order is lexical on filename, matching the render order the prep script
    uses. Frames must all share ``[H, W]`` — a mixed-size sample is refused,
    not silently cropped, because a shape disagreement is a rendering bug and
    hiding it would make the gate report on a fixture that never existed.
    """
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz frames under {root}")
    rgbs, depths = [], []
    for f in files:
        payload = np.load(f)
        if "rgb" not in payload or "depth_metres" not in payload:
            raise KeyError(
                f"{f}: expected keys 'rgb' and 'depth_metres', got "
                f"{list(payload.keys())}"
            )
        rgbs.append(payload["rgb"])
        depths.append(payload["depth_metres"])
    shapes = {r.shape for r in rgbs}
    if len(shapes) != 1:
        raise ValueError(f"mixed frame shapes in {root}: {shapes}")
    frames = np.stack(rgbs).astype(np.uint8)
    gt_depth = np.stack(depths).astype(np.float32)
    return frames, gt_depth


def build_depth_predictor(weights: Path):
    """Return a ``callable(frame_rgb_uint8) -> disparity_hxw`` or ``None``.

    Uses the same DA-V2 wrapper the rest of the project uses. Kept behind a
    factory because the gate needs a *predictor callable* rather than a bound
    wrapper — see ``gate_depth`` in ``src/data/validity.py``. Returns ``None``
    when weights are missing; the caller must exit 2 rather than run the gate
    without a predictor, because a REFUSE from a missing model would look
    identical to a REFUSE from a bad fixture.
    """
    if not weights.exists():
        return None
    from src.models.dav2_wrapper import DAv2Wrapper

    wrapper = DAv2Wrapper(str(weights), encoder="vits")
    wrapper.load()

    def predict(frame_rgb_uint8: np.ndarray) -> np.ndarray:
        # DAv2Wrapper.predict takes BGR (OpenCV convention); the gate hands us
        # RGB. Convert once here rather than at every call site.
        bgr = frame_rgb_uint8[..., ::-1].copy()
        depth_field = wrapper.predict({"image": bgr})["depth"]
        return np.asarray(depth_field.data, dtype=np.float64)

    return predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Directory of Infinigen frames prepared by infinigen_generate.py.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/day11/infinigen_gate.json"),
    )
    args = parser.parse_args(argv)

    try:
        frames, gt_depth = load_sample(args.sample)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR loading sample: {exc}", file=sys.stderr)
        return 1

    print(
        f"loaded {frames.shape[0]} frames at "
        f"{frames.shape[1]}x{frames.shape[2]} from {args.sample}"
    )

    predictor = build_depth_predictor(args.weights)
    if predictor is None:
        print(
            f"depth weights not at {args.weights} — a REFUSE from a missing "
            "model is indistinguishable from a REFUSE from a bad fixture, so "
            "the depth gate is not run at all. Fetch with "
            "scripts/fetch_weights.py depth_anything_v2_small.",
            file=sys.stderr,
        )
        return 2

    dataset = "Infinigen-Indoors-depth-candidate"
    verdicts = {}
    for capability, inputs in (
        ("motion_geometry", {"frames": frames}),
        ("appearance_semantics", {"frames": frames}),
        (
            "depth",
            {"frames": frames, "gt_depth": gt_depth, "predictor": predictor},
        ),
    ):
        result = validity.evaluate(capability, dataset, **inputs)
        verdicts[capability] = result.as_dict()
        verdict = "PASS" if result.passed else "REFUSE"
        print(f"{verdict}  {capability}: {result.reason}")
        for key, value in result.evidence.items():
            print(f"    {key}: {value}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdicts, indent=2, sort_keys=True))
    print(f"\nverdicts written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
