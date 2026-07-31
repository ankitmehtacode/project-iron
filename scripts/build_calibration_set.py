"""Build an INT8 calibration set from lane-C deployment footage.

Why the lane check is the whole point
-------------------------------------
INT8 quantization records activation ranges observed on calibration frames.
Those ranges are then baked into the model, so calibration data is not a
convenience sample — it decides the model's numerical behaviour everywhere.
Calibrating on the wrong domain costs more accuracy than every other export bug
combined, and calibrating on research-licensed data means shipping a model
derived from data that forbids it, with the quantization tables as evidence.

So this refuses lane R, by name, with the exact message the strategy document
specifies. It also refuses lane S: synthetic frames have no sensor noise, no
rolling shutter, no compression artefacts and no real lighting, and a model
whose activation ranges were fitted to clean renders clips real ones.

Stratification
--------------
512-1024 frames spread across declared conditions, not the first N frames of
one recording. A calibration set of one bright empty corridor produces
activation ranges that describe a bright empty corridor, and every crowded
evening scene then falls outside them.

    python scripts/build_calibration_set.py --source data/site-zero/session-003
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import IronConfig
from src.data.golden import Condition
from src.data.registry import DatasetRegistry, LaneViolation, RegistryError

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"

MIN_FRAMES = 512
MAX_FRAMES = 1024

# The message the data strategy specifies verbatim.
LANE_REFUSAL = "INT8 calibration requires lane C domain data"

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
CONDITIONS_FILENAME = "conditions.json"


class CalibrationError(RuntimeError):
    """Raised when a calibration set cannot be built honestly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_lane_c(registry: DatasetRegistry, dataset: str) -> None:
    """Refuse anything that is not consented, self-collected, in-domain data.

    Raises:
        LaneViolation: for lanes R and S, each with its own reason. They are
            different mistakes: lane R is a licensing violation, lane S is a
            domain mismatch that silently costs accuracy.
    """
    entry = registry.get(dataset)
    if entry.lane == "C":
        return
    if entry.lane == "R":
        raise LaneViolation(
            f"{LANE_REFUSAL}. {entry.name!r} is lane R (research-licensed, "
            "eval only). Quantization tables derived from it would ship a "
            "model built on data whose license forbids exactly that."
        )
    raise LaneViolation(
        f"{LANE_REFUSAL}. {entry.name!r} is lane S (synthetic). Synthetic "
        "frames carry no sensor noise, rolling shutter, compression artefacts "
        "or real lighting, so activation ranges fitted to them clip on real "
        "footage. Synthetic data is fine for training; calibration needs the "
        "deployment domain."
    )


def read_condition_tags(source: Path) -> dict[str, list[str]]:
    """Read ``conditions.json``: frame filename -> condition tags.

    Required, not optional. Without declared conditions there is nothing to
    stratify by, and an unstratified calibration set is the failure this script
    exists to prevent — so its absence is an error rather than a fallback to
    "just take the first 512".

    Raises:
        CalibrationError: if it is missing, malformed, or names an unknown
            condition.
    """
    path = source / CONDITIONS_FILENAME
    if not path.exists():
        raise CalibrationError(
            f"{path} is missing. A calibration set must be stratified across "
            "declared conditions; without tags there is nothing to stratify "
            "by, and an unstratified set produces activation ranges that "
            "describe one scene. Expected a JSON mapping of frame filename to "
            f"a list of condition tags from: "
            f"{', '.join(sorted(c.value for c in Condition))}"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibrationError(f"{path}: expected a JSON object at the top level")

    known = {c.value for c in Condition}
    tags: dict[str, list[str]] = {}
    for frame, values in payload.items():
        if not isinstance(values, list) or not values:
            raise CalibrationError(f"{path}: {frame!r} has no condition tags")
        unknown = sorted(set(values) - known)
        if unknown:
            raise CalibrationError(
                f"{path}: {frame!r} declares unknown condition(s) {unknown}. "
                f"Valid: {', '.join(sorted(known))}"
            )
        tags[frame] = list(values)
    return tags


def stratified_sample(
    frames: list[Path],
    tags: dict[str, list[str]],
    target: int,
    seed: int,
) -> list[Path]:
    """Round-robin across conditions until the target is met.

    Round-robin rather than proportional: a condition that is rare in the
    source is often the one that stresses the activation ranges hardest — the
    dark corridor, the glare, the crowded lobby. Proportional sampling would
    reproduce the source's own imbalance, which is exactly what makes an
    unstratified set useless.
    """
    rng = random.Random(seed)

    buckets: dict[str, list[Path]] = defaultdict(list)
    for frame in frames:
        for tag in tags.get(frame.name, []):
            buckets[tag].append(frame)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    chosen: list[Path] = []
    seen: set[Path] = set()
    order = sorted(buckets)
    cursors = {tag: 0 for tag in order}

    while len(chosen) < target:
        progressed = False
        for tag in order:
            if len(chosen) >= target:
                break
            bucket = buckets[tag]
            while cursors[tag] < len(bucket):
                candidate = bucket[cursors[tag]]
                cursors[tag] += 1
                if candidate not in seen:
                    seen.add(candidate)
                    chosen.append(candidate)
                    progressed = True
                    break
        if not progressed:
            break  # every bucket exhausted
    return chosen


def build(args: argparse.Namespace) -> int:
    config = IronConfig.load()
    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 2

    registry = DatasetRegistry.load(REGISTRY_PATH)
    try:
        require_lane_c(registry, args.dataset)
    except (LaneViolation, RegistryError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    frames = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in FRAME_SUFFIXES and path.is_file()
    )
    if not frames:
        print(f"No frames found under {source}", file=sys.stderr)
        return 2

    tags = read_condition_tags(source)
    untagged = [f.name for f in frames if f.name not in tags]
    if untagged:
        raise CalibrationError(
            f"{len(untagged)} frame(s) have no condition tags, e.g. "
            f"{untagged[:3]}. Every frame must declare its conditions, or the "
            "stratification is over a subset while the set claims coverage."
        )

    target = max(MIN_FRAMES, min(args.frames, MAX_FRAMES))
    chosen = stratified_sample(frames, tags, target, args.seed)

    if len(chosen) < MIN_FRAMES:
        print(
            f"REFUSED: only {len(chosen)} frames available, minimum is "
            f"{MIN_FRAMES}. Too few calibration frames produce activation "
            "ranges fitted to noise; capture more before quantizing.",
            file=sys.stderr,
        )
        return 1

    output = (
        Path(args.output)
        if args.output
        else (
            config.paths.resolved_data_dir
            / "calibration"
            / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
    )
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for frame in chosen:
        digest = sha256_file(frame)
        shutil.copy2(frame, frames_dir / frame.name)
        for tag in tags[frame.name]:
            coverage[tag] += 1
        records.append(
            {"file": frame.name, "sha256": digest, "conditions": tags[frame.name]}
        )

    set_sha = hashlib.sha256(
        json.dumps(sorted(r["sha256"] for r in records), separators=(",", ":")).encode()
    ).hexdigest()

    manifest = {
        "schema_version": "1.0",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/build_calibration_set.py",
        "dataset": args.dataset,
        "lane": "C",
        "lane_note": (
            "INT8 calibration is restricted to lane C: calibration decides the "
            "model's numerical behaviour everywhere, so it must come from the "
            "deployment domain and from data we are licensed to ship."
        ),
        "source": str(source),
        "seed": args.seed,
        "frame_count": len(records),
        "calibration_set_sha": set_sha,
        "condition_coverage": dict(sorted(coverage.items())),
        "conditions_uncovered": sorted(
            c.value for c in Condition if coverage.get(c.value, 0) == 0
        ),
        "frames": records,
        "config_sha": config.config_sha(),
    }
    manifest_path = output / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Calibration set: {len(records)} frames -> {output}")
    print(f"  set sha    : {set_sha[:16]}...")
    print(f"  conditions : {len(coverage)} covered")
    if manifest["conditions_uncovered"]:
        print(
            f"  UNCOVERED  : {', '.join(manifest['conditions_uncovered'])}\n"
            "               activation ranges will not reflect these; capture "
            "them before quantizing for production."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="directory of frames")
    parser.add_argument(
        "--dataset",
        default="site-zero",
        help="registry name of the source; must be lane C",
    )
    parser.add_argument("--frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    try:
        return build(args)
    except CalibrationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
