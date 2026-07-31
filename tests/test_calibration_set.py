"""Tests for the INT8 calibration-set builder.

The refusals matter more than the happy path here. Calibration decides the
model's numerical behaviour everywhere, so a set built from the wrong lane is
not a slightly worse set — it is a licensing violation (lane R) or a silent
accuracy loss (lane S).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_calibration_set as builder  # noqa: E402

from src.data.golden import Condition  # noqa: E402
from src.data.registry import DatasetRegistry, LaneViolation  # noqa: E402


def make_source(root: Path, count: int, conditions: list[str]) -> Path:
    """A frame directory with declared condition tags."""
    source = root / "session"
    source.mkdir(parents=True, exist_ok=True)
    tags: dict[str, list[str]] = {}
    for index in range(count):
        name = f"frame_{index:05d}.png"
        # Distinct bytes per frame so the shas differ, as real frames would.
        (source / name).write_bytes(b"PNG" + str(index).encode())
        tags[name] = [conditions[index % len(conditions)]]
    (source / "conditions.json").write_text(json.dumps(tags))
    return source


ALL_CONDITIONS = [c.value for c in Condition]


# ---------------------------------------------------------------------------
# Lane refusals
# ---------------------------------------------------------------------------


def test_lane_r_is_refused_with_the_exact_message() -> None:
    """The message the data strategy specifies, verbatim."""
    registry = DatasetRegistry.load(builder.REGISTRY_PATH)
    with pytest.raises(LaneViolation) as excinfo:
        builder.require_lane_c(registry, "NTU-RGBD-120")
    assert "INT8 calibration requires lane C domain data" in str(excinfo.value)


def test_lane_r_refusal_names_the_licensing_reason() -> None:
    registry = DatasetRegistry.load(builder.REGISTRY_PATH)
    with pytest.raises(LaneViolation, match="license forbids"):
        builder.require_lane_c(registry, "Market-1501")


def test_lane_s_is_also_refused_for_a_different_reason() -> None:
    """Synthetic is a domain mismatch, not a licensing problem.

    Distinguishing them matters: someone told only "wrong lane" would
    reasonably reach for the synthetic set, which is exactly the wrong fix.
    """
    registry = DatasetRegistry.load(builder.REGISTRY_PATH)
    with pytest.raises(LaneViolation) as excinfo:
        builder.require_lane_c(registry, "Kubric")
    message = str(excinfo.value)
    assert "INT8 calibration requires lane C domain data" in message
    assert "sensor noise" in message


def test_lane_c_is_accepted() -> None:
    registry = DatasetRegistry.load(builder.REGISTRY_PATH)
    builder.require_lane_c(registry, "site-zero")  # must not raise


def test_end_to_end_refusal_exits_nonzero(tmp_path: Path) -> None:
    source = make_source(tmp_path, 600, ALL_CONDITIONS)
    code = builder.main(
        ["--source", str(source), "--dataset", "NTU-RGBD-120", "--frames", "512"]
    )
    assert code == 2


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def test_builds_a_stratified_set(tmp_path: Path) -> None:
    source = make_source(tmp_path, 900, ALL_CONDITIONS)
    output = tmp_path / "calib"
    code = builder.main(
        [
            "--source",
            str(source),
            "--dataset",
            "site-zero",
            "--frames",
            "512",
            "--output",
            str(output),
        ]
    )
    assert code == 0

    manifest = json.loads((output / "calibration_manifest.json").read_text())
    assert manifest["frame_count"] == 512
    assert manifest["lane"] == "C"
    assert manifest["calibration_set_sha"]
    assert len(list((output / "frames").glob("*.png"))) == 512

    # Every condition present in the source must appear in the set: an
    # unstratified set is the failure this script exists to prevent.
    assert manifest["conditions_uncovered"] == []
    assert len(manifest["condition_coverage"]) == len(ALL_CONDITIONS)


def test_rare_conditions_are_not_swamped(tmp_path: Path) -> None:
    """Round-robin, not proportional.

    A condition that is rare in the source is often the one that stresses the
    activation ranges hardest. Proportional sampling would reproduce the
    source's imbalance, which is what makes a calibration set useless.
    """
    source = tmp_path / "session"
    source.mkdir()
    tags: dict[str, list[str]] = {}
    for index in range(1000):
        name = f"frame_{index:05d}.png"
        (source / name).write_bytes(b"PNG" + str(index).encode())
        # 995 daylight frames, 5 glare frames.
        tags[name] = ["glare"] if index < 5 else ["daylight"]
    (source / "conditions.json").write_text(json.dumps(tags))

    frames = sorted(source.glob("*.png"))
    chosen = builder.stratified_sample(frames, tags, target=512, seed=1)

    glare = {f"frame_{i:05d}.png" for i in range(5)}
    picked_glare = sum(1 for path in chosen if path.name in glare)
    assert picked_glare == 5, (
        f"only {picked_glare} of 5 rare glare frames selected; proportional "
        "sampling would have taken ~2"
    )


def test_sampling_is_deterministic(tmp_path: Path) -> None:
    """Same source and seed, same set — calibration must be reproducible."""
    source = make_source(tmp_path, 700, ALL_CONDITIONS)
    frames = sorted(source.glob("*.png"))
    tags = builder.read_condition_tags(source)
    first = builder.stratified_sample(frames, tags, 512, seed=7)
    second = builder.stratified_sample(frames, tags, 512, seed=7)
    assert first == second


# ---------------------------------------------------------------------------
# Condition tags
# ---------------------------------------------------------------------------


def test_missing_conditions_file_is_an_error(tmp_path: Path) -> None:
    """No tags means nothing to stratify by; falling back would defeat it."""
    source = tmp_path / "session"
    source.mkdir()
    (source / "frame_00000.png").write_bytes(b"PNG")
    code = builder.main(["--source", str(source), "--dataset", "site-zero"])
    assert code == 2


def test_unknown_condition_tag_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "session"
    source.mkdir()
    (source / "frame_00000.png").write_bytes(b"PNG")
    (source / "conditions.json").write_text(json.dumps({"frame_00000.png": ["dusk"]}))
    with pytest.raises(builder.CalibrationError, match="unknown condition"):
        builder.read_condition_tags(source)


def test_too_few_frames_is_refused(tmp_path: Path) -> None:
    """Ranges fitted to a hundred frames describe noise."""
    source = make_source(tmp_path, 100, ALL_CONDITIONS)
    code = builder.main(
        [
            "--source",
            str(source),
            "--dataset",
            "site-zero",
            "--output",
            str(tmp_path / "c"),
        ]
    )
    assert code == 1


def test_uncovered_conditions_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gap in coverage must be visible, not inferred from an absence."""
    source = make_source(tmp_path, 600, ["daylight", "crowded"])
    builder.main(
        [
            "--source",
            str(source),
            "--dataset",
            "site-zero",
            "--output",
            str(tmp_path / "calib"),
        ]
    )
    manifest = json.loads(
        (tmp_path / "calib" / "calibration_manifest.json").read_text()
    )
    assert "lens_smudge" in manifest["conditions_uncovered"]
    assert "UNCOVERED" in capsys.readouterr().out
