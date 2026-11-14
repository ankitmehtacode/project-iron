"""The site-zero-solo scoring path, run end to end on the synthetic stand-in
(Day 46). No real person and no real frame: the stand-in is rendered from a
known camera, so every answer below is known before the run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.site_zero_eval as script
from src.cascade.motion import MotionGateConfig
from src.data import site_zero_synthetic as syn
from src.eval import site_zero


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("site_zero") / "run"
    assert script.main(["--synthetic", "--out", str(out)]) == 0
    payload: dict[str, Any] = json.loads((out / "site_zero_report.json").read_text())
    return payload


def _bottom_centre_floor_point(x: float, y: float) -> np.ndarray:
    """Where the image box's bottom-centre lands on the floor, from the true
    camera: the oracle box is the figure's corner bounding box, so this is
    the exact answer a perfect calibration must give."""
    corners = syn.figure_corners(x, y)
    u = (corners[:, 0].min() + corners[:, 0].max()) / 2
    v = corners[:, 1].max()
    w = np.linalg.inv(syn.floor_to_image()) @ np.array([u, v, 1.0])
    out: np.ndarray = w[:2] / w[2]
    return out


@pytest.mark.slow
def test_calibration_from_the_clicks_recovers_the_camera(
    report: dict[str, Any]
) -> None:
    residual = report["known_answers"]["calibration"]["residual_m"]
    assert len(residual) == len(syn.CALIB_MARKS)
    assert max(residual.values()) < 1e-5


@pytest.mark.slow
def test_per_mark_error_is_the_known_box_anchor_error(report: dict[str, Any]) -> None:
    rows = {r["mark"]: r for r in report["per_mark"]}
    assert set(rows) == set(syn.TEST_MARKS)
    for name, (x, y) in syn.TEST_MARKS.items():
        row = rows[name]
        # the localisation clip runs with the gate off: every window frame sampled
        assert row["frames_with_a_sample"] == row["window_frames"] > 0
        expected = float(np.linalg.norm(_bottom_centre_floor_point(x, y) - (x, y)))
        assert row["raw_error_m"]["p50"] == pytest.approx(expected, abs=1e-4)
        assert row["raw_error_m"]["max"] == pytest.approx(expected, abs=1e-4)
        # the estimator smooths the same samples; it stays near them
        assert row["estimator_error_m"]["p50"] < expected + 0.05
        assert row["estimator_error_m"]["p50"] < row["baseline_centroid_error_m"]


@pytest.mark.slow
def test_empty_stretches_have_no_person_and_bounded_wakes(
    report: dict[str, Any]
) -> None:
    stay = MotionGateConfig().stay_awake_frames
    rows = report["empty_stretches"]
    assert {r["clip"] for r in rows} == {"A", "C"}
    for r in rows:
        assert r["gate_on"]
        assert r["person_detections"] == 0
        assert r["track_points"] == 0
        # after the figure leaves, the gate may stay awake its hysteresis
        assert r["detector_kept_frames"] <= stay
    assert [r["detector_kept_frames"] for r in rows if r["clip"] == "A"] == [0]


@pytest.mark.slow
def test_entries_and_exits_match_the_notes(report: dict[str, Any]) -> None:
    c = report["entries_exits"]["C"]
    assert c["expected"] == {"room": {"entered": 2, "exited": 2}}
    assert c["observed"] == c["expected"]
    assert c["matches"] is True
    assert report["entries_exits"]["A"]["events_total"] == 0


def test_real_and_synthetic_paths_share_one_scoring_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STRUCTURAL: the real-capture branch and the synthetic branch both
    produce their report by calling src.eval.site_zero.run_site_zero, and
    the script holds that exact function. Diverge either and this fails."""
    assert script.run_site_zero is site_zero.run_site_zero
    calls: list[str] = []

    def spy(data_dir: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(str(data_dir))
        return {"spy": True}

    monkeypatch.setattr(script, "run_site_zero", spy)
    # the stand-in's rendering is exercised by the slow tests above
    monkeypatch.setattr(
        script, "synthetic_inputs", lambda d: (d / "c.yaml", object(), "Person", {})
    )
    import src.main

    monkeypatch.setattr(src.main, "pretrained_detector_factory", lambda: object())
    real_out, syn_out = tmp_path / "real", tmp_path / "syn"
    assert (
        script.main(
            ["--data", str(tmp_path), "--calibration", "c.yaml", "--out", str(real_out)]
        )
        == 0
    )
    assert script.main(["--synthetic", "--out", str(syn_out)]) == 0
    assert len(calls) == 2 and calls[0] == str(tmp_path)
    for out in (real_out, syn_out):
        written = json.loads((out / "site_zero_report.json").read_text())
        assert written["spy"] is True


def test_notes_are_validated(tmp_path: Path) -> None:
    bad = tmp_path / "notes.yaml"
    head = "camera_id: c\nclips:\n  A: {file: a.mp4, segments: "
    bad.write_text(head + "[{kind: walk, start_s: 0, end_s: 1}]}\n")
    with pytest.raises(ValueError, match="segment kind"):
        site_zero.load_notes(bad)
    bad.write_text(head + "[{kind: on_mark, start_s: 0, end_s: 1}]}\n")
    with pytest.raises(ValueError, match="without a mark"):
        site_zero.load_notes(bad)
