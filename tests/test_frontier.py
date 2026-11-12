"""Detector-schedule frontier (src/pipeline/frontier.py), Day 43."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.frontier import (
    ADAPTIVE_IDLE_STRIDE,
    ADAPTIVE_LIVE_STRIDE,
    CANDIDATES,
    FrameDets,
    Unit,
    bootstrap,
    decide,
    schedule,
)

EMPTY = FrameDets(np.zeros((0, 4)), np.zeros(0), ())


def person(x: float = 100.0) -> FrameDets:
    return FrameDets(
        np.array([[x, 100.0, x + 40, 200.0]]), np.array([0.9]), ("person",)
    )


def test_fixed_schedules() -> None:
    frames = [EMPTY] * 31
    gate = np.zeros(31, dtype=bool)
    gate[[3, 4, 20]] = True
    assert schedule("gate_off", frames, gate, "person").tolist() == list(range(31))
    assert schedule("gate_default", frames, gate, "person").tolist() == [3, 4, 20]
    assert schedule("stride_6", frames, gate, "person").tolist() == [
        0,
        6,
        12,
        18,
        24,
        30,
    ]
    assert schedule("stride_15", frames, gate, "person").tolist() == [0, 15, 30]


def test_keepalive_runs_through_gate_sleep_while_a_track_lives() -> None:
    # Gate awake on 0-4 only; a person on every frame. The shadow tracker
    # holds a live track, so keepalive keeps running after the gate sleeps.
    frames = [person()] * 20
    gate = np.zeros(20, dtype=bool)
    gate[:5] = True
    runs = schedule("gate_keepalive", frames, gate, "person")
    assert runs.tolist() == list(range(20))


def test_keepalive_lets_the_track_die_when_nobody_is_there() -> None:
    frames = [person()] * 3 + [EMPTY] * 60
    gate = np.zeros(63, dtype=bool)
    gate[:3] = True
    runs = schedule("gate_keepalive", frames, gate, "person").tolist()
    # The track ends after max_missed (30) empty runs; the gate is asleep, so
    # runs stop there instead of continuing to frame 62.
    assert runs[:3] == [0, 1, 2]
    assert runs[-1] < 62


def test_adaptive_idle_and_live_strides() -> None:
    gate = np.ones(61, dtype=bool)
    idle = schedule("adaptive", [EMPTY] * 61, gate, "person")
    assert np.diff(idle).tolist() == [ADAPTIVE_IDLE_STRIDE] * 4
    live = schedule("adaptive", [person()] * 61, gate, "person")
    assert set(np.diff(live).tolist()) == {ADAPTIVE_LIVE_STRIDE}


def test_motion_adaptive_never_runs_on_a_sleeping_gate() -> None:
    frames = [person()] * 90
    gate = np.zeros(90, dtype=bool)
    gate[10:40] = True
    runs = schedule("motion_adaptive", frames, gate, "person")
    assert runs.size and gate[runs].all()


def test_schedule_rejects_unknown_and_misshapen() -> None:
    with pytest.raises(ValueError, match="unknown candidate"):
        schedule("stride_7", [EMPTY], np.ones(1, bool), "person")
    with pytest.raises(ValueError, match="gate mask"):
        schedule("gate_off", [EMPTY] * 2, np.ones(3, bool), "person")
    assert len(CANDIDATES) == 7


def test_bootstrap_is_seeded_and_brackets_the_point() -> None:
    units = [Unit(gt_events=2, matched=1, predicted=1)] * 10 + [
        Unit(gt_events=1, matched=1, predicted=2)
    ] * 10
    a, b = bootstrap(units), bootstrap(units)
    assert a == b
    lo, hi = a["recall"]
    assert lo is not None and hi is not None
    assert lo <= 20 / 30 <= hi


def _block(inv: int, lo: float, hi: float) -> dict[str, object]:
    return {
        "pooled": {
            "invocations": inv,
            "bootstrap95_over_gt_tracks": {"recall": [lo, hi]},
        }
    }


def test_decision_rule_picks_fewest_overlapping_else_stride_6() -> None:
    result = {
        "gate_off": _block(100, 0.9, 1.0),
        "stride_6": _block(20, 0.85, 0.95),
        "stride_15": _block(8, 0.5, 0.7),
        "adaptive": _block(15, 0.88, 0.97),
    }
    assert decide(result)["chosen"] == "adaptive"
    none_pass = {"gate_off": _block(100, 0.9, 1.0), "stride_15": _block(8, 0.5, 0.7)}
    assert decide(none_pass)["chosen"] == "stride_6"


def test_keepalive_reads_live_tracks_before_the_frame_and_steps_only_on_runs() -> None:
    # detect_camera's order: gate decides, then a live track (as it stood
    # BEFORE this frame) keeps the detector awake, and the shadow tracker
    # steps only on frames the detector ran on. A person at frame 45, while
    # the gate sleeps and no track lives, is therefore never seen; any
    # ordering that stepped the tracker on non-run frames would see it and
    # keep running after it.
    frames = [person()] * 3 + [EMPTY] * 42 + [person()] + [EMPTY] * 30
    gate = np.zeros(len(frames), dtype=bool)
    gate[[0, 1, 2, 60, 61, 62]] = True
    runs = schedule("gate_keepalive", frames, gate, "person").tolist()
    # Track born at 0, matched at 1-2, ends after 30 missed runs (3..32).
    assert runs == list(range(33)) + [60, 61, 62]


def test_bootstrap_clusters_one_person_across_cameras() -> None:
    from src.pipeline.frontier import cluster

    units = [
        Unit(key="gt/7", gt_events=2, matched=1, predicted=1),
        Unit(key="gt/7", gt_events=1, matched=1, predicted=2),
        Unit(key="pred/Camera_0000/t3", predicted=1),
    ]
    merged = {u.key: u for u in cluster(units)}
    assert set(merged) == {"gt/7", "pred/Camera_0000/t3"}
    assert (merged["gt/7"].gt_events, merged["gt/7"].matched) == (3, 2)
    assert merged["gt/7"].predicted == 3


def test_the_product_tracker_is_iou_only_and_kalman_is_behind_a_flag() -> None:
    from src.pipeline.tracker import (
        IOU_ONLY,
        KALMAN_TWO_STAGE,
        PRODUCT_TRACKER,
        TRACKERS,
    )

    assert PRODUCT_TRACKER == IOU_ONLY
    assert not PRODUCT_TRACKER.use_kalman and not PRODUCT_TRACKER.second_stage
    assert TRACKERS == {"iou": IOU_ONLY, "kalman": KALMAN_TWO_STAGE}
    assert KALMAN_TWO_STAGE.use_kalman and KALMAN_TWO_STAGE.second_stage
