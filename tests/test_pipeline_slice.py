"""The Day-43 end-to-end slice: floor projection, tracker, event compiler, and
one run over a synthetic video with no dataset needed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.events.schema import Verb
from src.pipeline.events import EventRules, Trajectory, Zone, compile_trajectory
from src.pipeline.floor import CalibrationError, FloorProjector, from_floor_points
from src.pipeline.tracker import IOU_ONLY, Tracker, TrackerConfig

# --- floor projection -------------------------------------------------------


def _write_floor_file(path: Path, homography: np.ndarray, camera: str) -> None:
    """Four floor points, imaged through ``inverse(homography)``."""
    floor = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])
    to_image = np.linalg.inv(homography)
    h = np.c_[floor, np.ones(4)] @ to_image.T
    image = h[:, :2] / h[:, 2:]
    path.write_text(
        json.dumps(
            {
                "camera_id": camera,
                "image_points": image.tolist(),
                "floor_points": floor.tolist(),
                "camera_xyz_m": [2.0, -5.0, 3.0],
            }
        )
    )


TRUE_H = np.array([[0.01, 0.002, -3.0], [0.0005, 0.02, -2.0], [0.0001, 0.001, 1.0]])


def test_four_point_floor_file_recovers_the_homography(tmp_path: Path) -> None:
    path = tmp_path / "cam.yaml"
    _write_floor_file(path, TRUE_H, "cam")
    projector = from_floor_points(path, "cam")
    floor = np.array([[1.0, 2.0], [3.5, 5.0]])
    h = np.c_[floor, np.ones(2)] @ np.linalg.inv(TRUE_H).T
    xy, valid = projector.project(h[:, :2] / h[:, 2:])
    assert valid.all()
    np.testing.assert_allclose(xy, floor, atol=1e-6)
    assert projector.twin_rev >= 0 and projector.source == "floor_points"


def test_floor_file_for_another_camera_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cam.yaml"
    _write_floor_file(path, TRUE_H, "cam")
    with pytest.raises(CalibrationError):
        from_floor_points(path, "other")


def test_a_point_above_the_horizon_is_invalid_not_guessed() -> None:
    projector = FloorProjector(
        camera_id="c",
        image_to_floor=np.eye(3),
        camera_xyz_m=(0.0, 0.0, 1.0),
        twin_rev=1,
        source="test",
        front_sign=1.0,
    )
    # w = 1 for every pixel under the identity; flip the sign for one row.
    projector2 = FloorProjector(
        camera_id="c",
        image_to_floor=np.diag([1.0, 1.0, -1.0]),
        camera_xyz_m=(0.0, 0.0, 1.0),
        twin_rev=1,
        source="test",
        front_sign=1.0,
    )
    _, ok = projector.project(np.array([[1.0, 1.0]]))
    xy, bad = projector2.project(np.array([[1.0, 1.0]]))
    assert ok.all() and not bad.any() and np.isnan(xy).all()


# --- tracker ----------------------------------------------------------------


def _box(cx: float, cy: float, w: float = 20.0, h: float = 60.0) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _ids_by_object(assignments, n_objects):  # type: ignore[no-untyped-def]
    out: dict[int, set[int]] = {k: set() for k in range(n_objects)}
    for a in assignments:
        out[a.detection_index].add(a.track_id)
    return out


def test_two_walkers_keep_their_ids() -> None:
    tracker = Tracker()
    for f in range(40):
        boxes = np.array([_box(100 + 3 * f, 200), _box(400 - 3 * f, 260)])
        tracker.step(f, boxes, np.array([0.9, 0.9]))
    ids = _ids_by_object(tracker.finish(), 2)
    assert all(len(v) == 1 for v in ids.values()) and ids[0] != ids[1]


def test_a_sleeping_gate_does_not_end_a_track() -> None:
    """Frames the detector did not run on are not misses."""
    tracker = Tracker(TrackerConfig(max_missed=3))
    tracker.step(0, np.array([_box(100, 100)]), np.array([0.9]))
    tracker.step(1, np.array([_box(101, 100)]), np.array([0.9]))
    # gate asleep for 500 frames: no step calls
    tracker.step(502, np.array([_box(102, 100)]), np.array([0.9]))
    out = tracker.finish()
    assert {a.track_id for a in out} == {1} and len(out) == 3


def test_second_stage_keeps_a_track_through_a_low_score() -> None:
    def run(config: TrackerConfig) -> set[int]:
        tracker = Tracker(config)
        for f in range(12):
            score = 0.3 if 4 <= f < 8 else 0.9
            tracker.step(f, np.array([_box(100 + 2 * f, 100)]), np.array([score]))
        return {a.track_id for a in tracker.finish()}

    assert run(TrackerConfig()) == {1}
    # Without stage 2 the low scores are dropped; with max_missed=2 the
    # track ends and a new one starts.
    assert len(run(TrackerConfig(second_stage=False, max_missed=2))) == 2


def test_tentative_tracks_are_dropped() -> None:
    tracker = Tracker(TrackerConfig(min_hits=3))
    tracker.step(0, np.array([_box(50, 50)]), np.array([0.9]))
    tracker.step(1, np.array([_box(500, 500)]), np.array([0.9]))
    assert tracker.finish() == []


def test_iou_only_baseline_is_one_stage_no_motion() -> None:
    assert not IOU_ONLY.use_kalman and not IOU_ONLY.second_stage


def test_out_of_order_runs_are_refused() -> None:
    tracker = Tracker()
    tracker.step(5, np.zeros((0, 4)), np.zeros(0))
    with pytest.raises(ValueError):
        tracker.step(5, np.zeros((0, 4)), np.zeros(0))


# --- event compiler ---------------------------------------------------------

ZONE = Zone("z", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
RULES = EventRules(min_presence_s=0.5, dwell_s=3.0, max_gap_s=2.0)


def _trajectory(xs: list[float], fps: float = 10.0) -> Trajectory:
    n = len(xs)
    return Trajectory(
        subject_id="s",
        ts_ns=np.array([int(i * 1e9 / fps) for i in range(n)], dtype=np.int64),
        xy=np.c_[np.array(xs, dtype=float), np.full(n, 5.0)],
        frame_refs=tuple(f"cam/frame-{i}" for i in range(n)),
        score=1.0,
    )


def _verbs(events) -> list[tuple[Verb, str]]:  # type: ignore[no-untyped-def]
    return [(e.verb, e.frame_ref) for e in events]


def test_walk_in_and_out() -> None:
    walk_in = [-5 + 0.5 * i for i in range(10)]
    walk_out = [5 + 0.5 * i for i in range(30)]
    xs = walk_in + [5.0] * 40 + walk_out
    events = compile_trajectory(_trajectory(xs), [ZONE], RULES)
    verbs = [e.verb for e in events]
    assert verbs == [Verb.ENTERED, Verb.DWELLED, Verb.EXITED]
    assert events[0].frame_ref == "cam/frame-10"  # first inside sample
    assert events[1].ts_ns - events[0].ts_ns == 3_000_000_000


def test_edge_flicker_is_not_an_event() -> None:
    xs = [-1.0] * 10 + [0.1, -0.1] * 10 + [-1.0] * 10
    assert compile_trajectory(_trajectory(xs), [ZONE], RULES) == []


def test_a_track_that_starts_inside_did_not_enter() -> None:
    events = compile_trajectory(_trajectory([5.0] * 50), [ZONE], RULES)
    assert [e.verb for e in events] == [Verb.DWELLED]


def test_a_long_gap_resets_state() -> None:
    t = _trajectory([-5.0] * 10 + [5.0] * 10)
    ts = t.ts_ns.copy()
    ts[10:] += 5_000_000_000  # 5 s blind gap between outside and inside
    gapped = Trajectory(t.subject_id, ts, t.xy, t.frame_refs, t.score)
    assert compile_trajectory(gapped, [ZONE], RULES) == []


# --- one run, end to end ----------------------------------------------------


def _synthetic(  # type: ignore[no-untyped-def]
    tmp_path: Path, n: int = 30, still_from: int | None = None
):
    """A box walks left to right across a 320x240 frame at 10 fps. From
    ``still_from`` on it stands still: no pixel changes, so the motion gate
    falls asleep while the oracle still reports the person."""
    import cv2

    from src.detect.backends import GroundTruthBoxes, OracleDetector
    from src.detect.contract import ClassVocabulary
    from src.pipeline.events import ZoneConfig
    from src.pipeline.run import CameraInput

    fps, w, h = 10.0, 320, 240
    video = tmp_path / "cam.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    boxes: dict[int, np.ndarray] = {}
    for f in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        x = 10 + 9 * (f if still_from is None else min(f, still_from))
        boxes[f] = np.array([[x, 100, x + 20, 180]], dtype=float)
        frame[100:180, x : x + 20] = 255
        writer.write(frame)
    writer.release()

    # Floor = image pixels / 10 (a 32 m x 24 m floor seen straight down).
    calibration = tmp_path / "cam.yaml"
    calibration.write_text(
        json.dumps(
            {
                "camera_id": "cam",
                "image_points": [[0, 0], [320, 0], [320, 240], [0, 240]],
                "floor_points": [[0, 0], [32, 0], [32, 24], [0, 24]],
                "camera_xyz_m": [16.0, 12.0, 10.0],
            }
        )
    )
    camera = CameraInput("cam", video, from_floor_points(calibration, "cam"))
    vocab = ClassVocabulary("test", "1", ("Person",))
    truth = GroundTruthBoxes(
        boxes=boxes, labels={f: ["Person"] for f in range(n)}, source_sha="t"
    )
    zones = ZoneConfig(
        site_id="test",
        zones=(Zone("right", ((16.0, 0.0), (40.0, 0.0), (40.0, 24.0), (16.0, 24.0))),),
        rules=EventRules(min_presence_s=0.3, dwell_s=0.5, max_gap_s=2.0),
    )
    return camera, lambda _c: OracleDetector(truth, vocab), zones


def test_run_end_to_end_on_a_synthetic_video(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from src.model.events import read_events_v2_parquet
    from src.pipeline.run import run

    n = 30
    camera, factory, zones = _synthetic(tmp_path, n)
    out = tmp_path / "run"
    manifest = run(
        [camera],
        factory,
        "Person",
        zones,
        out,
        max_seconds=None,
        progress=lambda _m: None,
        gate_mode="off",
        cache_root=tmp_path / "cache",
        allow_dirty=True,
    )

    tracks = pq.read_table(out / "tracks.parquet").to_pylist()
    assert {r["track_id"] for r in tracks} == {1} and len(tracks) == n
    # Bottom-centre of the box at frame 0 is (20, 180) px -> (2.0, 18.0) m.
    assert tracks[0]["raw_x"] == pytest.approx(2.0, abs=1e-6)
    assert tracks[0]["raw_y"] == pytest.approx(18.0, abs=1e-6)
    events = read_events_v2_parquet(out / "events.parquet")
    assert [e.verb for e in events] == [Verb.ENTERED, Verb.DWELLED]
    assert all(e.manifest_sha == manifest["manifest_sha"] for e in events)
    assert manifest["cameras"]["cam"]["detector_runs"] == n
    assert "git_dirty" in manifest["identity"]

    with pytest.raises(FileExistsError):
        run(
            [camera],
            factory,
            "Person",
            zones,
            out,
            max_seconds=None,
            progress=lambda _m: None,
            cache_root=tmp_path / "cache",
            allow_dirty=True,
        )


def _run_mode(  # type: ignore[no-untyped-def]
    tmp_path: Path, camera, factory, zones, mode: str
):
    from src.pipeline.run import Detections, run

    out = tmp_path / f"run_{mode}"
    manifest = run(
        [camera],
        factory,
        "Person",
        zones,
        out,
        max_seconds=None,
        progress=lambda _m: None,
        gate_mode=mode,
        cache_root=tmp_path / "cache",
        allow_dirty=True,
    )
    return manifest, Detections.load(out / "cam" / "detections.npz")


def test_gate_default_is_all_frame_detections_masked_by_the_gate(
    tmp_path: Path,
) -> None:
    from src.cascade.motion import MotionGate, MotionGateConfig
    from src.cascade.stage import StageContext

    import cv2

    camera, factory, zones = _synthetic(tmp_path, n=60, still_from=30)
    m_off, off = _run_mode(tmp_path, camera, factory, zones, "off")
    m_def, gated = _run_mode(tmp_path, camera, factory, zones, "default")

    # The detector ran once per frame, on the first run only.
    assert m_off["cameras"]["cam"]["detection_cache"]["cache_misses"] == 60
    assert m_def["cameras"]["cam"]["detection_cache"]["cache_misses"] == 0
    assert m_def["cameras"]["cam"]["detection_cache"]["cache_hits"] == 60

    # The kept frames are exactly the frames a standalone gate wakes on.
    gate = MotionGate(MotionGateConfig())
    capture = cv2.VideoCapture(str(camera.video))
    awake = []
    for i in range(60):
        ok, image = capture.read()
        assert ok
        out = gate.process(image[None], StageContext(frame_index=i, ts_ns=0))
        if gate.should_wake_next(out):
            awake.append(i)
    assert gated.run_frames.tolist() == awake
    assert 0 < len(awake) < 60  # the gate sleeps once the box stands still

    # And on those frames the boxes are the all-frame boxes.
    sel = np.isin(off.frame, gated.run_frames)
    np.testing.assert_array_equal(gated.boxes, off.boxes[sel])
    np.testing.assert_array_equal(gated.frame, off.frame[sel])


def test_keepalive_keeps_frames_while_a_track_lives(tmp_path: Path) -> None:
    camera, factory, zones = _synthetic(tmp_path, n=60, still_from=30)
    _, gated = _run_mode(tmp_path, camera, factory, zones, "default")
    _, kept = _run_mode(tmp_path, camera, factory, zones, "keepalive")
    assert set(gated.run_frames) <= set(kept.run_frames)
    # The person never leaves, so once tracked every later frame is kept.
    first = int(kept.run_frames[0])
    assert kept.run_frames.tolist() == list(range(first, 60))
    assert len(kept.run_frames) > len(gated.run_frames)


def test_dirty_paths_counts_tracked_changes_and_untracked_code() -> None:
    from src.pipeline.run import dirty_paths

    porcelain = "\n".join(
        [
            " M README.md",
            "A  docs/new.md",
            "?? src/pipeline/new_module.py",
            "?? configs/zones/x.yaml",
            "?? artifacts/pytest/run.log",
            "?? .cache/thing.npz",
        ]
    )
    assert dirty_paths(porcelain) == [
        "README.md",
        "docs/new.md",
        "src/pipeline/new_module.py",
        "configs/zones/x.yaml",
    ]
    assert dirty_paths("?? notes.txt\n") == []


@pytest.mark.parametrize("dirty", [True, None])
def test_run_refuses_a_dirty_tree_without_allow_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty: bool | None
) -> None:
    import src.pipeline.run as run_module

    monkeypatch.setattr(
        run_module,
        "git_state",
        lambda _root: {"commit": "c0ffee", "dirty": dirty, "dirty_paths": ["x.py"]},
    )
    camera, factory, zones = _synthetic(tmp_path, n=5)
    kwargs = dict(
        max_seconds=None,
        progress=lambda _m: None,
        gate_mode="off",
        cache_root=tmp_path / "cache",
    )
    with pytest.raises(run_module.DirtyTreeError):
        run_module.run([camera], factory, "Person", zones, tmp_path / "a", **kwargs)
    assert not (tmp_path / "a").exists()
    manifest = run_module.run(
        [camera], factory, "Person", zones, tmp_path / "b", allow_dirty=True, **kwargs
    )
    assert manifest["identity"]["git_dirty"] is dirty
    assert manifest["git_dirty_paths"] == ["x.py"]


def test_cli_help_runs() -> None:
    from src.main import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


# --- evaluation: velocity and coverage ---------------------------------------


def _truth_walking(n: int = 10, fps: float = 10.0):  # type: ignore[no-untyped-def]
    """One person walking +x at 2 m/s, 15 m from a camera at the origin."""
    from src.pipeline.truth import CameraTruth

    frames = np.arange(n)
    xy = np.c_[15.0 + 2.0 * frames / fps, np.zeros(n)]
    return CameraTruth(
        frame=frames.astype(np.int64),
        object_id=np.zeros(n, np.int64),
        box_xyxy=np.tile([0.0, 0.0, 10.0, 10.0], (n, 1)),
        floor_xy=xy,
    )


def test_gt_velocity_is_the_central_difference() -> None:
    from src.pipeline.evaluate import gt_velocity

    v = gt_velocity(_truth_walking(), fps=10.0)
    np.testing.assert_allclose(v, np.tile([2.0, 0.0], (10, 1)))


def test_velocity_errors_and_coverage_on_a_gated_track() -> None:
    from src.pipeline.evaluate import coverage_counts, raw_velocity, velocity_errors
    from src.pipeline.run import TrackPoint

    truth = _truth_walking()
    # The tracker saw frames 0, 2, 3 and 7 (a gate that slept), the raw
    # projection exact, the estimator velocity 0.5 m/s too slow.
    kept = [0, 2, 3, 7]
    points = [
        TrackPoint(
            camera_id="c",
            track_id=1,
            frame_index=f,
            ts_ns=0,
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
            score=1.0,
            raw_xy=tuple(truth.floor_xy[f]),  # type: ignore[arg-type]
            est_xy=tuple(truth.floor_xy[f]),  # type: ignore[arg-type]
            est_sigma_m=0.1,
            est_v=(1.5, 0.0),
        )
        for f in kept
    ]
    raw = raw_velocity(points, fps=10.0)
    assert np.isnan(raw[0]).all()  # a track's first point has no difference
    np.testing.assert_allclose(raw[1:], np.tile([2.0, 0.0], (3, 1)))

    matches = [(f, f, 1, i) for i, f in enumerate(kept)]
    errors = velocity_errors(truth, points, matches, np.zeros(2), fps=10.0)
    assert errors is not None
    assert errors["[10,20)"]["estimator"] == pytest.approx([0.5] * 3)
    assert errors["[10,20)"]["raw_projection"] == pytest.approx([0.0] * 3)
    assert coverage_counts(truth, matches, np.zeros(2)) == {"[10,20)": [10, 4]}

    old = [TrackPoint(**{**p.__dict__, "est_v": None}) for p in points]
    assert velocity_errors(truth, old, matches, np.zeros(2), fps=10.0) is None
