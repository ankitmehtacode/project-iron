"""Per-camera exclusion masks (Day 46, Objective 6): detections anchored in a
mask are dropped before tracking and counted, and a masked region is
unobserved, so no absence can be proven there."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.model.coverage import (
    Absence,
    CannotEstablish,
    Coverage,
    Interval,
    prove_absence,
)
from src.pipeline.events import ExclusionMask, ZoneConfig, load_zone_config
from src.pipeline.run import Detections, apply_exclusion_masks

SQUARE = ((100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0))


def _detections() -> Detections:
    boxes = np.array(
        [
            [140.0, 50.0, 160.0, 150.0],  # anchor (150, 150): inside
            [140.0, 50.0, 160.0, 250.0],  # anchor (150, 250): outside
            [10.0, 10.0, 30.0, 120.0],  # anchor (20, 120): outside
        ]
    )
    return Detections(
        camera_id="cam",
        fps=10.0,
        n_frames=3,
        width=320,
        height=320,
        run_frames=np.array([0, 1, 2]),
        frame=np.array([0, 1, 2]),
        boxes=boxes,
        scores=np.ones(3),
        labels=np.array(["Person", "Person", "Forklift"]),
        producer=("a", "b", "c"),
        backend="test",
    )


def test_detections_anchored_in_a_mask_are_dropped_and_counted() -> None:
    mask = ExclusionMask("cam", "doorway", SQUARE)
    kept, counts = apply_exclusion_masks(_detections(), [mask])
    assert counts == {"doorway": 1}
    assert kept.boxes.shape == (2, 4) and list(kept.frame) == [1, 2]
    assert list(kept.run_frames) == [0, 1, 2]  # the detector did run there
    untouched, none = apply_exclusion_masks(_detections(), [])
    assert none == {} and untouched.boxes.shape == (3, 4)


def test_a_detection_in_two_overlapping_masks_is_counted_once() -> None:
    a = ExclusionMask("cam", "a", SQUARE)
    b = ExclusionMask("cam", "b", SQUARE)
    _, counts = apply_exclusion_masks(_detections(), [a, b])
    assert sum(counts.values()) == 1


def test_site_config_reads_masks_per_camera(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(
        "site_id: s\nzones:\n  - {name: z, polygon: [[0, 0], [1, 0], [1, 1]]}\n"
        "exclusion_masks:\n  cam_a:\n    - {name: window, polygon: "
        "[[0, 0], [10, 0], [10, 10]]}\n"
    )
    config = load_zone_config(path)
    assert [m.region_id for m in config.masks] == ["cam_a/mask/window"]
    assert config.masks_for("cam_b") == ()
    with pytest.raises(ValueError):
        ExclusionMask("cam", "bad", ((0.0, 0.0), (1.0, 1.0)))


def _coverage(masked: tuple[str, ...]) -> Coverage:
    return Coverage(
        subject_id="cam",
        subject_kind="camera",
        interval=Interval(0, 100),
        status="live",
        envelope_ref="env",
        gaps=(),
        manifest_sha="m",
        masked_regions=masked,
    )


def test_absence_cannot_be_established_inside_or_across_a_mask() -> None:
    never = lambda _s, _i: False  # noqa: E731
    log = [_coverage(("cam/mask/doorway",))]
    inside = prove_absence("cam/mask/doorway", Interval(10, 20), never, log)
    assert isinstance(inside, CannotEstablish) and inside.reason == "masked_region"
    whole = prove_absence("cam", Interval(10, 20), never, log)
    assert isinstance(whole, CannotEstablish) and whole.reason == "masked_region"
    unmasked = prove_absence("cam", Interval(10, 20), never, [_coverage(())])
    assert isinstance(unmasked, Absence)


SCENE = (
    Path(__file__).resolve().parent.parent
    / "data/raw/PhysicalAI-SmartSpaces/2cbe9563cbe9f47f846e5c871ee994572bbbc60e"
    / "paths-0deaa0fe541fbaf1/MTMC_Tracking_2026/val/Warehouse_022"
)
CAMS = ("Camera_0000", "Camera_0001")


def _run(tmp: Path, zones: ZoneConfig, name: str) -> tuple[dict[str, Any], Path]:
    from src.detect.backends import OracleDetector
    from src.main import camera_input
    from src.pipeline.run import run
    from src.pipeline.truth import load_truth, oracle_boxes, vocabulary

    gt = load_truth(SCENE / "ground_truth.json")
    cameras = [
        camera_input(SCENE / "videos" / f"{c}.mp4", SCENE / "calibration.json", c)
        for c in CAMS
    ]
    out = tmp / name
    manifest = run(
        cameras,
        lambda cam: OracleDetector(oracle_boxes(gt, cam.camera_id), vocabulary(gt)),
        "Person",
        zones,
        out,
        max_seconds=20.0,
        progress=lambda _m: None,
        gate_mode="off",
        cache_root=tmp / "cache",
        allow_dirty=True,
    )
    return manifest, out


@pytest.mark.slow
@pytest.mark.skipif(not SCENE.exists(), reason=f"SmartSpaces scene absent: {SCENE}")
def test_events_outside_the_mask_are_unchanged_on_smartspaces(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from src.pipeline.events import load_zone_config

    base = load_zone_config(
        Path(__file__).resolve().parent.parent / "configs/zones/warehouse_022.yaml"
    )
    plain, plain_dir = _run(tmp_path, base, "plain")
    tracks = pq.read_table(plain_dir / "tracks.parquet").to_pylist()
    cam0 = [t for t in tracks if t["camera_id"] == CAMS[0]]
    assert cam0, "the fixture needs Camera_0000 tracks"
    # mask the region around one track's first 20 anchors: chosen from the
    # unmasked run, so it is known to contain detections
    first = [t for t in cam0 if t["track_id"] == cam0[0]["track_id"]][:20]
    u = [(t["x0"] + t["x1"]) / 2 for t in first]
    v = [t["y1"] for t in first]
    polygon = (
        (min(u) - 5, min(v) - 5),
        (max(u) + 5, min(v) - 5),
        (max(u) + 5, max(v) + 5),
        (min(u) - 5, max(v) + 5),
    )
    mask = ExclusionMask(CAMS[0], "test", polygon)
    zones = ZoneConfig(base.site_id, base.zones, base.rules, masks=(mask,))
    masked, masked_dir = _run(tmp_path, zones, "masked")
    assert masked["cameras"][CAMS[0]]["suppressed_by_mask"]["test"] > 0
    assert masked["cameras"][CAMS[1]]["suppressed_by_mask"] == {}

    touching = {
        (t["camera_id"], t["track_id"])
        for t in tracks
        if t["camera_id"] == CAMS[0]
        and mask.contains(np.array([[(t["x0"] + t["x1"]) / 2, t["y1"]]]))[0]
    }

    def events(path: Path, skip: set[tuple[str, int]]) -> Counter[tuple[Any, ...]]:
        out: Counter[tuple[Any, ...]] = Counter()
        for e in pq.read_table(path / "events.parquet").to_pylist():
            cam, tid = e["subject"]["id"].rsplit("/t", 1)
            if (cam, int(tid)) in skip:
                continue
            out[(cam, e["verb"], (e["zone"] or {}).get("id"), e["ts_ns"])] += 1
        return out

    outside = events(plain_dir, touching)
    after = events(masked_dir, set())
    assert outside, "no event outside the mask: the comparison would be vacuous"
    assert not (outside - after), "an event outside the mask changed"
    other = Counter({k: n for k, n in after.items() if k[0] == CAMS[1]})
    assert other == Counter({k: n for k, n in outside.items() if k[0] == CAMS[1]})


def test_a_mask_for_a_camera_the_run_lacks_is_refused(tmp_path: Path) -> None:
    from tests.test_pipeline_slice import _synthetic

    from src.pipeline.run import run

    camera, factory, zones = _synthetic(tmp_path, n=5)
    typo = ExclusionMask("cam_typo", "door", SQUARE)
    masked = ZoneConfig(zones.site_id, zones.zones, zones.rules, masks=(typo,))
    with pytest.raises(ValueError, match="cam_typo"):
        run(
            [camera],
            factory,
            "Person",
            masked,
            tmp_path / "run",
            max_seconds=None,
            progress=lambda _m: None,
            gate_mode="off",
            cache_root=tmp_path / "cache",
            allow_dirty=True,
        )
