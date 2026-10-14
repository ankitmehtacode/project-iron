"""The SmartSpaces readers fail loudly on schema drift (Day 41, Objective 1).

Hand-built fixtures, no dataset needed: each test writes the smallest file
that exercises one way the format could drift and checks the reader raises
rather than skipping.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.data.smartspaces import (
    CARD_DOCUMENTED_GT_KEYS,
    GT_OBJECT_KEYS,
    SchemaDrift,
    build_scene_ground_truth,
    iter_ground_truth,
    load_calibration,
    load_scene_ground_truth,
    twin_rev_from_sha256,
)

_OBJECT: dict[str, Any] = {
    "object type": "Person",
    "object id": 7,
    "3d location": [1.0, 2.0, 0.9],
    "3d bounding box scale": [0.5, 0.4, 1.8],
    "3d bounding box rotation": [0.0, 0.0, 1.57],
    "2d bounding box visible": {"Camera_0000": [10, 20, 30, 80]},
}

_SENSOR: dict[str, Any] = {
    "type": "camera",
    "id": "Camera_0000",
    "coordinates": {"x": 1.0, "y": 2.0},
    "scaleFactor": 9.5,
    "translationToGlobalCoordinates": {"x": 3.0, "y": 4.0},
    "attributes": [
        {"name": "fps", "value": "30"},
        {"name": "direction", "value": "93.5"},
        {"name": "direction3d", "value": "-20.1,-79.8,-110.4"},
        {"name": "frameWidth", "value": "1920"},
        {"name": "frameHeight", "value": "1080"},
    ],
    "intrinsicMatrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
    "extrinsicMatrix": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 5.0]],
    "cameraMatrix": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0.2, 1.0]],
    "homography": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
}


def _write_gt(tmp_path: Path, frames: dict[str, Any] | str) -> Path:
    path = tmp_path / "ground_truth.json"
    path.write_text(frames if isinstance(frames, str) else json.dumps(frames))
    return path


def _write_cal(tmp_path: Path, sensors: list[dict[str, Any]]) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"calibrationType": "cartesian", "sensors": sensors}))
    return path


def test_a_well_formed_file_parses_completely(tmp_path: Path) -> None:
    other = {**_OBJECT, "object id": 8, "object type": "Forklift"}
    path = _write_gt(tmp_path, {"0": [_OBJECT, other], "1": [_OBJECT]})
    frames = list(iter_ground_truth(path))
    assert [f for f, _ in frames] == [0, 1]
    assert frames[0][1][1].object_type == "Forklift"
    assert frames[0][1][0].visible == {"Camera_0000": (10, 20, 30, 80)}

    scene = build_scene_ground_truth(path)
    assert scene.class_names == ("Person", "Forklift")
    assert scene.frame.tolist() == [0, 0, 1]
    assert scene.box_xyxy.shape == (3, 4)


def test_the_card_spelling_is_refused_because_the_data_does_not_use_it(
    tmp_path: Path,
) -> None:
    """The card documents underscores; the files use spaces. A file in the
    card's spelling would be a format change, and must raise."""
    assert len(GT_OBJECT_KEYS) == len(CARD_DOCUMENTED_GT_KEYS)
    assert not GT_OBJECT_KEYS & CARD_DOCUMENTED_GT_KEYS
    card_object = {k.replace(" ", "_"): v for k, v in _OBJECT.items()}
    assert set(card_object) == CARD_DOCUMENTED_GT_KEYS
    with pytest.raises(SchemaDrift, match="unknown keys"):
        list(iter_ground_truth(_write_gt(tmp_path, {"0": [card_object]})))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda o: o.update({"velocity": [0, 0, 0]}), "unknown keys"),
        (lambda o: o.pop("3d bounding box rotation"), "missing keys"),
        (lambda o: o.update({"object id": "7"}), "not an int"),
        (lambda o: o.update({"object id": True}), "not an int"),
        (lambda o: o.update({"3d location": [1.0, 2.0]}), "expected 3"),
        (lambda o: o.update({"3d location": [1.0, 2.0, None]}), "number"),
        (
            lambda o: o.update({"2d bounding box visible": {"Camera_0000": [1, 2, 3]}}),
            "not 4 ints",
        ),
        (
            lambda o: o.update(
                {"2d bounding box visible": {"Camera_0000": [1.5, 2, 3, 4]}}
            ),
            "not 4 ints",
        ),
        (lambda o: o.update({"2d bounding box visible": []}), "not an object"),
    ],
    ids=[
        "unknown-key",
        "missing-key",
        "id-string",
        "id-bool",
        "location-2d",
        "location-null",
        "box-3",
        "box-float",
        "visible-list",
    ],
)
def test_object_drift_raises(tmp_path: Path, mutate: Any, match: str) -> None:
    obj = copy.deepcopy(_OBJECT)
    mutate(obj)
    with pytest.raises(SchemaDrift, match=match):
        list(iter_ground_truth(_write_gt(tmp_path, {"0": [obj]})))


@pytest.mark.parametrize(
    "text, match",
    [
        ('{"0": [], "0": []}', "appears twice"),
        ('{"a": []}', "not a decimal"),
        ('{"0": []} {}', "trailing"),
        ('[{"0": []}]', "expected '{'"),
        ('{"0": {}}', "expected a list"),
    ],
    ids=[
        "duplicate-frame",
        "non-numeric-frame",
        "trailing",
        "not-object",
        "frame-dict",
    ],
)
def test_file_structure_drift_raises(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(SchemaDrift, match=match):
        list(iter_ground_truth(_write_gt(tmp_path, text)))


def test_an_object_id_twice_in_one_frame_raises(tmp_path: Path) -> None:
    path = _write_gt(tmp_path, {"0": [_OBJECT, _OBJECT]})
    with pytest.raises(SchemaDrift, match="twice"):
        build_scene_ground_truth(path)


def test_calibration_parses_attribute_strings(tmp_path: Path) -> None:
    calibration = load_calibration(_write_cal(tmp_path, [_SENSOR]))
    camera = calibration.cameras["Camera_0000"]
    assert (camera.fps, camera.frame_width, camera.frame_height) == (30.0, 1920, 1080)
    assert camera.extrinsic.shape == (3, 4)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.update({"distortion": [0.0]}), "unknown keys"),
        (lambda s: s.pop("homography"), "missing keys"),
        (lambda s: s["attributes"].pop(), "attributes"),
        (lambda s: s.update({"type": "lidar"}), "expected 'camera'"),
        (lambda s: s.update({"extrinsicMatrix": [[1.0, 0, 0]] * 3}), "cells"),
        (lambda s: s["attributes"][3].update({"value": "1920.5"}), "integer string"),
        (lambda s: s["attributes"][0].update({"value": 30}), "expected a string"),
    ],
    ids=[
        "unknown",
        "missing",
        "attr-missing",
        "type",
        "matrix-shape",
        "width-float",
        "attr-number",
    ],
)
def test_calibration_drift_raises(tmp_path: Path, mutate: Any, match: str) -> None:
    sensor = copy.deepcopy(_SENSOR)
    mutate(sensor)
    with pytest.raises(SchemaDrift, match=match):
        load_calibration(_write_cal(tmp_path, [sensor]))


def test_duplicate_camera_ids_raise(tmp_path: Path) -> None:
    with pytest.raises(SchemaDrift, match="duplicate camera id"):
        load_calibration(_write_cal(tmp_path, [_SENSOR, _SENSOR]))


def test_twin_rev_is_a_content_hash_of_the_calibration(tmp_path: Path) -> None:
    a = load_calibration(_write_cal(tmp_path, [_SENSOR]))
    again = load_calibration(_write_cal(tmp_path, [_SENSOR]))
    moved = copy.deepcopy(_SENSOR)
    moved["extrinsicMatrix"][2][3] = 5.5
    b = load_calibration(_write_cal(tmp_path, [moved]))
    assert a.twin_rev == again.twin_rev
    assert a.twin_rev != b.twin_rev
    assert 0 <= a.twin_rev < 2**60
    with pytest.raises(ValueError):
        twin_rev_from_sha256("abc")


def test_the_column_cache_is_rebuilt_when_the_source_changes(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "scene.npz"
    path = _write_gt(tmp_path, {"0": [_OBJECT]})
    first = load_scene_ground_truth(path, cache)
    assert cache.exists() and first.frame.tolist() == [0]

    moved = {**_OBJECT, "3d location": [9.0, 9.0, 9.0]}
    _write_gt(tmp_path, {"0": [moved], "1": [moved]})
    second = load_scene_ground_truth(path, cache)
    assert second.frame.tolist() == [0, 1]
    np.testing.assert_array_equal(second.location[0], [9.0, 9.0, 9.0])
    assert second.source_sha256 != first.source_sha256
