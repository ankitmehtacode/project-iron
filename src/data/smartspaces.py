"""Strict readers for NVIDIA PhysicalAI-SmartSpaces ``MTMC_Tracking_2026``.

Lane S, synthetic (``configs/datasets.yaml``, commit ``2cbe9563``). Not
evidence about real crowding or real sensors.

Fail loudly on schema drift
---------------------------
Every reader here raises :class:`SchemaDrift` on an unknown key, a missing
key, or a value of the wrong shape or type, rather than skipping it. A
tolerant loader would turn a format change into silently missing data,
and the first thing anyone would see is a plausible-looking number.

That rule fired on the first file read (Day 41). The dataset card
documents the ground-truth keys with underscores (``object_type``,
``3d_location``, ...); every 2026 file on disk uses spaces
(``"object type"``, ``"3d location"``, ...). :data:`GT_OBJECT_KEYS` is the
spelling the data actually has, and :data:`CARD_DOCUMENTED_GT_KEYS` keeps
the card's spelling beside it so the disagreement stays visible.

Memory
------
:func:`iter_ground_truth` decodes one frame at a time with the standard
library's ``JSONDecoder.raw_decode``, so a scene costs the file's text plus
one frame of objects. A whole-file ``json.load`` of the largest scene
(282 MB) peaked at 1.54 GB resident (``/usr/bin/time -l``, Day 41). That
is a tooling measurement for choosing a parser, not a product claim. No
streaming-JSON dependency is added for it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

GT_OBJECT_KEYS: frozenset[str] = frozenset(
    {
        "object type",
        "object id",
        "3d location",
        "3d bounding box scale",
        "3d bounding box rotation",
        "2d bounding box visible",
    }
)
"""The per-object keys every 2026 ``ground_truth.json`` actually uses."""

CARD_DOCUMENTED_GT_KEYS: frozenset[str] = frozenset(
    {
        "object_type",
        "object_id",
        "3d_location",
        "3d_bounding_box_scale",
        "3d_bounding_box_rotation",
        "2d_bounding_box_visible",
    }
)
"""The spelling the dataset card's README documents (raw README at
``2cbe9563``, "Ground Truth Format"). Not what the files contain."""

CALIBRATION_TOP_KEYS: frozenset[str] = frozenset({"calibrationType", "sensors"})
SENSOR_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "id",
        "coordinates",
        "scaleFactor",
        "translationToGlobalCoordinates",
        "attributes",
        "intrinsicMatrix",
        "extrinsicMatrix",
        "cameraMatrix",
        "homography",
    }
)
SENSOR_ATTRIBUTES: frozenset[str] = frozenset(
    {"fps", "direction", "direction3d", "frameWidth", "frameHeight"}
)


class SchemaDrift(ValueError):
    """A SmartSpaces file does not have the shape this reader was built for."""


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraCalibration:
    """One sensor from ``calibration.json``, every field parsed and checked.

    ``attributes`` values are JSON strings on disk (``"30"``, ``"1920"``)
    where the card documents numbers; they are parsed here and the string
    form is not kept.
    """

    camera_id: str
    intrinsic: FloatArray  # [3, 3]
    extrinsic: FloatArray  # [3, 4]
    camera_matrix: FloatArray  # [3, 4]
    homography: FloatArray  # [3, 3]
    fps: float
    frame_width: int
    frame_height: int
    direction_deg: float
    direction3d: tuple[float, float, float]
    coordinates: tuple[float, float]
    scale_factor: float
    translation_to_global: tuple[float, float]


@dataclass(frozen=True)
class SceneCalibration:
    path: Path
    sha256: str
    cameras: dict[str, CameraCalibration]

    @property
    def twin_rev(self) -> int:
        """See :func:`twin_rev_from_sha256`."""
        return twin_rev_from_sha256(self.sha256)


def twin_rev_from_sha256(sha256_hex: str) -> int:
    """A ``twin_rev`` for a scene: the first 60 bits of its
    ``calibration.json`` sha256.

    ``WorldPosition.twin_rev`` is an ``int``, and this project never
    defaults it. A content hash makes the revision a fact about the file
    rather than a counter someone has to remember to bump: two scenes
    share a ``twin_rev`` exactly when their calibration bytes are equal.
    60 bits keeps it a non-negative int64."""
    if len(sha256_hex) != 64:
        raise ValueError(f"expected a sha256 hex digest, got {sha256_hex!r}")
    return int(sha256_hex[:15], 16)


def load_calibration(path: Path) -> SceneCalibration:
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    _require_keys(data, CALIBRATION_TOP_KEYS, f"{path.name} top level")
    if data["calibrationType"] != "cartesian":
        raise SchemaDrift(
            f"{path}: calibrationType {data['calibrationType']!r}, expected 'cartesian'"
        )
    cameras: dict[str, CameraCalibration] = {}
    for index, sensor in enumerate(_require_list(data["sensors"], "sensors")):
        where = f"{path.name} sensors[{index}]"
        _require_keys(sensor, SENSOR_KEYS, where)
        if sensor["type"] != "camera":
            raise SchemaDrift(f"{where}: type {sensor['type']!r}, expected 'camera'")
        attrs = _parse_attributes(sensor["attributes"], where)
        camera_id = _require_str(sensor["id"], f"{where}.id")
        if camera_id in cameras:
            raise SchemaDrift(f"{where}: duplicate camera id {camera_id!r}")
        direction3d = tuple(
            _finite(float(v), where) for v in attrs["direction3d"].split(",")
        )
        if len(direction3d) != 3:
            raise SchemaDrift(f"{where}: direction3d has {len(direction3d)} parts")
        cameras[camera_id] = CameraCalibration(
            camera_id=camera_id,
            intrinsic=_matrix(
                sensor["intrinsicMatrix"], (3, 3), f"{where}.intrinsicMatrix"
            ),
            extrinsic=_matrix(
                sensor["extrinsicMatrix"], (3, 4), f"{where}.extrinsicMatrix"
            ),
            camera_matrix=_matrix(
                sensor["cameraMatrix"], (3, 4), f"{where}.cameraMatrix"
            ),
            homography=_matrix(sensor["homography"], (3, 3), f"{where}.homography"),
            fps=_finite(float(attrs["fps"]), where),
            frame_width=_int_string(attrs["frameWidth"], f"{where}.frameWidth"),
            frame_height=_int_string(attrs["frameHeight"], f"{where}.frameHeight"),
            direction_deg=_finite(float(attrs["direction"]), where),
            direction3d=(direction3d[0], direction3d[1], direction3d[2]),
            coordinates=_xy(sensor["coordinates"], f"{where}.coordinates"),
            scale_factor=_number(sensor["scaleFactor"], f"{where}.scaleFactor"),
            translation_to_global=_xy(
                sensor["translationToGlobalCoordinates"],
                f"{where}.translationToGlobalCoordinates",
            ),
        )
    return SceneCalibration(
        path=path, sha256=hashlib.sha256(raw_bytes).hexdigest(), cameras=cameras
    )


def _parse_attributes(value: Any, where: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in _require_list(value, f"{where}.attributes"):
        _require_keys(item, frozenset({"name", "value"}), f"{where}.attributes[]")
        name = _require_str(item["name"], f"{where}.attributes[].name")
        if name in out:
            raise SchemaDrift(f"{where}: duplicate attribute {name!r}")
        out[name] = _require_str(item["value"], f"{where}.attributes[{name}].value")
    if set(out) != SENSOR_ATTRIBUTES:
        raise SchemaDrift(
            f"{where}: attributes {sorted(out)} != expected {sorted(SENSOR_ATTRIBUTES)}"
        )
    return out


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GtObject:
    object_type: str
    object_id: int
    location: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float]
    visible: dict[str, tuple[int, int, int, int]]
    """camera id -> ``[xmin, ymin, xmax, ymax]`` in pixels."""


def iter_ground_truth(path: Path) -> Iterator[tuple[int, list[GtObject]]]:
    """Yield ``(frame_id, objects)`` per frame, in file order, validating
    every object. Raises :class:`SchemaDrift` at the first deviation."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = _expect(text, _skip_ws(text, 0), "{", path)
    pos = _skip_ws(text, pos)
    if text.startswith("}", pos):
        _require_end(text, pos + 1, path)
        return
    seen: set[int] = set()
    while True:
        key, pos = decoder.raw_decode(text, pos)
        if not isinstance(key, str) or not key.isdigit():
            raise SchemaDrift(f"{path}: frame key {key!r} is not a decimal string")
        frame_id = int(key)
        if frame_id in seen:
            raise SchemaDrift(f"{path}: frame {frame_id} appears twice")
        seen.add(frame_id)
        pos = _expect(text, _skip_ws(text, pos), ":", path)
        value, pos = decoder.raw_decode(text, _skip_ws(text, pos))
        where = f"{path.name} frame {frame_id}"
        yield frame_id, [
            _parse_object(obj, f"{where}[{i}]")
            for i, obj in enumerate(_require_list(value, where))
        ]
        pos = _skip_ws(text, pos)
        if text.startswith(",", pos):
            pos = _skip_ws(text, pos + 1)
            continue
        _require_end(text, _expect(text, pos, "}", path), path)
        return


def _parse_object(obj: Any, where: str) -> GtObject:
    _require_keys(obj, GT_OBJECT_KEYS, where)
    object_id = obj["object id"]
    if isinstance(object_id, bool) or not isinstance(object_id, int):
        raise SchemaDrift(f"{where}: object id {object_id!r} is not an int")
    visible_raw = obj["2d bounding box visible"]
    if not isinstance(visible_raw, dict):
        raise SchemaDrift(f"{where}: 2d bounding box visible is not an object")
    visible: dict[str, tuple[int, int, int, int]] = {}
    for camera_id, box in visible_raw.items():
        items = _require_list(box, f"{where} box {camera_id}")
        if len(items) != 4 or any(
            isinstance(v, bool) or not isinstance(v, int) for v in items
        ):
            raise SchemaDrift(f"{where} box {camera_id}: {box!r} is not 4 ints")
        visible[camera_id] = (items[0], items[1], items[2], items[3])
    return GtObject(
        object_type=_require_str(obj["object type"], f"{where}.object type"),
        object_id=object_id,
        location=_vec3(obj["3d location"], f"{where}.3d location"),
        scale=_vec3(obj["3d bounding box scale"], f"{where}.3d bounding box scale"),
        rotation=_vec3(
            obj["3d bounding box rotation"], f"{where}.3d bounding box rotation"
        ),
        visible=visible,
    )


@dataclass(frozen=True)
class SceneGroundTruth:
    """A scene's ground truth as columns, one row per (frame, object).

    ``box_*`` arrays have one row per (frame, object, camera) visible box;
    ``box_row`` indexes back into the object rows.
    """

    source_sha256: str
    class_names: tuple[str, ...]
    camera_names: tuple[str, ...]
    frame: npt.NDArray[np.int32]
    object_id: npt.NDArray[np.int64]
    class_index: npt.NDArray[np.int16]
    location: FloatArray  # [N, 3]
    scale: FloatArray  # [N, 3]
    rotation: FloatArray  # [N, 3]
    box_row: npt.NDArray[np.int64]
    box_camera: npt.NDArray[np.int16]
    box_xyxy: npt.NDArray[np.int32]  # [M, 4]
    frame_ids_in_file_order: npt.NDArray[np.int32]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial.npz")
        np.savez(
            partial,
            parser_version=np.array(PARSER_VERSION),
            source_sha256=np.array(self.source_sha256),
            class_names=np.array(self.class_names),
            camera_names=np.array(self.camera_names),
            frame=self.frame,
            object_id=self.object_id,
            class_index=self.class_index,
            location=self.location,
            scale=self.scale,
            rotation=self.rotation,
            box_row=self.box_row,
            box_camera=self.box_camera,
            box_xyxy=self.box_xyxy,
            frame_ids_in_file_order=self.frame_ids_in_file_order,
        )
        partial.rename(path)


PARSER_VERSION = "smartspaces-gt-columns-v1"


def build_scene_ground_truth(path: Path) -> SceneGroundTruth:
    """Parse ``ground_truth.json`` strictly into :class:`SceneGroundTruth`."""
    source_sha256 = _sha256_file(path)
    classes: dict[str, int] = {}
    cameras: dict[str, int] = {}
    frames: list[int] = []
    order: list[int] = []
    ids: list[int] = []
    cls: list[int] = []
    loc: list[tuple[float, float, float]] = []
    scl: list[tuple[float, float, float]] = []
    rot: list[tuple[float, float, float]] = []
    box_row: list[int] = []
    box_cam: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for frame_id, objects in iter_ground_truth(path):
        order.append(frame_id)
        seen_ids: set[int] = set()
        for obj in objects:
            if obj.object_id in seen_ids:
                raise SchemaDrift(
                    f"{path.name} frame {frame_id}: object id {obj.object_id} twice"
                )
            seen_ids.add(obj.object_id)
            row = len(ids)
            frames.append(frame_id)
            ids.append(obj.object_id)
            cls.append(classes.setdefault(obj.object_type, len(classes)))
            loc.append(obj.location)
            scl.append(obj.scale)
            rot.append(obj.rotation)
            for camera_id, box in obj.visible.items():
                box_row.append(row)
                box_cam.append(cameras.setdefault(camera_id, len(cameras)))
                boxes.append(box)
    return SceneGroundTruth(
        source_sha256=source_sha256,
        class_names=tuple(classes),
        camera_names=tuple(cameras),
        frame=np.asarray(frames, dtype=np.int32),
        object_id=np.asarray(ids, dtype=np.int64),
        class_index=np.asarray(cls, dtype=np.int16),
        location=np.asarray(loc, dtype=np.float64).reshape(-1, 3),
        scale=np.asarray(scl, dtype=np.float64).reshape(-1, 3),
        rotation=np.asarray(rot, dtype=np.float64).reshape(-1, 3),
        box_row=np.asarray(box_row, dtype=np.int64),
        box_camera=np.asarray(box_cam, dtype=np.int16),
        box_xyxy=np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
        frame_ids_in_file_order=np.asarray(order, dtype=np.int32),
    )


def load_scene_ground_truth(gt_path: Path, cache_path: Path) -> SceneGroundTruth:
    """The columns for ``gt_path``, from ``cache_path`` when that cache was
    built from these exact bytes by this parser version, else rebuilt.
    The cache is derived data under the ignored ``data/`` tree."""
    source_sha256 = _sha256_file(gt_path)
    if cache_path.exists():
        with np.load(cache_path) as data:
            if (
                str(data["source_sha256"]) == source_sha256
                and str(data["parser_version"]) == PARSER_VERSION
            ):
                return SceneGroundTruth(
                    source_sha256=source_sha256,
                    class_names=tuple(str(c) for c in data["class_names"]),
                    camera_names=tuple(str(c) for c in data["camera_names"]),
                    frame=data["frame"],
                    object_id=data["object_id"],
                    class_index=data["class_index"],
                    location=data["location"],
                    scale=data["scale"],
                    rotation=data["rotation"],
                    box_row=data["box_row"],
                    box_camera=data["box_camera"],
                    box_xyxy=data["box_xyxy"],
                    frame_ids_in_file_order=data["frame_ids_in_file_order"],
                )
    scene = build_scene_ground_truth(gt_path)
    scene.save(cache_path)
    return scene


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_keys(obj: Any, expected: frozenset[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise SchemaDrift(f"{where}: expected an object, got {type(obj).__name__}")
    keys = set(obj)
    if keys != expected:
        raise SchemaDrift(
            f"{where}: unknown keys {sorted(keys - expected)}, "
            f"missing keys {sorted(expected - keys)}"
        )


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaDrift(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise SchemaDrift(f"{where}: expected a string, got {value!r}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaDrift(f"{where}: expected a number, got {value!r}")
    return _finite(float(value), where)


def _finite(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise SchemaDrift(f"{where}: non-finite value {value!r}")
    return value


def _int_string(value: str, where: str) -> int:
    if not value.isdigit():
        raise SchemaDrift(f"{where}: {value!r} is not an integer string")
    return int(value)


def _vec3(value: Any, where: str) -> tuple[float, float, float]:
    items = _require_list(value, where)
    if len(items) != 3:
        raise SchemaDrift(f"{where}: expected 3 numbers, got {len(items)}")
    return (
        _number(items[0], where),
        _number(items[1], where),
        _number(items[2], where),
    )


def _xy(value: Any, where: str) -> tuple[float, float]:
    _require_keys(value, frozenset({"x", "y"}), where)
    return (_number(value["x"], where), _number(value["y"], where))


def _matrix(value: Any, shape: tuple[int, int], where: str) -> FloatArray:
    rows = _require_list(value, where)
    if len(rows) != shape[0]:
        raise SchemaDrift(f"{where}: {len(rows)} rows, expected {shape[0]}")
    out = np.empty(shape, dtype=np.float64)
    for r, row in enumerate(rows):
        cells = _require_list(row, where)
        if len(cells) != shape[1]:
            raise SchemaDrift(
                f"{where}: row {r} has {len(cells)} cells, expected {shape[1]}"
            )
        for c, cell in enumerate(cells):
            out[r, c] = _number(cell, where)
    return out


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def _expect(text: str, pos: int, char: str, path: Path) -> int:
    if not text.startswith(char, pos):
        raise SchemaDrift(f"{path}: expected {char!r} at offset {pos}")
    return pos + 1


def _require_end(text: str, pos: int, path: Path) -> None:
    if _skip_ws(text, pos) != len(text):
        raise SchemaDrift(f"{path}: trailing content after the top-level object")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
