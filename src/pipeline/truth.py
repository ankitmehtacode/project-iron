"""SmartSpaces ground truth, cut the ways the slice needs it.

Used by the oracle detector and by evaluation only. Nothing on the product
path (gate, pretrained detector, projector, tracker, estimator, compiler)
imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.data.smartspaces import (
    MEASURED_GEOMETRY,
    SceneGroundTruth,
    load_scene_ground_truth,
)
from src.detect.backends import GroundTruthBoxes
from src.detect.contract import ClassVocabulary

FloatArray = npt.NDArray[np.float64]

PERSON = "Person"


def load_truth(gt_path: Path) -> SceneGroundTruth:
    """Parsed columns, cached under the ignored ``data/derived`` tree (the
    loader rebuilds when the source bytes or parser version change)."""
    from src.config import IronConfig

    cache = (
        IronConfig.load().paths.resolved_data_dir
        / "derived"
        / "smartspaces_gt"
        / f"{gt_path.parent.name}.npz"
    )
    return load_scene_ground_truth(gt_path, cache)


def vocabulary(gt: SceneGroundTruth) -> ClassVocabulary:
    return ClassVocabulary(
        name="smartspaces", version=gt.source_sha256[:12], labels=gt.class_names
    )


def _camera_index(gt: SceneGroundTruth, camera_id: str) -> int:
    if camera_id not in gt.camera_names:
        raise KeyError(f"{camera_id} not in ground truth cameras {gt.camera_names}")
    return gt.camera_names.index(camera_id)


def oracle_boxes(gt: SceneGroundTruth, camera_id: str) -> GroundTruthBoxes:
    """Every visible GT box on one camera, all classes, by frame."""
    sel = np.flatnonzero(gt.box_camera == _camera_index(gt, camera_id))
    frames = gt.frame[gt.box_row[sel]]
    boxes: dict[int, list[FloatArray]] = {}
    labels: dict[int, list[str]] = {}
    order = np.argsort(frames, kind="stable")
    for i in sel[order]:
        f = int(gt.frame[gt.box_row[i]])
        boxes.setdefault(f, []).append(gt.box_xyxy[i].astype(np.float64))
        labels.setdefault(f, []).append(gt.class_names[gt.class_index[gt.box_row[i]]])
    return GroundTruthBoxes(
        boxes={
            f: np.array(b, dtype=np.float64).reshape(-1, 4) for f, b in boxes.items()
        },
        labels=labels,
        source_sha=gt.source_sha256,
    )


@dataclass(frozen=True)
class CameraTruth:
    """Persons visible on one camera: one row per (frame, person) box."""

    frame: npt.NDArray[np.int64]
    object_id: npt.NDArray[np.int64]
    box_xyxy: FloatArray
    floor_xy: FloatArray


def person_truth(gt: SceneGroundTruth, camera_id: str, max_frame: int) -> CameraTruth:
    """Visible person boxes on ``camera_id`` with frame < ``max_frame``, and
    each person's floor point (centre of the 3D box's bottom face)."""
    person = gt.class_names.index(PERSON)
    sel = np.flatnonzero(gt.box_camera == _camera_index(gt, camera_id))
    rows = gt.box_row[sel]
    keep = (gt.class_index[rows] == person) & (gt.frame[rows] < max_frame)
    sel, rows = sel[keep], rows[keep]
    floor = MEASURED_GEOMETRY.floor_point(gt.location[rows], gt.scale[rows])
    order = np.lexsort((gt.object_id[rows], gt.frame[rows]))
    return CameraTruth(
        frame=gt.frame[rows][order].astype(np.int64),
        object_id=gt.object_id[rows][order].astype(np.int64),
        box_xyxy=gt.box_xyxy[sel][order].astype(np.float64),
        floor_xy=floor[order, :2],
    )
