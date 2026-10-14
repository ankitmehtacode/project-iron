"""SmartSpaces ground truth agrees with its calibration under the declared
convention, per scene, and a deliberately wrong convention fails (Day 41,
Objective 2).

Reads the fetched dataset; skips with a reason when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from smartspaces_integrity import DATA_ROOT, load_scene, scene_dirs  # noqa: E402

from src.data.smartspaces import (  # noqa: E402
    MEASURED_GEOMETRY,
    GroundTruthAxes,
    LocationReference,
    ProjectionSource,
    SmartSpacesGeometry,
    box_containment,
    camera_intrinsics,
    load_calibration,
    world_positions,
)
from src.contracts.errors import AxisConventionMismatch  # noqa: E402

pytestmark = pytest.mark.requires_dataset

SCENES = [p.name for p in scene_dirs()] or ["(no scenes fetched)"]

CONTAINED_AT = 0.9
"""A visible box agrees with its object when >= 90% of its area lies inside
the projected 3D box."""
AGREEMENT_TOLERANCE = 0.99
"""Declared before the full run: at least 99% of sampled visible boxes must
agree, per (scene, class)."""
SAMPLE_PER_CLASS = 3_000
MIN_BOX_SIDE_PX = 4

KNOWN_PAYLOAD_DIVERGENCE = {
    ("Warehouse_010", "Transporter"),
    ("Warehouse_011", "Transporter"),
    ("Warehouse_012", "Transporter"),
    ("Warehouse_013", "Transporter"),
    ("Warehouse_014", "Transporter"),
    ("Warehouse_015", "Transporter"),
    ("Warehouse_016", "Transporter"),
    ("Warehouse_019", "Forklift"),
}
"""Where the 2D box and the 3D box label different extents of the same
object, found Day 41 and checked by eye on video crops: a Transporter
carrying a loaded pallet has a 2D box around robot AND cargo, but a 3D box
around the 0.23 m robot base only (the failing boxes extend 3.6 projected
box-heights above it; their bottoms align like everyone else's); per object
id the split is all-or-nothing (Warehouse_010: 6 of 12 ids agree on 100%,
6 on 0-8%). Warehouse_019's one "Forklift" is a stacker whose 2D box takes
in mast, forks and load around a 0.72 x 0.78 m body box. Every other
(scene, class) pair agrees on >= 99%. These are asserted to STILL diverge,
so a data revision that changes them is noticed, not absorbed."""


def _agreement(scene_name: str, swap_xy: bool) -> dict[str, float]:
    scene = next(p for p in scene_dirs() if p.name == scene_name)
    gt = load_scene(scene)
    calibration = load_calibration(scene / "calibration.json")
    box = gt.box_xyxy
    usable = ((box[:, 2] - box[:, 0]) >= MIN_BOX_SIDE_PX) & (
        (box[:, 3] - box[:, 1]) >= MIN_BOX_SIDE_PX
    )
    row_class = gt.class_index[gt.box_row]
    rng = np.random.default_rng(41)
    out = {}
    for k, name in enumerate(gt.class_names):
        candidates = np.flatnonzero(usable & (row_class == k))
        if candidates.size == 0:
            continue
        index = np.sort(
            rng.choice(
                candidates, size=min(SAMPLE_PER_CLASS, candidates.size), replace=False
            )
        )
        location = gt.location[gt.box_row[index]].copy()
        if swap_xy:
            location[:, [0, 1]] = location[:, [1, 0]]
        containment, _ = box_containment(
            gt, calibration, MEASURED_GEOMETRY.projection, index, location
        )
        scored = np.isfinite(containment)
        if swap_xy:
            # A wrong convention that throws objects behind the camera has
            # failed for those boxes; they count against it.
            out[name] = float(np.mean(np.nan_to_num(containment) >= CONTAINED_AT))
            continue
        # Under the declared convention, a 3D box with a corner behind the
        # camera has no image rectangle and is unscored (as in
        # scripts/smartspaces_reprojection.py); a class must keep at least
        # half its sample scored.
        assert scored.mean() >= 0.5, (name, float(scored.mean()))
        out[name] = float(np.mean(containment[scored] >= CONTAINED_AT))
    return out


def _require_data() -> None:
    if not scene_dirs():
        pytest.skip(
            f"PhysicalAI-SmartSpaces is not fetched at {DATA_ROOT}; run "
            "scripts/fetch_dataset.py (see configs/datasets.yaml)"
        )


@pytest.mark.parametrize("scene_name", SCENES)
def test_ground_truth_reprojects_inside_its_visible_boxes(scene_name: str) -> None:
    _require_data()
    agreement = _agreement(scene_name, swap_xy=False)
    for name, value in agreement.items():
        if (scene_name, name) in KNOWN_PAYLOAD_DIVERGENCE:
            assert value < AGREEMENT_TOLERANCE, (name, value)
        else:
            assert value >= AGREEMENT_TOLERANCE, (name, value)


@pytest.mark.parametrize("scene_name", SCENES)
def test_a_swapped_axis_convention_fails_the_same_check(scene_name: str) -> None:
    """The check can fail: with x and y swapped, no class passes."""
    _require_data()
    agreement = _agreement(scene_name, swap_xy=True)
    assert agreement
    assert all(v < AGREEMENT_TOLERANCE for v in agreement.values()), agreement


def test_contracts_carry_the_declared_conventions() -> None:
    _require_data()
    scene = scene_dirs()[0]
    calibration = load_calibration(scene / "calibration.json")
    camera = next(iter(calibration.cameras.values()))
    intrinsics = camera_intrinsics(camera)
    assert (intrinsics.valid_for.width, intrinsics.valid_for.height) == (1920, 1080)
    assert intrinsics.calibrated and intrinsics.is_undistorted

    positions = world_positions(np.zeros((2, 3)), calibration, MEASURED_GEOMETRY)
    assert positions.twin_rev == calibration.twin_rev

    y_up = SmartSpacesGeometry(
        projection=ProjectionSource.INTRINSIC_EXTRINSIC,
        location_reference=LocationReference.BOX_CENTRE,
        axes=GroundTruthAxes.X_Y_UP_Z,
        height_scale_index=2,
    )
    with pytest.raises(AxisConventionMismatch):
        world_positions(np.zeros((2, 3)), calibration, y_up)


def test_floor_point_drops_half_the_height_from_the_centre() -> None:
    location = np.array([[1.0, 2.0, 0.9]])
    scale = np.array([[0.5, 0.4, 1.8]])
    np.testing.assert_allclose(
        MEASURED_GEOMETRY.floor_point(location, scale), [[1.0, 2.0, 0.0]]
    )
