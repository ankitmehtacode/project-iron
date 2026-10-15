"""pycocotools agrees with a hand-computed AP before it scores anything
(Day 42, hypothesis H3).

Fixture: one image, three GT person boxes (A, B, C), four detections in
score order: d1 exact on A, d2 far from everything, d3 exact on B, d4 far
from everything. C is never detected. Matched boxes are exact, so every
IoU threshold from 0.50 to 0.95 sees the same matches and AP == AP50.

By hand: precision/recall after each detection is (1, 1/3), (1/2, 1/3),
(2/3, 2/3), (1/2, 2/3). COCO interpolates precision as the max over recall
>= r and samples it at r = 0.00, 0.01, ..., 1.00 (101 points), taking the
first detection whose recall reaches r. r <= 0.33: 34 points at 1.0;
0.34 <= r <= 0.66: 33 points at 2/3; r >= 0.67: 34 points at 0.
AP = (34 + 33 * 2/3) / 101 = 56 / 101.
"""

from __future__ import annotations

import contextlib
import io

import pytest

pytest.importorskip(
    "pycocotools", reason="pycocotools is a Part-B runtime pin (lockfile only)"
)

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

HAND_AP = 56 / 101

GT_BOXES = [[100, 100, 50, 100], [300, 100, 50, 100], [500, 100, 50, 100]]
DETECTIONS = [
    ([100, 100, 50, 100], 0.9),  # exact on A
    ([800, 500, 40, 40], 0.8),  # nowhere near any GT
    ([300, 100, 50, 100], 0.7),  # exact on B
    ([900, 600, 40, 40], 0.6),  # nowhere near any GT
]


def _evaluate() -> COCOeval:
    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1, "width": 1920, "height": 1080}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            for i, box in enumerate(GT_BOXES)
        ],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(
            [
                {"image_id": 1, "category_id": 1, "bbox": box, "score": score}
                for box, score in DETECTIONS
            ]
        )
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def test_pycocotools_ap_matches_the_hand_computed_value() -> None:
    ev = _evaluate()
    assert ev.stats[0] == pytest.approx(HAND_AP, abs=1e-9)  # AP@[.5:.95]
    assert ev.stats[1] == pytest.approx(HAND_AP, abs=1e-9)  # AP50


def test_pycocotools_recall_matches_the_hand_computed_value() -> None:
    ev = _evaluate()
    assert ev.stats[8] == pytest.approx(2 / 3, abs=1e-9)  # AR@100
