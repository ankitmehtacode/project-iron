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


def test_the_detection_harness_agrees_with_the_hand_fixture() -> None:
    """Lead-order rule 3, harness half: scripts/eval_detection_zero_shot.py's
    height-bucket route (every area overwritten with height**2, pycocotools'
    own area ranges) must reproduce the hand value on the same fixture, in the
    'all' range, and COCO's size-range rule in the one bucket holding all
    three GT boxes (100 px tall: [64, 128)). That rule ignores an unmatched
    detection whose own size is outside the range, so the two 40 px false
    positives do not count there: TP, TP gives precision 1 up to recall 2/3,
    i.e. (34 + 33) / 101 = 67/101 by hand. Per-bucket AP never charges a
    bucket for false positives of other sizes."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import eval_detection_zero_shot as harness

    payload = {
        "images": [{"id": 1, "width": 1920, "height": 1080}],
        "categories": [{"id": 1, "name": "Person"}],
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
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": box, "score": score}
        for box, score in DETECTIONS
    ]
    ev = harness._cocoeval(payload, dets, 1.0, use_cats=True)
    rows = harness._ap_recall(ev, harness._gt_counts(payload, 1.0))
    for bucket, hand in (("all", HAND_AP), ("[64,128)", 67 / 101)):
        assert rows[bucket]["AP"] == pytest.approx(hand, abs=1e-9)
        assert rows[bucket]["AP50"] == pytest.approx(hand, abs=1e-9)
        assert rows[bucket]["recall_iou50"] == pytest.approx(2 / 3, abs=1e-9)
    assert rows["[0,32)"] == {"gt": 0}


def test_a_box_exactly_on_a_bucket_edge_is_counted_once() -> None:
    """Review finding (Day 42): pycocotools' area ranges are closed, the
    buckets half-open. A GT 64 px tall belongs to [64,128) only."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import eval_detection_zero_shot as harness

    box = [100, 100, 30, 64]
    payload = {
        "images": [{"id": 1, "width": 1920, "height": 1080}],
        "categories": [{"id": 1, "name": "Person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": box,
                "area": 1920.0,
                "iscrowd": 0,
            }
        ],
    }
    dets = [{"image_id": 1, "category_id": 1, "bbox": box, "score": 0.9}]
    ev = harness._cocoeval(payload, dets, 1.0, use_cats=True)
    rows = harness._ap_recall(ev, harness._gt_counts(payload, 1.0))
    assert rows["[64,128)"]["recall_iou50"] == pytest.approx(1.0)
    recall = ev.eval["recall"]
    lower = ev.params.areaRngLbl.index("[32,64)")
    assert (recall[:, :, lower, -1] == -1).all(), "64 px box leaked into [32,64)"
