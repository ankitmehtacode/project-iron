"""Score relative depth on annotated pixel pairs (DA-2K protocol).

Why this benchmark specifically. Day 9 showed that a scale-and-shift alignment
can take a near-constant prediction, fit it to whatever depth dominates the
frame, and report AbsRel 0.1538 while the model ordered pixels backwards. Every
metric-depth benchmark has that hole, because ``disparity_rel`` has no scale and
one must be fitted before the metric can be computed.

Pairwise relative depth has no such hole. The question is only "is point A
nearer than point B", the prediction is compared with itself, and **no alignment
is performed at all** — so there is nothing for a fit to launder. It is the
honest first real-data depth measurement for a model whose output carries
neither scale nor shift.

The data is not present. This adapter is written and tested against a
synthesized fixture so that it runs the moment a human clears the licence; see
``configs/datasets.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DepthPair:
    """One annotated comparison: which of two pixels is nearer.

    Attributes:
        point_a: ``(row, col)`` in the image.
        point_b: ``(row, col)``.
        nearer: ``"a"`` or ``"b"`` — the annotated answer.
    """

    point_a: tuple[int, int]
    point_b: tuple[int, int]
    nearer: str

    def __post_init__(self) -> None:
        if self.nearer not in ("a", "b"):
            raise ValueError(
                f"nearer must be 'a' or 'b', got {self.nearer!r}. DA-2K pairs "
                "are annotated with a strict answer; there is no 'equal' class "
                "and inventing one would change the protocol."
            )


def score_pairs(disparity: np.ndarray, pairs: Iterable[DepthPair]) -> dict[str, Any]:
    """Fraction of annotated pairs the prediction orders correctly.

    Args:
        disparity: ``[H, W]`` relative inverse depth — larger is nearer, which
            is the ``disparity_rel`` convention and is confirmed rather than
            assumed (day 9: on real footage the floor reads nearer than the
            ceiling).
        pairs: annotated comparisons.

    Returns:
        Accuracy plus the counts behind it. Chance is 0.5, and it is reported
        alongside so a result near chance cannot read as a result.
    """
    pairs = list(pairs)
    if not pairs:
        return {
            "pairs": 0,
            "accuracy": float("nan"),
            "correct": 0,
            "chance": 0.5,
            "note": "no annotated pairs; nothing was measured",
        }

    height, width = disparity.shape
    correct = 0
    skipped = 0
    for pair in pairs:
        (row_a, col_a), (row_b, col_b) = pair.point_a, pair.point_b
        inside = (
            0 <= row_a < height
            and 0 <= col_a < width
            and 0 <= row_b < height
            and 0 <= col_b < width
        )
        if not inside:
            # Out of bounds is a broken annotation, not a wrong answer. Counting
            # it as either would move the accuracy for a reason unrelated to the
            # model.
            skipped += 1
            continue
        value_a = float(disparity[row_a, col_a])
        value_b = float(disparity[row_b, col_b])
        if not (np.isfinite(value_a) and np.isfinite(value_b)):
            skipped += 1
            continue
        predicted = "a" if value_a > value_b else "b"
        correct += int(predicted == pair.nearer)

    scored = len(pairs) - skipped
    return {
        "pairs": len(pairs),
        "scored": scored,
        "skipped": skipped,
        "correct": correct,
        "accuracy": (correct / scored) if scored else float("nan"),
        "chance": 0.5,
        "alignment": "none — pairwise comparison needs no scale or shift",
    }
