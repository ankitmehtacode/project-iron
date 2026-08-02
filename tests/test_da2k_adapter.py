"""The pairwise-depth adapter, ready before the data arrives.

DA-2K is not fetched — its licence has not been read. The adapter is written
and tested now against a synthesized fixture so that clearing the licence is
the only remaining step, and so the protocol is pinned before there is any
result to be tempted by.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.da2k_adapter import DepthPair, score_pairs


def _disparity() -> np.ndarray:
    """A ramp where larger values are nearer, per the disparity_rel contract."""
    field = np.zeros((20, 20), dtype=np.float64)
    for row in range(20):
        # Larger disparity is nearer, so the value must RISE towards the
        # bottom of the frame for the floor to read as the near surface.
        field[row, :] = row + 1
    return field


def _five_pairs() -> list[DepthPair]:
    """Five annotated comparisons a correct model gets right."""
    return [
        DepthPair((0, 5), (19, 5), "b"),
        DepthPair((18, 2), (1, 2), "a"),
        DepthPair((3, 10), (15, 10), "b"),
        DepthPair((17, 7), (4, 7), "a"),
        DepthPair((10, 1), (2, 1), "a"),
    ]


def test_a_correct_prediction_scores_one() -> None:
    result = score_pairs(_disparity(), _five_pairs())
    assert result["scored"] == 5
    assert result["accuracy"] == 1.0
    assert result["chance"] == 0.5


def test_an_inverted_prediction_scores_zero() -> None:
    """The day-9 failure mode, which no alignment can hide here.

    A model that orders depth backwards scored a respectable AbsRel once a
    scale and shift were fitted to it. Pairwise comparison fits nothing, so an
    inverted prediction scores zero and says so.
    """
    result = score_pairs(-_disparity(), _five_pairs())
    assert result["accuracy"] == 0.0


def test_a_constant_prediction_cannot_beat_chance() -> None:
    """The other day-9 failure mode: near-constant output.

    Under metric alignment a constant fits the dominant depth and scores well.
    Here it has no ordering information at all and lands at chance.
    """
    flat = np.full((20, 20), 3.0)
    result = score_pairs(flat, _five_pairs())
    assert result["accuracy"] <= 0.5


def test_no_alignment_is_performed() -> None:
    """Scale and shift must not change the answer.

    That invariance is the whole reason this benchmark suits ``disparity_rel``:
    there is no fit, so there is nothing for a fit to launder.
    """
    base = score_pairs(_disparity(), _five_pairs())["accuracy"]
    scaled = score_pairs(_disparity() * 37.0 + 11.0, _five_pairs())["accuracy"]
    assert base == scaled == 1.0


def test_out_of_bounds_annotations_are_skipped_not_counted() -> None:
    """A broken annotation is not a wrong answer.

    Counting it either way would move the accuracy for a reason that has
    nothing to do with the model.
    """
    pairs = _five_pairs() + [DepthPair((999, 999), (0, 0), "a")]
    result = score_pairs(_disparity(), pairs)
    assert result["pairs"] == 6
    assert result["scored"] == 5
    assert result["skipped"] == 1
    assert result["accuracy"] == 1.0


def test_empty_pair_set_reports_nothing_measured() -> None:
    result = score_pairs(_disparity(), [])
    assert result["pairs"] == 0
    assert np.isnan(result["accuracy"]), "no pairs means no accuracy, never 0.0"


def test_an_invalid_annotation_is_refused() -> None:
    with pytest.raises(ValueError, match="must be 'a' or 'b'"):
        DepthPair((0, 0), (1, 1), "equal")
