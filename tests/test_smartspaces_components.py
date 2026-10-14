"""The vectorised component kernel used on SmartSpaces matches Day 26's
own coupling definition exactly (Day 41, Objective 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure_component_sparsity import (  # noqa: E402
    PROXIMITY_THRESHOLD_SWEEP_M,
    _connected_components,
    _pairwise_adjacency,
)
from measure_smartspaces_components import (  # noqa: E402
    proximity_mst,
    sweep_components,
)


def _as_partition(labels: np.ndarray) -> set[frozenset[int]]:
    return {frozenset(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)}


@pytest.mark.parametrize("seed", range(12))
def test_mst_components_equal_day26_components_at_every_threshold(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 40))
    # A cramped floor so every threshold in the sweep changes something.
    positions = rng.uniform([0, 0, 0.3], [12, 12, 1.5], size=(n, 3))
    if n > 3:
        positions[1] = positions[0]  # coincident entities: distance 0
        positions[3] = positions[2] + [1.1, 0, 0]  # exactly at the knee
    maxima, labels = sweep_components(
        n,
        proximity_mst(positions),
        PROXIMITY_THRESHOLD_SWEEP_M,
        PROXIMITY_THRESHOLD_SWEEP_M,
    )
    for threshold, largest in zip(PROXIMITY_THRESHOLD_SWEEP_M, maxima):
        expected = {
            frozenset(c)
            for c in _connected_components(n, _pairwise_adjacency(positions, threshold))
        }
        assert _as_partition(labels[threshold]) == expected, threshold
        assert largest == max((len(c) for c in expected), default=0), threshold


def test_a_distance_exactly_at_the_threshold_does_not_couple() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    _, labels = sweep_components(
        2, proximity_mst(positions), (1.5, 1.5000001), (1.5, 1.5000001)
    )
    assert len(_as_partition(labels[1.5])) == 2
    assert len(_as_partition(labels[1.5000001])) == 1
