"""Day 26, Objective 2 -- the coupling-graph machinery, isolated from any
golden-set I/O (that path is exercised manually against real data; these
tests cover the connectivity/statistics logic on small, fully-controlled
inputs where the right answer is known by construction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_component_sparsity as mcs  # noqa: E402


def test_connected_components_all_isolated() -> None:
    components = mcs._connected_components(4, set())
    assert sorted(components) == [[0], [1], [2], [3]]


def test_connected_components_one_pair() -> None:
    components = mcs._connected_components(3, {(0, 1)})
    assert sorted(components) == [[0, 1], [2]]


def test_connected_components_transitive_chain() -> None:
    """A~B and B~C but NOT A~C directly must still merge into one
    component -- this is exactly the mechanism that makes a factor graph's
    joint-solve block bigger than any single pairwise relationship: chained
    coupling, not just direct coupling."""
    components = mcs._connected_components(3, {(0, 1), (1, 2)})
    assert components == [[0, 1, 2]]


def test_connected_components_full_merge() -> None:
    edges = {(i, j) for i in range(6) for j in range(i + 1, 6)}
    components = mcs._connected_components(6, edges)
    assert components == [[0, 1, 2, 3, 4, 5]]


def test_pairwise_adjacency_respects_threshold() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    edges = mcs._pairwise_adjacency(positions, threshold_m=2.0)
    assert edges == {(0, 1)}


def test_pairwise_adjacency_strict_inequality_at_exact_threshold() -> None:
    """A pair exactly AT the threshold distance is not counted as coupled --
    matching apply_velocity_covariance_floor's own convention elsewhere in
    this codebase of stating boundary behaviour precisely rather than
    leaving it to floating-point luck."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = mcs._pairwise_adjacency(positions, threshold_m=2.0)
    assert edges == set()
    edges_inside = mcs._pairwise_adjacency(positions, threshold_m=2.0001)
    assert edges_inside == {(0, 1)}


def test_distribution_empty_is_nan_not_a_crash() -> None:
    dist = mcs._distribution([])
    assert dist["n"] == 0
    assert np.isnan(dist["p50"])
    assert np.isnan(dist["p95"])
    assert np.isnan(dist["max"])


def test_distribution_known_values() -> None:
    dist = mcs._distribution([1, 1, 1, 2, 6])
    assert dist["n"] == 5
    assert dist["p50"] == 1.0
    assert dist["max"] == 6.0


def test_default_threshold_is_declared_not_fitted_and_within_sweep() -> None:
    """The default threshold must be one of the values the sensitivity
    sweep actually tests -- otherwise the single-threshold report and the
    sweep table could silently disagree about what "default" means."""
    assert mcs.DEFAULT_PROXIMITY_THRESHOLD_M in mcs.PROXIMITY_THRESHOLD_SWEEP_M
