"""Cross-camera association by geometry (Day 46, Objective 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.crosscam import (
    NO_PARTNER,
    REGION_K,
    Tracklet,
    associate,
    ls_intersection,
    pair_evidence,
)


def _track(
    key: str, cam: str, cam_xy: tuple[float, float], xy: np.ndarray, sigma: float
) -> Tracklet:
    n = len(xy)
    return Tracklet(
        key=key,
        camera_id=cam,
        camera_xy=np.array(cam_xy, float),
        frames=np.arange(n, dtype=np.int64),
        foot_xy=np.asarray(xy, float),
        sigma_m=np.full(n, sigma),
    )


def _walk(start: tuple[float, float], n: int = 40) -> np.ndarray:
    return np.array([[start[0] + 0.05 * i, start[1]] for i in range(n)])


def test_region_constant_is_the_stated_95_percent_disk() -> None:
    # chi-square(2) at 0.95: exp(-r^2/2) = 0.05
    assert np.exp(-(REGION_K**2) / 2) == pytest.approx(0.05)


def test_the_same_person_on_two_cameras_is_fused() -> None:
    rng = np.random.default_rng(0)
    path = _walk((5.0, 5.0))
    a = _track("c0/t1", "c0", (0, 0), path + rng.normal(0, 0.1, path.shape), 0.2)
    b = _track("c1/t1", "c1", (10, 0), path + rng.normal(0, 0.1, path.shape), 0.2)
    far = _track("c1/t2", "c1", (10, 0), _walk((5.0, 15.0)), 0.2)
    result = associate([a, b, far])
    assert result.accepted == [("c0/t1", "c1/t1")]
    assert result.components["c0/t1"] == result.components["c1/t1"]
    assert result.components["c1/t2"] != result.components["c0/t1"]
    assert pair_evidence(a, far).compatible == 0


def test_two_people_inside_the_gate_stay_ambiguous_and_unfused() -> None:
    path = _walk((5.0, 5.0))
    a = _track("c0/t1", "c0", (0, 0), path, 0.3)
    b1 = _track("c1/t1", "c1", (10, 0), path + [0.0, 0.1], 0.3)
    b2 = _track("c1/t2", "c1", (10, 0), path - [0.0, 0.1], 0.3)
    result = associate([a, b1, b2])
    verdict = next(v for v in result.verdicts if v["tracklet"] == "c0/t1")
    assert verdict["kind"] == "ambiguous" and verdict["winner"] is None
    assert result.accepted == []
    assert len(set(result.components.values())) == 3


def test_a_lone_tracklet_prefers_no_partner_over_a_poor_match() -> None:
    path = _walk((5.0, 5.0), n=5)
    a = _track("c0/t1", "c0", (0, 0), path, 0.2)
    # inside the gate on one step only, far outside on the rest
    off = path + [0.0, 3.0]
    off[0] = path[0] + [0.0, 0.5]
    b = _track("c1/t1", "c1", (10, 0), off, 0.2)
    result = associate([a, b])
    verdict = next(v for v in result.verdicts if v["tracklet"] == "c0/t1")
    assert verdict["kind"] == "decisive" and verdict["winner"] == NO_PARTNER
    assert result.accepted == []


def test_ls_intersection_of_two_rays_and_of_parallel_ones() -> None:
    x = ls_intersection(
        np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[1.0, 1.0], [9.0, 1.0]])
    )
    assert x is not None and np.allclose(x, [5.0, 5.0])
    assert (
        ls_intersection(
            np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 1.0]])
        )
        is None
    )


def test_tracklet_rejects_inconsistent_arrays() -> None:
    with pytest.raises(ValueError):
        Tracklet("k", "c", np.zeros(2), np.array([1, 0]), np.zeros((2, 2)), np.ones(2))
    with pytest.raises(ValueError):
        Tracklet("k", "c", np.zeros(2), np.array([0, 1]), np.zeros((2, 2)), np.zeros(2))
