"""Day 20, Objective 4 — scripts/eval_estimator.py's own logic.

Not a re-test of the filter (that's tests/test_estimator_filter.py); this
covers the evaluation script's own machinery: camera-position inversion,
RMSE, and the three-method comparison on a small, fully-controlled
synthetic track where the right answer is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import eval_estimator as ee  # noqa: E402

from src.estimator.measurement_model import measurement_model_for  # noqa: E402
from src.estimator.motion_model import motion_model_for  # noqa: E402


def _identity_world_to_camera_at(position: tuple[float, float, float]) -> np.ndarray:
    """A camera at ``position`` with identity rotation: world-to-camera
    translation is -position."""
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = -np.array(position)
    return extrinsics


def test_camera_position_world_inverts_translation_only_extrinsics() -> None:
    extrinsics = _identity_world_to_camera_at((5.0, -2.0, 1.5))
    recovered = ee._camera_position_world(extrinsics)
    np.testing.assert_allclose(recovered, [5.0, -2.0, 1.5])


def test_camera_position_world_inverts_rotated_extrinsics() -> None:
    # 90-degree rotation about Z: x' = -y, y' = x, z' = z
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera_pos = np.array([3.0, 4.0, 0.0])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = -rotation @ camera_pos
    recovered = ee._camera_position_world(extrinsics)
    np.testing.assert_allclose(recovered, camera_pos, atol=1e-10)


def test_camera_depth_m_is_positive_distance_along_optical_axis() -> None:
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, 0.0))
    depth = ee._camera_depth_m(np.array([0.0, 0.0, 5.0]), extrinsics)
    assert depth == pytest.approx(5.0)


def test_rmse_of_empty_list_is_nan() -> None:
    assert np.isnan(ee._rmse([]))


def test_config_descriptions_covers_every_config_spec() -> None:
    """Day 24, Objective 3 audit: CONFIG_SPECS and CONFIG_DESCRIPTIONS are
    two separately hand-maintained dicts keyed by the same config labels
    (A/B/C/D) -- the same shape as the CAPABILITIES/GATES defect (Day 16,
    Day 23), just not yet caught drifting. A label present in one and
    missing from the other currently fails with a bare KeyError deep in
    main() the first time that config is scored, not at the boundary.
    Cheap structural guard against a fifth config being added to one dict
    and not the other."""
    assert set(ee.CONFIG_SPECS) == set(ee.CONFIG_DESCRIPTIONS)


def test_rmse_known_values() -> None:
    # errors of 3 and 4 -> RMSE = sqrt((9+16)/2) = sqrt(12.5)
    assert ee._rmse([9.0, 16.0]) == pytest.approx(np.sqrt(12.5))


# ---------------------------------------------------------------------------
# _evaluate_track: a small, fully-controlled synthetic track
# ---------------------------------------------------------------------------


def test_evaluate_track_on_a_straight_line_walker() -> None:
    """A noiseless straight-line walker: the filter should track it near-
    exactly and beat both trivial baselines by a wide margin."""
    frames = 10
    step_m = 0.5
    fps = 12.0
    track = np.zeros((frames, 3))
    track[:, 0] = np.arange(frames) * step_m
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, -10.0))  # camera 10m back
    rng = np.random.default_rng(1)

    result = ee._evaluate_track(
        track,
        extrinsics,
        "test-sensor",
        fps,
        motion_model_for("person"),
        measurement_model_for("test-sensor"),
        rng,
    )

    assert result is not None
    assert len(result.frames) == frames - ee.FIRST_COMPARABLE_INDEX
    filter_rmse = ee._rmse([r.position_sq_error for r in result.frames])
    copy_previous_rmse = ee._rmse([r.copy_previous_sq_error for r in result.frames])
    cv_rmse = ee._rmse([r.cv_no_update_sq_error for r in result.frames])

    # On a genuinely constant-velocity track with small measurement noise,
    # the filter should be competitive with (not necessarily always beat --
    # noise is real) copy-previous, since copy-previous's zero-velocity
    # assumption is a poor fit to a walker.
    assert filter_rmse < copy_previous_rmse
    # constant_velocity_no_update, uncorrected, should not be dramatically
    # better than the filter on a track with no dynamics it needs correcting
    # for -- this just confirms the baseline computed something sane, not a
    # NaN or a wildly divergent number.
    assert np.isfinite(cv_rmse)
    assert all(0.0 <= r.distance_m for r in result.frames)


def test_evaluate_track_records_posterior_velocity_variance() -> None:
    """Day 25, Objective 1: FrameRecord.velocity_variance_diag_mps2 must be
    the REAL posterior covariance the filter produced (post-floor, if a
    floor is enabled) -- not re-derived separately, so it can never drift
    from what the coverage numbers are already computed against."""
    frames = 10
    fps = 12.0
    track = np.zeros((frames, 3))
    track[:, 0] = np.arange(frames) * 0.5
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, -10.0))

    result = ee._evaluate_track(
        track,
        extrinsics,
        "test-sensor",
        fps,
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model_for("test-sensor"),
        np.random.default_rng(1),
    )
    assert result is not None

    from src.estimator.motion_model import pedestrian_velocity_covariance_floor_mps2

    floor = pedestrian_velocity_covariance_floor_mps2(1.0 / fps)
    for record in result.frames:
        assert len(record.velocity_variance_diag_mps2) == 3
        # A floor-enabled config's posterior can never read below the
        # floor on any axis -- exactly the check Day 25 Objective 1 ran
        # against a real golden set; here it runs against a synthetic
        # track so it stays in the fast suite.
        for variance in record.velocity_variance_diag_mps2:
            assert variance >= floor - 1e-9


def test_sigma_v_distribution_reports_min_p50_max() -> None:
    records = [
        ee.FrameRecord(
            regime="static",
            distance_m=1.0,
            position_sq_error=0.0,
            axis_sq_error=np.zeros(3),
            velocity_sq_error=0.0,
            nees=ee.compute_nees(np.zeros(6), np.eye(6)),
            standardized_innovation_x=None,
            copy_previous_sq_error=0.0,
            cv_no_update_sq_error=0.0,
            cv_no_update_velocity_sq_error=0.0,
            velocity_variance_diag_mps2=variance_diag,
        )
        for variance_diag in [(1.0, 1.0, 1.0), (4.0, 4.0, 4.0), (9.0, 9.0, 9.0)]
    ]
    # sigma_v per frame = sqrt(mean of the 3 diagonal entries): 1.0, 2.0, 3.0
    dist = ee._sigma_v_distribution(records)
    assert dist["n"] == 3
    assert dist["min_mps"] == pytest.approx(1.0)
    assert dist["p50_mps"] == pytest.approx(2.0)
    assert dist["max_mps"] == pytest.approx(3.0)


def test_sigma_v_distribution_empty_is_nan_not_a_crash() -> None:
    dist = ee._sigma_v_distribution([])
    assert dist["n"] == 0
    assert np.isnan(dist["min_mps"])
    assert np.isnan(dist["p50_mps"])
    assert np.isnan(dist["max_mps"])
    assert dist["frames_at_floor"] == 0


def _record_with_sigma_v(sigma_v_mps: float) -> "ee.FrameRecord":
    variance = sigma_v_mps**2
    return ee.FrameRecord(
        regime="static",
        distance_m=1.0,
        position_sq_error=0.0,
        axis_sq_error=np.zeros(3),
        velocity_sq_error=0.0,
        nees=ee.compute_nees(np.zeros(6), np.eye(6)),
        standardized_innovation_x=None,
        copy_previous_sq_error=0.0,
        cv_no_update_sq_error=0.0,
        cv_no_update_velocity_sq_error=0.0,
        velocity_variance_diag_mps2=(variance, variance, variance),
    )


def test_sigma_v_distribution_counts_frames_pinned_at_the_floor() -> None:
    """Day 30, Objective 3. Day 25 read `min == p50 == max == 1.5000` as
    confirmation that the clamp fires — which it is, and which is also the
    signature of a filter whose velocity uncertainty has stopped varying
    at all. The fraction pinned is what distinguishes the two, and it was
    one line away the whole time."""
    floor = ee._floor_sigma_v_mps()
    records = [
        _record_with_sigma_v(floor),
        _record_with_sigma_v(floor),
        _record_with_sigma_v(floor * 1.2),
    ]
    dist = ee._sigma_v_distribution(records)
    assert dist["floor_mps"] == pytest.approx(floor)
    assert dist["frames_at_floor"] == 2
    assert dist["fraction_at_floor"] == pytest.approx(2 / 3)


def test_sigma_v_distribution_reports_nothing_pinned_when_nothing_is() -> None:
    """The falsifiability half: a filter converging well below the floor
    (config A's actual behaviour, p50 0.25-0.60 m/s) must report ZERO
    frames at the floor, or the statistic would be counting something
    other than what it claims."""
    floor = ee._floor_sigma_v_mps()
    dist = ee._sigma_v_distribution(
        [_record_with_sigma_v(floor * 0.2), _record_with_sigma_v(floor * 0.4)]
    )
    assert dist["frames_at_floor"] == 0
    assert dist["fraction_at_floor"] == pytest.approx(0.0)


def test_velocity_floor_is_absolute_so_the_probe_timestep_cannot_matter() -> None:
    """`_floor_sigma_v_mps` calls the floor function with a fixed probe
    dt. Since Day 24's correction the floor ignores dt entirely; this
    pins that, so a future re-introduction of dt-scaling breaks here
    rather than silently changing every pinned-fraction number."""
    from src.estimator.motion_model import pedestrian_velocity_covariance_floor_mps2

    values = {
        pedestrian_velocity_covariance_floor_mps2(dt) for dt in (1 / 120, 1 / 12, 1.0)
    }
    assert len(values) == 1


def test_evaluate_track_too_short_returns_none() -> None:
    track = np.zeros((2, 3))  # fewer frames than FIRST_COMPARABLE_INDEX + 1
    extrinsics = _identity_world_to_camera_at((0.0, 0.0, -10.0))
    result = ee._evaluate_track(
        track,
        extrinsics,
        "test-sensor",
        12.0,
        motion_model_for("person"),
        measurement_model_for("test-sensor"),
        np.random.default_rng(1),
    )
    assert result is None


def test_score_golden_set_handles_missing_version_gracefully(tmp_path: Path) -> None:
    from src.config import IronConfig

    config = IronConfig.load()
    report = ee._score_golden_set("does-not-exist-version", tmp_path, config)
    assert report is None


# ---------------------------------------------------------------------------
# Day 22, Objective 3 -- the generalized no-trade verdict
# Day 25, Objective 3 -- made DIRECTIONAL: overconfidence fails outright,
# underconfidence is a weighed, numeric cost. See ADR 0010's Day 25
# revision for the rationale.
# ---------------------------------------------------------------------------


def _regime_block(
    n: int,
    coverage_95: float,
    mixture_coverage: float | None = None,
    nees_pass: float | None = None,
) -> dict:
    block = {
        "n": n,
        "filter_rmse_m": 0.1,
        "margin_vs_copy_previous_m": 0.05,
        "margin_vs_constant_velocity_m": 1.0,
        "thin_evidence": n < ee.MIN_REGIME_FRAMES_FOR_A_CONCLUSION,
        "consistency": {
            "empirical_coverage_95": coverage_95,
            "nees_pass_rate_within_95": (
                nees_pass if nees_pass is not None else coverage_95
            ),
        },
    }
    if mixture_coverage is not None:
        block["consistency_mixture"] = {
            "empirical_coverage_by_sampling_95": mixture_coverage,
            "mixture_nees_coverage_95": mixture_coverage,
        }
    return block


def _fake_report(by_regime: dict) -> dict:
    return {"version": "v-test", "refused": False, "by_regime": by_regime}


def test_regime_coverage_prefers_mixture_when_present() -> None:
    block = _regime_block(20, coverage_95=0.5, mixture_coverage=0.9)
    assert ee._regime_coverage(block) == pytest.approx(0.9)


def test_regime_coverage_falls_back_to_nees_when_no_mixture() -> None:
    block = _regime_block(20, coverage_95=0.85)
    assert ee._regime_coverage(block) == pytest.approx(0.85)


def test_no_trade_unscoreable_when_cessation_too_thin() -> None:
    baseline = _fake_report(
        {
            "cessation": _regime_block(3, 0.0),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(3, 0.9),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradeUnscoreable)
    assert verdict.n_cessation == 3


def test_no_trade_pass_when_cessation_improves_and_steady_holds() -> None:
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.0, mixture_coverage=None),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.5, mixture_coverage=0.8),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "D", candidate)
    assert isinstance(verdict, ee.NoTradePass)
    assert verdict.cessation_delta_toward_nominal > 0


def test_no_trade_fails_overconfident_when_steady_regime_drops_below_nominal() -> None:
    """The dangerous direction: a steady regime's coverage moves BELOW
    nominal -- the filter's stated uncertainty understates its true error.
    Non-negotiable, regardless of cessation's own improvement."""
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.0),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.9, mixture_coverage=0.9),
            "static": _regime_block(50, 0.5, mixture_coverage=0.5),  # overconfident
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradeFailOverconfident)
    assert verdict.regime == "static"
    assert verdict.coverage_after == pytest.approx(0.5)


def test_no_trade_pass_with_cost_when_steady_regime_moves_above_nominal() -> None:
    """The safe direction: a steady regime's coverage moves further ABOVE
    nominal (more conservative, not misleading). This must be a weighed
    cost, not the same outright failure an overconfident move gets."""
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.0),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.9, mixture_coverage=0.9),
            "static": _regime_block(50, 0.99, mixture_coverage=0.99),  # underconfident
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradePassWithCost)
    assert verdict.regime == "static"
    assert verdict.magnitude == pytest.approx(0.04)
    assert verdict.costs == (("static", pytest.approx(0.04)),)


def test_no_trade_no_improvement_when_cessation_does_not_improve_materially() -> None:
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.90),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.91, mixture_coverage=0.91),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradeNoImprovement)


def test_no_trade_degradation_tolerance_absorbs_small_wiggle() -> None:
    """A steady regime moving slightly further from nominal, within
    NO_TRADE_DEGRADATION_TOLERANCE, must not count as a degradation in
    EITHER direction."""
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.0),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.9, mixture_coverage=0.9),
            "static": _regime_block(50, 0.94, mixture_coverage=0.94),  # tiny wiggle
            "sustained": _regime_block(50, 0.95),
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradePass)


def test_no_trade_overconfident_fail_wins_over_a_simultaneous_cost() -> None:
    """A candidate degrading one steady regime toward overconfidence and
    another toward underconfidence must fail outright -- the cost side
    cannot offset the non-negotiable side."""
    baseline = _fake_report(
        {
            "cessation": _regime_block(20, 0.0),
            "static": _regime_block(50, 0.95),
            "sustained": _regime_block(50, 0.95),
        }
    )
    candidate = _fake_report(
        {
            "cessation": _regime_block(20, 0.9, mixture_coverage=0.9),
            "static": _regime_block(50, 0.5, mixture_coverage=0.5),  # overconfident
            "sustained": _regime_block(50, 0.99, mixture_coverage=0.99),  # underconf.
        }
    )
    verdict = ee._no_trade_verdict("A", baseline, "C", candidate)
    assert isinstance(verdict, ee.NoTradeFailOverconfident)
    assert verdict.regime == "static"


# ---------------------------------------------------------------------------
# Day 22, Objective 3 -- CONFIG_SPECS
# ---------------------------------------------------------------------------


def test_config_specs_cover_all_four_configs() -> None:
    assert set(ee.CONFIG_SPECS) == {"A", "B", "C", "D"}
    assert ee.CONFIG_SPECS["A"] == ("single_model", False)
    assert ee.CONFIG_SPECS["B"] == ("single_model", True)
    assert ee.CONFIG_SPECS["C"] == ("imm", False)
    assert ee.CONFIG_SPECS["D"] == ("imm", True)
