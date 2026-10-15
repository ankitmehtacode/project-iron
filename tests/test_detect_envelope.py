"""The detect envelope and R from it (Day 42, Objective B6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.detect.envelope import (
    SYNTHETIC_DERIVED,
    DetectEnvelope,
    DetectEnvelopeError,
    ResolutionCurves,
    position_envelope_for,
)
from src.estimator.measurement_model import MeasurementModel
from src.model.envelope import EnvelopeCurve


def _curve(points: tuple[tuple[float, float], ...]) -> EnvelopeCurve:
    return EnvelopeCurve(independent_variable="object_height_px", points=points)


def _envelope(centre: tuple[tuple[float, float], ...]) -> DetectEnvelope:
    return DetectEnvelope(
        producer_model_sha="m",
        producer_preprocess_sha="p",
        object_class="Person",
        curves={
            "1080p": ResolutionCurves(
                resolution="1080p",
                recall=_curve(((16, 0.2), (300, 0.95))),
                centre_sigma_px=_curve(centre),
                size_sigma_px=_curve(((16, 6.0), (300, 9.0))),
                matched_pairs=500,
            )
        },
        fit_on="test",
        manifest_sha="sha",
    )


def test_r_grows_as_object_height_shrinks() -> None:
    """Even with the pixel error falling for small boxes, the metric error
    grows: a pixel at range z subtends z / fx metres."""
    env = _envelope(((16.0, 2.5), (48.0, 3.0), (96.0, 4.0), (192.0, 5.0)))
    fx, height_m = 1000.0, 1.7
    model = MeasurementModel(
        position_envelope_for(env, "1080p", "cam", fx, height_m, twin_rev=0)
    )
    heights_px = [192.0, 96.0, 48.0, 16.0]
    ranges = [fx * height_m / h for h in heights_px]
    r = [float(np.trace(model.R(z))) for z in ranges]
    assert all(b > a for a, b in zip(r, r[1:])), r


def test_box_uncertainty_is_read_from_the_measured_curve() -> None:
    env = _envelope(((16.0, 2.0), (200.0, 6.0)))
    u = env.source("1080p", "m", "p", 1080).uncertainty(0, 0, 10, 108)  # 108 tall
    assert u.kind == "gaussian_px"
    assert u.get("sigma_x") == pytest.approx(2.0 + (108 - 16) / (200 - 16) * 4.0)


def test_a_relabelled_envelope_is_refused() -> None:
    with pytest.raises(DetectEnvelopeError, match="synthetic-derived"):
        DetectEnvelope(
            producer_model_sha="m",
            producer_preprocess_sha="p",
            object_class="Person",
            curves=_envelope(((16.0, 2.0), (200.0, 6.0))).curves,
            fit_on="t",
            manifest_sha="s",
            label="valid for real cameras",
        )
    assert SYNTHETIC_DERIVED.startswith("synthetic-derived")


def test_unmeasured_resolution_is_refused_not_borrowed() -> None:
    with pytest.raises(DetectEnvelopeError, match="no curves measured"):
        _envelope(((16.0, 2.0), (200.0, 6.0))).at("720p")


def test_curves_must_be_height_indexed() -> None:
    with pytest.raises(DetectEnvelopeError, match="object_height_px"):
        ResolutionCurves(
            resolution="1080p",
            recall=EnvelopeCurve("range_m", ((1, 0.1), (2, 0.2))),
            centre_sigma_px=_curve(((16, 2.0), (200, 6.0))),
            size_sigma_px=_curve(((16, 2.0), (200, 6.0))),
            matched_pairs=10,
        )


def test_r_from_the_measured_smartspaces_envelope_grows_as_height_shrinks() -> None:
    """The real Day-42 curves (Person, val fit), at every measured resolution."""
    import json
    from pathlib import Path

    from src.detect.envelope import from_artifact

    path = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "smartspaces"
        / "day42_detect_envelope.json"
    )
    env = from_artifact(json.loads(path.read_text()))
    assert set(env.curves) == {"1080p", "720p", "360p"}
    for res, curves in env.curves.items():
        model = MeasurementModel(
            position_envelope_for(env, res, "cam", 1000.0, 1.7, twin_rev=0)
        )
        heights = [h for h, _ in curves.centre_sigma_px.points][::-1]
        r = [float(np.trace(model.R(1000.0 * 1.7 / h))) for h in heights]
        assert all(b > a for a, b in zip(r, r[1:])), (res, r)


def test_a_box_outside_the_measured_heights_says_so() -> None:
    """Review finding (Day 42): clamped boxes were labelled within_envelope."""
    src = _envelope(((16.0, 2.0), (200.0, 6.0))).source("1080p", "m", "p", 1080)
    assert src.status(0, 0, 5, 100) == "within_envelope"
    assert src.status(0, 0, 5, 10) == "outside_envelope"
    assert src.status(0, 0, 5, 400) == "outside_envelope"


def test_observations_carry_the_envelope_status_per_box() -> None:
    from src.contracts.frames import FrameGeometry
    from src.detect.contract import ClassVocabulary, DetectionSet, ProducerShas
    from src.model.evidence import UncalibratedScore

    detections = DetectionSet(
        frame=FrameGeometry(1920, 1080),
        boxes_xyxy=np.array([[0.0, 0.0, 20.0, 100.0], [0.0, 0.0, 5.0, 10.0]]),
        scores=(UncalibratedScore(0.9), UncalibratedScore(0.9)),
        class_ids=np.array([0, 0]),
        vocabulary=ClassVocabulary("t", "1", ("person",)),
        ts_ns=1,
        producer=ProducerShas("m", "p", "c"),
    )
    src = _envelope(((16.0, 2.0), (200.0, 6.0))).source("1080p", "m", "p", 1080)
    obs = detections.to_observations("cam", "f", uncertainty_for=src)
    assert [o.envelope_status for o in obs] == ["within_envelope", "outside_envelope"]
    assert all(o.uncertainty.kind == "gaussian_px" for o in obs)


def test_an_envelope_refuses_another_producer_or_resolution() -> None:
    env = _envelope(((16.0, 2.0), (200.0, 6.0)))
    with pytest.raises(DetectEnvelopeError, match="different producer"):
        env.source("1080p", "other-model", "p", 1080)
    with pytest.raises(DetectEnvelopeError, match="different producer"):
        env.source("1080p", "m", "other-preprocess", 1080)
    with pytest.raises(DetectEnvelopeError, match="720"):
        env.source("1080p", "m", "p", 720)
