"""The detector contract, its geometry, and the two weight-free backends
(Day 42, Objective B3)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cascade.stages import DetectionWakePolicy, DetectorStage
from src.cascade.stage import StageContext
from src.contracts.frames import FrameGeometry
from src.detect.backends import (
    GroundTruthBoxes,
    MotionBlobConfig,
    MotionBlobDetector,
    OracleDetector,
)
from src.detect.contract import (
    ClassVocabulary,
    DetectionError,
    DetectionSet,
    FrameInput,
    ProducerShas,
)
from src.detect.preprocess import (
    CANONICAL_SPEC_DIR,
    RESIZE_POLICIES,
    DetectorPreprocessError,
    DetectorPreprocessSpec,
)
from src.model.evidence import UncalibratedScore

FRAME = FrameGeometry(width=1920, height=1080)
VOCAB = ClassVocabulary(name="test", version="1", labels=("Person", "Forklift"))
PRODUCER = ProducerShas(model_sha="m", preprocess_sha="p", config_sha="c")


def _set(boxes: list[list[float]], frame: object = FRAME) -> DetectionSet:
    return DetectionSet(
        frame=frame,  # type: ignore[arg-type]
        boxes_xyxy=np.array(boxes, dtype=float).reshape(-1, 4),
        scores=tuple(UncalibratedScore(0.5) for _ in boxes),
        class_ids=np.zeros(len(boxes), dtype=int),
        vocabulary=VOCAB,
        ts_ns=1,
        producer=PRODUCER,
    )


def _spec(policy: str, hw: tuple[int, int] = (640, 640)) -> DetectorPreprocessSpec:
    return DetectorPreprocessSpec(
        input_hw=hw,
        channel_order="BGR",
        resize_policy=policy,  # type: ignore[arg-type]
        pad_value=114.0,
        rescale_factor=1.0,
        mean=None,
        std=None,
        interpolation="cv2_linear",
    )


# --- the envelope refuses -------------------------------------------------


def test_boxes_without_a_declared_frame_are_refused() -> None:
    with pytest.raises(DetectionError, match="FrameGeometry"):
        _set([[0, 0, 10, 10]], frame=(1920, 1080))


def test_a_box_outside_the_frame_is_refused() -> None:
    with pytest.raises(DetectionError, match="outside"):
        _set([[1900, 0, 1950, 10]])


def test_a_bare_float_score_is_refused() -> None:
    with pytest.raises(DetectionError, match="Uncalibrated"):
        DetectionSet(
            frame=FRAME,
            boxes_xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
            scores=(0.9,),  # type: ignore[arg-type]
            class_ids=np.array([0]),
            vocabulary=VOCAB,
            ts_ns=1,
            producer=PRODUCER,
        )


def test_class_ids_must_index_the_vocabulary() -> None:
    with pytest.raises(DetectionError, match="vocabulary"):
        DetectionSet(
            frame=FRAME,
            boxes_xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]),
            scores=(UncalibratedScore(0.5),),
            class_ids=np.array([2]),
            vocabulary=VOCAB,
            ts_ns=1,
            producer=PRODUCER,
        )


def test_empty_producer_sha_is_refused() -> None:
    with pytest.raises(DetectionError):
        ProducerShas(model_sha="", preprocess_sha="p", config_sha="c")


def test_observations_carry_unmeasured_uncertainty_never_zero() -> None:
    obs = _set([[10, 20, 50, 120]]).to_observations("cam", "frame:1")
    assert len(obs) == 1
    assert obs[0].uncertainty.kind == "unmeasured"
    assert obs[0].envelope_status == "unknown"
    assert obs[0].measurement.bbox_px == (10.0, 20.0, 50.0, 120.0)
    assert obs[0].producer_shas == ("m", "p", "c")


# --- geometry: every policy round-trips -----------------------------------

SOURCES = [FrameGeometry(1920, 1080), FrameGeometry(1280, 720), FrameGeometry(640, 360)]


@settings(max_examples=200, deadline=None)
@given(
    policy=st.sampled_from(RESIZE_POLICIES),
    source=st.sampled_from(SOURCES),
    x0=st.floats(0, 0.9),
    y0=st.floats(0, 0.9),
    w=st.floats(0.001, 0.1),
    h=st.floats(0.001, 0.1),
)
def test_a_known_box_through_resize_and_back_returns_within_0_01_px(
    policy: str, source: FrameGeometry, x0: float, y0: float, w: float, h: float
) -> None:
    """The forward map (canonical -> model input) is computed here from each
    policy's definition, independently of ``to_canonical``; mapping the result
    home must return the box. (Composing ``to_canonical`` with its own inverse
    would be the identity for ANY transform — Day-42 review.)"""
    spec = _spec(policy)
    in_h, in_w = spec.input_hw
    W, H = source.width, source.height
    if policy == "stretch":
        sx, sy, ox, oy = in_w / W, in_h / H, 0.0, 0.0
    else:
        r = min(in_h / H, in_w / W)
        rw, rh = round(W * r), round(H * r)
        sx, sy = rw / W, rh / H
        ox, oy = (
            ((in_w - rw) // 2, (in_h - rh) // 2)
            if policy == "letterbox_center"
            else (0, 0)
        )
    box = np.array([x0 * W, y0 * H, (x0 + w) * W, (y0 + h) * H])
    in_model = np.array(
        [[box[0] * sx + ox, box[1] * sy + oy, box[2] * sx + ox, box[3] * sy + oy]]
    )
    back = spec.boxes_to_canonical(in_model, source)
    assert np.abs(back[0] - box).max() < 0.01


def test_transform_agrees_with_direct_computation() -> None:
    """Lead-order rule 4: the AffineTransform against arithmetic by hand."""
    src = FrameGeometry(1920, 1080)
    point = np.array([[320.0, 180.0]])  # model-input pixels
    stretch = _spec("stretch").to_canonical(src).apply(point)
    assert np.allclose(stretch, [[320 * 1920 / 640, 180 * 1080 / 640]])
    # letterbox: r = min(640/1080, 640/1920) = 1/3, content 640x360
    top_left = _spec("letterbox_top_left").to_canonical(src).apply(point)
    assert np.allclose(top_left, [[960.0, 540.0]])
    centre = _spec("letterbox_center").to_canonical(src).apply(point)
    assert np.allclose(centre, [[960.0, (180 - 140) * 3.0]])  # pad (640-360)/2


@pytest.mark.parametrize("policy", RESIZE_POLICIES)
@pytest.mark.parametrize("src", SOURCES, ids=str)
def test_apply_puts_content_where_the_transform_says(
    policy: str, src: FrameGeometry
) -> None:
    """A white block drawn at a known canonical box lands, after apply(), at
    the model-input box the inverse transform predicts: the pixels and the
    mapping cannot disagree, on every source raster."""
    image = np.zeros((src.height, src.width, 3), np.uint8)
    y0, y1 = src.height * 5 // 18, src.height * 10 // 18
    x0, x1 = src.width * 5 // 16, src.width * 10 // 16
    image[y0:y1, x0:x1] = 255
    spec = _spec(policy)
    tensor = spec.apply(image)[0, 0]
    ys, xs = np.nonzero(tensor > 200)
    to_input = spec.to_canonical(src).inverse()
    (ex0, ey0), (ex1, ey1) = to_input.apply(
        np.array([[float(x0), float(y0)], [float(x1), float(y1)]])
    )
    assert abs(xs.min() - ex0) <= 1 and abs(xs.max() + 1 - ex1) <= 1
    assert abs(ys.min() - ey0) <= 1 and abs(ys.max() + 1 - ey1) <= 1


def test_a_backend_whose_declared_preprocessing_differs_refuses() -> None:
    spec = DetectorPreprocessSpec.load(
        CANONICAL_SPEC_DIR / "rtdetr_v2_r18vd.preprocess.json"
    )
    declared = json.loads(
        json.dumps(
            {
                "do_resize": True,
                "size": {"height": 640, "width": 640},
                "do_pad": False,
                "do_rescale": True,
                "rescale_factor": 1 / 255,
                "do_normalize": False,
                "resample": 2,
            }
        )
    )
    spec.assert_matches(DetectorPreprocessSpec.from_hf_image_processor(declared, "t"))
    declared["size"] = {"height": 800, "width": 800}
    with pytest.raises(DetectorPreprocessError, match="refusing to load"):
        spec.assert_matches(
            DetectorPreprocessSpec.from_hf_image_processor(declared, "t")
        )
    declared["do_pad"] = True
    with pytest.raises(DetectorPreprocessError, match="padded"):
        DetectorPreprocessSpec.from_hf_image_processor(declared, "t")


@pytest.mark.requires_weights  # heavy runtime library: transformers
def test_apply_matches_the_huggingface_processor_pixel_for_pixel() -> None:
    transformers = pytest.importorskip("transformers")
    spec = DetectorPreprocessSpec.load(
        CANONICAL_SPEC_DIR / "rtdetr_v2_r18vd.preprocess.json"
    )
    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    processor = transformers.RTDetrImageProcessor(
        do_resize=True,
        size={"height": 640, "width": 640},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=False,
        do_pad=False,
    )
    theirs = processor(images=image_bgr[:, :, ::-1].copy(), return_tensors="np")
    ours = spec.apply(image_bgr)
    assert ours.shape == theirs["pixel_values"].shape
    assert np.abs(ours - theirs["pixel_values"]).max() < 1e-5


# --- backends -------------------------------------------------------------


def _frame(image: np.ndarray, index: int) -> FrameInput:
    return FrameInput(
        image_bgr=image,
        ts_ns=index * 33_333_333,
        frame_index=index,
        geometry=FrameGeometry(width=image.shape[1], height=image.shape[0]),
    )


def test_oracle_returns_the_ground_truth_clipped_to_the_frame() -> None:
    truth = GroundTruthBoxes(
        boxes={3: np.array([[10.0, 20.0, 50.0, 120.0], [1900.0, 0, 1990.0, 40.0]])},
        labels={3: ["Person", "Forklift"]},
        source_sha="gt",
    )
    det = OracleDetector(truth, VOCAB)
    out = det.detect(_frame(np.zeros((1080, 1920, 3), np.uint8), 3))
    assert out.labels() == ["Person", "Forklift"]
    assert out.boxes_xyxy[1, 2] == 1920.0
    assert len(det.detect(_frame(np.zeros((1080, 1920, 3), np.uint8), 4))) == 0


def test_motion_blob_finds_a_moving_square_in_canonical_pixels() -> None:
    det = MotionBlobDetector(MotionBlobConfig())
    rng = np.random.default_rng(1)
    background = rng.integers(60, 80, (1080, 1920, 3), dtype=np.uint8)
    out = None
    for index in range(40):
        image = background.copy()
        x = 400 + 12 * index
        image[500:700, x : x + 120] = 230
        out = det.detect(_frame(image, index))
    assert out is not None and len(out) >= 1
    box = out.boxes_xyxy[np.argmax(out.heights())]
    x = 400 + 12 * 39
    assert abs(box[1] - 500) < 12 and abs(box[3] - 700) < 12
    assert abs(box[2] - (x + 120)) < 24
    assert out.labels()[0] == "moving_object"


def test_detector_stage_calls_the_protocol_and_applies_the_policy() -> None:
    truth = GroundTruthBoxes(
        boxes={0: np.array([[10.0, 20.0, 50.0, 120.0]])},
        labels={0: ["Forklift"]},
        source_sha="gt",
    )
    image = np.zeros((1, 1080, 1920, 3), np.uint8)
    person_only = DetectorStage(
        OracleDetector(truth, VOCAB), DetectionWakePolicy(frozenset({"Person"}), 0.3)
    )
    ctx = StageContext(frame_index=0, ts_ns=0)
    out = person_only.process(image, ctx)
    assert not out.wake_next and len(ctx.payload["detections"]) == 1
    forklift = DetectorStage(
        OracleDetector(truth, VOCAB), DetectionWakePolicy(frozenset({"Forklift"}), 0.3)
    )
    assert forklift.process(image, StageContext(frame_index=0, ts_ns=0)).wake_next


def test_detector_and_policy_are_bound_together() -> None:
    with pytest.raises(ValueError):
        DetectorStage(detector=None, policy=DetectionWakePolicy(frozenset({"P"}), 0.1))
