"""The signed RT-DETRv2-S checkpoint behind the contract (Day 42, B3).

Needs the weights fetched through the registry, so ``requires_weights``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_weights

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from src.contracts.frames import FrameGeometry  # noqa: E402
from src.detect.backends import PretrainedDetector  # noqa: E402
from src.detect.contract import DetectionError, FrameInput  # noqa: E402


@pytest.fixture(scope="module")
def detector() -> PretrainedDetector:
    try:
        return PretrainedDetector(
            "RT-DETRv2-S-COCO", "rtdetr_v2_r18vd", score_floor=0.0, threads=2
        )
    except DetectionError as exc:
        pytest.skip(f"RT-DETRv2-S weights not fetched: {exc}")


def _image() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = rng.integers(40, 90, (1080, 1920, 3), dtype=np.uint8)
    image[300:900, 800:1000] = (30, 30, 160)  # a tall dark-red block
    return image


def test_boxes_mapped_home_agree_with_huggingface_post_processing(
    detector: PretrainedDetector,
) -> None:
    """Lead-order rule 4: our AffineTransform route against the library's own
    direct computation (normalised cxcywh x target size) on the same forward
    pass; the 50 highest-scoring boxes, to 1e-3 px."""
    image = _image()
    geometry = FrameGeometry(width=1920, height=1080)
    ours = detector.detect(FrameInput(image, 0, 0, geometry))
    pixels = torch.from_numpy(detector.spec.apply(image))
    with torch.no_grad():
        out = detector._model(pixel_values=pixels)
    processor = transformers.RTDetrImageProcessor()
    theirs = processor.post_process_object_detection(
        out, threshold=0.0, target_sizes=[(1080, 1920)]
    )[0]
    order = np.argsort(-ours.score_array())[:50]
    their_boxes = theirs["boxes"].numpy()[:50]
    their_boxes[:, [0, 2]] = their_boxes[:, [0, 2]].clip(0, 1920)
    their_boxes[:, [1, 3]] = their_boxes[:, [1, 3]].clip(0, 1080)
    assert np.allclose(
        ours.score_array()[order], theirs["scores"].numpy()[:50], atol=1e-6
    )
    assert np.abs(ours.boxes_xyxy[order] - their_boxes).max() < 1e-3


def test_vocabulary_is_coco_and_has_no_warehouse_classes(
    detector: PretrainedDetector,
) -> None:
    labels = set(detector.vocabulary.labels)
    assert "person" in labels and len(labels) == 80
    assert not {"forklift", "Forklift", "robot", "pallet truck"} & labels


def test_producer_shas_name_the_signed_weights(detector: PretrainedDetector) -> None:
    out = detector.detect(FrameInput(_image(), 0, 0, FrameGeometry(1920, 1080)))
    assert out.producer.preprocess_sha == detector.spec.preprocess_sha()
    assert len(out.producer.model_sha) == 64
