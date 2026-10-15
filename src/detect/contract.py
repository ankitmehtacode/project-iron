"""The detector contract: one typed envelope, any backend (Day 42, B3).

Three backends sit behind :class:`Detector` — ground truth
(:class:`~src.detect.backends.OracleDetector`), MOG2 foreground blobs
(:class:`~src.detect.backends.MotionBlobDetector`) and a signed pretrained
model (:class:`~src.detect.backends.PretrainedDetector`). Anything built on
:class:`DetectionSet` can swap them, so the gap between the oracle and a real
backend IS the measured end-to-end cost of detector error.

What the envelope refuses to carry
----------------------------------
- Boxes without a declared frame. ``frame`` is required, and every box must
  lie inside it: a box is canonical-frame pixels or it is not constructable.
- A score that reads as a probability. Scores are
  :class:`~src.model.evidence.UncalibratedScore` until a
  :class:`~src.model.evidence.CalibrationRecord` exists for this producer.
- Class ids without their vocabulary. An id means nothing outside the
  versioned label list it indexes.
- A producer it cannot name. Model, preprocessing and config shas travel
  with every set, so two sets are comparable only if they say so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from src.contracts.frames import AffineTransform, FrameGeometry
from src.model.evidence import UncalibratedScore
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import CameraFrameMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid
from src.model.world import UNREGISTERED
from src.model.uncertainty import Uncertainty

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
FrameArray = npt.NDArray[np.uint8]


class DetectionError(ValueError):
    """Raised when a detection envelope violates its contract."""


@dataclass(frozen=True)
class ClassVocabulary:
    """A versioned label list. Class ids index :attr:`labels`."""

    name: str
    version: str
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise DetectionError("a vocabulary needs a name and a version")
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise DetectionError(f"labels must be non-empty and unique: {self.labels}")

    @property
    def sha(self) -> str:
        payload = json.dumps([self.name, self.version, list(self.labels)])
        return hashlib.sha256(payload.encode()).hexdigest()

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise DetectionError(
                f"{label!r} is not in vocabulary {self.name}@{self.version}"
            ) from None


@dataclass(frozen=True)
class ProducerShas:
    """What produced a detection set. Every field required and non-empty."""

    model_sha: str
    preprocess_sha: str
    config_sha: str

    def __post_init__(self) -> None:
        for name in ("model_sha", "preprocess_sha", "config_sha"):
            if not getattr(self, name):
                raise DetectionError(f"ProducerShas.{name} must not be empty")

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.model_sha, self.preprocess_sha, self.config_sha)


@dataclass(frozen=True)
class DetectionSet:
    """Every detection one backend made on one frame, in canonical pixels.

    Attributes:
        frame: The canonical frame (the source video's raster). Required.
        boxes_xyxy: ``[N, 4]`` float64, ``(x0, y0, x1, y1)`` in ``frame``
            pixels under the half-integer pixel-centre convention, so a box
            spans ``[0, width] x [0, height]`` at most.
        scores: One :class:`UncalibratedScore` per box.
        class_ids: ``[N]`` indices into ``vocabulary.labels``.
        vocabulary: The versioned label list.
        ts_ns: When the frame was captured (not processed).
        producer: Model, preprocessing and config shas.
    """

    frame: FrameGeometry
    boxes_xyxy: FloatArray
    scores: tuple[UncalibratedScore, ...]
    class_ids: IntArray
    vocabulary: ClassVocabulary
    ts_ns: int
    producer: ProducerShas
    tolerance_px: float = field(default=0.5)
    """How far outside the frame a box edge may fall before construction
    refuses (sub-pixel rounding in a backend's own arithmetic)."""

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FrameGeometry):
            raise DetectionError("boxes without a declared FrameGeometry are refused")
        boxes = np.asarray(self.boxes_xyxy, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise DetectionError(f"boxes must be [N, 4], got {boxes.shape}")
        ids = np.asarray(self.class_ids, dtype=np.int64)
        n = boxes.shape[0]
        if ids.shape != (n,) or len(self.scores) != n:
            raise DetectionError(
                f"{n} boxes, {ids.shape} class ids and {len(self.scores)} scores"
            )
        if not np.isfinite(boxes).all():
            raise DetectionError("boxes contain non-finite coordinates")
        if n and (
            np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])
        ):
            raise DetectionError("degenerate box (x1 <= x0 or y1 <= y0)")
        t = self.tolerance_px
        w, h = self.frame.width, self.frame.height
        if n and (
            boxes[:, [0, 1]].min() < -t
            or boxes[:, 2].max() > w + t
            or boxes[:, 3].max() > h + t
        ):
            raise DetectionError(
                f"a box lies outside the declared {self.frame} frame; boxes are "
                "canonical-frame pixels, mapped home through an AffineTransform"
            )
        if n and (ids.min() < 0 or ids.max() >= len(self.vocabulary.labels)):
            raise DetectionError("class id outside the vocabulary")
        if not all(isinstance(s, UncalibratedScore) for s in self.scores):
            raise DetectionError("scores must be UncalibratedScore until calibrated")
        object.__setattr__(self, "boxes_xyxy", boxes)
        object.__setattr__(self, "class_ids", ids)

    def __len__(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    def labels(self) -> list[str]:
        return [self.vocabulary.labels[i] for i in self.class_ids]

    def score_array(self) -> FloatArray:
        return np.array([s.score for s in self.scores], dtype=np.float64)

    def heights(self) -> FloatArray:
        return self.boxes_xyxy[:, 3] - self.boxes_xyxy[:, 1]

    def to_observations(
        self,
        sensor_id: str,
        frame_ref: str,
        uncertainty_for: "UncertaintySource | None" = None,
        twin_rev: int = UNREGISTERED,
    ) -> list[Observation]:
        """One :class:`Observation` per box. ``uncertainty_for`` maps a box
        to its measured :class:`Uncertainty` (B6's envelope); absent, every
        box is :meth:`Uncertainty.unmeasured` — never zero."""
        frame_of_reference = FrameOfReference(
            geometry=self.frame,
            to_canonical=AffineTransform.identity(),
            twin_rev=twin_rev,
        )
        out = []
        for box, label in zip(self.boxes_xyxy, self.labels()):
            x0, y0, x1, y1 = (float(v) for v in box)
            uncertainty = (
                uncertainty_for.uncertainty(x0, y0, x1, y1)
                if uncertainty_for is not None
                else Uncertainty.unmeasured()
            )
            out.append(
                Observation(
                    observation_id=generate_ulid(self.ts_ns),
                    sensor_id=sensor_id,
                    ts_ns=self.ts_ns,
                    frame_ref=frame_ref,
                    measurement=CameraFrameMeasurement(
                        bbox_px=(x0, y0, x1, y1), label=label
                    ),
                    uncertainty=uncertainty,
                    frame_of_reference=frame_of_reference,
                    producer_shas=self.producer.as_tuple(),
                    envelope_status=(
                        "unknown" if uncertainty_for is None else "within_envelope"
                    ),
                )
            )
        return out


class UncertaintySource(Protocol):
    """Anything that can state a box's measurement uncertainty (B6)."""

    def uncertainty(self, x0: float, y0: float, x1: float, y1: float) -> Uncertainty:
        ...


@dataclass(frozen=True)
class FrameInput:
    """What a backend is given for one frame. ``frame_index`` is for the
    oracle's GT lookup only; product backends must not read it."""

    image_bgr: FrameArray
    ts_ns: int
    frame_index: int
    geometry: FrameGeometry

    def __post_init__(self) -> None:
        if self.image_bgr.shape[:2] != self.geometry.shape:
            raise DetectionError(
                f"image is {self.image_bgr.shape[:2]} but geometry says {self.geometry}"
            )


class Detector(Protocol):
    """One frame in, one :class:`DetectionSet` out, canonical pixels."""

    @property
    def name(self) -> str:
        ...

    @property
    def vocabulary(self) -> ClassVocabulary:
        ...

    def detect(self, frame: FrameInput) -> DetectionSet:
        ...
