"""Three detector backends behind one contract (Day 42, B3).

- :class:`OracleDetector`: ground-truth visible boxes. Score 1.0, never a
  product: it is the upper bound the other two are measured against.
- :class:`MotionBlobDetector`: MOG2 foreground with the gate's parameters
  (``Mog2Backend``), connected components as boxes. Grey-then-downscale, where
  the gate downscales colour first: equal to the gate's raster only up to
  uint8 rounding. Licence-free and appearance-blind: the trivial
  baseline a learned detector must clearly beat.
- :class:`PretrainedDetector`: a registry-signed RT-DETR(v2) checkpoint on
  the pinned torch/transformers. Refuses to load unless its licence is
  signed, its files hash to what the registry recorded, and its declared
  preprocessing equals the committed spec.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.cascade.motion import Mog2Backend, _downscale, _to_gray
from src.contracts.frames import AffineTransform, FrameGeometry
from src.detect.contract import (
    ClassVocabulary,
    DetectionError,
    DetectionSet,
    FrameInput,
    ProducerShas,
)
from src.detect.preprocess import CANONICAL_SPEC_DIR, DetectorPreprocessSpec
from src.model.evidence import UncalibratedScore

FloatArray = npt.NDArray[np.float64]


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --- oracle -----------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthBoxes:
    """Visible GT boxes for one camera, by frame index."""

    boxes: Mapping[int, FloatArray]
    """frame index -> ``[N, 4]`` xyxy canonical pixels."""
    labels: Mapping[int, Sequence[str]]
    source_sha: str


class OracleDetector:
    """Returns the ground truth. Reads ``frame.frame_index``; nothing else may."""

    def __init__(self, truth: GroundTruthBoxes, vocabulary: ClassVocabulary) -> None:
        self._truth = truth
        self._vocabulary = vocabulary
        self._producer = ProducerShas(
            model_sha=f"oracle:{truth.source_sha}",
            preprocess_sha="identity",
            config_sha=_sha({"backend": "oracle", "vocab": vocabulary.sha}),
        )

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def vocabulary(self) -> ClassVocabulary:
        return self._vocabulary

    def detect(self, frame: FrameInput) -> DetectionSet:
        boxes = np.asarray(
            self._truth.boxes.get(frame.frame_index, np.zeros((0, 4))), np.float64
        ).reshape(-1, 4)
        labels = list(self._truth.labels.get(frame.frame_index, []))
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.geometry.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.geometry.height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return DetectionSet(
            frame=frame.geometry,
            boxes_xyxy=boxes[keep],
            scores=tuple(UncalibratedScore(1.0) for _ in range(int(keep.sum()))),
            class_ids=np.array(
                [self._vocabulary.index(lb) for lb, k in zip(labels, keep) if k],
                dtype=np.int64,
            ),
            vocabulary=self._vocabulary,
            ts_ns=frame.ts_ns,
            producer=self._producer,
        )


# --- motion blobs -----------------------------------------------------------

MOTION_BLOB_VOCABULARY = ClassVocabulary(
    name="motion-blob", version="1", labels=("moving_object",)
)


@dataclass(frozen=True)
class MotionBlobConfig:
    """Blob extraction on the gate's MOG2 mask. Defaults are stated, not tuned."""

    gate_width: int = 320
    gate_height: int = 180
    """The raster MOG2 runs at (the product gate's). 0 = native."""
    open_px: int = 1
    """Morphological opening radius (gate px) removing speckle."""
    close_px: int = 2
    """Closing radius (gate px) joining a body split by low-contrast clothing."""
    min_area_px: int = 4
    """Smallest component kept, gate px. A blob smaller than this cannot be
    told from MOG2 noise at 320x180."""


class MotionBlobDetector:
    """Foreground blobs from the gate's MOG2, as boxes, mapped home.

    Stateful (the background model) and appearance-blind. Score is the
    component's fill ratio of its own box, an uncalibrated number in [0, 1].
    """

    def __init__(self, config: MotionBlobConfig | None = None) -> None:
        self._config = config or MotionBlobConfig()
        self._backend = Mog2Backend()
        self._producer = ProducerShas(
            model_sha="mog2:history=250,varThreshold=16,shadows=excluded",
            preprocess_sha=_sha(
                {"gate": [self._config.gate_width, self._config.gate_height]}
            ),
            config_sha=_sha(self._config.__dict__),
        )

    @property
    def name(self) -> str:
        return "motion_blob"

    @property
    def vocabulary(self) -> ClassVocabulary:
        return MOTION_BLOB_VOCABULARY

    def detect(self, frame: FrameInput) -> DetectionSet:
        import cv2

        c = self._config
        gray = _to_gray(frame.image_bgr)
        if c.gate_width and c.gate_height:
            gray = _downscale(gray, c.gate_height, c.gate_width)
        mask = self._backend.foreground_mask(gray).astype(np.uint8)
        if c.open_px:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c.open_px + 1,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if c.close_px:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c.close_px + 1,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rows = [
            stats[i]
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= c.min_area_px
        ]
        gate = FrameGeometry(width=mask.shape[1], height=mask.shape[0])
        home = AffineTransform.for_resize(frame.geometry, gate)
        if rows:
            st = np.array(rows, dtype=np.float64)
            gate_boxes = np.stack(
                [st[:, 0], st[:, 1], st[:, 0] + st[:, 2], st[:, 1] + st[:, 3]], axis=1
            )
            boxes = home.apply(gate_boxes.reshape(-1, 2)).reshape(-1, 4)
            fill = st[:, 4] / (st[:, 2] * st[:, 3])
        else:
            boxes, fill = np.zeros((0, 4)), np.zeros(0)
        return DetectionSet(
            frame=frame.geometry,
            boxes_xyxy=boxes,
            scores=tuple(UncalibratedScore(float(f)) for f in fill.clip(0, 1)),
            class_ids=np.zeros(len(boxes), dtype=np.int64),
            vocabulary=MOTION_BLOB_VOCABULARY,
            ts_ns=frame.ts_ns,
            producer=self._producer,
        )


# --- pretrained -------------------------------------------------------------


class PretrainedDetector:
    """A signed RT-DETR / RT-DETRv2 checkpoint (transformers, CPU).

    Load-time refusals, each raising rather than warning:
    - the registry entry is unsigned (``require_fetchable``);
    - the weights on disk do not hash to the registry's ``content_sha``;
    - the model's own ``preprocessor_config.json`` declares a transform
      different from the committed spec beside this repository.
    """

    def __init__(
        self,
        registry_name: str,
        spec_name: str,
        score_floor: float = 0.05,
        threads: int | None = None,
    ) -> None:
        from src.config import IronConfig
        from src.data.registry import DatasetRegistry

        root = Path(__file__).resolve().parent.parent.parent
        registry = DatasetRegistry.load(root / "configs" / "datasets.yaml")
        entry = registry.open_for_eval(registry_name)
        snapshot = entry.license_snapshot
        assert snapshot is not None  # open_for_eval guarantees it
        if not snapshot.resolved_commit_sha or not entry.content_sha:
            raise DetectionError(
                f"{registry_name!r} has no pinned commit or no fetched content_sha"
            )
        weights = (
            IronConfig.load().paths.resolved_models_dir
            / "weights"
            / entry.name
            / snapshot.resolved_commit_sha
        )
        hashes = {}
        for name in entry.hf_files:
            path = weights / name
            if not path.exists():
                raise DetectionError(f"{path} missing; fetch it through the registry")
            hashes[name] = _file_sha256(path)
        combined = _sha(hashes)
        if combined != entry.content_sha:
            raise DetectionError(
                f"weights under {weights} hash to {combined[:12]}, registry "
                f"recorded {entry.content_sha[:12]}: refusing to load"
            )

        spec = DetectorPreprocessSpec.load(
            CANONICAL_SPEC_DIR / f"{spec_name}.preprocess.json"
        )
        declared = DetectorPreprocessSpec.from_hf_image_processor(
            json.loads((weights / "preprocessor_config.json").read_text()),
            source=f"{entry.name}@{snapshot.resolved_commit_sha}",
        )
        spec.assert_matches(declared)
        if not spec.verified_against_official_config:
            raise DetectionError(f"spec {spec_name} is not marked verified")

        import torch
        import transformers

        if threads is not None:
            torch.set_num_threads(threads)
        config = json.loads((weights / "config.json").read_text())
        model_cls = {
            "rt_detr": transformers.RTDetrForObjectDetection,
            "rt_detr_v2": transformers.RTDetrV2ForObjectDetection,
        }.get(config.get("model_type", ""))
        if model_cls is None:
            raise DetectionError(f"unsupported model_type {config.get('model_type')!r}")
        self._model = model_cls.from_pretrained(str(weights)).eval()
        self._torch = torch
        labels = tuple(
            config["id2label"][str(i)] for i in range(len(config["id2label"]))
        )
        self._vocabulary = ClassVocabulary(
            name=f"coco-{entry.name}",
            version=snapshot.resolved_commit_sha[:12],
            labels=labels,
        )
        self._spec = spec
        self._score_floor = score_floor
        self._name = entry.name
        self._producer = ProducerShas(
            model_sha=entry.content_sha,
            preprocess_sha=spec.preprocess_sha(),
            config_sha=_sha({"score_floor": score_floor, "config": config}),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocabulary(self) -> ClassVocabulary:
        return self._vocabulary

    @property
    def spec(self) -> DetectorPreprocessSpec:
        return self._spec

    def detect(self, frame: FrameInput) -> DetectionSet:
        torch = self._torch
        pixels = torch.from_numpy(self._spec.apply(frame.image_bgr))
        with torch.no_grad():
            out = self._model(pixel_values=pixels)
        # RT-DETR's own head: sigmoid class scores, top-k over (query, class),
        # normalised cxcywh in model-input space.
        logits = out.logits[0].sigmoid()
        num_classes = logits.shape[1]
        scores, flat = logits.flatten().topk(min(300, logits.numel()))
        query = (flat // num_classes).numpy()
        classes = (flat % num_classes).numpy()
        scores_np = scores.numpy().astype(np.float64)
        cxcywh = out.pred_boxes[0].numpy().astype(np.float64)[query]
        in_h, in_w = self._spec.input_hw
        xyxy_in = np.stack(
            [
                (cxcywh[:, 0] - cxcywh[:, 2] / 2) * in_w,
                (cxcywh[:, 1] - cxcywh[:, 3] / 2) * in_h,
                (cxcywh[:, 0] + cxcywh[:, 2] / 2) * in_w,
                (cxcywh[:, 1] + cxcywh[:, 3] / 2) * in_h,
            ],
            axis=1,
        )
        boxes = self._spec.boxes_to_canonical(xyxy_in, frame.geometry)
        keep = (
            (scores_np >= self._score_floor)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        return DetectionSet(
            frame=frame.geometry,
            boxes_xyxy=boxes[keep],
            scores=tuple(
                UncalibratedScore(float(s)) for s in scores_np[keep].clip(0, 1)
            ),
            class_ids=classes[keep].astype(np.int64),
            vocabulary=self._vocabulary,
            ts_ns=frame.ts_ns,
            producer=self._producer,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
