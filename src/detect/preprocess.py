"""Detector preprocessing: the spec that travels with the weights (Day 42, B3).

Same conventions as :mod:`src.models.preprocess` (sidecar beside the
weights, a sha over the transform-defining fields only, a
``verified_against_official_config`` flag), but a separate type: that spec
describes a video encoder (frames, stride, tubelet, patch size, a
``[B, T, C, H, W]`` clip), and a single-image detector has none of those and
two things the encoder spec cannot say — a resize policy that changes
GEOMETRY (stretch vs letterbox) and a pad value.

Geometry is the point. A detector reports boxes in its own input raster;
:meth:`DetectorPreprocessSpec.to_canonical` is the typed
:class:`~src.contracts.frames.AffineTransform` that carries them back to the
source frame, derived from the same sizes :meth:`apply` actually produced.
No backend computes that mapping by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.contracts.frames import AffineTransform, FrameGeometry

ResizePolicy = Literal["stretch", "letterbox_top_left", "letterbox_center"]
RESIZE_POLICIES: tuple[str, ...] = ("stretch", "letterbox_top_left", "letterbox_center")

SPEC_SUFFIX = ".preprocess.json"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_SPEC_DIR = REPO_ROOT / "configs" / "preprocess"


class DetectorPreprocessError(RuntimeError):
    """Raised when a detector's preprocessing is missing, malformed, or
    disagrees with what the model declares."""


@dataclass(frozen=True)
class DetectorPreprocessSpec:
    """Everything that turns a decoded BGR frame into detector input.

    Attributes:
        input_hw: ``(height, width)`` of the model input.
        channel_order: Channel order the model expects (OpenCV decodes BGR).
        resize_policy: ``stretch`` (each axis scaled independently),
            ``letterbox_top_left`` or ``letterbox_center`` (one scale, the
            rest padded with :attr:`pad_value`).
        pad_value: Fill for letterbox padding, in 0-255 units.
        rescale_factor: Multiplier applied to 0-255 values (1/255 or 1.0).
        mean, std: Per-channel standardisation after rescaling, or ``None``
            for a model that takes rescaled values as they are.
        interpolation: ``pil_bilinear`` (Pillow, antialiased on downscale:
            what HuggingFace image processors use) or ``cv2_linear``.
        source: Provenance of the values.
        verified_against_official_config: True only after every value was
            checked against the model's own shipped config.
    """

    input_hw: tuple[int, int]
    channel_order: Literal["RGB", "BGR"]
    resize_policy: ResizePolicy
    pad_value: float
    rescale_factor: float
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    interpolation: Literal["pil_bilinear", "cv2_linear"]
    source: str = "unspecified"
    verified_against_official_config: bool = False

    def __post_init__(self) -> None:
        if len(self.input_hw) != 2 or min(self.input_hw) <= 0:
            raise DetectorPreprocessError(f"bad input_hw {self.input_hw}")
        if self.resize_policy not in RESIZE_POLICIES:
            raise DetectorPreprocessError(
                f"unknown resize_policy {self.resize_policy!r}"
            )
        if self.channel_order not in ("RGB", "BGR"):
            raise DetectorPreprocessError(f"bad channel_order {self.channel_order!r}")
        if self.interpolation not in ("pil_bilinear", "cv2_linear"):
            raise DetectorPreprocessError(f"bad interpolation {self.interpolation!r}")
        if (self.mean is None) != (self.std is None):
            raise DetectorPreprocessError("mean and std are both set or both None")
        if self.std is not None and any(s <= 0 for s in self.std):
            raise DetectorPreprocessError(f"std must be positive, got {self.std}")
        if self.rescale_factor <= 0:
            raise DetectorPreprocessError("rescale_factor must be positive")

    # -- identity ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_hw"] = list(self.input_hw)
        for key in ("mean", "std"):
            payload[key] = None if payload[key] is None else list(payload[key])
        return payload

    def preprocess_sha(self) -> str:
        """sha256 over the fields that change the model input; provenance
        fields are excluded, as in :mod:`src.models.preprocess`."""
        payload = self.to_dict()
        payload.pop("source")
        payload.pop("verified_against_official_config")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectorPreprocessSpec":
        known = {k: v for k, v in payload.items() if not k.startswith("_")}
        try:
            return cls(
                input_hw=(int(known["input_hw"][0]), int(known["input_hw"][1])),
                channel_order=known["channel_order"],
                resize_policy=known["resize_policy"],
                pad_value=float(known["pad_value"]),
                rescale_factor=float(known["rescale_factor"]),
                mean=None if known["mean"] is None else tuple(known["mean"]),
                std=None if known["std"] is None else tuple(known["std"]),
                interpolation=known["interpolation"],
                source=str(known.get("source", "unspecified")),
                verified_against_official_config=bool(
                    known.get("verified_against_official_config", False)
                ),
            )
        except KeyError as exc:
            raise DetectorPreprocessError(f"spec is missing {exc}") from None

    @classmethod
    def load(cls, path: Path) -> "DetectorPreprocessSpec":
        if not path.exists():
            raise DetectorPreprocessError(
                f"no preprocessing spec at {path}: a detector without its spec "
                "does not load"
            )
        return cls.from_dict(json.loads(path.read_text()))

    # -- geometry ---------------------------------------------------------

    def resized_hw(self, source: FrameGeometry) -> tuple[int, int]:
        """The raster the image content occupies inside the model input."""
        in_h, in_w = self.input_hw
        if self.resize_policy == "stretch":
            return (in_h, in_w)
        r = min(in_h / source.height, in_w / source.width)
        return (
            max(1, int(round(source.height * r))),
            max(1, int(round(source.width * r))),
        )

    def pad_offset(self, source: FrameGeometry) -> tuple[float, float]:
        """``(x, y)`` of the content's top-left corner inside the input."""
        if self.resize_policy != "letterbox_center":
            return (0.0, 0.0)
        in_h, in_w = self.input_hw
        h, w = self.resized_hw(source)
        return ((in_w - w) // 2, (in_h - h) // 2)

    def to_canonical(self, source: FrameGeometry) -> AffineTransform:
        """Model-input pixel coordinates -> ``source`` (canonical) pixels.

        Built from the sizes :meth:`apply` produces, so the mapping and the
        image cannot disagree. Pure scale plus offset under the half-integer
        pixel-centre convention (see :mod:`src.contracts.frames`).
        """
        h, w = self.resized_hw(source)
        ox, oy = self.pad_offset(source)
        sx, sy = source.width / w, source.height / h
        return AffineTransform(
            scale_x=sx, scale_y=sy, offset_x=-ox * sx, offset_y=-oy * sy
        )

    def boxes_to_canonical(
        self, boxes_xyxy: npt.NDArray[np.float64], source: FrameGeometry
    ) -> npt.NDArray[np.float64]:
        """Map ``[N, 4]`` model-input boxes home and clip to the frame."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        t = self.to_canonical(source)
        corners = t.apply(boxes.reshape(-1, 2)).reshape(-1, 4)
        corners[:, [0, 2]] = corners[:, [0, 2]].clip(0.0, source.width)
        corners[:, [1, 3]] = corners[:, [1, 3]].clip(0.0, source.height)
        return corners

    # -- pixels -----------------------------------------------------------

    def apply(self, image_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """``[H, W, 3]`` uint8 BGR -> ``[1, 3, in_h, in_w]`` float32."""
        if (
            image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
        ):
            raise DetectorPreprocessError(
                f"expected HxWx3 uint8 BGR, got {image_bgr.shape} {image_bgr.dtype}"
            )
        source = FrameGeometry(width=image_bgr.shape[1], height=image_bgr.shape[0])
        image = image_bgr[:, :, ::-1] if self.channel_order == "RGB" else image_bgr
        h, w = self.resized_hw(source)
        if self.interpolation == "pil_bilinear":
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(np.ascontiguousarray(image)).resize(
                    (w, h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        else:
            import cv2

            resized = cv2.resize(
                np.ascontiguousarray(image), (w, h), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        in_h, in_w = self.input_hw
        canvas = np.full((in_h, in_w, 3), self.pad_value, dtype=np.float32)
        ox, oy = (int(v) for v in self.pad_offset(source))
        canvas[oy : oy + h, ox : ox + w] = resized
        canvas *= self.rescale_factor
        if self.mean is not None and self.std is not None:
            canvas = (canvas - np.asarray(self.mean, np.float32)) / np.asarray(
                self.std, np.float32
            )
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[np.newaxis])

    # -- agreement with the model's own declaration -----------------------

    def assert_matches(self, declared: "DetectorPreprocessSpec") -> None:
        """Refuse to load when the model's own config declares a different
        transform than the spec this backend will apply."""
        if self.preprocess_sha() != declared.preprocess_sha():
            mine, theirs = self.to_dict(), declared.to_dict()
            diffs = [
                f"{k}: spec {mine[k]!r} vs model {theirs[k]!r}"
                for k in mine
                if k not in ("source", "verified_against_official_config")
                and mine[k] != theirs[k]
            ]
            raise DetectorPreprocessError(
                "the model declares different preprocessing than its spec; "
                "refusing to load: " + "; ".join(diffs)
            )

    @classmethod
    def from_hf_image_processor(
        cls, config: dict[str, Any], source: str
    ) -> "DetectorPreprocessSpec":
        """What a HuggingFace RT-DETR ``preprocessor_config.json`` declares.

        Only the shape this repository has verified is accepted: a fixed
        ``height x width`` resize with no padding (a stretch). Anything else
        is refused rather than approximated.
        """
        size = config.get("size") or {}
        if not config.get("do_resize", True) or set(size) != {"height", "width"}:
            raise DetectorPreprocessError(f"unsupported resize declaration {size!r}")
        if config.get("do_pad", False):
            raise DetectorPreprocessError("padded HF processors are not supported")
        if config.get("resample", 2) != 2:
            raise DetectorPreprocessError(
                f"resample {config.get('resample')!r} is not PIL bilinear (2)"
            )
        normalise = bool(config.get("do_normalize", False))
        return cls(
            input_hw=(int(size["height"]), int(size["width"])),
            channel_order="RGB",
            resize_policy="stretch",
            pad_value=0.0,
            rescale_factor=(
                float(config.get("rescale_factor", 1 / 255))
                if config.get("do_rescale", True)
                else 1.0
            ),
            mean=tuple(config["image_mean"]) if normalise else None,
            std=tuple(config["image_std"]) if normalise else None,
            interpolation="pil_bilinear",
            source=source,
            verified_against_official_config=True,
        )
