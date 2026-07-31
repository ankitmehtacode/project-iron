"""Preprocessing specifications that travel with the model.

Why this is a file and not constants
------------------------------------
Preprocessing is part of the model, not part of the calling code. Mean, std,
resolution, frame count and tubelet are properties of the checkpoint that
produced a set of weights; encode them in Python and they immediately drift
from whatever model is actually on disk. That is how ``CLIP_FRAMES = 4`` came
to sit in an entrypoint while the fetched checkpoint was a 64-frame model, and
how the OpenVINO path ended up applying no standardisation at all while an
unused wrapper next to it applied the correct one.

So the spec is serialized **next to the model file** — ``model.xml`` gets
``model.preprocess.json`` — and the wrapper loads and asserts it at
construction. There is no hardcoded mean or std anywhere in this package, and
``tests/test_preprocess_spec.py`` greps the tree to keep it that way.

The sha is a provenance key
---------------------------
Two embeddings are comparable only if they were produced by the same encoder
*and* the same preprocessing. ``preprocess_sha`` is what lets a FAISS index or
an embedding parquet declare which preprocessing produced its vectors, and lets
the query path refuse when that no longer matches. Changing any field here
invalidates every vector derived under the old one — see :mod:`src.artifacts`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

ChannelOrder = Literal["RGB", "BGR"]

SPEC_SUFFIX = ".preprocess.json"

# Canonical specs shipped with the repository. These are templates: the
# authoritative copy lives beside the weights, because that is the thing they
# describe.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_SPEC_DIR = _REPO_ROOT / "configs" / "preprocess"


class PreprocessError(RuntimeError):
    """Raised when a preprocessing spec is missing, malformed, or mismatched."""


@dataclass(frozen=True)
class PreprocessSpec:
    """Everything needed to turn decoded frames into encoder input.

    Attributes:
        frames: Frames the encoder consumes per clip.
        stride: Frame sampling stride. Together with ``frames`` this fixes the
            real-time span a clip covers, which is the thing that actually
            matters — 4 consecutive frames at 30 fps is 0.13 s, a static image
            at video-model prices.
        resolution: ``(height, width)`` the encoder was exported for.
        mean: Per-channel mean subtracted after scaling to [0, 1].
        std: Per-channel standard deviation divided after mean subtraction.
        channel_order: Channel order the encoder expects. OpenCV decodes BGR;
            feeding that to an RGB-trained model is a silent quality loss no
            shape check catches.
        resize_policy: How a frame reaches ``resolution``. Recorded rather than
            assumed, because letterbox and stretch produce different geometry
            and only one of them preserves aspect.
        tubelet: Frames per temporal token. 4 frames at tubelet 2 is 2 temporal
            slots, not 4.
        patch_size: Spatial patch edge in pixels.
        source: Where these values came from, for provenance.
        verified_against_official_config: ``False`` until someone has confirmed
            the values against the shipped model config with the weights in
            hand. A spec asserting values nobody checked is a guess wearing a
            manifest, so the flag is surfaced by
            :meth:`assert_ready_for_production`.
    """

    frames: int
    stride: int
    resolution: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    channel_order: ChannelOrder
    resize_policy: str
    tubelet: int
    patch_size: int
    source: str = "unspecified"
    verified_against_official_config: bool = False

    def __post_init__(self) -> None:
        if self.frames <= 0 or self.stride <= 0:
            raise PreprocessError(
                f"frames and stride must be positive, got frames={self.frames} "
                f"stride={self.stride}"
            )
        if len(self.resolution) != 2 or any(v <= 0 for v in self.resolution):
            raise PreprocessError(
                f"resolution must be a positive (height, width), got {self.resolution}"
            )
        if len(self.mean) != 3 or len(self.std) != 3:
            raise PreprocessError(
                f"mean and std must have three channels, got mean={self.mean} "
                f"std={self.std}"
            )
        if any(s <= 0 for s in self.std):
            raise PreprocessError(
                f"std must be positive in every channel, got {self.std}; a zero "
                "would divide every pixel of that channel by zero"
            )
        if self.channel_order not in ("RGB", "BGR"):
            raise PreprocessError(
                f"channel_order must be 'RGB' or 'BGR', got {self.channel_order!r}"
            )
        if self.tubelet <= 0 or self.patch_size <= 0:
            raise PreprocessError("tubelet and patch_size must be positive")
        if self.frames % self.tubelet != 0:
            raise PreprocessError(
                f"frames ({self.frames}) is not a multiple of tubelet "
                f"({self.tubelet}); the trailing frames would be silently "
                "dropped by the encoder"
            )

    # -- identity ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolution"] = list(self.resolution)
        payload["mean"] = list(self.mean)
        payload["std"] = list(self.std)
        return payload

    def preprocess_sha(self) -> str:
        """sha256 over the values that change the encoder's input.

        ``source`` and ``verified_against_official_config`` are excluded: they
        are documentation about the spec, not part of the transform. Two specs
        that produce identical input must hash identically regardless of who
        wrote them down.
        """
        payload = self.to_dict()
        payload.pop("source", None)
        payload.pop("verified_against_official_config", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -- persistence ------------------------------------------------------

    @staticmethod
    def sidecar_path_for(model_path: Path) -> Path:
        """Where the spec for ``model_path`` lives: ``<model stem>.preprocess.json``."""
        return model_path.with_suffix("").with_name(model_path.stem + SPEC_SUFFIX)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessSpec":
        try:
            return cls(
                frames=int(payload["frames"]),
                stride=int(payload["stride"]),
                resolution=(
                    int(payload["resolution"][0]),
                    int(payload["resolution"][1]),
                ),
                mean=tuple(float(v) for v in payload["mean"]),  # type: ignore[arg-type]
                std=tuple(float(v) for v in payload["std"]),  # type: ignore[arg-type]
                channel_order=payload["channel_order"],
                resize_policy=str(payload["resize_policy"]),
                tubelet=int(payload["tubelet"]),
                patch_size=int(payload["patch_size"]),
                source=str(payload.get("source", "unspecified")),
                verified_against_official_config=bool(
                    payload.get("verified_against_official_config", False)
                ),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise PreprocessError(
                f"preprocessing spec is missing or malformed field: {exc}"
            ) from exc

    @classmethod
    def load(cls, path: Path) -> "PreprocessSpec":
        if not path.exists():
            raise PreprocessError(f"preprocessing spec not found at {path}")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PreprocessError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def load_for_model(cls, model_path: Path) -> "PreprocessSpec":
        """Load the spec sitting beside ``model_path``.

        Deliberately does **not** fall back to a canonical template. A silent
        fallback would let an arbitrary model be used with preprocessing
        belonging to a different one, producing embeddings that are wrong in a
        way nothing downstream can detect — which is the entire failure this
        module exists to prevent. The error names the template to copy.
        """
        sidecar = cls.sidecar_path_for(model_path)
        if not sidecar.exists():
            raise PreprocessError(
                f"no preprocessing spec beside the model: expected {sidecar}. "
                f"Copy the canonical template from "
                f"{CANONICAL_SPEC_DIR}/ and verify its values against the "
                "model's own config before use. Refusing to guess: applying "
                "another model's preprocessing produces embeddings that are "
                "wrong in a way no shape or finiteness check detects."
            )
        return cls.load(sidecar)

    # -- use --------------------------------------------------------------

    def apply(self, clip: FloatArray) -> FloatArray:
        """Standardise a ``[B, T, C, H, W]`` clip whose values are in [0, 1].

        Args:
            clip: Float array, channels in :attr:`channel_order`, values in
                [0, 1].

        Returns:
            ``(clip - mean) / std``, float32, same shape.

        Raises:
            PreprocessError: if the shape or channel count disagrees with this
                spec, or the values are not in [0, 1]. Passing 0-255 data here
                silently produces activations roughly 255x too large, so the
                range is checked rather than assumed.
        """
        array = np.asarray(clip, dtype=np.float32)
        if array.ndim != 5:
            raise PreprocessError(
                f"expected a [B, T, C, H, W] clip, got shape {array.shape}"
            )
        _, frames, channels, height, width = array.shape
        if channels != 3:
            raise PreprocessError(f"expected 3 channels, got {channels}")
        if (height, width) != self.resolution:
            raise PreprocessError(
                f"clip is {height}x{width} but this spec describes "
                f"{self.resolution[0]}x{self.resolution[1]}; resize before "
                "standardising, and attach the transform"
            )
        if frames != self.frames:
            raise PreprocessError(
                f"clip has {frames} frames but this spec describes {self.frames}"
            )

        finite = np.isfinite(array)
        if not finite.all():
            raise PreprocessError("clip contains non-finite values")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise PreprocessError(
                f"clip values span [{array.min():.4g}, {array.max():.4g}] but "
                "must be in [0, 1] before standardisation. Passing 0-255 data "
                "here produces activations about 255x too large, and nothing "
                "downstream would raise."
            )

        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        return ((array - mean) / std).astype(np.float32)

    # -- consistency ------------------------------------------------------

    def assert_matches_pipeline(self, pipeline: Any) -> None:
        """Raise unless a :class:`~src.config.PipelineConfig` agrees with this spec.

        The model decides these values; the config must follow. A mismatch
        means the pipeline is feeding the encoder something it was not exported
        for, which produces plausible output and degraded quality.
        """
        problems: list[str] = []
        if pipeline.clip_frames != self.frames:
            problems.append(
                f"clip_frames={pipeline.clip_frames} but the model expects "
                f"{self.frames}"
            )
        if (pipeline.clip_h, pipeline.clip_w) != self.resolution:
            problems.append(
                f"clip is {pipeline.clip_h}x{pipeline.clip_w} but the model "
                f"expects {self.resolution[0]}x{self.resolution[1]}"
            )
        if pipeline.tubelet != self.tubelet:
            problems.append(
                f"tubelet={pipeline.tubelet} but the model uses {self.tubelet}"
            )
        if pipeline.patch_size != self.patch_size:
            problems.append(
                f"patch_size={pipeline.patch_size} but the model uses "
                f"{self.patch_size}"
            )
        if problems:
            raise PreprocessError(
                "pipeline configuration disagrees with the model's "
                "preprocessing spec: " + "; ".join(problems)
            )

    def assert_ready_for_production(self) -> None:
        """Raise if this spec has not been checked against the model's own config.

        Raises:
            PreprocessError: when
                :attr:`verified_against_official_config` is False.
        """
        if not self.verified_against_official_config:
            raise PreprocessError(
                "this preprocessing spec is marked unverified "
                f"(source: {self.source}). Confirm every value against the "
                "model's shipped config with the weights in hand, then set "
                "verified_against_official_config to true. Embeddings produced "
                "under an unverified spec cannot be trusted for indexing."
            )

    def describe(self) -> str:
        status = (
            "verified" if self.verified_against_official_config else "UNVERIFIED"
        )
        return (
            f"{self.frames}f/stride{self.stride} @ "
            f"{self.resolution[0]}x{self.resolution[1]} {self.channel_order}, "
            f"tubelet {self.tubelet}, patch {self.patch_size}, "
            f"sha {self.preprocess_sha()[:12]}... [{status}]"
        )
