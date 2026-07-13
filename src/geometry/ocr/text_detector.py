from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    apply_threshold: bool = True
    denoise: bool = False
    scale_factor: float = 1.0
    threshold_method: str = "otsu"  # "otsu" | "adaptive"


@dataclass
class OCRResult:
    text: str
    bbox: tuple[int, int, int, int]
    conf: int


# Tesseract configuration
_TESS_CONFIG = "--psm 11 --oem 3"

# Minimum confidence threshold (0-100).  Tokens below this are discarded.
_MIN_CONFIDENCE: int = 40


# Internal helpers


def _preprocess(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Upscale — cubic interpolation preserves edges better for text
    if cfg.scale_factor != 1.0:
        new_w = int(gray.shape[1] * cfg.scale_factor)
        new_h = int(gray.shape[0] * cfg.scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Denoise (optional — adds ~5-10 ms per frame; skip for real-time use)
    if cfg.denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Binarisation
    if cfg.apply_threshold:
        if cfg.threshold_method == "adaptive":
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:  # default: Otsu
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray


def _scale_bbox(
    bbox: tuple[int, int, int, int], scale: float
) -> tuple[int, int, int, int]:
    """
    Map a bounding box from the (possibly upscaled) preprocessed image back
    to the original frame's coordinate space.
    """
    if scale == 1.0:
        return bbox
    x, y, w, h = bbox
    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


# Public API — single frame


def detect_text(
    frame: np.ndarray,
    min_conf: int = _MIN_CONFIDENCE,
    tess_cfg: str = _TESS_CONFIG,
    preprocess: Optional[PreprocessConfig] = None,
) -> list[OCRResult]:
    """
    Run OCR on a single video frame and return structured detections.
    ValueError  : If ``frame`` does not have shape (H, W, 3).
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame of shape (H, W, 3), got {frame.shape}.")

    cfg = preprocess if preprocess is not None else PreprocessConfig()

    # ── Preprocessing ────────────────────────────────────────────────
    processed = _preprocess(frame, cfg)

    data = pytesseract.image_to_data(
        processed,
        config=tess_cfg,
        output_type=pytesseract.Output.DICT,
    )

    texts = np.array(data["text"])
    confs = np.array(data["conf"], dtype=int)
    lefts = np.array(data["left"], dtype=int)
    tops = np.array(data["top"], dtype=int)
    widths = np.array(data["width"], dtype=int)
    heights = np.array(data["height"], dtype=int)

    mask = (confs >= min_conf) & (np.char.strip(texts.astype(str)) != "")

    results: list[OCRResult] = [
        OCRResult(
            text=str(texts[i]).strip(),
            bbox=_scale_bbox(
                (int(lefts[i]), int(tops[i]), int(widths[i]), int(heights[i])),
                cfg.scale_factor,
            ),
            conf=int(confs[i]),
        )
        for i in np.where(mask)[0]
    ]

    return results


# Public API — video tensor


def detect_text_in_video(
    frames: np.ndarray,
    frame_skip: int = 1,
    min_conf: int = _MIN_CONFIDENCE,
    tess_cfg: str = _TESS_CONFIG,
    preprocess: Optional[PreprocessConfig] = None,
) -> dict[int, list[OCRResult]]:
    """
    Batch OCR over a video tensor with optional frame skipping.
    ValueError : If ``frames`` does not have shape (T, H, W, 3).
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"Expected tensor of shape (T, H, W, 3), got {frames.shape}.")

    results: dict[int, list[OCRResult]] = {}

    # Slice the frame axis by step=frame_skip; no copy made (view).
    indices = range(0, frames.shape[0], frame_skip)

    for idx in indices:
        detections = detect_text(
            frames[idx],
            min_conf=min_conf,
            tess_cfg=tess_cfg,
            preprocess=preprocess,
        )
        if detections:
            results[idx] = detections

    return results


# Quick smoke-test (run: python text_detector.py)
if __name__ == "__main__":
    # Synthesise a white canvas with black text for a dependency-free test.
    canvas = np.ones((120, 400, 3), dtype=np.uint8) * 255
    cv2.putText(
        canvas,
        "Hello OCR 2025",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cfg = PreprocessConfig(apply_threshold=True, scale_factor=2.0)
    detections = detect_text(canvas, preprocess=cfg)

    print(f"Detected {len(detections)} token(s):")
    for d in detections:
        print(f"  text={d.text!r:15s}  bbox={d.bbox}  conf={d.conf}")
