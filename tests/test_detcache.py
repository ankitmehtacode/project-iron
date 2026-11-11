"""The per-frame detection cache: resumable, torn-write tolerant, and refusing
anything it cannot vouch for."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.detcache import (
    CacheKey,
    DetectionCache,
    DetectionCacheError,
    FrameRecord,
)

KEY = CacheKey("v" * 64, "m" * 64, "p" * 64, (640, 640))
LABELS = ("person", "chair")


def _record(i: int) -> FrameRecord:
    return FrameRecord(
        boxes_xyxy=np.array([[i, 2.5, i + 10.123456789012345, 40.0]]),
        scores=np.array([0.1 + i / 1000]),
        class_ids=np.array([i % 2]),
    )


def _open(root: Path, config_sha: str = "c") -> DetectionCache:
    return DetectionCache.open(root, KEY, config_sha, LABELS, (1080, 1920))


def test_records_round_trip_exactly_and_resume(tmp_path: Path) -> None:
    with _open(tmp_path) as cache:
        assert len(cache) == 0
        for i in range(5):
            cache.append(i, _record(i))
    with _open(tmp_path) as cache:
        assert len(cache) == 5
        for i in range(5):
            got, want = cache.get(i), _record(i)
            assert got is not None
            np.testing.assert_array_equal(got.boxes_xyxy, want.boxes_xyxy)
            np.testing.assert_array_equal(got.scores, want.scores)
            np.testing.assert_array_equal(got.class_ids, want.class_ids)
        assert cache.get(5) is None
        cache.append(5, _record(5))
    with _open(tmp_path) as cache:
        assert len(cache) == 6


@pytest.mark.parametrize(
    "tail", ['{"f": 3, "b": [[1, 2', '{"f": 3, "b": [], "s": [], "c": []}']
)
def test_a_torn_last_record_is_dropped_and_overwritten(
    tmp_path: Path, tail: str
) -> None:
    with _open(tmp_path) as cache:
        for i in range(3):
            cache.append(i, _record(i))
        path = cache.path
    with open(path, "a") as handle:
        handle.write(tail)  # the process died mid-write: no newline
    with _open(tmp_path) as cache:
        assert len(cache) == 3
        cache.append(3, _record(3))
    with _open(tmp_path) as cache:
        assert len(cache) == 4


def test_damage_before_the_last_line_is_refused(tmp_path: Path) -> None:
    with _open(tmp_path) as cache:
        for i in range(3):
            cache.append(i, _record(i))
        path = cache.path
    lines = path.read_text().splitlines()
    lines[2] = "not json"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DetectionCacheError, match="damaged"):
        _open(tmp_path)


def test_a_changed_detector_config_is_refused(tmp_path: Path) -> None:
    with _open(tmp_path, config_sha="c") as cache:
        cache.append(0, _record(0))
    with pytest.raises(DetectionCacheError, match="changed detector config"):
        _open(tmp_path, config_sha="other")


def test_every_key_field_separates_caches(tmp_path: Path) -> None:
    variants = [
        KEY,
        CacheKey("w" * 64, KEY.model_sha, KEY.preprocess_sha, KEY.input_hw),
        CacheKey(KEY.video_sha256, "n" * 64, KEY.preprocess_sha, KEY.input_hw),
        CacheKey(KEY.video_sha256, KEY.model_sha, "q" * 64, KEY.input_hw),
        CacheKey(KEY.video_sha256, KEY.model_sha, KEY.preprocess_sha, (512, 512)),
        CacheKey(KEY.video_sha256, KEY.model_sha, KEY.preprocess_sha, None),
    ]
    assert len({k.path(tmp_path) for k in variants}) == len(variants)


def test_frames_must_be_contiguous(tmp_path: Path) -> None:
    with _open(tmp_path) as cache:
        cache.append(0, _record(0))
        with pytest.raises(DetectionCacheError):
            cache.append(2, _record(2))


def test_a_second_writer_is_refused(tmp_path: Path) -> None:
    with _open(tmp_path):
        with pytest.raises(DetectionCacheError, match="locked"):
            _open(tmp_path)
