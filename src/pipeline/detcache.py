"""Per-frame, resumable detection cache.

The detector is the slice's only expensive step (about a second a frame for
RT-DETRv2-S on CPU). It is stateless per frame, so its output on a frame is a
pure function of::

    (video sha256, frame index, model sha, preprocess sha, input size)

which is the cache key. One file holds one (video, model, preprocess, input
size) combination; each record in it is one frame. Records are appended and
fsynced one frame at a time, so a run killed at any point (a forced power-off
ended the first Day-43 pretrained run) loses at most the frame in flight and
the next run resumes from the first frame not on disk.

The detector's ``config_sha`` (score floor, model config) is not in the key
but it changes the output, so it is recorded in the file header and a
mismatch is refused rather than mixed.

File layout (JSON lines)::

    {"header": {...key, config_sha, vocabulary, frame size...}}
    {"f": 0, "b": [[x0, y0, x1, y1], ...], "s": [...], "c": [...]}
    {"f": 1, ...}

Frames are contiguous from 0, because the slice always decodes from the start.
A torn last line (the write the process died in) is truncated on open; any
other damage raises.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class DetectionCacheError(RuntimeError):
    """The cache file cannot be trusted for this key."""


@dataclass(frozen=True)
class CacheKey:
    video_sha256: str
    model_sha: str
    preprocess_sha: str
    input_hw: tuple[int, int] | None
    """Model input ``(height, width)``; ``None`` = the native frame size."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_sha256": self.video_sha256,
            "model_sha": self.model_sha,
            "preprocess_sha": self.preprocess_sha,
            "input_hw": list(self.input_hw) if self.input_hw else "native",
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()

    def path(self, root: Path) -> Path:
        return root / self.video_sha256[:16] / f"{self.digest()[:24]}.jsonl"


@dataclass(frozen=True)
class FrameRecord:
    boxes_xyxy: FloatArray
    scores: FloatArray
    class_ids: npt.NDArray[np.int64]


class DetectionCache:
    """Open with :meth:`open`; use as a context manager so the lock and the
    file are released."""

    def __init__(
        self,
        path: Path,
        header: dict[str, Any],
        records: list[FrameRecord],
        handle: IO[str],
    ) -> None:
        self.path = path
        self.header = header
        self._records = records
        self._handle = handle

    @classmethod
    def open(
        cls,
        root: Path,
        key: CacheKey,
        config_sha: str,
        labels: tuple[str, ...],
        frame_hw: tuple[int, int],
    ) -> "DetectionCache":
        path = key.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "key": key.as_dict(),
            "config_sha": config_sha,
            "labels": list(labels),
            "frame_hw": list(frame_hw),
        }
        handle = open(path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise DetectionCacheError(
                f"{path} is locked by another process writing the same cache"
            ) from exc
        try:
            records = cls._load(path, handle, header)
        except BaseException:
            handle.close()
            raise
        return cls(path, header, records, handle)

    @staticmethod
    def _load(path: Path, handle: IO[str], header: dict[str, Any]) -> list[FrameRecord]:
        handle.seek(0)
        raw = handle.read()
        if not raw:
            handle.write(json.dumps({"header": header}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return []
        lines = raw.split("\n")
        good_bytes = 0
        records: list[FrameRecord] = []
        for i, line in enumerate(lines):
            is_last = i == len(lines) - 1
            if is_last and line == "":
                break  # the file ends with a newline: nothing torn
            try:
                item = json.loads(line)
                if is_last:
                    # Complete JSON but no trailing newline: the write died
                    # between the record and its newline. Treat as torn.
                    raise ValueError("unterminated last record")
            except ValueError:
                if not is_last:
                    raise DetectionCacheError(
                        f"{path}: line {i + 1} is not JSON and is not the last "
                        "line; the cache is damaged, delete it to rebuild"
                    ) from None
                handle.truncate(good_bytes)
                handle.seek(0, os.SEEK_END)
                if good_bytes == 0:
                    # The torn line was the header itself: write it again,
                    # or frame 0 would be read back as the header.
                    handle.write(json.dumps({"header": header}, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            if i == 0:
                stored = item.get("header")
                if stored != header:
                    raise DetectionCacheError(
                        f"{path} was written for {stored}, not {header}; a "
                        "changed detector config is a new cache, delete this one"
                    )
            else:
                if item["f"] != len(records):
                    raise DetectionCacheError(
                        f"{path}: frame {item['f']} where {len(records)} was "
                        "expected; frames must be contiguous from 0"
                    )
                records.append(
                    FrameRecord(
                        boxes_xyxy=np.asarray(item["b"], np.float64).reshape(-1, 4),
                        scores=np.asarray(item["s"], np.float64),
                        class_ids=np.asarray(item["c"], np.int64),
                    )
                )
            good_bytes += len(line.encode("utf-8")) + 1
        return records

    def __len__(self) -> int:
        """Frames on disk; frame ``len(self)`` is the first miss."""
        return len(self._records)

    def get(self, frame_index: int) -> FrameRecord | None:
        if frame_index < len(self._records):
            return self._records[frame_index]
        return None

    def append(self, frame_index: int, record: FrameRecord) -> None:
        if frame_index != len(self._records):
            raise DetectionCacheError(
                f"append frame {frame_index}, expected {len(self._records)}"
            )
        line = json.dumps(
            {
                "f": frame_index,
                "b": record.boxes_xyxy.tolist(),
                "s": record.scores.tolist(),
                "c": record.class_ids.tolist(),
            }
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._records.append(record)

    def close(self) -> None:
        if not self._handle.closed:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()

    def __enter__(self) -> "DetectionCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
