"""Per-camera multi-object tracker: two-stage IoU association on Kalman-
predicted boxes (the ByteTrack recipe, minimal), numpy + scipy only.

Each detector run:

1. Predict every live track's box forward by the frames since its last run.
2. Stage 1: Hungarian-match high-score detections to tracks on IoU.
3. Stage 2: match low-score detections to the tracks stage 1 left
   unmatched (a partly occluded person scores low; dropping the detection
   would end the track).
4. Unmatched high-score detections start tentative tracks; a track that
   misses ``max_missed`` consecutive detector runs ends.

A frame the motion gate slept through is NOT a detector run: nothing was
looked at, so no track is predicted forward and none is charged a miss. A
person standing still absorbs into the background model and the gate sleeps;
treating that as "detector saw nothing" would end every stationary track and
fire a false ``exited``.

``use_kalman=False, second_stage=False`` is the IoU-only baseline the ID
switch number is reported against: last box as the prediction, one stage.

Tracks with fewer than ``min_hits`` associated detections are dropped at
:meth:`finish` (a batch run: the whole video is seen before output).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TrackerConfig:
    """Stated, not tuned on the evaluation scene."""

    high_score: float = 0.5
    low_score: float = 0.1
    stage1_min_iou: float = 0.2
    stage2_min_iou: float = 0.4
    max_missed: int = 30
    """Consecutive detector runs without a match before a track ends (1 s at
    30 fps with the gate awake)."""
    min_hits: int = 3
    max_extrapolation_frames: int = 1
    """How far a box is predicted across frames the detector did not run on.
    The gate sleeping is itself evidence that little moved; carrying a
    walking velocity across a 500-frame sleep puts the box hundreds of
    pixels from a person who stood still (Day 43, found by
    ``test_a_sleeping_gate_does_not_end_a_track``)."""
    use_kalman: bool = True
    second_stage: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_score <= self.high_score <= 1.0:
            raise ValueError("need 0 <= low_score <= high_score <= 1")
        if min(self.max_missed, self.min_hits, self.max_extrapolation_frames) < 1:
            raise ValueError(
                "max_missed, min_hits and max_extrapolation_frames must be >= 1"
            )


IOU_ONLY = TrackerConfig(use_kalman=False, second_stage=False)


@dataclass(frozen=True)
class Assignment:
    """One detection attached to one track at one frame."""

    track_id: int
    frame_index: int
    detection_index: int
    box_xyxy: tuple[float, float, float, float]
    score: float


def iou_matrix(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU of ``[N, 4]`` and ``[M, 4]`` xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2])
        - np.maximum(a[:, None, 0], b[None, :, 0]),
        0,
        None,
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3])
        - np.maximum(a[:, None, 1], b[None, :, 1]),
        0,
        None,
    )
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def _match(
    iou: FloatArray, min_iou: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian on ``1 - IoU``; pairs below ``min_iou`` are left unmatched."""
    rows, cols = iou.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    r, c = linear_sum_assignment(1.0 - iou)
    pairs = [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= min_iou]
    matched_r = {i for i, _ in pairs}
    matched_c = {j for _, j in pairs}
    return (
        pairs,
        [i for i in range(rows) if i not in matched_r],
        [j for j in range(cols) if j not in matched_c],
    )


# Box Kalman filter on (cx, cy, w, h) and their per-frame velocities. Noise
# scales with box height, as in SORT/ByteTrack: a fixed pixel noise is wrong
# for a person 40 px tall and one 400 px tall alike.
_POS_STD = 1.0 / 20.0
_VEL_STD = 1.0 / 160.0


def _to_cxcywh(box: FloatArray) -> FloatArray:
    x0, y0, x1, y1 = box
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])


def _to_xyxy(state: FloatArray) -> FloatArray:
    cx, cy, w, h = state[:4]
    w, h = max(w, 1.0), max(h, 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class _Track:
    track_id: int
    mean: FloatArray
    cov: FloatArray
    last_box: FloatArray
    hits: int = 1
    missed: int = 0
    last_run_frame: int = 0
    assignments: list[Assignment] = field(default_factory=list)

    def predicted_box(self, use_kalman: bool) -> FloatArray:
        """The box to associate against: the Kalman state after
        :meth:`predict`, or the last detection for the IoU-only baseline."""
        return _to_xyxy(self.mean) if use_kalman else self.last_box

    def predict(self, frame_index: int, max_frames: int) -> None:
        dt = float(min(frame_index - self.last_run_frame, max_frames))
        if dt <= 0:
            return
        F = np.eye(8)
        F[:4, 4:] = dt * np.eye(4)
        h = max(self.mean[3], 1.0)
        q = np.r_[[_POS_STD * h] * 4, [_VEL_STD * h] * 4] ** 2 * dt
        self.mean = F @ self.mean
        self.cov = F @ self.cov @ F.T + np.diag(q)
        self.last_run_frame = frame_index

    def update(self, box: FloatArray) -> None:
        z = _to_cxcywh(box)
        H = np.zeros((4, 8))
        H[:, :4] = np.eye(4)
        r = np.diag(np.full(4, (_POS_STD * max(z[3], 1.0)) ** 2))
        s = H @ self.cov @ H.T + r
        k = self.cov @ H.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ (z - H @ self.mean)
        self.cov = (np.eye(8) - k @ H) @ self.cov
        self.last_box = np.asarray(box, dtype=np.float64)


class Tracker:
    """One camera's tracks. Feed it every detector run in frame order."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._live: list[_Track] = []
        self._ended: list[_Track] = []
        self._next_id = 1
        self._last_frame = -1

    def step(
        self, frame_index: int, boxes_xyxy: FloatArray, scores: FloatArray
    ) -> None:
        """One detector run: the detections (already filtered to the tracked
        class) at ``frame_index``."""
        if frame_index <= self._last_frame:
            raise ValueError(
                f"frame {frame_index} after {self._last_frame}: runs must be in order"
            )
        self._last_frame = frame_index
        c = self.config
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if c.use_kalman:
            for track in self._live:
                track.predict(frame_index, c.max_extrapolation_frames)
        predicted = (
            np.array([t.predicted_box(c.use_kalman) for t in self._live])
            if self._live
            else np.zeros((0, 4))
        )

        high = np.flatnonzero(scores >= c.high_score)
        low = np.flatnonzero((scores >= c.low_score) & (scores < c.high_score))
        pairs, free_tracks, free_high = _match(
            iou_matrix(predicted, boxes[high]), c.stage1_min_iou
        )
        matched = [(ti, int(high[dj])) for ti, dj in pairs]
        if c.second_stage and free_tracks and low.size:
            sub = predicted[free_tracks]
            pairs2, still_free, _ = _match(
                iou_matrix(sub, boxes[low]), c.stage2_min_iou
            )
            matched += [(free_tracks[ti], int(low[dj])) for ti, dj in pairs2]
            free_tracks = [free_tracks[i] for i in still_free]

        for ti, di in matched:
            track = self._live[ti]
            if c.use_kalman:
                track.update(boxes[di])
            else:
                track.last_box = boxes[di]
            track.last_run_frame = frame_index
            track.hits += 1
            track.missed = 0
            track.assignments.append(
                self._assignment(track, frame_index, di, boxes, scores)
            )
        for ti in free_tracks:
            self._live[ti].missed += 1

        for dj in free_high:
            di = int(high[dj])
            mean = np.r_[_to_cxcywh(boxes[di]), np.zeros(4)]
            h = max(mean[3], 1.0)
            cov = np.diag(
                np.r_[[(2 * _POS_STD * h) ** 2] * 4, [(10 * _VEL_STD * h) ** 2] * 4]
            )
            track = _Track(
                track_id=self._next_id,
                mean=mean,
                cov=cov,
                last_box=boxes[di],
                last_run_frame=frame_index,
            )
            self._next_id += 1
            track.assignments.append(
                self._assignment(track, frame_index, di, boxes, scores)
            )
            self._live.append(track)

        still_live: list[_Track] = []
        for track in self._live:
            (self._ended if track.missed >= c.max_missed else still_live).append(track)
        self._live = still_live

    @staticmethod
    def _assignment(
        track: _Track, frame_index: int, di: int, boxes: FloatArray, scores: FloatArray
    ) -> Assignment:
        x0, y0, x1, y1 = (float(v) for v in boxes[di])
        return Assignment(
            track_id=track.track_id,
            frame_index=frame_index,
            detection_index=di,
            box_xyxy=(x0, y0, x1, y1),
            score=float(scores[di]),
        )

    def finish(self) -> list[Assignment]:
        """End every track; return the confirmed tracks' assignments, ordered
        by (track_id, frame_index)."""
        tracks = self._ended + self._live
        self._ended, self._live = [], []
        out = [
            a
            for t in sorted(tracks, key=lambda t: t.track_id)
            if t.hits >= self.config.min_hits
            for a in t.assignments
        ]
        return out
