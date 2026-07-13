import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Dict, List, Tuple


class TemporalStitcher:
    def __init__(self, cotracker_model, window_size: int = 16, overlap: int = 4):
        self.tracker = cotracker_model
        self.window_size = window_size
        self.overlap = overlap

    def _get_windows(self, total_frames: int) -> List[Tuple[int, int]]:
        step = self.window_size - self.overlap
        windows = []
        for start_idx in range(0, total_frames - self.overlap, step):
            end_idx = min(start_idx + self.window_size, total_frames)
            windows.append((start_idx, end_idx))
            if end_idx == total_frames:
                break
        return windows

    def stitch_video_tracks(
        self, video: np.ndarray, grid_size: int = 30
    ) -> Dict[str, np.ndarray]:
        T = video.shape[0]
        windows = self._get_windows(T)

        all_tracks = None
        all_visibility = None
        current_queries = None

        for i, (start_idx, end_idx) in enumerate(windows):
            chunk = video[start_idx:end_idx]
            inputs = {"video": chunk}

            if i == 0:
                inputs["grid_size"] = grid_size
            else:
                inputs["queries"] = current_queries

            result = self.tracker.predict(inputs)
            chunk_tracks = result["tracks"]
            chunk_vis = result["visibility"]

            N = chunk_tracks.shape[1]

            if i == 0:
                all_tracks = np.zeros((T, N, 2), dtype=np.float32)
                all_visibility = np.zeros((T, N), dtype=np.float32)

            all_tracks[start_idx:end_idx] = chunk_tracks
            all_visibility[start_idx:end_idx] = chunk_vis

            if i < len(windows) - 1:
                overlap_start = self.window_size - self.overlap
                last_known_points = chunk_tracks[overlap_start]

                t_col = np.zeros((N, 1), dtype=np.float32)
                current_queries = np.concatenate([t_col, last_known_points], axis=-1)

        return {"tracks": all_tracks, "visibility": all_visibility}
