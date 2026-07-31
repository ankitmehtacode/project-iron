import numpy as np
from src.interface.temporal_stitching import TemporalStitcher


class MockTracker:
    def predict(self, inputs):
        video_len = inputs["video"].shape[0]
        N_points = 100
        return {
            "tracks": np.random.rand(video_len, N_points, 2).astype(np.float32),
            "visibility": np.ones((video_len, N_points), dtype=np.float32),
        }


def test_windowing():
    tracker = MockTracker()
    stitcher = TemporalStitcher(tracker, window_size=16, overlap=4)
    windows = stitcher._get_windows(30)
    assert windows == [(0, 16), (12, 28), (24, 30)]


def test_stitching_shapes():
    tracker = MockTracker()
    stitcher = TemporalStitcher(tracker, window_size=16, overlap=4)
    dummy_video = np.zeros((30, 10, 10, 3), dtype=np.uint8)

    result = stitcher.stitch_video_tracks(dummy_video, grid_size=10)

    assert result["tracks"].shape == (30, 100, 2)
    assert result["visibility"].shape == (30, 100)
