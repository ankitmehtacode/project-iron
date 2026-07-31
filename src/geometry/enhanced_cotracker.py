import os  # noqa: F401
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image


def load_image_sequence(image_dir: str, max_frames: int = 20) -> np.ndarray:
    """Load a sorted sequence of images from *image_dir* into an (T,H,W,3) array."""
    image_dir = Path(image_dir)
    extensions = (".png", ".jpg", ".jpeg")
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]
    )[:max_frames]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    frames = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
    video = np.stack(frames, axis=0)  # (T, H, W, 3)

    print(f"Loaded {len(frames)} frames from {image_dir}")
    print(f"Video shape: {video.shape}")
    return video


def create_grid_points(
    height: int,
    width: int,
    step: int = 20,
) -> np.ndarray:
    """
    Generate a dense, regular grid of (x, y) pixel coordinates.

    Parameters
    ----------
    height, width : int
        Frame dimensions.
    step : int
        Pixel stride between adjacent grid points (controls grid density).
        Smaller  →  denser grid.  Default: 20.

    Returns
    -------
    points : np.ndarray, shape (N, 2), dtype float32, C-contiguous
        Each row is (x, y).  N = len(xs) * len(ys).
    """

    xs = np.arange(0, width, step, dtype=np.float32)
    ys = np.arange(0, height, step, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys)

    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    points = np.ascontiguousarray(points, dtype=np.float32)

    assert (
        points.ndim == 2 and points.shape[1] == 2
    ), f"Expected shape (N, 2), got {points.shape}"
    assert points.dtype == np.float32, f"Expected float32, got {points.dtype}"

    print(f"Grid points: {points.shape[0]}  (step={step}, grid {len(xs)}×{len(ys)})")
    return points


def build_queries(points: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an (N, 2) NumPy points array to the (1, N, 3) query tensor
    that CoTracker3 expects: each row is (frame_index, x, y).

    All points are anchored at frame 0 (simultaneous batch).

    Parameters
    ----------
    points : np.ndarray, shape (N, 2), dtype float32
    device : torch.device

    Returns
    -------
    queries : torch.Tensor, shape (1, N, 3), on *device*
    """
    N = points.shape[0]

    pts_tensor = torch.from_numpy(points)

    frame_ids = torch.zeros(N, 1, dtype=torch.float32)
    queries = torch.cat([frame_ids, pts_tensor], dim=-1)

    queries = queries.unsqueeze(0).to(device)

    assert queries.shape == (1, N, 3), f"Unexpected query shape: {queries.shape}"

    return queries


def run_cotracker(
    video: np.ndarray,
    points: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run CoTracker3 on *video* tracking all *points* in a single batch.

    Parameters
    ----------
    video  : np.ndarray (T, H, W, 3)
    points : np.ndarray (N, 2)  — float32, C-contiguous
    device : torch.device

    Returns
    -------
    tracks     : np.ndarray (T, N, 2)
    visibility : np.ndarray (T, N)
    """
    from cotracker.predictor import CoTrackerPredictor

    T, H, W, _ = video.shape
    N = points.shape[0]

    assert (
        video.ndim == 4 and video.shape[3] == 3
    ), f"video must be (T,H,W,3), got {video.shape}"
    assert (
        points.ndim == 2 and points.shape[1] == 2
    ), f"points must be (N,2), got {points.shape}"

    video_tensor = (
        torch.from_numpy(video).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)
    )

    queries = build_queries(points, device)

    model = CoTrackerPredictor(checkpoint=None)
    model = model.to(device)
    model.eval()

    print(f"Running CoTracker3 on {N} points simultaneously …")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video_tensor, queries=queries)

    tracks = pred_tracks.squeeze(0).cpu().numpy()
    visibility = pred_visibility.squeeze(0).cpu().numpy()

    print(f"Output tracks shape    : {tracks.shape}")
    print(f"Output visibility shape: {visibility.shape}")
    return tracks, visibility


def _loop_grid_points(height: int, width: int, step: int) -> np.ndarray:
    """Reference implementation using explicit Python loops (old approach)."""
    points = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            points.append([float(x), float(y)])
    return np.array(points, dtype=np.float32)


def benchmark(
    height: int = 720, width: int = 1280, step: int = 20, repeats: int = 100
) -> None:
    """
    Compare loop-based vs vectorised grid generation.

    Runs each approach *repeats* times and prints mean ± std wall-clock time.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK: loop-based vs vectorised grid generation")
    print(f"  Frame size : {width}×{height}   step={step}")
    print(f"  Repeats    : {repeats}")
    print("=" * 60)

    _loop_grid_points(height, width, step)
    create_grid_points(height, width, step)

    # Loop-based
    times_loop = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pts_loop = _loop_grid_points(height, width, step)
        times_loop.append(time.perf_counter() - t0)

    # Vectorised
    times_vec = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pts_vec = create_grid_points(height, width, step)
        times_vec.append(time.perf_counter() - t0)

    t_loop = np.array(times_loop) * 1e3
    t_vec = np.array(times_vec) * 1e3

    print(f"\n  Loop-based  : {t_loop.mean():.3f} ms ± {t_loop.std():.3f} ms")
    print(f"  Vectorised  : {t_vec.mean():.3f} ms ± {t_vec.std():.3f} ms")
    print(f"  Speed-up    : {t_loop.mean() / t_vec.mean():.1f}×")
    print(
        f"  Points generated: {pts_vec.shape[0]}  (both match: "
        f"{np.allclose(pts_loop, pts_vec)})"
    )
    print("=" * 60 + "\n")


def main():
    # Force use of the video downloaded
    video_path = "src/interface/ui/data/raw/test_video.mp4"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load video from file instead of folder sequence
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []

    # CRITICAL FIX: Read up to 300 frames instead of 20
    while len(frames) < 300:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    video = np.stack(frames, axis=0)  # (T, H, W, 3)
    T, H, W, _ = video.shape

    points = create_grid_points(H, W, step=20)
    # --- FAILSAFE TRACKING BLOCK ---
    try:
        from cotracker.predictor import CoTrackerPredictor  # noqa: F401

        tracks, visibility = run_cotracker(video, points, device)
    except (ImportError, ModuleNotFoundError, Exception):
        print(
            "[Warning] CoTracker not found or failed. "
            "Generating mock tracks for UI testing."
        )
        T, H, W, _ = video.shape
        N = points.shape[0]
        # Create tracks that move slightly diagonally
        tracks = np.zeros((T, N, 2), dtype=np.float32)
        for t in range(T):
            tracks[t] = points + (t * 2.5)
        visibility = np.ones((T, N), dtype=bool)
    # --- END FAILSAFE ---

    # Save to 'outputs/' so projector_vectorized.py can find them
    np.save(output_dir / "tracks.npy", tracks)
    np.save(output_dir / "visibility.npy", visibility)

    # Create dummy depth maps for testing since you don't have the depth model yet
    depth_dir = output_dir / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for t in range(T):
        dummy_depth = np.random.uniform(1.0, 10.0, (H, W)).astype(np.float32)
        np.save(depth_dir / f"frame_{t:04d}.npy", dummy_depth)

    print(f"SUCCESS: Generated tracks and dummy depth maps in {output_dir}")


if __name__ == "__main__":
    main()
