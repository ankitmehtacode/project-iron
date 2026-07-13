import time
import numpy as np
import pandas as pd
from pathlib import Path


def load_data(tracks_path: str, depth_dir: str) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.load(tracks_path)
    print(f"Tracks shape: {tracks.shape}")

    depth_paths = sorted(Path(depth_dir).glob("frame_*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth maps found in {depth_dir}")

    depth_maps = np.stack([np.load(p) for p in depth_paths], axis=0)
    print(f"Depth maps shape: {depth_maps.shape}")

    return tracks, depth_maps


def compute_intrinsics(H: int, W: int) -> tuple[float, float, float, float]:
    fx = fy = float(max(H, W))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


# vectorized depth
def project_points_to_3d(
    points: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mask_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project N 2D points into 3D using a depth map and camera intrinsics.

    Parameters
    ----------
    points     : (N, 2) float32  — (x, y) pixel coordinates
    depth_map  : (H, W) float32  — per-pixel depth values
    fx, fy     : focal lengths
    cx, cy     : principal point
    mask_invalid : if True, zero/NaN depth entries are flagged in the mask

    Returns
    -------
    points_3d : (N, 3) float32  — (X, Y, Z) in camera space
    valid_mask: (N,)   bool     — True where depth is finite and > 0
    """
    H, W = depth_map.shape
    N = points.shape[0]

    assert (
        points.ndim == 2 and points.shape[1] == 2
    ), f"points must be (N,2), got {points.shape}"
    assert depth_map.ndim == 2, f"depth_map must be (H,W), got {depth_map.shape}"

    # Ensure float32 contiguous input — avoids implicit copies later
    points = np.ascontiguousarray(points, dtype=np.float32)
    depth_map = np.ascontiguousarray(depth_map, dtype=np.float32)

    # Separate x and y columns — views, no copy
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Convert to integer pixel indices and clip to valid range in one pass
    xs = np.clip(np.round(x_coords), 0, W - 1).astype(np.int32)
    ys = np.clip(np.round(y_coords), 0, H - 1).astype(np.int32)

    # Vectorized depth lookup: fetch all N depth values simultaneously
    Z = depth_map[ys, xs]

    # Optional: build validity mask — no loop, pure boolean array
    if mask_invalid:
        valid_mask = np.isfinite(Z) & (Z > 0.0)  # (N,) bool
    else:
        valid_mask = np.ones(N, dtype=bool)

    # Vectorized 3D projection using broadcasting:
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy

    # Stack into (N, 3) — np.stack operates on existing arrays, no extra alloc
    points_3d = np.stack([X, Y, Z], axis=-1)

    assert points_3d.shape == (N, 3), f"Unexpected output shape: {points_3d.shape}"

    return points_3d, valid_mask


# REFACTORED: project_to_3d now delegates to the vectorized helper above
def project_to_3d(tracks: np.ndarray, depth_maps: np.ndarray) -> pd.DataFrame:
    T, N, _ = tracks.shape
    _, H, W = depth_maps.shape
    fx, fy, cx, cy = compute_intrinsics(H, W)

    # Pre-allocate output arrays for all frames × points — single allocation
    all_X = np.empty((T, N), dtype=np.float32)
    all_Y = np.empty((T, N), dtype=np.float32)
    all_Z = np.empty((T, N), dtype=np.float32)
    all_valid = np.empty((T, N), dtype=bool)

    # One vectorized call per frame (outer loop over frames is unavoidable
    for t in range(T):
        pts_3d, mask = project_points_to_3d(
            tracks[t],
            depth_maps[t],
            fx,
            fy,
            cx,
            cy,
            mask_invalid=True,
        )
        all_X[t] = pts_3d[:, 0]
        all_Y[t] = pts_3d[:, 1]
        all_Z[t] = pts_3d[:, 2]
        all_valid[t] = mask

    # Flatten to 1-D for the DataFrame — ravel() returns a view, no copy
    frame_ids = np.repeat(np.arange(T), N)
    point_ids = np.tile(np.arange(N), T)

    return pd.DataFrame(
        {
            "frame_id": frame_ids,
            "point_id": point_ids,
            "X": all_X.ravel(),
            "Y": all_Y.ravel(),
            "Z": all_Z.ravel(),
            "valid": all_valid.ravel(),
        }
    )


def _loop_project(
    points: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Reference loop-based implementation for benchmarking only."""
    H, W = depth_map.shape
    results = []
    for x, y in points:
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        Z = float(depth_map[yi, xi])
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        results.append([X, Y, Z])
    return np.array(results, dtype=np.float32)


def benchmark(N: int = 10_000, H: int = 720, W: int = 1280, repeats: int = 50) -> None:
    rng = np.random.default_rng(42)
    points = rng.uniform([0, 0], [W, H], size=(N, 2)).astype(np.float32)
    depth_map = rng.uniform(0.5, 50.0, size=(H, W)).astype(np.float32)
    fx = fy = float(max(H, W))
    cx, cy = W / 2.0, H / 2.0

    print("\n" + "=" * 58)
    print("BENCHMARK: loop-based vs vectorised depth projection")
    print(f"  Points : {N:,}   Frame : {W}×{H}   Repeats : {repeats}")
    print("=" * 58)

    # Warm-up
    _loop_project(points[:10], depth_map, fx, fy, cx, cy)
    project_points_to_3d(points[:10], depth_map, fx, fy, cx, cy)

    times_loop = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _loop_project(points, depth_map, fx, fy, cx, cy)
        times_loop.append(time.perf_counter() - t0)

    times_vec = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        project_points_to_3d(points, depth_map, fx, fy, cx, cy)
        times_vec.append(time.perf_counter() - t0)

    tl = np.array(times_loop) * 1e3
    tv = np.array(times_vec) * 1e3

    ref, vec = (
        _loop_project(points, depth_map, fx, fy, cx, cy),
        project_points_to_3d(points, depth_map, fx, fy, cx, cy)[0],
    )
    match = np.allclose(ref, vec, atol=1e-4)

    print(f"\n  Loop-based   : {tl.mean():.2f} ms ± {tl.std():.2f} ms")
    print(f"  Vectorised   : {tv.mean():.2f} ms ± {tv.std():.2f} ms")
    print(f"  Speed-up     : {tl.mean() / tv.mean():.1f}×")
    print(f"  Results match: {match}")
    print("=" * 58 + "\n")


def main():
    tracks_path = "outputs/tracks.npy"
    depth_dir = "outputs/depth_maps"
    output_path = Path("outputs/point_cloud_tracks.csv")

    tracks, depth_maps = load_data(tracks_path, depth_dir)

    T_track, T_depth = tracks.shape[0], depth_maps.shape[0]
    if T_track != T_depth:
        print(
            f"Warning: {T_track} track frames vs {T_depth} depth frames — truncating."
        )
        T = min(T_track, T_depth)
        tracks = tracks[:T]
        depth_maps = depth_maps[:T]

    benchmark()

    df = project_to_3d(tracks, depth_maps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved point cloud CSV to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
