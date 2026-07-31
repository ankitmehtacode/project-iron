"""
performance_eval.py

Measures how fast our CV pipeline runs on CPU.
Tracks latency (ms), throughput (FPS), and checks if we stay under 150ms.
"""

import statistics
import time


# defaults - can be changed when calling the functions
DEFAULT_NUM_RUNS = 20
DEFAULT_WARMUP_RUNS = 2
LATENCY_THRESHOLD_MS = 150.0


def _time_callable(fn):
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return (end - start) * 1000.0


def run_pipeline_with_breakdown(stages):
    if not stages:
        raise ValueError("Need at least one stage to measure.")

    timings = {}

    for name, fn in stages.items():
        if not callable(fn):
            raise TypeError(f"Stage '{name}' is not callable.")
        timings[name] = _time_callable(fn)

    timings["total"] = sum(timings.values())
    return timings


def measure_pipeline_performance(
    pipeline_fn,
    num_runs=DEFAULT_NUM_RUNS,
    warmup_runs=DEFAULT_WARMUP_RUNS,
    latency_threshold_ms=LATENCY_THRESHOLD_MS,
    stages=None,
    print_results=True,
):
    if not callable(pipeline_fn):
        raise TypeError("pipeline_fn must be callable.")
    if warmup_runs >= num_runs:
        raise ValueError(
            f"warmup_runs ({warmup_runs}) must be less than "
            f"num_runs ({num_runs}) - need at least one measured run."
        )

    # run the pipeline and collect all timings
    all_latencies = []
    for _ in range(num_runs):
        latency = _time_callable(pipeline_fn)
        all_latencies.append(latency)

    # drop the warm-up runs
    measured = all_latencies[warmup_runs:]

    # compute stats
    avg_latency = statistics.mean(measured)
    std_latency = statistics.stdev(measured) if len(measured) > 1 else 0.0
    min_latency = min(measured)
    max_latency = max(measured)

    # FPS calculation:
    # if one frame takes avg_latency ms, then in 1000ms (1 second)
    # we can fit 1000 / avg_latency frames
    fps = 1000.0 / avg_latency if avg_latency > 0 else float("inf")

    # check the latency constraint
    constraint_passed = avg_latency < latency_threshold_ms

    # run stage-wise breakdown if stages were provided
    stage_breakdown = None
    if stages is not None:
        stage_breakdown = run_pipeline_with_breakdown(stages)

    # pack everything into a dict
    results = {
        "num_runs": num_runs,
        "warmup_runs": warmup_runs,
        "measured_runs": len(measured),
        "latency_per_run_ms": measured,
        "avg_latency_ms": round(avg_latency, 4),
        "std_latency_ms": round(std_latency, 4),
        "min_latency_ms": round(min_latency, 4),
        "max_latency_ms": round(max_latency, 4),
        "fps": round(fps, 2),
        "latency_threshold_ms": latency_threshold_ms,
        "constraint_passed": constraint_passed,
    }

    if stage_breakdown is not None:
        results["stage_breakdown"] = stage_breakdown

    if print_results:
        _print_results(results)

    return results


def _print_results(results):
    """Print the benchmark results in a readable format."""
    sep = "=" * 55

    print(f"\n{sep}")
    print("  PIPELINE PERFORMANCE EVALUATION")
    print(sep)

    print(
        f"\n  Runs        : {results['num_runs']}  "
        f"(warm-up: {results['warmup_runs']}, "
        f"measured: {results['measured_runs']})"
    )

    print(f"\n  Average Latency : {results['avg_latency_ms']:.2f} ms")
    print(f"  Std Deviation   : {results['std_latency_ms']:.2f} ms")
    print(f"  Min Latency     : {results['min_latency_ms']:.2f} ms")
    print(f"  Max Latency     : {results['max_latency_ms']:.2f} ms")
    print(f"  Throughput      : {results['fps']:.2f} FPS")

    # print stage breakdown if we have it
    breakdown = results.get("stage_breakdown")
    if breakdown:
        print("\n  Stage-wise Breakdown:")
        for stage_name, latency in breakdown.items():
            if stage_name == "total":
                continue
            print(f"    {stage_name:20s} : {latency:.2f} ms")
        print(f"    {'total':20s} : {breakdown['total']:.2f} ms")

    # constraint check
    threshold = results["latency_threshold_ms"]
    passed = results["constraint_passed"]
    status = "PASS" if passed else "FAIL"

    print(f"\n  Constraint Check ({threshold:.0f} ms threshold):")
    print(f"    {status}")

    print(f"\n{sep}\n")


# demo - runs a quick test with simulated stages
def _demo():
    import numpy as np

    try:
        import torch
    except ImportError:
        print("[WARN] PyTorch not available, skipping torch-based stages.")
        torch = None

    print("Running performance evaluation demo with simulated stages...\n")

    # create some dummy data (prepared once, outside the timing loop)
    rng = np.random.default_rng(42)
    N_POINTS = 100
    EMBED_DIM = 64

    dummy_video = rng.random((1, 4, 3, 224, 224), dtype=np.float32)
    dummy_tracks = rng.random((4, N_POINTS, 2), dtype=np.float32) * 224
    dummy_depth = rng.uniform(0.5, 50.0, size=(4, 224, 224)).astype(np.float32)
    dummy_positions = rng.random((N_POINTS, 3)).astype(np.float32)
    dummy_embeddings = rng.random((N_POINTS, EMBED_DIM)).astype(np.float32)
    dummy_track_ids = np.arange(N_POINTS)

    # simulate embedding extraction (V-JEPA + CoTracker style)
    def simulate_embedding_extraction():
        mat = dummy_video.reshape(dummy_video.shape[0], -1)
        _ = mat @ mat.T

    # simulate depth-based 3D projection
    def simulate_3d_projection():
        from src.geometry.projector_vectorized import (
            project_points_to_3d,
            compute_intrinsics,
        )

        fx, fy, cx, cy = compute_intrinsics(224, 224)
        for t in range(dummy_tracks.shape[0]):
            project_points_to_3d(dummy_tracks[t], dummy_depth[t], fx, fy, cx, cy)

    # simulate fusion graph construction
    def simulate_fusion_graph():
        if torch is not None:
            from src.graph.fusion_graph import build_fusion_graph

            pos = torch.from_numpy(dummy_positions)
            tids = torch.from_numpy(dummy_track_ids).long()
            emb = torch.from_numpy(dummy_embeddings)
            build_fusion_graph(pos, tids, embeddings=emb, radius=1.5)
        else:
            from scipy.spatial import cKDTree

            tree = cKDTree(dummy_positions)
            _ = tree.query_ball_point(dummy_positions, r=1.5)

    # full pipeline = all stages in sequence
    def run_full_pipeline():
        simulate_embedding_extraction()
        simulate_3d_projection()
        simulate_fusion_graph()

    stages = {
        "embedding": simulate_embedding_extraction,
        "projection": simulate_3d_projection,
        "graph": simulate_fusion_graph,
    }

    results = measure_pipeline_performance(
        pipeline_fn=run_full_pipeline,
        num_runs=20,
        warmup_runs=2,
        stages=stages,
        print_results=True,
    )

    return results


if __name__ == "__main__":
    _demo()
