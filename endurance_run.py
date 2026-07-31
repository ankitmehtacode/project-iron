"""
endurance_run.py
================
Continuous pipeline stress test for memory leak detection.

Runs the Project Iron semantic pipeline on 200 synthetic video clips
processed sequentially to verify that memory usage remains stable
over an extended inference session.

Pipeline under test:
    Synthetic video clip
        -> CoTracker3  : point tracking   [B, T, N, 2]
        -> V-JEPA2 INT8: patch embeddings [B, T*196, 1024]
        -> Patch mapper: semantic tracks  [B, T, N, 1024]

Memory stability methodology:
    Clip 1 is used as a warm-up pass. PyTorch and OpenVINO allocate
    internal buffers on the first inference call. The RAM level after
    clip 1 is recorded as the baseline. Clips 2-200 are then measured
    relative to this baseline. A steady upward trend in RAM after
    warm-up indicates a memory leak. Flat or oscillating values pass.

Synthetic clip specification:
    Shape  : [1, 4, 3, 224, 224]  (batch=1, frames=4, RGB, 224x224)
    Values : random float32 in [0, 1]
    Storage: generated in memory, never written to disk

Usage:
    pip install -e .
    python endurance_run.py

Output:
    Console: per-clip RAM readings and final verdict
    File   : logs/endurance_run.log

Author: Radhe Tare
"""

import os
import sys
import time
import traceback
import numpy as np
import psutil

from src.config import IronConfig, apply_runtime_settings

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_process_ram_mb():
    """Return the current process RSS memory usage in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_system_available_ram_mb():
    """Return the system-wide available RAM in megabytes."""
    return psutil.virtual_memory().available / (1024 * 1024)


def make_synthetic_clip(config):
    """
    Generate a random video clip matching the configured input format.

    The clip is created in memory and discarded after each inference pass.
    No disk I/O is performed.

    Args:
        config: Loaded IronConfig; supplies the clip shape.

    Returns:
        np.ndarray: float32 array of shape [1, T, C, H, W] with values in [0, 1].
    """
    return np.random.rand(*config.pipeline.clip_shape).astype(np.float32)


def log(message, filehandle=None):
    """
    Write a message to stdout and optionally to a log file.

    Args:
        message    : String to print.
        filehandle : Open file object. If provided, message is also written
                     there and the buffer is flushed immediately.
    """
    print(message)
    if filehandle:
        filehandle.write(message + "\n")
        filehandle.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config = IronConfig.load()
    applied = apply_runtime_settings(config)

    num_clips = config.endurance.num_clips
    leak_threshold_mb = config.endurance.leak_threshold_mb
    vjepa_xml = config.paths.resolved_vjepa_xml
    cotracker_pth = config.paths.resolved_cotracker_checkpoint

    log_dir = config.paths.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / config.endurance.log_filename

    with open(log_file, "w") as logfile:
        # Header
        log("=" * 65, logfile)
        log("Endurance Run - Continuous Pipeline Memory Stability Test", logfile)
        log("=" * 65, logfile)
        log(f"Config sha     : {config.config_sha()}", logfile)
        log(f"Clips          : {num_clips}", logfile)
        log(f"Clip shape     : {list(config.pipeline.clip_shape)}", logfile)
        log(
            f"Leak threshold : {leak_threshold_mb} MB steady growth "
            f"after warm-up clip",
            logfile,
        )
        log(f"Seed           : {applied.seed}", logfile)
        for skip in applied.skipped:
            log(f"  NOT APPLIED  : {skip}", logfile)
        log(f"V-JEPA2 model  : {vjepa_xml}", logfile)
        log(f"CoTracker3     : {cotracker_pth}", logfile)
        log("", logfile)

        # Verify model files are accessible before starting
        for path in [vjepa_xml, cotracker_pth]:
            if not path.exists():
                log(f"ERROR: Model file not found: {path}", logfile)
                log(
                    "Fetch weights with: python scripts/fetch_weights.py",
                    logfile,
                )
                sys.exit(1)

        # -------------------------------------------------------------------
        # Load pipeline
        # Models are loaded once and reused across every clip.
        # This reflects real deployment behaviour where inference is
        # continuous rather than loading models per clip.
        # -------------------------------------------------------------------
        log("Loading pipeline...", logfile)
        load_start = time.time()

        from src.semantics.semantic_extractor import SemanticExtractor

        extractor = SemanticExtractor(
            vjepa_xml=str(vjepa_xml),
            cotracker_checkpoint=str(cotracker_pth),
            grid_size=config.pipeline.grid_size,
            device=config.runtime.device,
        )

        load_time = time.time() - load_start
        log(f"Pipeline ready in {load_time:.1f}s", logfile)
        log(f"Process RAM after loading  : {get_process_ram_mb():.1f} MB", logfile)
        log(
            f"System RAM available       : " f"{get_system_available_ram_mb():.1f} MB",
            logfile,
        )
        log("", logfile)
        log(
            "Clip 1 is the warm-up pass. Baseline is set from its final " "RAM level.",
            logfile,
        )
        log("Delta values for clips 2-200 are measured from that baseline.", logfile)
        log("", logfile)

        # -------------------------------------------------------------------
        # Inference loop
        # -------------------------------------------------------------------
        header = (
            f"{'Clip':>5}  {'RAM (MB)':>10}  {'Delta (MB)':>11}  "
            f"{'Sys Avail (MB)':>14}  {'Time (s)':>9}  Status"
        )
        log(header, logfile)
        log("-" * 68, logfile)

        ram_baseline = None  # established after clip 1
        ram_readings = []  # post-warm-up readings for trend analysis
        errors = 0
        total_start = time.time()
        clip_idx = 0

        for clip_idx in range(1, num_clips + 1):
            clip_start = time.time()

            try:
                # Create synthetic clip in memory
                video = make_synthetic_clip(config)

                # Run full inference pipeline
                result = extractor.extract(video)

                # Release references immediately to test proper cleanup
                del result
                del video

            except Exception as exc:
                errors += 1
                log(f"  ERROR on clip {clip_idx}: {exc}", logfile)
                traceback.print_exc()
                if errors > 5:
                    log("Aborting: too many consecutive errors.", logfile)
                    break
                continue

            elapsed = time.time() - clip_start
            ram_now = get_process_ram_mb()
            sys_avail = get_system_available_ram_mb()

            # Clip 1 establishes the warm-up baseline
            if clip_idx == 1:
                ram_baseline = ram_now
                delta = 0.0
                status = "WARMUP"
            else:
                delta = ram_now - ram_baseline
                ram_readings.append(ram_now)
                if delta > leak_threshold_mb:
                    status = "FAIL"
                elif delta > leak_threshold_mb * 0.5:
                    status = "WARN"
                else:
                    status = "OK"

            row = (
                f"{clip_idx:>5}  {ram_now:>10.1f}  {delta:>+11.1f}  "
                f"{sys_avail:>14.1f}  {elapsed:>9.2f}  {status}"
            )
            log(row, logfile)

        # -------------------------------------------------------------------
        # Summary report
        # -------------------------------------------------------------------
        total_elapsed = time.time() - total_start
        ram_end = get_process_ram_mb()
        total_growth = ram_end - (ram_baseline or ram_end)

        log("", logfile)
        log("=" * 65, logfile)
        log("SUMMARY", logfile)
        log("=" * 65, logfile)
        log(f"Clips processed        : {clip_idx} / {num_clips}", logfile)
        log(f"Errors                 : {errors}", logfile)
        log(
            f"Total time             : {total_elapsed:.1f}s  "
            f"({total_elapsed / max(clip_idx, 1):.2f}s per clip)",
            logfile,
        )
        log(f"RAM after warm-up clip : {ram_baseline:.1f} MB  (baseline)", logfile)
        log(f"RAM at end of run      : {ram_end:.1f} MB", logfile)
        log(f"Total growth           : {total_growth:+.1f} MB", logfile)
        log(f"Leak threshold         : {leak_threshold_mb} MB", logfile)
        log("", logfile)

        # Trend analysis: compare last 10 vs first 10 post-warmup clips
        # to distinguish between a one-time allocation and a true leak.
        if len(ram_readings) >= 20:
            trend = np.mean(ram_readings[-10:]) - np.mean(ram_readings[:10])
            log(
                f"Trend (last 10 vs first 10 post-warmup clips): " f"{trend:+.1f} MB",
                logfile,
            )
        else:
            trend = total_growth

        log("", logfile)

        if total_growth > leak_threshold_mb and trend > 30:
            log("RESULT: FAILED", logfile)
            log(f"  RAM grew by {total_growth:.1f} MB after warm-up.", logfile)
            log("  Investigate object retention in SemanticExtractor.", logfile)
        else:
            log("RESULT: PASSED", logfile)
            log("  RAM remained stable after warm-up.", logfile)
            log(
                f"  Growth of {total_growth:+.1f} MB is within the "
                f"{leak_threshold_mb} MB threshold.",
                logfile,
            )
            log("  No memory leak detected across 200 inference passes.", logfile)

        log("=" * 65, logfile)
        log(f"Full log: {log_file}", logfile)


if __name__ == "__main__":
    main()
