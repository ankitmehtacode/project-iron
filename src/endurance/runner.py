"""Endurance run orchestration: steady-state and re-init loops.

Two loops, because framework memory leaks live in two different places. The
steady-state loop holds one extractor and pushes clips through it, which finds
per-inference retention. The re-init loop constructs and destroys the extractor
repeatedly, which finds load/unload leaks — overwhelmingly where OpenVINO and
torch actually leak, and completely invisible to a loop that loads once.

Error policy
------------
Any exception fails the run, immediately, with a full traceback in the log and
a nonzero exit code. The previous harness tolerated five errors before aborting
and logged the count as "consecutive errors" while counting cumulatively. Both
halves of that were wrong: a soak test that continues past a failure is
measuring the memory behaviour of a pipeline that is not working, and reporting
the result as a pass.
"""

from __future__ import annotations

import enum
import gc
import json
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

from src.config import IronConfig, apply_runtime_settings
from src.endurance.gates import (
    LatencySummary,
    LeakVerdict,
    Verdict,
    evaluate_leak,
    summarize_latency,
)
from src.endurance.memory import MemorySample, MemorySampler, platform_fidelity_note
from src.provenance import ManifestError, RunManifest

DETERMINISM_COSINE_TOLERANCE = 1e-5


class ExitCode(enum.IntEnum):
    """Process exit codes, chosen so CI can distinguish the failure modes.

    A missing checkpoint is an environment problem, not a regression, and must
    not look like one on a dashboard.
    """

    OK = 0
    GATE_FAILED = 1
    PREREQUISITES_MISSING = 2
    PIPELINE_RAISED = 3
    MANIFEST_UNWRITABLE = 4


class WeightsUnavailable(RuntimeError):
    """Raised when a required model file or runtime library is absent."""


class Extractor(Protocol):
    """The surface the harness needs from the pipeline under test."""

    def extract(self, video: np.ndarray) -> dict[str, Any]:
        ...


ExtractorFactory = Callable[[], Extractor]


class RunLogger:
    """Writes to stdout and a log file at once, flushing every line.

    Flushing eagerly matters: a soak run that dies at hour three must leave the
    lines it had already produced on disk, not in a buffer.
    """

    def __init__(self, handle: TextIO | None = None) -> None:
        self._handle = handle

    def __call__(self, message: str = "") -> None:
        print(message)
        if self._handle is not None:
            self._handle.write(message + "\n")
            self._handle.flush()


class JsonlWriter:
    """Appends one JSON record per line, flushing each."""

    def __init__(self, handle: TextIO | None) -> None:
        self._handle = handle

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()


@dataclass(frozen=True)
class DeterminismResult:
    """Whether the same input twice produced the same embeddings.

    Reports *how* it matched, not just that it did. Byte-identical and
    "identical to within 1e-5 cosine" are different claims: the second is
    consistent with thread-count-dependent reduction order in INT8 kernels,
    which is worth knowing before anyone tries to reproduce a result on
    different hardware.
    """

    checked: bool
    byte_identical: bool
    max_cosine_distance: float
    passed: bool
    detail: str


def check_determinism(extractor: Extractor, clip: np.ndarray) -> DeterminismResult:
    """Run one clip through the pipeline twice and compare the embeddings.

    Args:
        extractor: The pipeline under test.
        clip: Input clip, reused verbatim for both passes.

    Returns:
        A :class:`DeterminismResult` naming which form of equality held.
    """
    first = np.asarray(extractor.extract(clip)["semantic_tracks"], dtype=np.float64)
    second = np.asarray(extractor.extract(clip)["semantic_tracks"], dtype=np.float64)

    if first.shape != second.shape:
        return DeterminismResult(
            checked=True,
            byte_identical=False,
            max_cosine_distance=float("nan"),
            passed=False,
            detail=(
                f"shape changed between identical runs: {first.shape} then "
                f"{second.shape}"
            ),
        )

    byte_identical = bool(np.array_equal(first, second))
    if byte_identical:
        return DeterminismResult(
            checked=True,
            byte_identical=True,
            max_cosine_distance=0.0,
            passed=True,
            detail="byte-identical embeddings across two runs of the same input",
        )

    a = first.reshape(-1, first.shape[-1])
    b = second.reshape(-1, second.shape[-1])
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    usable = norms > 0
    if not np.any(usable):
        return DeterminismResult(
            checked=True,
            byte_identical=False,
            max_cosine_distance=float("nan"),
            passed=False,
            detail="all embeddings are zero vectors; cosine is undefined",
        )

    cosine_similarity = np.sum(a[usable] * b[usable], axis=1) / norms[usable]
    max_distance = float(np.max(1.0 - cosine_similarity))
    passed = max_distance <= DETERMINISM_COSINE_TOLERANCE
    return DeterminismResult(
        checked=True,
        byte_identical=False,
        max_cosine_distance=max_distance,
        passed=passed,
        detail=(
            f"not byte-identical; max cosine distance {max_distance:.3e} "
            f"({'within' if passed else 'EXCEEDS'} the "
            f"{DETERMINISM_COSINE_TOLERANCE:.0e} tolerance)"
        ),
    )


@dataclass
class RunOutcome:
    """Everything a run produced, so the caller decides the exit code."""

    mode: str
    clips_completed: int
    leak: LeakVerdict
    latency: LatencySummary
    determinism: DeterminismResult | None = None
    samples: list[MemorySample] = field(default_factory=list)

    def exit_code(self) -> ExitCode:
        if self.leak.failed:
            return ExitCode.GATE_FAILED
        if self.determinism is not None and not self.determinism.passed:
            return ExitCode.GATE_FAILED
        return ExitCode.OK


def make_synthetic_clip(config: IronConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate one synthetic clip matching the configured input shape.

    Note for whoever extends this: uniform noise exercises no data-dependent
    branch in any of these models. It is adequate for finding allocator leaks
    and nothing else. Real endurance means hours of real footage with
    adversarial clips injected on a schedule.
    """
    return rng.random(config.pipeline.clip_shape, dtype=np.float32)


def default_extractor_factory(config: IronConfig) -> ExtractorFactory:
    """Build a factory that constructs the real SemanticExtractor.

    Raises :class:`WeightsUnavailable` at call time — not import time — so the
    caller can turn it into a clean message and exit code 2 rather than a
    traceback.
    """

    def factory() -> Extractor:
        for path in (
            config.paths.resolved_vjepa_xml,
            config.paths.resolved_cotracker_checkpoint,
        ):
            if not path.exists():
                raise WeightsUnavailable(f"weights not found at {path}")
        try:
            from src.semantics.semantic_extractor import SemanticExtractor
        except ImportError as exc:
            raise WeightsUnavailable(
                f"pipeline dependencies are not installed: {exc}. "
                "Install locking-requirements.txt."
            ) from exc

        return SemanticExtractor(
            vjepa_xml=str(config.paths.resolved_vjepa_xml),
            cotracker_checkpoint=str(config.paths.resolved_cotracker_checkpoint),
            grid_size=config.pipeline.grid_size,
            device=config.runtime.device,
        )

    return factory


class EnduranceRunner:
    """Drives an endurance run and applies the gates.

    The extractor is supplied by a factory rather than constructed here so the
    harness itself can be tested without model weights, and so the re-init loop
    can build a fresh one per cycle through the same path production uses.
    """

    def __init__(
        self,
        config: IronConfig,
        factory: ExtractorFactory,
        log: RunLogger,
        metrics: JsonlWriter,
        manifest_sha: str,
    ) -> None:
        self._config = config
        self._factory = factory
        self._log = log
        self._metrics = metrics
        self._manifest_sha = manifest_sha
        self._sampler = MemorySampler(
            track_allocations=True,
            top_allocations=config.endurance.tracemalloc_top,
        )

    def run(self, mode: str, iterations: int) -> RunOutcome:
        """Run in ``steady`` or ``reinit`` mode for ``iterations`` cycles."""
        if mode == "steady":
            return self._run_steady(iterations)
        if mode == "reinit":
            return self._run_reinit(iterations)
        raise ValueError(f"unknown mode {mode!r}; expected 'steady' or 'reinit'")

    # -- steady state -----------------------------------------------------

    def _run_steady(self, clips: int) -> RunOutcome:
        endurance = self._config.endurance
        rng = np.random.default_rng(self._config.runtime.seed)

        self._log("Loading pipeline...")
        load_start = time.perf_counter()
        extractor = self._factory()
        self._log(f"Pipeline ready in {time.perf_counter() - load_start:.1f}s")
        self._log(f"Memory fidelity: {platform_fidelity_note()}")
        self._log("")

        warmup = endurance.warmup_clips
        durations: list[float] = []
        post_warmup_rss: list[float] = []
        samples: list[MemorySample] = []
        determinism: DeterminismResult | None = None

        header = (
            f"{'Clip':>6}  {'RSS (MB)':>10}  {'USS (MB)':>10}  "
            f"{'Time (s)':>9}  Phase"
        )
        self._log(header)
        self._log("-" * len(header))

        for clip_index in range(1, clips + 1):
            video = make_synthetic_clip(self._config, rng)

            with self._fail_on_exception(clip_index):
                # The determinism probe runs on the second clip's input, once
                # the first clip has warmed the framework buffers. Doing it on
                # clip 1 would measure lazy initialisation rather than
                # reproducibility. It sits inside the failure guard because it
                # calls the pipeline: an exception raised here is an exception
                # raised by the run, and must be logged and fatal like any
                # other.
                if determinism is None and clip_index == 2:
                    determinism = check_determinism(extractor, video)
                    self._log(f"  determinism: {determinism.detail}")

                start = time.perf_counter()
                result = extractor.extract(video)
                del result
                elapsed = time.perf_counter() - start
            del video

            deep = clip_index % endurance.tracemalloc_interval_clips == 0
            sample = self._sampler.sample(count_objects=deep)
            samples.append(sample)
            durations.append(elapsed)

            phase = "warmup" if clip_index <= warmup else "measured"
            if clip_index > warmup:
                post_warmup_rss.append(sample.rss_mb)

            uss = "n/a" if sample.uss_mb is None else f"{sample.uss_mb:10.1f}"
            self._log(
                f"{clip_index:>6}  {sample.rss_mb:>10.1f}  {uss:>10}  "
                f"{elapsed:>9.3f}  {phase}"
            )

            self._metrics.write(
                {
                    "mode": "steady",
                    "manifest_sha": self._manifest_sha,
                    "clip": clip_index,
                    "phase": phase,
                    "duration_s": round(elapsed, 6),
                    "config_sha": self._config.config_sha(),
                    **sample.as_dict(),
                }
            )

            if deep:
                for line in self._sampler.allocation_diff():
                    self._log(f"    tracemalloc {line}")

        latency = summarize_latency(durations)
        leak = evaluate_leak(
            post_warmup_rss,
            seconds_per_clip=latency.mean_ms / 1000.0 if latency.count else 0.0,
            threshold_mb_per_hour=endurance.leak_mb_per_hour_max,
        )
        return RunOutcome(
            mode="steady",
            clips_completed=clips,
            leak=leak,
            latency=latency,
            determinism=determinism,
            samples=samples,
        )

    # -- re-init cycles ---------------------------------------------------

    def _run_reinit(self, cycles: int) -> RunOutcome:
        """Construct and destroy the extractor repeatedly.

        RSS is sampled after the extractor is dropped and the collector has
        run, so what the regression sees is memory the load/unload path failed
        to give back — not the steady-state working set.
        """
        endurance = self._config.endurance
        rng = np.random.default_rng(self._config.runtime.seed)

        self._log(f"Memory fidelity: {platform_fidelity_note()}")
        self._log("")
        header = f"{'Cycle':>6}  {'RSS (MB)':>10}  {'USS (MB)':>10}  {'Time (s)':>9}"
        self._log(header)
        self._log("-" * len(header))

        durations: list[float] = []
        post_warmup_rss: list[float] = []
        samples: list[MemorySample] = []

        for cycle in range(1, cycles + 1):
            start = time.perf_counter()
            with self._fail_on_exception(cycle):
                extractor = self._factory()
                video = make_synthetic_clip(self._config, rng)
                result = extractor.extract(video)
                del result
                del video
                del extractor
            gc.collect()
            elapsed = time.perf_counter() - start

            sample = self._sampler.sample(count_objects=True)
            samples.append(sample)
            durations.append(elapsed)
            if cycle > endurance.warmup_clips:
                post_warmup_rss.append(sample.rss_mb)

            uss = "n/a" if sample.uss_mb is None else f"{sample.uss_mb:10.1f}"
            self._log(f"{cycle:>6}  {sample.rss_mb:>10.1f}  {uss:>10}  {elapsed:>9.3f}")
            self._metrics.write(
                {
                    "mode": "reinit",
                    "manifest_sha": self._manifest_sha,
                    "cycle": cycle,
                    "duration_s": round(elapsed, 6),
                    "config_sha": self._config.config_sha(),
                    **sample.as_dict(),
                }
            )

        latency = summarize_latency(durations)
        leak = evaluate_leak(
            post_warmup_rss,
            seconds_per_clip=latency.mean_ms / 1000.0 if latency.count else 0.0,
            threshold_mb_per_hour=endurance.leak_mb_per_hour_max,
        )
        return RunOutcome(
            mode="reinit",
            clips_completed=cycles,
            leak=leak,
            latency=latency,
            samples=samples,
        )

    # -- shared -----------------------------------------------------------

    @contextmanager
    def _fail_on_exception(self, index: int) -> Iterator[None]:
        """Log a full traceback and abort the run on any exception.

        There is deliberately no tolerance counter. A soak test that survives a
        failure is measuring the memory behaviour of a broken pipeline and
        reporting it as a result.
        """
        try:
            yield
        except Exception:
            self._log("")
            self._log(f"FATAL: iteration {index} raised. Aborting the run.")
            for line in traceback.format_exc().rstrip().splitlines():
                self._log(f"  {line}")
            raise

    def close(self) -> None:
        self._sampler.stop()


def report(outcome: RunOutcome, log: RunLogger) -> None:
    """Write the human-readable summary block."""
    log("")
    log("=" * 72)
    log(f"SUMMARY ({outcome.mode})")
    log("=" * 72)
    log(f"Iterations completed : {outcome.clips_completed}")
    log(f"Latency              : {outcome.latency.summary()}")
    log(f"Memory gate          : {outcome.leak.summary()}")
    log(f"  reason             : {outcome.leak.reason}")
    if outcome.determinism is not None:
        status = "PASS" if outcome.determinism.passed else "FAIL"
        log(f"Determinism          : {status} — {outcome.determinism.detail}")
    else:
        log("Determinism          : not checked (run was too short)")

    if outcome.leak.verdict is Verdict.INCONCLUSIVE:
        log("")
        log(
            "NOTE: the memory gate is INCONCLUSIVE, which is not a pass. "
            "Run more clips to get a usable trend."
        )
    log("=" * 72)


def resolve_log_paths(config: IronConfig) -> tuple[Path, Path]:
    """Return ``(log_path, metrics_path)``, creating the directory."""
    log_dir = config.paths.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return (
        log_dir / config.endurance.log_filename,
        log_dir / config.endurance.metrics_filename,
    )


def execute(
    config: IronConfig,
    mode: str,
    iterations: int,
    factory: ExtractorFactory | None = None,
) -> ExitCode:
    """Run the harness end to end and return the process exit code.

    Separated from the CLI so tests can drive a full run with an injected
    factory and assert on the exit code without spawning a process.
    """
    # Validated before any file is opened. An unrecognised mode is a caller
    # bug, not a run outcome, and must not be laundered into an exit code that
    # looks like a pipeline failure.
    if mode not in ("steady", "reinit"):
        raise ValueError(f"unknown mode {mode!r}; expected 'steady' or 'reinit'")

    log_path, metrics_path = resolve_log_paths(config)
    applied = apply_runtime_settings(config)

    with open(log_path, "w") as log_handle, open(metrics_path, "w") as metrics_handle:
        log = RunLogger(log_handle)
        metrics = JsonlWriter(metrics_handle)

        log("=" * 72)
        log("Endurance run")
        log("=" * 72)
        log(f"Mode        : {mode}")
        log(f"Iterations  : {iterations}")
        log(f"Config sha  : {config.config_sha()}")
        log(f"Clip shape  : {list(config.pipeline.clip_shape)}")
        log(f"Leak limit  : {config.endurance.leak_mb_per_hour_max} MB/hour")
        log(f"Seed        : {applied.seed}")
        for skipped in applied.skipped:
            log(f"  NOT APPLIED: {skipped}")
        log(f"Metrics     : {metrics_path}")
        log("")

        # The manifest is written before any processing. A run that cannot
        # record what produced its results must not produce them: the outputs
        # would be unattributable to a model version, config, or commit, and
        # therefore unusable for any later comparison.
        try:
            manifest = RunManifest.capture(config)
            manifest_path = manifest.write(
                config.paths.resolved_output_dir / "manifest.json"
            )
        except ManifestError as exc:
            log(f"ERROR: {exc}")
            return ExitCode.MANIFEST_UNWRITABLE

        log(manifest.summary())
        log(f"manifest     : {manifest_path}")
        if manifest.git.available and manifest.git.dirty:
            log(
                "WARNING: the working tree is dirty, so this run cannot be "
                "reproduced from its commit alone."
            )
        log("")

        runner = EnduranceRunner(
            config,
            factory or default_extractor_factory(config),
            log,
            metrics,
            manifest.manifest_sha,
        )
        try:
            outcome = runner.run(mode, iterations)
        except WeightsUnavailable as exc:
            log(f"ERROR: {exc}")
            log("Fetch weights with: python scripts/fetch_weights.py")
            return ExitCode.PREREQUISITES_MISSING
        except Exception:
            # The per-iteration guard has already logged a traceback for
            # anything raised inside the loop. This catches whatever it could
            # not reach — extractor construction, teardown — and logs there
            # too, so no failure path exits without leaving evidence.
            if "Traceback" not in log_path.read_text():
                for line in traceback.format_exc().rstrip().splitlines():
                    log(f"  {line}")
            return ExitCode.PIPELINE_RAISED
        finally:
            runner.close()

        report(outcome, log)
        return outcome.exit_code()
