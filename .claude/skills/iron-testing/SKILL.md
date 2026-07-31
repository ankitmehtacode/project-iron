---
name: iron-testing
description: Test taxonomy, pytest markers, determinism, endurance/leak methodology, and fault injection for project-iron. MUST be consulted when writing ANY test, debugging flaky tests, adding CI jobs, testing memory stability, or verifying a bug fix. Triggers on: "write a test", "add tests", "test this", "flaky", "memory leak", "endurance", "soak", "stress test", "CI failing", "coverage", "fault injection", "determinism". Encodes how this repo proves things instead of asserting them.
---

# Iron Testing — How This Repo Proves Things

## The Test Taxonomy (pick deliberately; a change usually needs several)

| Kind | Lives in | Proves |
|---|---|---|
| Property (Hypothesis) | `test_contracts_properties.py` | algebraic invariants over generated inputs — transforms round-trip, composition, rejection of illegal states |
| Known-bug (red-first) | `test_known_bugs.py` | a suspected defect exists; stays as regression guard after the fix |
| Golden-vector | per wrapper | fixed input → stored reference embedding within tolerance; catches silent export drift |
| Boundary integration | one per module boundary | two teams' modules agree (e.g. checkerboard clip → full pipeline → known 3D position within tolerance) |
| Metric eval | `make eval` | scorecard on frozen golden/synthetic sets (see iron-eval-discipline) |
| Endurance/soak | `src/endurance` | memory + latency stability over hours on REAL footage |
| Fault injection | `tests/test_faults.py` | defined behavior under corrupt frame, truncated video, disk full, SIGTERM, model file missing, clock jump, camera reconnect |

## Markers (registered in pyproject; unregistered markers fail CI)

- `@pytest.mark.requires_weights` — skips with a clear reason when checkpoints absent; NEVER
  fails on missing weights, NEVER fake-passes. CI default runs `-m "not requires_weights"`.
- `@pytest.mark.known_bug` + `xfail(strict=False)` — encodes a confirmed/suspected defect with
  a comment naming the audit finding. When the fix lands, the xfail is REMOVED in the same PR
  so the test guards forever. An xfail that quietly starts passing is a finding — investigate.
- Weight-dependent + slow tests run nightly, not per-PR.

## Determinism Law

Seeds set from config (`np`, `torch`) at every entrypoint. Thread counts pinned (see
iron-cascade-runtime) — INT8 determinism is thread-count-dependent. The determinism test:
identical input twice → byte-identical or ≤1e-5 cosine, and the test states which. A flaky
test is a bug with a stack trace you haven't read yet; `@pytest.mark.flaky` retries are
forbidden as a fix.

## Memory-Leak Methodology (the only accepted one)

1. Warm-up pass establishes baseline (first inference allocates framework buffers — expected).
2. Fit linear regression of RSS vs iteration post-warmup; FAIL iff the **upper bound of the
   95% CI on the slope**, in MB/hour, exceeds the configured threshold. Magic-number deltas
   are rejected in review.
3. RSS alone is insufficient: also USS/PSS (`/proc/self/smaps_rollup`), `tracemalloc` diffs,
   GC object counts, `malloc_trim(0)` before readings, live OpenVINO infer-request count.
4. **Any exception fails the run.** No error tolerance counters.
5. Two loops always: steady-state inference AND construct/destroy re-init cycles — framework
   leaks live overwhelmingly in load/unload.
6. Endurance means hours of real footage with adversarial clips injected on schedule — not
   seconds of random noise. `np.random.rand` exercises no data-dependent branch and proves
   nothing about the pipeline.

## Test Data Rules

Synthetic GT (Kubric) for anything needing exact numbers; golden real clips for realism;
degenerate clips (all-black, blown-out, 1-frame, T%tubelet≠0) in every suite that touches
video. Test inputs that only cover the happy path are half a test.
