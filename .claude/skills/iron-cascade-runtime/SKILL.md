---
name: iron-cascade-runtime
description: Runtime architecture, compute budgets, threading, and latency law for project-iron's CPU-only cascade (motion gate → detector → tracker → semantics). MUST be consulted when writing or modifying pipeline execution, stage wake logic, queues, threading, batching, model loading, memory management, or anything performance-related. Triggers on: "slow", "optimize", "latency", "fps", "CPU usage", "threads", "queue", "pipeline stage", "orchestrator", "memory manager", "real-time", "parallelize". The product's Tier-1 economics live or die on the budgets in this file.
---

# Iron Cascade Runtime — Budgets Are Requirements, Not Aspirations

Tier 1 sells "a layer of intelligence on your existing hardware." That claim is exactly as
true as these budgets are met on the reference box (NUC-class, 8×720p streams).

## The Wake Hierarchy

Stage 0 motion gate (background model, ~zero cost, always on) → Stage 1 INT8 person/object
detector (wakes on motion) → Stage 2 tracker + metric geometry (wakes on detection) → Stage 3
heavy semantics: embeddings, pose, HOI (wakes per active track, per policy). Rules:

- Every stage logs its wake decisions and per-stage p50/p99 into `CascadeStats` JSONL. A stage
  without wake accounting is unfinished.
- Hysteresis on every gate (stay-awake N frames) — wake flapping costs more than staying awake.
- Policy (customer config) compiles to wake thresholds; customers never touch model internals.

## The Budgets (reference hardware, enforced in CI bench)

| Condition | Budget |
|---|---|
| Idle scene, per camera | < 3% of one core |
| 5-person scene, full pipeline, 8 cams | within T1 box total |
| Critical alert, glass-to-alert | p99 ≤ 800 ms (T3 fast lane) |
| Scripting events | p99 ≤ 2.5 s |
| Cross-camera clock sync | < 50 ms sustained |

A change that busts a budget is a regression regardless of what it improves — surface the
tradeoff, don't ship it quietly.

## Threading Law (CPU-only truth)

OpenVINO and PyTorch each grab all cores by default; together they oversubscribe and halve
throughput. `INFERENCE_NUM_THREADS` and `torch.set_num_threads` are set explicitly from
config at startup, with a documented core budget per stage. Never leave either at default.
INT8 determinism holds only at fixed thread count — perf experiments that change threads
invalidate golden-vector comparisons (see iron-model-export).

## Concurrency Model

Stage-parallel processes with **bounded queues** over shared memory — not sequential
load/unload per clip (model loading would dominate wall time), not unbounded queues (that's
an OOM with a delay). Backpressure policy is explicit: drop frames if required, and every drop
writes a `gap` record (see iron-provenance). Batch each stage across a whole clip where the
latency lane allows.

## Two-Lane Rule (T3)

Fast lane: lightweight detector path, sub-second alerts. Accurate lane: full pipeline, richer
events, seconds. Both write the same event log; an accurate-lane event may supersede a
fast-lane one via correction record. "No latency, highest precision" is implemented as two
lanes, because it is not implementable as one.

## Measurement Rules

Latency is reported p50/p95/p99/max, never mean alone. mmap'd model paging is measured at p99
on the reference disk. Memory-leak claims use the regression-slope gate from iron-testing, not
eyeballed RSS. The load/unload cycle gets its own leak loop — that's where OpenVINO leaks live.
