---
name: iron-provenance
description: Provenance, manifests, and artifact/version coupling for project-iron. MUST be consulted whenever code writes results, metrics, parquet rows, FAISS indexes, logs, model exports, calibration files, or reads them back — and whenever models, quantization, PCA matrices, preprocessing, or intrinsics change. Triggers on: "save results", "write parquet", "build index", "export model", "re-quantize", "recalibrate", "load index", "compare runs". Prevents the class of disaster where an index built by one model version is queried by another.
---

# Iron Provenance — A Result Without a Manifest Is Inadmissible

## The Manifest Rule

Every pipeline run writes `manifest.json` BEFORE processing begins: git sha + dirty flag,
sha256 of every model file loaded, `config_sha`, `preprocess_sha`, library versions, hardware,
thread settings, UTC start. A run refuses to start if the manifest can't be written. Every
output record (parquet row, JSONL metric, event) carries `manifest_sha`. Results lacking one
are inadmissible in any comparison, report, or debugging session — treat them as if they don't
exist.

## The Coupling Law (the disaster this skill prevents)

Derived artifacts are only valid against the exact producers that made them:

| Artifact | Coupled to | On mismatch |
|---|---|---|
| FAISS index | `encoder_sha` + `preprocess_sha` + `pca_sha` | query path REFUSES (raise, log, alert) |
| Parquet z values | `intrinsics_id` + `scale_shift_id` + `anchor_method` | rows flagged, excluded from metric queries |
| Camera registration | `twin_rev` | auto re-registration request |
| INT8 model | calibration-set sha | export invalid, re-gate required |

Re-exporting or re-quantizing a model **silently invalidates every vector in every index built
from it**. Therefore: index metadata stores the producer shas; the query path asserts equality
at load; rebuild is a scripted, timed, documented operation (`scripts/rebuild_index.py`), never
an ad-hoc notebook. If you touch the encoder, your PR checklist includes the index rebuild plan.

## Versioned-Artifact Rules

- Model checkpoints, calibration files, PCA matrices, twin reconstructions, golden sets:
  content-addressed, immutable, referenced by id/sha — never by "latest", never overwritten
  in place.
- Corrections propagate, they don't overwrite: identity revocations and track merges write
  correction records; the log is append-only. An event log that can be silently edited cannot
  be evidence.
- Dropped frames under backpressure write `gap` records. A timeline that omits what it dropped
  is a timeline that lies, precisely when it matters most.

## Comparing Runs

Two runs are comparable iff their manifests differ ONLY in the dimension under test. A speed
comparison across different thread settings, or an accuracy comparison across different
preprocess shas, is invalid — say so rather than reporting it.
