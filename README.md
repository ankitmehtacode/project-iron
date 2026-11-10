# Project Iron

[![CI](https://github.com/ankitmehtacode/project-iron/actions/workflows/ci.yaml/badge.svg)](https://github.com/ankitmehtacode/project-iron/actions/workflows/ci.yaml)

Fixed-camera video in, per-person tracks and zone events out. One pipeline
runs today, per camera, on synthetic footage:

```
video -> motion gate -> person detector -> tracker -> floor projection
      -> position estimator -> event compiler -> events.parquet -> Inspector
```

| Stage | Module | What it is |
| --- | --- | --- |
| Motion gate | `src/cascade/motion.py` | MOG2 at 320x180; the detector runs only when it wakes |
| Detector | `src/detect/` | RT-DETRv2-S (COCO, CPU, licence-gated registry weights), or the ground-truth oracle |
| Tracker | `src/pipeline/tracker.py` | Two-stage IoU association on Kalman-predicted boxes, per camera |
| Floor projection | `src/pipeline/floor.py` | Box bottom-centre to floor metres: calibration `K @ E`, or a four-point floor file |
| Estimator | `src/estimator/` | Constant-velocity Kalman per track (ADR 0010 config A) |
| Event compiler | `src/pipeline/events.py` | `entered`, `exited`, `dwelled` over world-coordinate zone polygons |
| Evaluation | `src/pipeline/evaluate.py` | Position RMSE, ID switches, event precision/recall, each against a baseline |
| Inspector | `src/inspector/` | Read-only local viewer over a run's outputs |

## What does not exist yet

- **Real footage.** Every number is NVIDIA PhysicalAI-SmartSpaces (synthetic,
  licence lane S), with calibration given. None is evidence about a real site.
- **Cross-camera identity.** The same person on two cameras is two subjects.
- **Calibrated scores.** Event `confidence` is the mean *uncalibrated* detector
  score of the track; it is not a probability.
- **A measured position-noise envelope.** The estimator uses an illustrative,
  unmeasured range-to-noise curve.
- **Timing claims.** Nothing here has been timed on reference hardware.
- **Other verbs.** The schema has 19; the compiler emits three.

The results and the open problems are in `docs/day43/REPORT.md`.

## Run it

Environment: Python 3.10, `pip install -r locking-requirements.txt` (CPU torch).
The data and weights come through the licence-gated registry
(`configs/datasets.yaml`); both licence signatures are already recorded.

```bash
# one SmartSpaces scene (calibration, ground truth, four 1080p videos)
python scripts/fetch_dataset.py PhysicalAI-SmartSpaces \
    --revision 2cbe9563cbe9f47f846e5c871ee994572bbbc60e \
    --hf-path "MTMC_Tracking_2026/val/Warehouse_022/*"
# detector weights, hash-checked against the registry on load
python scripts/fetch_dataset.py RT-DETRv2-S-COCO

S=data/raw/PhysicalAI-SmartSpaces/<commit>/<paths-dir>/MTMC_Tracking_2026/val/Warehouse_022

python -m iron run --video $S/videos/Camera_0001.mp4 \
    --calibration $S/calibration.json --out outputs/runs/my-run
python -m iron eval --run outputs/runs/my-run --ground-truth $S/ground_truth.json
python -m iron inspect --run outputs/runs/my-run      # http://127.0.0.1:8899
```

`--video` repeats for several cameras. `--detector oracle --ground-truth GT`
swaps in ground-truth boxes. `--gate off` runs the detector on every frame,
the reference for what the gate's sleeping costs. Zones default to
`configs/zones/warehouse_022.yaml`. A run directory is never overwritten.

Outputs: `manifest.json` (what ran, every sha), `tracks.parquet`,
`events.parquet` (event schema v2), and per camera `detections.npz` and
`cascade.jsonl` (gate and detector wake accounting). `eval` writes
`eval/eval.<stamp>.json` beside them.

## Develop

```bash
make test     # pytest, without weights-dependent tests
make lint     # black --check, flake8
make types    # mypy --strict over the scope in mypy.ini
```

CI runs the same three on pull requests and on pushes to `main` and
`foundation/**`.

Scorecards, run manifests and endurance logs are written to run-stamped names
and opened exclusively (`src/runstamp.py`): no run overwrites another's record.

## Directory structure

<!-- BEGIN TREE -->
<!-- regenerate with: python scripts/gen_tree.py -->

```
project-iron/
├── .claude/
│   └── skills/
│       ├── iron-cascade-runtime/
│       │   └── SKILL.md
│       ├── iron-contracts/
│       │   └── SKILL.md
│       ├── iron-eval-discipline/
│       │   └── SKILL.md
│       ├── iron-events/
│       │   └── SKILL.md
│       ├── iron-model-export/
│       │   └── SKILL.md
│       ├── iron-privacy-security/
│       │   └── SKILL.md
│       ├── iron-provenance/
│       │   └── SKILL.md
│       └── iron-testing/
│           └── SKILL.md
├── .github/
│   └── workflows/
│       └── ci.yaml
├── artifacts/
│   ├── chirla/
│   │   └── day40_untrained_baselines.json
│   ├── pytest/
│   │   ├── day32_full.log
│   │   ├── day32_full.xml
│   │   ├── day32_gated.log
│   │   ├── day32_gated.xml
│   │   ├── day32_summary.json
│   │   ├── day33_quick.log
│   │   ├── day33_quick.xml
│   │   ├── day34_quick.log
│   │   ├── day34_quick.xml
│   │   ├── day35_quick.xml
│   │   ├── day36_quick.xml
│   │   ├── day37_quick.xml
│   │   ├── day38_quick.xml
│   │   ├── day39_quick.xml
│   │   ├── day40_fresh_clone.log
│   │   ├── day40_quick.xml
│   │   ├── day41_postformat_full.log
│   │   ├── day41_postformat_full.xml
│   │   ├── day41_preformat_full.log
│   │   ├── day41_preformat_full.xml
│   │   ├── day42_skips_did_not_run.md
│   │   ├── day42_skips_full.log
│   │   ├── day42_skips_full.xml
│   │   ├── day42_skips_gated.log
│   │   ├── day42_skips_gated.xml
│   │   ├── day42_skips_suite_report.out
│   │   ├── day42_skips_summary.json
│   │   ├── day42_venv_pinned_full.log
│   │   ├── day42_venv_pinned_full.xml
│   │   ├── fresh_clone_234bedc5b8d4.log
│   │   ├── fresh_clone_234bedc5b8d4.xml
│   │   ├── fresh_clone_4f3fc42f3621.log
│   │   └── fresh_clone_4f3fc42f3621.xml
│   └── smartspaces/
│       ├── day41_component_sizes.json
│       ├── day41_gate_followup.json
│       ├── day41_gate_parity.json
│       ├── day41_gate_scorecard.json
│       ├── day41_integrity.json
│       ├── day41_reprojection.json
│       ├── day42_coco_export.json
│       ├── day42_detect_envelope.json
│       ├── day42_detection_zero_shot.json
│       ├── day42_gate_miss_by_size.json
│       ├── day42_rerender_transient.json
│       └── day42_validity_row.json
├── configs/
│   ├── envelope/
│   │   └── gate_320x180.envelope.json
│   ├── golden/
│   │   ├── v1-driving.golden.json
│   │   ├── v2-indoor.golden.json
│   │   ├── v3-indoor.golden.json
│   │   ├── v4-gate.golden.json
│   │   ├── v4.1-gate.golden.json
│   │   ├── v5-cessation.golden.json
│   │   └── v6-motion.golden.json
│   ├── preprocess/
│   │   ├── rtdetr_v2_r18vd.preprocess.json
│   │   └── vjepa2_vitl.preprocess.json
│   ├── zones/
│   │   └── warehouse_022.yaml
│   ├── datasets.yaml
│   └── default.yaml
├── docs/
│   ├── adr/
│   │   ├── 0001-identity-adapter-architecture.md
│   │   ├── 0002-clip-length-policy.md
│   │   ├── 0003-event-class-hierarchy.md
│   │   ├── 0004-coverage-absence-primitives.md
│   │   ├── 0005-bitemporal-relationships.md
│   │   ├── 0006-episode-roles-not-interaction-primitive.md
│   │   ├── 0007-evidence-commitment-reproducibility-vs-erasure.md
│   │   ├── 0008-production-provenance-lost.md
│   │   ├── 0009-repository-history-divergence.md
│   │   ├── 0010-estimator-configuration.md
│   │   ├── 0011-multi-entity-factor-graph.md
│   │   ├── 0012-generic-adapters-on-lane-s.md
│   │   └── 0013-detector-fine-tuning-on-lane-s.md
│   ├── data_model/
│   │   └── v0.3.md
│   ├── day11/
│   │   ├── infinigen_empty_room_stderr.log
│   │   ├── infinigen_render_stall.log
│   │   ├── semantics_verdict.json
│   │   └── tracking_verdict.json
│   ├── day12/
│   │   ├── infinigen_throughput.md
│   │   ├── motion_gate_v3_with_baselines.json
│   │   └── semantics_gap_sweep.json
│   ├── day14/
│   │   ├── motion_gate_v3_gate_reframe.json
│   │   └── semantics_gap_sweep_rerun.json
│   ├── day16/
│   │   ├── motion_gate_v3_rescored.json
│   │   └── motion_gate_v4_gate.json
│   ├── day17/
│   │   ├── motion_gate_v3_rescored.json
│   │   ├── motion_gate_v4.1_gate.json
│   │   └── motion_gate_v4_gate_rescored.json
│   ├── day18/
│   │   ├── motion_gate_v3_rescored.json
│   │   └── motion_gate_v4.1_gate_corrected.json
│   ├── day20/
│   │   └── estimator_accuracy.json
│   ├── day21/
│   │   └── estimator_accuracy.json
│   ├── day42/
│   │   └── hypotheses.md
│   ├── blocker_ledger.yaml
│   ├── capture_runbook.md
│   ├── chirla_published_baselines.md
│   ├── chirla_verification_checklist.md
│   ├── dismissal_audit.md
│   ├── reference_hardware.md
│   ├── registry_deployment_scope_audit.md
│   ├── site_zero_consent_TEMPLATE.md
│   ├── smartspaces_integrity.md
│   └── weight_status.md
├── iron/
│   ├── __init__.py
│   └── __main__.py
├── outputs/
│   └── perception/
│       ├── depth_v3-indoor_1c9a975e3ee5.json
│       └── validity_matrix.json
├── scripts/
│   ├── associate_golden_set.py
│   ├── blocker_report.py
│   ├── build_calibration_set.py
│   ├── capture_dry_run.py
│   ├── cascade_bench.py
│   ├── check_lockfile.py
│   ├── chirla_reference_check.py
│   ├── cvat_project.py
│   ├── demo_events.py
│   ├── discover_cameras.py
│   ├── env_gate.py
│   ├── eval_chirla_baselines.py
│   ├── eval_depth.py
│   ├── eval_detection_zero_shot.py
│   ├── eval_estimator.py
│   ├── eval_joint_estimator.py
│   ├── eval_report.py
│   ├── eval_semantics.py
│   ├── eval_tracking.py
│   ├── export_cotracker3_onnx.py
│   ├── export_depth_anything_onnx.py
│   ├── export_vjepa_onnx.py
│   ├── export_vjepa_ov.py
│   ├── fetch_dataset.py
│   ├── fetch_weights.py
│   ├── fit_detect_envelope.py
│   ├── fresh_clone_check.sh
│   ├── gen_synthetic_indoor.py
│   ├── gen_tree.py
│   ├── infinigen_generate.py
│   ├── ingest_capture.py
│   ├── make_golden_vectors.py
│   ├── measure_acceleration_noise_floor.py
│   ├── measure_component_cap_cost.py
│   ├── measure_component_sparsity.py
│   ├── measure_envelope.py
│   ├── measure_gt_acceleration_distribution.py
│   ├── measure_gt_constraint_violations.py
│   ├── measure_slip_remedy_smoothing.py
│   ├── measure_smartspaces_components.py
│   ├── measure_substream_hypothesis.py
│   ├── migrate_events_v1_v2.py
│   ├── mock_pipeline.py
│   ├── probe_cycles_throughput.py
│   ├── prove_absence_query.py
│   ├── quantize_cotracker3.py
│   ├── quantize_depth_anything.py
│   ├── quantize_vjepa.py
│   ├── rebuild_index.py
│   ├── run_backbone_bakeoff_selftest.py
│   ├── run_erasure_drill_selftest.py
│   ├── run_infinigen_gate.py
│   ├── smartspaces_gate_followup.py
│   ├── smartspaces_gate_miss_by_size.py
│   ├── smartspaces_gate_parity.py
│   ├── smartspaces_integrity.py
│   ├── smartspaces_reprojection.py
│   ├── smartspaces_rerender_transient.py
│   ├── smartspaces_to_coco.py
│   ├── smartspaces_validity_row.py
│   ├── suite_report.py
│   ├── train_identity_adapter_selftest.py
│   ├── validity_matrix.py
│   ├── velocity_floor_binding_audit.py
│   ├── velocity_floor_frame_rate_sweep.py
│   ├── velocity_floor_pinning_audit.py
│   └── vjepa_wrapper.py
├── src/
│   ├── bench/
│   │   ├── __init__.py
│   │   └── environment.py
│   ├── cascade/
│   │   ├── __init__.py
│   │   ├── envelope.py
│   │   ├── motion.py
│   │   ├── runner.py
│   │   ├── stage.py
│   │   └── stages.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── fields.py
│   │   ├── frames.py
│   │   ├── geometry.py
│   │   ├── ground_truth.py
│   │   └── tokens.py
│   ├── data/
│   │   ├── converters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── ntu_skeleton.py
│   │   ├── __init__.py
│   │   ├── coco_export.py
│   │   ├── cvat_ingest.py
│   │   ├── da2k_adapter.py
│   │   ├── depth_eval.py
│   │   ├── golden.py
│   │   ├── registry.py
│   │   ├── scorecard.py
│   │   ├── smartspaces.py
│   │   └── validity.py
│   ├── detect/
│   │   ├── __init__.py
│   │   ├── backends.py
│   │   ├── contract.py
│   │   ├── envelope.py
│   │   └── preprocess.py
│   ├── endurance/
│   │   ├── __init__.py
│   │   ├── gates.py
│   │   ├── memory.py
│   │   └── runner.py
│   ├── estimator/
│   │   ├── __init__.py
│   │   ├── association_from_geometry.py
│   │   ├── consistency.py
│   │   ├── constraints.py
│   │   ├── diagnostics.py
│   │   ├── filter.py
│   │   ├── imm.py
│   │   ├── informativeness.py
│   │   ├── joint.py
│   │   ├── measurement_model.py
│   │   ├── motion_model.py
│   │   ├── regime.py
│   │   └── state.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── baselines.py
│   │   ├── intervals.py
│   │   └── retrieval.py
│   ├── events/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── enhanced_cotracker.py
│   │   └── projector_vectorized.py
│   ├── identity/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── backbone.py
│   │   ├── bakeoff.py
│   │   ├── checkpoint.py
│   │   ├── contracts.py
│   │   ├── erasure_drill.py
│   │   ├── lane_gate.py
│   │   ├── promotion.py
│   │   ├── selftest.py
│   │   └── trainer.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── rtcp.py
│   │   └── rtsp.py
│   ├── inspector/
│   │   ├── static/
│   │   │   ├── app.js
│   │   │   ├── index.html
│   │   │   └── style.css
│   │   ├── __init__.py
│   │   ├── artifacts.py
│   │   └── server.py
│   ├── interface/
│   │   ├── ui/
│   │   │   ├── components/
│   │   │   │   ├── Dashboard/
│   │   │   │   │   └── AnalyticsPanel.js
│   │   │   │   └── Visualizer/
│   │   │   │       └── GaussianSplat.js
│   │   │   ├── data/
│   │   │   │   └── raw/
│   │   │   │       └── test_video.mp4
│   │   │   ├── workers/
│   │   │   │   └── GaussianSorter.worker.js
│   │   │   ├── app.js
│   │   │   ├── index.html
│   │   │   ├── serve.py
│   │   │   ├── three.min.js
│   │   │   └── trajectory_data.json
│   │   ├── __init__.py
│   │   └── temporal_stitching.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── alert.py
│   │   ├── constraint.py
│   │   ├── coverage.py
│   │   ├── entity.py
│   │   ├── envelope.py
│   │   ├── episode.py
│   │   ├── events.py
│   │   ├── evidence.py
│   │   ├── frame_of_reference.py
│   │   ├── hypothesis.py
│   │   ├── measurement.py
│   │   ├── observation.py
│   │   ├── relationship.py
│   │   ├── ulid.py
│   │   ├── uncertainty.py
│   │   └── world.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cotracker3_wrapper.py
│   │   ├── dav2_wrapper.py
│   │   ├── model_wrapper.py
│   │   ├── preprocess.py
│   │   └── vjepa_wrapper.py
│   ├── pipeline/
│   │   ├── evaluate.py
│   │   ├── events.py
│   │   ├── floor.py
│   │   ├── run.py
│   │   ├── tracker.py
│   │   └── truth.py
│   ├── semantics/
│   │   ├── __init__.py
│   │   ├── patch_mapping.py
│   │   └── semantic_extractor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── parquet_writer.py
│   ├── __init__.py
│   ├── artifacts.py
│   ├── config.py
│   ├── main.py
│   ├── provenance.py
│   └── runstamp.py
├── tests/
│   ├── fixtures/
│   │   ├── cvat_miniature.xml
│   │   └── minimal_rtsp_server.py
│   ├── golden/
│   │   ├── clips/
│   │   │   ├── real_test_video.npz
│   │   │   ├── synthetic_seeded_noise.npz
│   │   │   ├── synthetic_spatial_gradient.npz
│   │   │   └── synthetic_tubelet_probe.npz
│   │   ├── reference/
│   │   │   ├── real_test_video.npz
│   │   │   ├── synthetic_seeded_noise.npz
│   │   │   ├── synthetic_spatial_gradient.npz
│   │   │   └── synthetic_tubelet_probe.npz
│   │   └── manifest.json
│   ├── conftest.py
│   ├── test_bench_environment.py
│   ├── test_biometric_data_hygiene.py
│   ├── test_blocker_ledger.py
│   ├── test_calibration_set.py
│   ├── test_cascade.py
│   ├── test_cascade_bench_artifact_gate.py
│   ├── test_check_lockfile.py
│   ├── test_coco_export.py
│   ├── test_coco_metrics.py
│   ├── test_config.py
│   ├── test_contracts_ground_truth.py
│   ├── test_contracts_properties.py
│   ├── test_converters.py
│   ├── test_cvat_roundtrip.py
│   ├── test_da2k_adapter.py
│   ├── test_data_model_citations.py
│   ├── test_data_registry.py
│   ├── test_dav2_depthfield.py
│   ├── test_depth_honesty.py
│   ├── test_detect_contract.py
│   ├── test_detect_envelope.py
│   ├── test_detect_pretrained.py
│   ├── test_discover_cameras.py
│   ├── test_endurance_gates.py
│   ├── test_env_gate.py
│   ├── test_envelope.py
│   ├── test_estimator_consistency.py
│   ├── test_estimator_constraints.py
│   ├── test_estimator_diagnostics.py
│   ├── test_estimator_filter.py
│   ├── test_estimator_imm.py
│   ├── test_estimator_informativeness.py
│   ├── test_estimator_joint.py
│   ├── test_estimator_models.py
│   ├── test_estimator_regime.py
│   ├── test_eval_baselines.py
│   ├── test_eval_estimator.py
│   ├── test_eval_joint_estimator.py
│   ├── test_events_schema.py
│   ├── test_falsification.py
│   ├── test_fetch_dataset.py
│   ├── test_gate_miss_by_size.py
│   ├── test_gate_reframe.py
│   ├── test_golden_sets.py
│   ├── test_golden_vectors.py
│   ├── test_identity_adapter.py
│   ├── test_identity_adr.py
│   ├── test_identity_bakeoff.py
│   ├── test_identity_erasure_drill.py
│   ├── test_identity_trainer.py
│   ├── test_ingest_capture.py
│   ├── test_inspector.py
│   ├── test_intervals.py
│   ├── test_known_bugs.py
│   ├── test_measure_component_cap_cost.py
│   ├── test_measure_component_sparsity.py
│   ├── test_migrate_events_v1_v2.py
│   ├── test_mint_acceptance.py
│   ├── test_model_alert.py
│   ├── test_model_constraint.py
│   ├── test_model_coverage.py
│   ├── test_model_episode.py
│   ├── test_model_events.py
│   ├── test_model_evidence.py
│   ├── test_model_hypothesis.py
│   ├── test_model_primitives.py
│   ├── test_model_registry.py
│   ├── test_model_relationship.py
│   ├── test_model_world.py
│   ├── test_motion_observability_split.py
│   ├── test_observability_partition.py
│   ├── test_patch_mapping.py
│   ├── test_pipeline_slice.py
│   ├── test_preprocess_spec.py
│   ├── test_provenance.py
│   ├── test_registry_writer.py
│   ├── test_report_did_not_run.py
│   ├── test_report_suite_provenance.py
│   ├── test_report_verdicts.py
│   ├── test_rerender_transient.py
│   ├── test_rtcp.py
│   ├── test_rtsp_ingest.py
│   ├── test_run_infinigen_gate.py
│   ├── test_runstamp.py
│   ├── test_scorecard_observability_proxy.py
│   ├── test_scorecard_rerender_exclusion.py
│   ├── test_smartspaces.py
│   ├── test_smartspaces_components.py
│   ├── test_smartspaces_reprojection.py
│   ├── test_substream_hypothesis.py
│   ├── test_suite_report.py
│   ├── test_synthetic_indoor.py
│   ├── test_temporal_stitching.py
│   ├── test_validity_detection.py
│   ├── test_validity_matrix.py
│   ├── test_validity_point_tracking.py
│   └── test_velocity_floor_frame_rate_sweep.py
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── architecture.md
├── conda_environment.yaml
├── endurance_run.py
├── FOUNDATION_REPORT.md
├── locking-requirements.txt
├── Makefile
├── mypy.ini
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── setup.py
```
<!-- END TREE -->

## Iron Inspector

`python -m iron inspect --run DIR` (or `make inspect` for the repository's own
artifacts) serves a local, read-only viewer at http://127.0.0.1:8899.

- **It reads real artifacts, only.** There is no mock data path; a test greps
  the serving code for fixture-shaped literals.
- **Absence is rendered, not elided.** A missing artifact shows the path it was
  looked for and the command that produces it. An unmeasured metric reads
  `unmeasured`, never `0`.
- **Every number names its source file.**
- **Incomparable scorecards are refused, not diffed.**
- **Standard library only**, bound to localhost.

| View | What it answers |
| --- | --- |
| **Run** | A slice run: per-camera detector runs, tracks and events, and each evaluation beside its baseline. |
| **Events** | The event log, each of the four event classes rendered distinctly (not by colour alone). |
| **Scorecard** | Gate metrics for the active synthetic golden set. |
| **Clip inspector** | Per frame: observability buckets, gate wake decisions, mover area against the envelope. |
| **Envelope** | The measured gate wake curve. |
| **Association / Identity**, **Coverage / Absence** | Association decisions and absence proofs over stored artifacts. |
| **Provenance** | Git sha and dirty flag, config, golden-set and envelope shas. |

## Licence

No licence is declared for this repository. Datasets and weights carry their
own terms, recorded per entry in `configs/datasets.yaml`; NVIDIA's terms for
SmartSpaces are unresolved, so its frames stay under the ignored `data/` tree.
