# Foundation Day 1 — Report

Branch `foundation/day-1`, ten objectives, one commit each, on top of baseline
`41e924c`. 81 files changed, +7,980 / −3,261.

Final state: **190 passed, 2 skipped, 2 xfailed**, 88.25% coverage on the new
packages against an 85% floor, `mypy --strict` clean on the six foundation
packages, and every CI command verified locally.

No model behaviour or numerical output was changed anywhere. The two `xfailed`
tests are confirmed defects left deliberately unfixed; the two `skipped` are
blocked on model weights that are not present in this environment.

---

## 1. Objectives

| # | Objective | Status | Commit | Deviations from spec |
|---|---|---|---|---|
| 1 | Repo hygiene | Done | `5ba8bf6` | Also converted **all** intra-project imports to `src.`-qualified, not only the three files with `sys.path` hacks — the transitive closure was required or the imports break. Also fixed a README reference to `scripts/pipeline_example.py`, which does not exist. |
| 2 | Typed configuration | Done | `6d04c4c` | Added a fourth section, `EnduranceConfig`. `config_sha` excludes `paths.project_root` — see §3. |
| 3 | Geometry contracts | Done | `351de37` | Property (d) is exhaustively parametrised over every non-metric unit rather than Hypothesis-sampled; the domain has two members, so enumeration is strictly stronger. A companion test fails if that list is ever emptied. |
| 4 | Red tests for known bugs | Done | `82e97d5` | Finding 5 tested against a characterisation replica, since the real logic was still inline; Objective 5 repointed it at the real harness. |
| 5 | Endurance harness rebuild | Done | `c1e9803` | `leak_mb_per_hour_max` lives on `EnduranceConfig`, not `RuntimeConfig`. Added an `INCONCLUSIVE` verdict — see §3. Dependencies from objectives 2 and 3 were declared here rather than incrementally. |
| 6 | Provenance manifest | Done | `7cb5137` | `manifest_sha` excludes the start timestamp, or it could not be stable across two calls as required. Adds exit code 4 for an unwritable manifest. |
| 7 | Event schema v1 | Done | `0bf209e` | None. |
| 8 | Cascade runtime skeleton | Done | `0d9a80d` | None. Bench passes its wake-rate assertion but surfaces a budget miss — see §4. |
| 9 | CI pipeline | Done | `a1e78f6` | Five jobs rather than four (added `cascade-bench`). `mypy.ini` does not pin `python_version` — see §3. Lint settings moved into shared config files. |
| 10 | This report | Done | (final commit) | — |

---

## 2. Known-bug findings

This is the most valuable output of the day. Five suspected defects were
encoded as tests written against the current code. **Three confirmed, one
confirmed and fixed, two blocked on weights** (finding 1 is both blocked and
partially corroborated by documentation evidence).

### Finding 3 — CONFIRMED: the encoder is fed un-standardised pixels

`tests/test_known_bugs.py::test_preprocess_normalization` → **XFAIL**

`src/semantics/semantic_extractor.py::_run_vjepa` passes its input array
straight into the V-JEPA2 OpenVINO IR. `extract()`'s docstring requires only
that input be "normalised to [0, 1]". V-JEPA2 was trained on
ImageNet-standardised input (mean 0.485/0.456/0.406, std 0.229/0.224/0.225).
Feeding it un-standardised pixels shifts every activation. Nothing raises, the
output shape is correct, and the embeddings look entirely reasonable.

Evidence, from the test's own output:

```
src/semantics/semantic_extractor.py: NO mean/std standardisation
src/models/vjepa_wrapper.py: 4 candidate line(s)
    src/models/vjepa_wrapper.py:19: self.processor = None
    src/models/vjepa_wrapper.py:29: self.processor = torch.hub.load(
    src/models/vjepa_wrapper.py:30: "facebookresearch/vjepa2", "vjepa2_preprocessor"
    src/models/vjepa_wrapper.py:69: video = self.processor(video)
scripts/vjepa_wrapper.py: NO mean/std standardisation
```

The sharpest part of this finding: **the correct code already exists and is not
what production runs.** `src/models/vjepa_wrapper.py` loads the official
`vjepa2_preprocessor` and applies it. That wrapper is unused. The path
`endurance_run.py` and `src/integration.py` actually reach is the OpenVINO one,
which has no preprocessing at all.

A methodological note, because it nearly went the other way: **the first
version of this test passed.** Its detector matched the word "normalis/ze", and
`semantic_extractor.py`'s docstring says input "Must be normalised to [0, 1]".
Scaling into [0, 1] is a range conversion, not subtracting a channel mean and
dividing by a channel standard deviation. The detector was reporting a live
defect as absent, which is worse than having no test. It now matches only
ImageNet constants, `Normalize(`, a preprocessor call, or `mean` and `std`
together, and `test_standardisation_detector_ignores_range_scaling` fails if it
ever accepts a bare range remark again.

### Finding 4 — CONFIRMED: relative disparity is published as metres

`tests/test_known_bugs.py::test_depth_output_declares_units` → **XFAIL**

Depth-Anything-V2 emits relative inverse depth on an arbitrary per-frame scale.
`DAv2Wrapper.predict` returns `{"depth": ...}` — a bare array with no units
declaration. That array flows into
`projector_vectorized.project_points_to_3d`, which treats it as `Z` in camera
space, and out to the Parquet `z` column, which is documented as:

```
README.md:230: | z | float32 | Depth (meters) |
```

Every 3D coordinate downstream therefore has arbitrary scale while claiming to
be metric. No constant fixes this: relative disparity has unknown scale **and**
unknown shift, so converting it without an anchor fabricates a metric result.

Two aggravating factors found while confirming this:

- `projector_vectorized.compute_intrinsics` invents the focal length as
  `fx = fy = max(H, W)`. It is not measured, not calibrated, and not recorded
  anywhere. Even with correctly-scaled depth, the resulting 3D points would be
  wrong.
- The same module samples depth with `np.round` to nearest pixel, which is the
  right choice, but averages nothing across depth discontinuities — worth
  preserving when this is migrated to the contract layer.

### Finding 5 — CONFIRMED, then FIXED in Objective 5

`tests/test_known_bugs.py::test_legacy_error_counter_was_cumulative_not_consecutive`
records the defect; `test_endurance_error_counter_semantics` now guards the fix.
Both **pass**.

`endurance_run.py` logged `"Aborting: too many consecutive errors"` but never
reset the counter after a successful clip. Measured against the replica: six
isolated failures, each immediately recovered from, aborted the run **at clip
16**. Fixed by deleting the tolerance entirely rather than making it count
consecutively — per iron-testing, any exception fails the run. The `known_bug`
marker was removed in the same change, so the test now guards the behaviour
permanently.

### Finding 1 — BLOCKED on weights (documentation evidence is unambiguous)

`tests/test_known_bugs.py::test_vjepa_token_count_matches_tubelet` → **SKIPPED**
(`openvino is not installed in this environment`)

The runtime check could not execute. The documentary conflict is not in doubt:
`semantic_extractor.py` documents V-JEPA2 output as `[B, T*196, 1024]` and
draws a data-flow diagram with one temporal slot per input frame. V-JEPA2 is a
tubelet encoder with tubelet 2, so a 4-frame clip yields **2** temporal slots,
not 4. When weights are available the test prints the observed shape against
both hypotheses, so whichever is true, the failure carries its own evidence.

Corroborating: `scripts/fetch_weights.py` fetches
`facebook/vjepa2-vitl-fpc64-256` — a **64-frame, 256px** checkpoint. The
pipeline feeds it 4 frames at 224px. That is a third, independent
inconsistency, and it is why Objective 2 preserved `clip_frames=4` and
`clip_h=clip_w=224` verbatim instead of "fixing" them: changing them is a
measured change, not a refactor.

### Finding 2 — BLOCKED on weights

`tests/test_known_bugs.py::test_patch_mapper_temporal_alignment` → **SKIPPED**

Static analysis of `_map_tracks_to_embeddings` shows the mechanism plainly:

```python
t_out = min(t, T_out - 1)
```

With 4 input frames and 2 temporal slots this sends frames 1, 2 and 3 all to
slot 1. Frame 1's semantics come from a slot covering frames 2–3. The test
probes with a clip that is black for frames 0–1 and white for frames 2–3 —
exactly the tubelet boundary — and asserts that frames sharing a tubelet share
their semantics.

Independent of the runtime check, `PatchTokens` in `src/contracts/tokens.py`
now makes this shape unrepresentable, and
`test_patch_tokens_rejects_the_repos_documented_shape` **passes**: constructing
tokens with the exact shape the repo documents (`[4, 196, 1024]` for a 4-frame
clip) is rejected with a message naming the `tubelet=1` assumption.

---

## 3. Design decisions that deviate from the brief

Each of these was a deliberate choice against a literal reading of the spec.

**`config_sha` excludes `paths.project_root`.** It is an absolute,
machine-specific path. Including it would give the same configuration different
hashes on a laptop and in CI, making every cross-machine run comparison
invalid — the opposite of what the hash is for. Hostname and absolute paths are
recorded in the run manifest instead.

**`manifest_sha` excludes the start timestamp.** The spec asked for a hash
"over all of it" and separately required the manifest to be stable across two
calls in an unchanged repo. Those are incompatible if the clock is included.
The hash answers "was this the same setup?", not "was this the same moment".

**The leak gate has three outcomes, not two.** `INCONCLUSIVE` exists because a
3-clip smoke run is not a leak test. Reporting it as PASS claims a clean bill of
health that was not earned; reporting FAIL is a false alarm, and false alarms
are how leak gates end up switched off. Runs with fewer than 10 post-warm-up
samples report INCONCLUSIVE and exit 0 with the status stated prominently.

**`leak_mb_per_hour_max` lives on `EnduranceConfig`, not `RuntimeConfig`.** It
describes a test harness, not the inference runtime. An operator tuning thread
counts should not have to read past leak thresholds to find them.

**`mypy.ini` does not pin `python_version`.** Pinning to the 3.10 floor makes
mypy parse installed stubs under 3.10 rules, and current numpy stubs use PEP 695
`type X = ...` syntax — so the pin fails inside numpy before reaching any
project code. mypy now analyses against the interpreter it runs on and CI pins
that interpreter to 3.10, so the floor is still checked where it is enforced.

---

## 4. Cascade bench numbers

`scripts/cascade_bench.py`, 60 s at 12 fps, 1280×720, static for 40 s then a
moving blob for 20 s. Backend: MOG2.

```
interval 0: 720 frames in 25.88s (27.8 fps)
  motion_gate      woke    720/720    (100.0%)  p50  24.682 ms  p99  28.701 ms

Woke next stage on : 240/720 frames (33.3%)
Expected           : ~33.3% (+/- 10 points, plus hysteresis)

Stage-0 cost per frame:
  p50 24.682 ms | p95 27.166 ms | p99 28.701 ms | max 73.450 ms
  29.62% of one core per camera at 12 fps (budget: under 3% idle)

PASS: wake rate 33.3% is within tolerance.
```

**The wake logic is correct and the cost is not.** 33.3% is exactly right. But
29.62% of one core per camera misses the Tier-1 idle budget of 3% by roughly
ten times, and at eight cameras that is 2.4 cores consumed before a single
detector runs.

Measured alternatives on the same sequence:

| Resolution / backend | p50 / frame | % of one core | wake rate |
|---|---|---|---|
| 1280×720 mog2 | 24.6 ms | 29.5% | 35% |
| 1280×720 frame-diff | 20.3 ms | 24.4% | **5.4%** |
| 640×360 mog2 | 6.0 ms | 7.2% | 35% |
| 320×180 mog2 | 2.0 ms | 2.3% | 35% |
| 320×180 frame-diff | 1.1 ms | 1.3% | 35% |

Gating at 320×180 meets the budget and, on this sequence, wakes on exactly the
same frames — motion gating does not need the resolution a detector needs. The
change was not made today because it alters gate behaviour and belongs in a
measured change against real footage rather than a synthetic blob.

Second result from the same table: frame differencing at 720p wakes on 5.4% of
frames against MOG2's 35%, because it sees only the moving *edges* of the blob
and the foreground fraction falls under the threshold.
`min_foreground_fraction` is therefore **not portable across resolutions or
backends** and must be re-tuned whenever either changes.

---

## 5. Endurance gate demo

The gate catching a synthetic leak, driven through the real `execute()` path
with an extractor that retains 2 MB per clip:

```
SUMMARY (steady)
Iterations completed : 40
Latency              : p50 0.0 ms | p95 0.0 ms | p99 0.0 ms | max 0.0 ms | mean 0.0 ms over 40 clips
Memory gate          : FAIL: +9095080.2 MB/hour (95% CI upper bound) against a 50.0 MB/hour limit;
                       point estimate +7534922.6 MB/hour, slope +0.0337 MB/clip over 39 clips, R^2=0.721
Determinism          : PASS — byte-identical embeddings across two runs of the same input
EXIT CODE: 1 -> GATE_FAILED
```

The absolute MB/hour figure is inflated because the fake extractor runs in
microseconds, so the per-hour extrapolation is enormous. With realistic clip
durations the number is meaningful; the verdict is correct either way.

The prerequisites path, which is what a developer without checkpoints sees:

```
$ python endurance_run.py --clips 3
ERROR: weights not found at .../models/int8/vjepa2_vitl_int8.xml
Fetch weights with: python scripts/fetch_weights.py
EXIT CODE: 2
```

Clean message, no traceback, distinct exit code. CI asserts on this.

Gate unit tests (`tests/test_endurance_gates.py`, 32 tests, all pass) include a
regression guard for the rule this replaced: a steady 0.5 MB/clip leak over 40
clips grows 20 MB total, so it passed the old `total_growth > 150 AND
trend > 30` rule — while being 1800 MB/hour.

Manifest hashes were verified against an independent `shasum -a 256`, matching
byte for byte including the `.bin` weights that an `.xml`-only hash would miss.

---

## 6. Tomorrow — prioritised

**1. Fix the missing input standardisation (finding 3).** Highest value per
hour of work in the repo. The correct preprocessing already exists in
`src/models/vjepa_wrapper.py`; the OpenVINO path needs the same mean/std
applied. Unblocks: every embedding, the FAISS index built from them, and any
similarity-search result. Until this lands, every stored vector is
systematically wrong in the same direction, so nothing built on them can be
evaluated. Requires a golden-vector test and a before/after eval per
iron-eval-discipline — and note that fixing it **invalidates every existing
index**, so plan the rebuild in the same change.

**2. Resolve findings 1 and 2 by installing weights and running the two skipped
tests.** They are written, they skip cleanly, and they need one environment with
`openvino` and the checkpoints. Until they run, the temporal off-by-one is
strongly evidenced but not measured. Fixing it means routing the mapper through
`PatchTokens`, which already rejects the wrong shape. Unblocks: any per-frame
semantic query, and any event whose timing comes from embeddings.

**3. Declare depth units at the boundary (finding 4).** Wrap `DAv2Wrapper`
output in `DepthField(units="disparity_rel")` and let `unproject` refuse it.
This will *break* the projector, which is the point — it currently produces
confident, arbitrary-scale 3D. Needs a depth-anchoring step and real intrinsics
before metric output can be claimed at all; `compute_intrinsics` inventing
`fx = max(H, W)` must go at the same time.

**4. Move the motion gate to a downscaled frame (§4).** ~12× cost reduction,
measured, no observed change in wake behaviour. Needs validation on real
footage, not the synthetic blob, and `min_foreground_fraction` re-tuned for the
new resolution. Unblocks the Tier-1 idle budget, which is currently missed by
10×.

**5. Reconcile clip geometry with the checkpoint.** The fetched checkpoint is
64-frame/256px; the pipeline feeds 4 frames at 224px. Decide deliberately, with
numbers, and update `configs/default.yaml` plus the pinning test in
`tests/test_config.py` in the same change.

Lower priority but cheap: migrate `projector_vectorized` onto the contract
layer; add the first golden-vector test for each wrapper; start moving legacy
modules into the `mypy --strict` and coverage scopes one at a time.

---

## 7. Things in the repo that contradicted the brief

- **The repo was not a git repository.** No `.git`, no remote, no history. A
  baseline commit (`41e924c`) was created so the ten objective commits had
  something to sit on. Nothing is pushed; there is no configured remote.
- **`setup.py` was empty** (0 bytes), so `pip install -e .` could not have
  worked, and `src/` had no `__init__.py`. The brief assumed a working editable
  install.
- **The README's directory tree described a different project.** It listed
  `docs/`, `configs/`, `data/` and `output/` that did not exist, omitted
  `src/graph/`, `src/interface/ui/`, `src/rag_agent.py`, `src/integration.py`
  and `src/vector_database.py`, and pointed at `scripts/pipeline_example.py`,
  which does not exist. It is now generated from `git ls-files`.
- **`psutil` was imported by `endurance_run.py` but never declared** in any
  requirements file, so the soak test could not run from a clean install.
- **Imports were broken repo-wide, not just in the three files with `sys.path`
  hacks.** `from model_wrapper import ModelWrapper` and similar flat imports
  only ever resolved because `src/` was injected onto `sys.path`. The full
  transitive closure had to be converted.
- **The environment has no torch, OpenVINO, or OpenCV-dependent model weights.**
  Two tests skip for that reason, stated explicitly rather than passed over.
- **`black` 23.12.1, pinned in `.pre-commit-config.yaml`, disagrees with current
  black** on five files, and the pre-existing `src/geometry/pipeline/metadata_fusion.py`
  was already unformatted against its own pinned hook. The tree is now formatted
  to the pinned version.

---

## 8. What exists now that did not this morning

`src/contracts/` (6 modules) · `src/endurance/` (4) · `src/events/` (2) ·
`src/cascade/` (5) · `src/config.py` · `src/provenance.py` ·
`configs/default.yaml` · `scripts/gen_tree.py`, `demo_events.py`,
`cascade_bench.py` · `.github/workflows/ci.yaml` · `mypy.ini` · `.flake8` ·
`requirements-dev.txt` · six new test modules (194 tests).

The single most durable line of the day is in `src/contracts/tokens.py`:

```python
if n_temporal != expected_temporal:
    raise ValueError(...)
```

That assertion makes the temporal off-by-2 bug class structurally
unrepresentable rather than merely fixed.
