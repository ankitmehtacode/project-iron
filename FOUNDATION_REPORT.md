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

---

# Day 2

Branch `foundation/day-2`, five commits on top of `1134a38`. 27 files changed,
+2,638 / −180.

Final state: **261 passed, 6 deselected, 2 xfailed**, 88.51% coverage against an
85% floor, `mypy --strict` clean, every CI command verified locally.

**The model weights did not arrive.** There is no `models/` directory, no
checkpoint anywhere on the machine, and neither `torch`, `openvino`,
`transformers` nor `cotracker` is installed. Nothing was faked or stubbed. That
determined the shape of the day: Objectives 1 and 2 skip as specified,
Objective 3's authorized behavior change is transitively blocked, and
Objectives 4 and 5 were completed in full.

| # | Objective | Status | Commit |
|---|---|---|---|
| 1 | Resolve blocked verdicts `[W]` | **Blocked** — no weights | `351fe92` |
| 2 | Golden-vector reference `[W]` | Harness + fixtures shipped; references blocked | `81db1be` |
| 3 | Normalization fix | **Partial** — machinery shipped, fix deferred (see §D2.3) | `b3b94be` |
| 4 | Depth honesty | Complete | `afd09b0` |
| 5 | Cascade gate budget | Complete, measured, **passes** | `58e9d73` |
| 6 | This report | Complete | (final) |

## D2.1 — The two blocked verdicts: still blocked

Both tests ran and both **SKIPPED**. No tensor shapes were observed, so no
verdict is claimed. Recording what a test did not measure as though it had is
the failure mode this whole exercise exists to avoid.

```
test_vjepa_token_count_matches_tubelet   SKIPPED
  2 unmet prerequisite(s): module 'openvino' is not installed;
  V-JEPA2 OpenVINO IR not found at models/int8/vjepa2_vitl_int8.xml

test_patch_mapper_temporal_alignment     SKIPPED
  5 unmet prerequisite(s): module 'openvino' is not installed; module 'torch'
  is not installed; module 'cotracker' is not installed; V-JEPA2 OpenVINO IR
  not found at models/int8/vjepa2_vitl_int8.xml; CoTracker3 checkpoint not
  found at models/weights/cotracker3/scaled_offline.pth
```

**Verdict on tubelet: UNRESOLVED.** Hypotheses A (tubelet=2 → 392 tokens) and B
(tubelet=1 → 784 tokens) remain untested against the export. The documentary
conflict is unchanged and unambiguous — `semantic_extractor.py` claims
`[B, T*196, 1024]` for T=4 while V-JEPA2 is a tubelet-2 encoder — but a
docstring is not a measurement.

**Verdict on the patch-mapper off-by-2: UNRESOLVED by execution.** Per the
brief, the finer-granularity red test was NOT written, because that instruction
was conditional on the off-by-2 being *confirmed*, and it was not. Writing a
test pinning an exact wrong index mapping that nobody has observed would invent
the finding it claims to document.

What did improve: the skip helpers now report every unmet prerequisite in one
pass instead of the first one found, so bringing an environment up is a single
read rather than a rerun-and-discover cycle. Objective 1's real deliverable
today is that these two tests are ready to answer the question the moment
weights land.

## D2.2 — Golden vectors: fixtures shipped, references blocked

Four input clips are committed (1.1 MB, uint8, compressed) with a sha manifest:

| clip | purpose |
|---|---|
| `synthetic_spatial_gradient` | static; both temporal slots see identical content, so a reference where they differ indicts temporal handling rather than spatial encoding |
| `synthetic_tubelet_probe` | black frames 0–1, white 2–3, straddling the tubelet boundary exactly |
| `synthetic_seeded_noise` | unstructured control |
| `real_test_video` | first 4 frames of the repo's own test video — an export defect can be invisible on noise and obvious on a scene |

**Golden cosine: NOT MEASURED.** The 4 comparisons SKIP. No reference
embeddings exist because producing them requires the official PyTorch
checkpoint and `torch`, neither present. `tests/golden/manifest.json` records
the three blockers explicitly rather than leaving an empty `references` block
ambiguous.

**This is the "before" number Objective 3 needed, and it does not exist.**

7 fixture-integrity tests *do* run and pass today: every clip is verified
against its recorded sha, the committed geometry is asserted against the live
pipeline config, and the tubelet probe is checked to actually be
black-then-white. A fixture that drifts without its sha changing invalidates
every reference later built from it, so that is guarded now rather than
discovered later.

The comparison, when it runs, gates on the **1st percentile** of per-patch
cosine rather than the mean. A mean over 196 patches stays comfortable while
the worst 2% are unusable, and the worst patches are where the objects are.

## D2.3 — The normalization fix: why it is not in this diff

Everything the fix needs is built, tested and committed. The fix itself is not,
and that was a deliberate call against a literal reading of the objective.

Objective 3 was authorized as a measured behavior change that "ships WITH its
measurement". Its measurement is Objective 2's golden cosine. That measurement
could not be taken. Wiring preprocessing into `semantic_extractor._run_vjepa`
would therefore have been an **unmeasured change to numerical output**, which
the global rules forbid — and it would have been unverifiable in exactly the
way that let this defect exist for months.

What did ship, all of it behavior-neutral and unit-tested:

- **`PreprocessSpec`** carrying `frames, stride, resolution, mean, std,
  channel_order, resize_policy, tubelet, patch_size`, serialized as
  `<model>.preprocess.json` beside the weights. `apply()` is verified to
  standardise, to reject unscaled 0–255 input (which would make activations
  ~255× too large, silently), and to reject non-finite values.
- **No hardcoded constants.** `test_no_hardcoded_normalization_constants_in_source`
  greps every `.py` for the ImageNet literals and fails if they appear outside
  the JSON.
- **`load_for_model` refuses to fall back** to the canonical template when the
  sidecar is missing. A silent fallback applies another model's preprocessing
  and produces embeddings wrong in a way no shape or finiteness check detects.
- **The template is marked `verified_against_official_config: false`**, and
  `assert_ready_for_production()` refuses on it. The values were transcribed
  without the checkpoint present; a spec asserting numbers nobody checked is a
  guess wearing a manifest.
- **`preprocess_sha` in the run manifest**, schema 1.0 → 1.1. When no spec is
  loaded the manifest prints `NONE — encoder path has no spec wired in`, so the
  current state is visible rather than blank.

### VOID EMBEDDINGS DECLARATION

**Every embedding and derived artifact produced before commit `b3b94be` is
void.** They were produced by the encoder path that applies no channel
standardisation, and there is no way to detect that from the vectors
themselves.

This is enforced in code, not documented as a convention. Every derived
artifact now requires a `<name>.meta.json` naming `encoder_sha`,
`preprocess_sha`, `pca_sha` and `manifest_sha`; `require_compatible()` refuses
on any mismatch **and on a missing sidecar**, because absence means the artifact
predates the mechanism. `VectorDatabase.load()` checks before
`FAISS.load_local`, not after — once an index is loaded the first query has
already returned confidently-ranked results computed in the wrong space.

`scripts/rebuild_index.py --audit` currently reports:

```
0 artifact(s) carry provenance, 22 are void or unverifiable.
```

(20 depth maps, `tracks.npy`, `visibility.npy`, all under `outputs/`.) The
rebuild path itself deliberately exits 2 rather than running: regenerating
today would write a *second* generation of void vectors, because the encoder
still applies no standardisation.

## D2.4 — Depth honesty

Audit finding 4 was two independent silent lies feeding each other. Both are
now closed at the publication boundary.

| | before | after |
|---|---|---|
| Parquet column | `z`, documented "Depth (meters)" | `disparity_rel` + a `depth_units` column on every row |
| README:230 | `\| z \| float32 \| Depth (meters) \|` | corrected, with the old claim recorded above it |
| Projector output | `X`, `Y`, `Z` | `X_uncalibrated`, `Y_uncalibrated`, `disparity_rel` |
| Intrinsics | `fx = fy = max(H,W)`, silent | `calibrated: bool`; `placeholder_intrinsics()` sets it False |
| `unproject` with a guess | produced confident 3D | raises `UncalibratedIntrinsics` |

Writing metric depth is still possible and now requires saying so:
`depth_units="meters"` is a claim someone makes, not the default reading of an
unlabelled column. The `IRON_ALLOW_UNCALIBRATED=1` escape hatch exists for
experiments and logs `UNCALIBRATED GEOMETRY … ARBITRARY SCALE` at WARNING on
every call. It matches exactly `"1"` — a test pins that `"true"`, `"yes"` and
`"TRUE"` do **not** enable it, because an override accepting any non-empty
string would be switched on by someone setting it to `"false"`.

`rescaled_to` preserves the flag, so a resize cannot launder a guess into a
calibration.

Finding 4's known-bug test remains **red, correctly**: `DAv2Wrapper.predict`
still returns a bare ndarray with no units. Its xfail reason now states
precisely what was fixed and what remains, rather than continuing to read as
wholly unaddressed. Wrapping that return in a `DepthField` is an API change to
a legacy module with three call sites and belongs in its own commit.

## D2.5 — Cascade gate: small-target parity and final cost

**PASS.** Wake decisions identical at both resolutions on all four scenarios,
and stage-0 cost inside the Tier-1 idle budget.

```
scenario          frames  wake@full  wake@gate   parity   p50 ms   % core
static               120       0.0%       0.0%    EXACT    2.478    2.97%
near_target          360      33.3%      33.3%    EXACT    2.476    2.97%
SMALL_TARGET         360      33.3%      33.3%    EXACT    2.449    2.94%
real_test_video      250      84.8%      84.8%    EXACT    2.181    2.62%

Worst stage-0 cost : 2.97% of one core per camera
Product budget     : 3.00%
Speedup vs full-res: 3.1x median (floor 3.0x)
```

**Small-target parity: EXACT, no divergence, no threshold retuned.** The
scenario is a blob ~15 px tall at gate resolution — a person about 60 px tall at
720p, someone across a car park. This was the case that would have stopped the
objective, and it did not diverge.

The real-footage scenario provides an additional unlabelled parity check: 250
frames at 320×176, also EXACT. No expected wake share is claimed for it, because
nobody has labelled that footage and inventing ground truth would be worse than
having none.

Cost went 29.5% → 22.5% → 4.4% → 2.97% of one core, via: gating at 320×180;
resizing the colour frame *before* the greyscale reduction (linear operations
commute, so identical results, but it moves the bulk work into one OpenCV call);
and computing the channel mean with `cv2.transform` rather than `numpy.mean`
(1.01 ms → 0.13 ms, same arithmetic).

**Two things measured rather than assumed.** My first bench ran every frame
through the gate twice — once directly, once via `CascadeRunner` — inflating
cost and feeding the MOG2 background model a sequence the scenario never
contained. And the `cv2.transform` switch rounds where numpy truncated: a
one-grey-level difference that changed **no** wake decision on any synthetic
scenario and **2 of 250** on real footage. That is a real behaviour change,
small and toward correctness, and it is recorded rather than glossed.

Measured and rejected: pinning OpenCV threads (1/2/4/12 all within 2% of each
other) and disabling MOG2 shadow detection (2% faster, but flipped a decision).

**The margin is thin — 2.97% against 3.00%.** So CI enforces two things: the
absolute budget, with a `--budget-scale 2.0` for shared runners that does not
change the product budget and always prints the unscaled number; and a
machine-independent floor of 3× cheaper than full-resolution gating, which is
the assertion that actually catches someone removing the downscale on any
hardware.

## D2.6 — Deviations from the brief

- **Objective 3's fix deferred.** Its authorization was conditional on shipping
  with a measurement that no weights made impossible. Reasoning in §D2.3.
- **Objective 1's additional red test not written.** Conditional on the off-by-2
  being confirmed; it was not confirmed.
- **`compute_intrinsics` kept rather than replaced.** Three legacy call sites
  still use it. It is now loudly documented and points at the typed replacement;
  migrating the call sites is a separate change.
- **Gate greyscale changed rounding behaviour** (§D2.5). Measured, 2/250 frames
  on real footage, reported rather than hidden.
- **`CascadeConfig` added** to `src/config.py` for config-driven gate
  resolution, mirroring the day-1 `EnduranceConfig` precedent.

## D2.7 — Tomorrow, ordered

1. **Install weights and run three things, in this order:** the two blocked
   known-bug tests, then `scripts/make_golden_vectors.py` for the reference
   embeddings, then `pytest tests/test_golden_vectors.py` for the "before"
   cosine. Nothing else on this list can be done honestly first, and all three
   are already written and waiting.
2. **Land the normalization fix with its before/after cosine.** One call site in
   `semantic_extractor._run_vjepa`, plus flipping the golden test green and
   removing its `known_bug` marker in the same commit. Then verify the canonical
   `PreprocessSpec` against the model's own config and set
   `verified_against_official_config: true`.
3. **Implement the rebuild path** in `scripts/rebuild_index.py`, and delete or
   regenerate the 22 void artifacts. Only meaningful after step 2.
4. **Resolve the temporal off-by-2** by routing the mapper through
   `PatchTokens`, which already rejects the wrong shape. Blocked on step 1's
   verdict.
5. **Wrap `DAv2Wrapper.predict` in a `DepthField`** to close finding 4 at the
   source rather than only at publication.
6. **Reconcile clip geometry with the checkpoint** — the fetched model is
   64-frame/256px, the pipeline sends 4 frames at 224px. `PreprocessSpec` will
   now refuse the mismatch once a real spec sits beside the weights, so this
   becomes a hard failure rather than silent degradation.
7. Lower priority: migrate the three `compute_intrinsics` call sites; get real
   camera calibration so metric depth is achievable at all; move legacy modules
   into the `mypy --strict` and coverage scopes one at a time.

## D2.8 — What contradicted the brief

- **The weights were not placed.** The brief opened by stating they had been.
  No `models/` directory exists, no checkpoint file exists anywhere on the
  machine, and the four runtime libraries needed to use them are absent.
- **`torch` being absent also blocks the golden generator**, which was implicitly
  assumed available for Objective 2 even in the weights-present case.
- **Objective 3 is not marked `[W]`** but is transitively weights-dependent: it
  requires flipping a weights-gated test green and reporting a weights-gated
  cosine.

---

# Day 3

Branch `foundation/day-3`. **STOPPED AT OBJECTIVE 0 — the environment gate
failed.**

Per the brief's hard stop, no other objective was attempted. Nothing was
stubbed, no verdict was improvised, and no measurement is reported that was not
taken. Objectives 1–4 remain exactly where Day 2 left them: written, waiting,
and blocked on the same thing.

## D3.0 — Environment gate: FAILED, 8 of 9 checks

`python scripts/env_gate.py` → exit 1.

```
CHECK                        STATUS  DETAIL
----------------------------------------------------------------------------
import torch                 FAIL    not installed
import openvino              FAIL    not installed
import cv2                   FAIL    version 5.0.0, pinned 4.8.1.78
import numpy                 FAIL    version 2.5.1, pinned 1.26.2
model vjepa_xml              FAIL    missing at models/int8/vjepa2_vitl_int8.xml
model vjepa_bin              FAIL    missing at models/int8/vjepa2_vitl_int8.bin
model cotracker_checkpoint   FAIL    missing at models/weights/cotracker3/scaled_offline.pth
pip check                    PASS    No broken requirements found.
seeds / threads              FAIL    torch not importable
```

### The exact missing-item checklist

**A. Runtime — install the pinned stack**

```bash
pip install -r locking-requirements.txt
```

| item | required | found |
|---|---|---|
| `torch` | 2.2.0 (CPU index) | not installed |
| `openvino` | 2024.6.0 | not installed |
| `numpy` | 1.26.2 | **2.5.1** |
| `opencv-python-headless` | 4.8.1.78 | **5.0.0** |

Also needed but not gate rows: `transformers` (4.36.0) and CoTracker3 from
`github.com/facebookresearch/co-tracker`, both required by the golden-vector
generator and the temporal-alignment test.

**B. Model artifacts — none of the three exist; `models/` itself is absent**

```bash
python scripts/fetch_weights.py                 # official checkpoints
python scripts/export_vjepa_onnx.py             # -> models/onnx/
python scripts/quantize_vjepa.py                # -> models/int8/*.xml + *.bin
```

| path | status |
|---|---|
| `models/int8/vjepa2_vitl_int8.xml` | missing |
| `models/int8/vjepa2_vitl_int8.bin` | missing |
| `models/weights/cotracker3/scaled_offline.pth` | missing |
| `models/weights/vjepa2_vitl/` (official PyTorch, for golden references) | missing |

A filesystem search across the home directory for any file over 1 MB matching
`*.pth`, `*.safetensors`, `*vjepa*` or `*cotracker*` returned nothing outside
the virtualenv. The weights are not merely at a different path — they are not
on this machine.

**C. Determinism** — follows from A. Seeds and thread caps cannot be applied
while torch is absent, and INT8 determinism holds only at a fixed thread count.

### A new finding: the dev environment was never on the pinned versions

Rows 3 and 4 are not simply "absent" — they are **present at the wrong
versions**. numpy 2.5.1 against a pinned 1.26.2, and OpenCV 5.0.0 against a
pinned 4.8.1.78. Those are major-version gaps.

This matters beyond today, and it is a caveat on Day 2:

- **The Day-2 cascade timings are not reference numbers.** The 2.97%-of-a-core
  figure was measured on OpenCV 5.0.0. OpenCV's resize and MOG2 implementations
  changed between 4.x and 5.x, so that number does not transfer to a machine
  running the pinned stack. The *parity* result is unaffected — it compares two
  gate resolutions within one environment — but the absolute cost must be
  re-measured once the pinned versions are installed.
- **Any golden vector recorded here would have been non-reproducible.** A
  reference embedding is only a reference for the runtime that produced it, and
  recording one under unpinned numpy would have quietly poisoned every later
  comparison. The gate catching this before Objective 2 ran is the gate doing
  its job.

`pip check` passes, which is worth noting as a limitation rather than
reassurance: it verifies that installed distributions satisfy each other's
constraints, not that they match this project's pins. Only the gate checks that.

## D3.0b — Environment rebuilt: gate now 7 of 9

After the checklist above was acted on, the gate stands at **7 PASS / 2 FAIL**.
Still failing, so Day 3 remains stopped — but the remaining two rows are a
different kind of blocker from the eight that started the day.

```
CHECK                        STATUS  DETAIL
----------------------------------------------------------------------------
import torch                 PASS    torch 2.2.0 (pinned)
import openvino              PASS    openvino 2024.6.0 (pinned)
import cv2                   PASS    opencv-python-headless 4.8.1.78 (pinned)
import numpy                 PASS    numpy 1.26.2 (pinned)
model vjepa_xml              FAIL    missing — production INT8 IR
model vjepa_bin              FAIL    missing — production INT8 IR
model cotracker_checkpoint   PASS    101.9 MB  sha256 2670d4562ed69326...
pip check                    PASS    No broken requirements found.
seeds / threads              PASS    seed=0 numpy=yes torch=yes torch_threads=6
```

Environment: a fresh `.venv-pinned` on **Python 3.10.20**, every runtime at its
pinned version. The full suite passes on it: **272 passed, 6 skipped,
2 xfailed**. That is a real result — all prior work was developed against
numpy 2.5 and OpenCV 5, and it holds on numpy 1.26.2 and OpenCV 4.8.1.78.

Checkpoint provenance recorded at download, for Objective 4's export manifest:

| artifact | size | sha256 |
|---|---|---|
| `cotracker3/scaled_offline.pth` | 101.9 MB | `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` |
| `vjepa2_vitl/model.safetensors` | 1.2 GB | see `models/weights/hashes.txt` |

### Three defects fixed to get here

**1. The lockfile had never been installable.** `mlflow==2.9.2` pins
`pyarrow<15` while the same file pins `pyarrow==18.1.0`. Unsatisfiable, on
every machine and every Python version — `pip install -r
locking-requirements.txt` has failed with `ResolutionImpossible` since the file
was written. Removing that one line resolves the whole file with every other
pin untouched. Safe because `grep -rn mlflow --include=*.py` matches zero
files: it was an intention recorded in a requirements file, and it was blocking
the entire install.

**2. The gate itself was wrong.** It compared pins against each module's
`__version__` attribute, and reported a correctly-pinned environment as
failing — `cv2.__version__` is `"4.8.1"` where the distribution is `4.8.1.78`,
and `openvino.__version__` is `"2024.6.0-17404-4c0f47d2335-releases/2024/6"`
where the distribution is `2024.6.0`. It now reads `importlib.metadata`, which
is what pip actually resolved and recorded. This is the third instance of the
same pattern in three days: the instrument was broken, not the code, and a gate
that cries wolf gets ignored — which is indistinguishable from having no gate.
Pinned by `test_version_comes_from_distribution_metadata_not_module_attribute`.

**3. `fetch_weights.py` wrote outside the repository.** `SAVE_DIR =
"../models/weights"`, resolved against the working directory. Run from
`scripts/` it lands correctly; run from the repository root — the documented
location — it writes to a *sibling* of the repo, where nothing that reads paths
through the config would ever find it. Same CWD-dependence class fixed
repo-wide on Day 1; this file was missed because it had no `sys.path` hack to
grep for. It also passed `token=True`, forcing Hugging Face authentication for
repositories that are all public, which fails outright on a machine with no
cached login.

### Why the last two rows are not being cleared

The remaining failures are the **production INT8 artifact**, and it is
deliberately not being generated.

Exporting and quantizing a fresh IR would turn both rows green in about ten
minutes. It would also destroy Objective 1. That objective is a forensic
verdict on *the artifact production actually used* — the only evidence of what
the stored embeddings were computed with. An artifact built today is Objective
4's replacement, a different thing entirely, and running the tubelet and
temporal-alignment tests against it while calling the result "the production
verdict" would fabricate exactly the kind of confident-wrong answer this gate
exists to prevent.

So the production `vjepa2_vitl_int8.xml` and `.bin` must be copied from
whichever machine ran production, with `sha256sum` taken before transfer so
corruption cannot be mistaken for an export defect. That is a human action with
latency outside this repository, and it is now the single blocker for Day 3.

## D3.1 — What was delivered

`scripts/env_gate.py`, run as `python scripts/env_gate.py` (add `--json` for a
machine-readable fingerprint). It checks all four rows the brief specifies,
reports **every** failure with a specific remedy rather than stopping at the
first, and exits nonzero. When the models are present it records size and
sha256 for each, which is the environment fingerprint Objective 0 asks to be
folded into the manifest — that step is deliberately not taken today, because
writing a fingerprint of an environment that has no models would record an
absence as though it were a provenance record.

`tests/test_env_gate.py` — 7 tests, passing. They pin the two ways the gate
could fail open: accepting a wrong version (it must compare, not merely
import), and reporting only the first problem (it must list all of them). One
test constructs a real artifact in a tmp dir and asserts the sha256 matches an
independent hash, so the fingerprint path is exercised even with no weights on
the machine.

## D3.2 — Unchanged from Day 2

Every deferred item stands exactly where it was, for exactly the same reason:

1. Tubelet verdict — **UNRESOLVED**, never executed.
2. Patch-mapper off-by-2 verdict — **UNRESOLVED**, never executed.
3. Golden before-cosine — **NOT MEASURED**, no reference embeddings exist.
4. Normalization fix — **NOT LANDED**; its authorization requires a measurement
   that requires this gate.
5. 22 void artifacts — still void, still refused by the artifact guard.

## D3.3 — Day 4, when the gate passes

The order is fixed by dependency, not preference. Nothing below can be honestly
done before the thing above it.

1. `python scripts/env_gate.py` until it exits 0. Everything else is blocked on
   this, and a partial pass is not a pass.
2. Re-measure the cascade bench on the pinned stack and correct the Day-2
   absolute figures in `src/cascade/motion.py`. The parity conclusion should
   hold; the cost number may not.
3. Objective 1 — run the two forensic tests, record actual tensor shapes,
   settle the tubelet verdict as Outcome A or B.
4. Objective 2 — generate reference embeddings, record the before-cosine
   (mean and p1 per patch).
5. Objective 3 — the one call site, with its before/after table, and the
   negative test proving the void-artifact refusal fires.
6. Objective 4 — scripted manifested FP32 export, then re-run Objective 1
   against the new artifact for the side-by-side table.

Still queued behind those, from Day 2: the mapper fix (Day 4 at the earliest,
and only after the tubelet verdict, one variable at a time); real-footage
wake-parity as a CI addition; and `DAv2Wrapper.predict` returning a `DepthField`.


---

# Day 3 (resumed, post-amendment)

Gate reads **PROCEED** with the production-artifact rows parked. Objectives 4,
2 and 3 ran in that order; Objective 1 stays parked.

**279 passed, 2 skipped, 1 xfailed** on the pinned venv (Python 3.10.20, torch
2.2.0, openvino 2024.6.0, numpy 1.26.2, opencv 4.8.1.78). `mypy --strict` clean.

## D3.4 — The headline: normalization fix, measured

Per-patch cosine against the official PyTorch reference, all four golden clips:

| | BEFORE | AFTER |
|---|---|---|
| **p1** (gated) | **0.331903** | **0.999987** |
| p50 | 0.746654 | 1.000000 |
| mean | 0.715315 | 0.999999 |
| min | 0.281222 | 0.999672 |

Per clip, before → after (p1):

| clip | before | after |
|---|---|---|
| `real_test_video` | 0.331928 | 1.000000 |
| `synthetic_seeded_noise` | 0.356102 | 1.000000 |
| `synthetic_spatial_gradient` | 0.324355 | 1.000000 |
| `synthetic_tubelet_probe` | 0.360102 | 0.999963 |

A p1 of 0.33 means the worst 1% of patches were nearly orthogonal to what the
model should have produced. The stop condition was p1 < 0.98 after the fix,
which would have meant a second export-level defect still in the path;
0.999987 clears it, so the fix is complete rather than partial.

The change is one call site: `_run_vjepa` applies the model's `PreprocessSpec`.
`known_bug` and `xfail` were removed from both the golden test and
`test_preprocess_normalization` in the same commit, so they now guard forever.

## D3.5 — Tubelet verdict: answered at the reference level

**Tubelet = 2. Confirmed three independent ways, none of them the production
artifact.**

1. `config.json` from the official checkpoint: `tubelet_size: 2`.
2. Reference forward pass, 4 frames @ 224px → `(1, 392, 1024)`.
   392 = (4 // 2) × (14 × 14).
3. The fresh export's token-count check, which aborts on mismatch:
   `392 = 2 temporal × 196 spatial`.

The repository's documented `[B, T*196, 1024]` — 784 tokens for T=4 — is
**refuted**. Hypothesis A holds, Hypothesis B does not.

This is *not yet* the Objective-1 verdict. It establishes what the official
model does, which is the ground truth the production artifact should match. The
production artifact could still deviate, and that question stays parked until
the file arrives.

## D3.6 — Export manifest summary

`models/export/2026-07-31/` — never the production path; the script refuses.

| field | value |
|---|---|
| precision | FP32 (INT8 deferred until the encoder choice is frozen) |
| encoder | context, `get_vision_features`, eager attention |
| geometry | `(4 // 2) * (224 // 16)^2 = 392` tokens |
| source checkpoint | `model.safetensors` sha `25466aef85727d16...` |
| artifact | `vjepa2_vitl_fp32.xml` sha `5ca2d3da552016a6...`, 1.2 GB `.bin` |
| preprocess_sha | `ea11d709b9241e68...` |
| IR vs PyTorch | relative 6.7e-06, per-token cosine p1 1.000000 |

Exported at 4f/224px while the checkpoint is native at 64f/256px, so position
embeddings are interpolated — which is exactly what the token-count check
verifies. That geometry was chosen to hold every variable but normalization
fixed for today's measurement. It is not defensible as a long-term choice:
4 frames at 12 fps is a third of a second, a still image at video-model prices.

## D3.7 — Defects found in code written today

**The export traced and verified on `torch.zeros`.** All-zero input makes every
attention key identical, so softmax is uniform and tiny numerical differences
compound through 24 layers. The same artifact scored max|IR−ref| **2.47** and
cosine p1 **0.9953** on zeros, versus **8.3e-04** and **1.000000** on random
input. Judging the export by the zeros number would have condemned a correct
conversion. Now seeded random, and the deviation is an **abort** criterion
rather than a number the script printed and moved past.

**Two errors in the Day-2 canonical PreprocessSpec**, caught the moment values
were read from the checkpoint instead of transcribed: it claimed 224×224 and 4
frames where the checkpoint says `image_size: 256` and `frames_per_clip: 64`.
mean/std were right. This is precisely why that file shipped marked
`verified_against_official_config: false`.

**A tension between two of my own tests.**
`test_preprocess_normalization` looked for inline mean/std;
`test_no_hardcoded_normalization_constants_in_source` forbids exactly that.
Both could not pass until the detector learned that delegation to a
`PreprocessSpec` *is* the required form — and better evidence than a constant
in a file, because a spec can be verified against the checkpoint.

**Fixture reproducibility.** The three synthetic clips are bit-identical across
numpy 2.5 → 1.26. `real_test_video` changed, because `cv2.resize(INTER_AREA)`
differs between OpenCV 5.0.0 and the pinned 4.8.1.78. The real-footage fixture
is OpenCV-version-dependent; it is now regenerated under the pin, and any
future reference must be regenerated with it.

## D3.8 — Void artifacts: refusal verified against real files

`scripts/rebuild_index.py --audit` → `0 artifact(s) carry provenance, 22 are
void or unverifiable.`

A negative test now asserts the guard refuses **all 22** real stale files under
`outputs/`, each error naming `rebuild_index.py`, while a freshly stamped
artifact still loads — a guard that refuses everything is as useless as one
that refuses nothing.

Those 22 were produced by the path that scored cosine 0.332. They are not
"probably fine".

## D3.9 — Day 4, ordered

1. **Objective 1, verbatim, the moment the production `xml`/`bin` arrive.**
   Append the side-by-side table: production artifact vs the 2026-07-31 export,
   token count and temporal alignment for each. That is the only remaining
   unknown, and no export substitutes for it.
2. **The mapper off-by-2**, once (1) gives a verdict. `t_out = min(t, T_out-1)`
   is still in `_map_tracks_to_embeddings`, and `PatchTokens` already rejects
   the wrong shape. One variable per day: this did not move today because the
   normalization change did.
3. **Re-measure the cascade bench on the pinned stack.** Day 2's 2.97%-of-a-core
   was taken on OpenCV 5.0.0; the parity conclusion holds, the absolute number
   does not transfer.
4. **Reconcile clip geometry with the checkpoint** — 4f/224 against a native
   64f/256 model. `PreprocessSpec` now refuses a mismatch, so this is a hard
   failure waiting to happen rather than silent degradation.
5. **Delete or regenerate the 22 void artifacts**, and implement the rebuild
   path now that the encoder is fixed.
6. **`DAv2Wrapper.predict` returning a `DepthField`**, closing finding 4 at the
   source rather than only at publication.
7. Pin `transformers` (unpinned; resolved to 4.57.6 and is now part of the
   reference provenance) and add `onnx` to the requirements — it is an
   undeclared build dependency of the export path.

---

# Day 5 — Data infrastructure

Branch `foundation/day-5`, seven objectives. Nothing downloaded, nothing
trained — by design: every fetch path requires a human license snapshot that
does not exist yet.

**361 passed, 6 skipped, 1 xfailed.** `mypy --strict` clean over 26 source
files (scope extended to `src/data`). Lint clean at the pinned versions.

## D5.0 — The finding this day implements

Every eval and calibration story in this repository was built on driving
footage while the product is indoor offices, shops and industrial floors. The
harness transfers; the footage does not. Two consequences are now enforced in
code rather than noted:

- The golden set is rebuilt for indoor (`v2-indoor`), and the driving set is
  demoted to `legacy` — retained for pipeline regression, refused for product
  metrics.
- Any INT8 calibration performed on driving frames is void. The calibration
  builder refuses anything that is not lane C.

## D5.1 — Registry contents

41 entries. **Zero verified.** Every `license_snapshot` is `null`, and every
stated class is recorded as `hypothesis_class` — a starting hypothesis from a
document written from memory, not a clearance.

| lane | count | meaning |
|---|---|---|
| R — research | 30 | eval and model selection only; never trained on, never in demos |
| S — synthetic | 7 | conditional: lane S pending per-asset commercial clearance |
| C — consented | 1 | `site-zero`, which does not exist yet |
| blocked | 3 | DukeMTMC, DukeMTMC-reID, MS-Celeb |

By subsystem: activity 7, re-ID 7, depth 5, anomaly 4, gait 4, detection 3,
twin 3, tracking 2, HOI 2.

The blocklist lives **in code**, not only in the YAML: deleting a blocked entry
from `configs/datasets.yaml` does not unblock it, and re-registering a blocked
name as `blocked: false` raises at load. Matching is fuzzy across case, dashes
and underscores, so `dukemtmc_reid` and `DukeMTMC-reID` both hit.

The gait row is worth stating plainly: **all four public gait datasets are lane
R, so there is no commercially usable public gait data at all.** The trainable
gait corpus is Site Zero or nothing. That is an independent confirmation of the
"gait as auxiliary, never sole identifier" posture — the data reality would not
support more even if the accuracy did.

## D5.2 — Human license verification required, by product impact

Nothing can be fetched until someone reads the actual text and records it with
`scripts/fetch_dataset.py --verify-license`. Ordered by what it unblocks:

1. **MEVA** — the single best public match to this product (multi-camera
   facility, hired consented actors, surveillance-style activity annotations).
   Its terms were historically unusual for this category; if verification
   confirms permissive, it may earn use beyond lane R, and it is the only
   public dataset that might. **Verify first.**
2. **MEVID** — built on MEVA, so its answer probably follows MEVA's. Clothing-
   change re-ID eval.
3. **The four synthetic re-ID sets** (RandPerson, UnrealPerson, ClonedPerson,
   PersonX) — currently lane S *conditionally*. They are the only re-ID data we
   could legitimately fine-tune on before Site Zero matures, so their status is
   what decides whether re-ID work can start at all. Watch for the SMPL trap:
   synthetic means no consent debt, not no license.
4. **Infinigen-Indoors and Kubric** — the depth/geometry eval story. Generator
   code is permissive; the per-asset terms are what need checking.
5. **NTU-RGB+D 120** — the POSE-verb benchmark the first converter targets.
6. Everything else, as it becomes relevant.

## D5.3 — Converter, golden set, CVAT

**Converter interface + first converter.** `src/data/converters/`: a `Converter`
protocol producing `CanonicalClips` — clips plus GT typed as
`src.events.Event`. The NTU-skeleton converter is implemented end to end
against a format-correct fixture synthesized in the test.

The action→verb mapping covers four classes whose NTU definitions match a verb
exactly. Everything else lands in `CanonicalClips.skipped` **with its reason**,
as part of the return type. Wrong GT is worse than missing GT: missing shrinks
the eval, wrong corrupts it while inflating the score. GT event ids are `uuid5`
over `(site, clip, verb)`, so re-conversion is idempotent and joins survive it.

**Golden sets.** `v1-driving` is `legacy` and `require_product_usable()` raises
for it; `v2-indoor` is `active` and **empty**. `make eval` exits 1 on an empty
set, because an empty set reporting no failures is not a passing grade.
Immutability is enforced: frozen dataclass, `with_clips()` mints a new version,
`write_golden_set` refuses to overwrite, and the `set_sha` is re-checked at
load so a hand-edited manifest is caught instead of silently invalidating every
number ever reported against it.

The repository had no `Makefile`; one now exists as a thin wrapper over scripts
that all work standalone.

**CVAT round trip.** The label config is *generated* from the `Verb` enum, so
the annotation spec and the production schema cannot drift. Verb/object
coherence is encoded into the CVAT form itself — INTERACTION labels carry a
required object attribute, POSE labels carry none — so an incoherent annotation
cannot be entered, not merely rejected later. Ingest reuses the schema's own
rules; the committed fixture contains three deliberate violations and each is
rejected with its CVAT track id so an annotator can find it.

**Calibration builder.** Refuses lane R with the specified message verbatim,
and refuses lane S with the *same headline but a different reason* — synthetic
frames have no sensor noise, rolling shutter, compression artefacts or real
lighting, so ranges fitted to clean renders clip on real footage. Someone told
only "wrong lane" would reasonably reach for the synthetic set, which is
precisely the wrong fix. Sampling is round-robin across declared conditions,
not proportional; a test seeds 995 daylight frames against 5 glare frames and
asserts all 5 rare frames are selected.

## D5.4 — Site Zero shopping list

**Cameras and mounts**

| item | qty | notes |
|---|---|---|
| IP cameras, 1080p, PoE, RTSP | 3 | the product claims both resolutions, so both must be measured |
| IP cameras, 720p, PoE, RTSP | 3 | the harder case, and the one the budget is quoted against |
| PoE switch, ≥8 ports | 1 | also the time-sync path |
| Wall/ceiling mounts, adjustable | 6 | realistic CCTV heights and angles, not desk height |
| Cat6 runs | 6 | length per site survey |
| NVR or a recording host with ≥4 TB | 1 | ~50 h raw across 6 cameras |
| Tripod + phone gimbal | 1 | twin walkthrough capture |

**Placement, which is a measurement decision rather than a convenience one**

- One **overlapping pair** — the only source of cross-camera association GT.
- One deliberate **blind-spot gap** between coverage zones — handoff eval, and
  the source of the `observed=false` inferred events the schema is built for.
- One **long-corridor** view at the far edge of the capability envelope, to
  populate the `far_field` condition.

**Volume targets (v1)**: 8–12 participants, 30–40 scripted sessions plus 2
weeks passive consented ambient, ~50 h raw, **5 h densely annotated**, a 30-clip
indoor golden set frozen out of it, and the calibration set extracted per §D5.3.

**Consent**: `docs/site_zero_consent_TEMPLATE.md`, marked **DRAFT — requires
counsel**. It covers purpose limitation, biometric handling, retention,
withdrawal, and per-purpose opt-in (demos are separately tickable and
refusable). It states one limitation plainly rather than eliding it: deleting
someone's footage does not remove what a trained model already learned from it,
and whether that satisfies the erasure obligation is flagged as the sharpest
open question for counsel.

Recruit beyond the team. Consent from eight colleagues who all know each other
is weak evidence of freely-given consent, and a poor sample of a real
deployment.

## D5.5 — Deviations

- **`make eval` had no Makefile to wire into**; one was created. Every target
  is a wrapper over a script that works standalone, so the Makefile is never
  the only way in.
- **`scripts/eval_report.py` reports the instrument, it does not compute
  metrics.** There is nothing to measure until Site Zero produces annotated
  clips. Building the part that decides *what* gets measured first is
  deliberate: it is the part that determines whether a number means anything.
- **`src/data/converters/base.py` defines `CanonicalClip` without media
  decoding.** NTU ships skeletons, not pixels, so the first converter needed no
  decoder. The ffmpeg-pinned mp4 path is specified in the dataclass and unbuilt
  until a converter needs it.
- **`Condition` lives in `src/data/golden.py`**, and the calibration builder
  imports it from there rather than duplicating the taxonomy.
- **`deterministic_event_id` added to the event schema.** GT needs stable ids
  across re-conversion; production events keep random `uuid4` because each
  detection is a new fact.

## D5.6 — Day 6, ordered

1. **Verify MEVA's license** (§D5.2). It gates the most product-relevant public
   data and may be the one dataset that earns broader use.
2. **Order cameras and run the site survey** — placement decides what GT is
   obtainable at all, and the overlapping pair plus blind-spot gap cannot be
   retrofitted by annotation.
3. **Counsel review of the consent template**, specifically the erasure
   limitation in §7 and whether employee consent is freely given for DPDP
   purposes.
4. Then, unchanged from Day 3 and still first in the engineering queue: the
   production `vjepa2_vitl_int8.xml`/`.bin` from whoever ran production, for
   the parked Objective-1 forensic verdict.

---

# Day 6 — Measurement

Seven objectives, all complete. Two authorized behaviour changes, both shipped
with their measurement. One first: `make eval` now produces a scorecard.

The theme, unplanned but unmistakable, is that **every significant finding came
from checking the measuring apparatus rather than the code**. The mapper bug was
found by disbelieving a token count; the cascade number was wrong because its
parity test was vacuous; the scorecard's own fixture had two defects found by
reading the scorecard; and the source tree turned out to be missing six files
from git because an ignore pattern was checking more than anyone intended.

## Objective 1 — Mapper vs 392 tokens (AUTHORIZED, measured)

**None of the three hypothesised states held.** The audit's framing was wrong,
and so was mine going in.

| Hop | What was checked | Result |
| --- | --- | --- |
| 0 | Config arithmetic | 2 slots × 196 patches = **392** expected |
| 1 | Encoder actual output | **`(1, 392, 1024)`** |
| 2 | Mapper's assumption | `T_out = features.shape[1] // NUM_PATCHES` |

The repo documents `T * 196 = 784`. That is **refuted**: V-JEPA2 applies a
tubelet of 2, so a 4-frame clip yields 2 temporal slots, not 4.

But the mapper never believed the 784. It read the token count dynamically and
derived `T_out = 2` correctly. The defect was one line further on — the
frame→slot rule was a **clamp**, `min(t, T_out - 1)`, where the encoder's layout
calls for integer division by the tubelet:

| frame | `min(t, T_out-1)` | `t // tubelet` |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | **1 ← wrong** | 0 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |

At 4 frames it misassigns frame 1 alone, which is why this survived as an
apparent edge case. The damage grows with clip length, because a clamp
saturates while integer division keeps advancing: **at 8 frames it misassigns 5
of 8**, and the checkpoint this pipeline uses is natively 64-frame.

Before/after on the real encoder, black frames 0–1 and white frames 2–3:

```
within black half   (f0 vs f1)   1.000000    was: frame 1 read the white slot
within white half   (f2 vs f3)   1.000000
across the boundary (f0 vs f2)   0.676467    correctly distinct
```

The consumer now takes `PatchTokens`, so `n_temporal == frames_covered //
tubelet` guards this path permanently — a 784-token block for a 4-frame clip
raises at the boundary instead of being reshaped into place.

Two things fixed in passing. The arithmetic moved to
`src/semantics/patch_mapping.py` and `src/semantics/__init__.py` no longer
imports at module scope: token indexing has nothing to do with point tracking,
but the coupling meant this code could not be run — let alone tested — without
a CoTracker checkpoint. **That is how a one-line index bug survived.** The 13
new tests need no weights. Patch sampling is also now bilinear rather than
`floor(x/patch_size)`, which had quantised every track to a 16-pixel cell.

## Objective 2 — Cascade cost on the pinned stack

**The product budget is MISSED, at 4.57% against a 3.00% target — 1.52× over.**
Reported, not closed. No threshold was retuned; the remedy is a separate
measured change.

| Scenario | p50/frame | % of one core | parity | speedup |
| --- | --- | --- | --- | --- |
| static (idle) | 2.70 ms | 3.24% | EXACT | 3.23× |
| near target | 2.58 ms | 3.10% | EXACT | 3.51× |
| SMALL target | 2.51 ms | 3.01% | EXACT | 3.33× |
| **real, upscaled 720p** | **3.81 ms** | **4.57%** | EXACT | 2.59× |
| real, native 320×176 | 2.29 ms | 2.75% | EXACT | 1.01× (no-op) |

Two corrections to the Day-2 numbers this supersedes.

**Day 2 measured 2.97% on OpenCV 5.0.0.** The pinned 4.8.1.78 is ~9% slower for
identical code — its resize and MOG2 implementations differ. A cost figure
without its stack is not a figure, so the stack now sits in config beside the
number.

**Day 2's real-footage parity was vacuous.** That clip is 320×176, already below
the 320×180 gate, so the downscale never ran and both "resolutions" processed a
bit-identical raster. Foreground fractions matched to 0.000000 across all 250
frames — which reads like a strong result and tested nothing. This settles the
ambiguity the brief asked about: **neither decisions nor intermediate values
diverged, because no downscale occurred.** Real content upscaled to 720p now
exercises the actual path, and parity there is genuinely EXACT on wake
decisions.

CI was re-based into two numbers, deliberately not collapsed into one:

- `idle_core_budget_fraction` (3.00%) — what Tier-1 hardware needs. Reported
  every run, **never** adjusted to match what the code does.
- `regression_ceiling_fraction` (5.50%) — what CI gates on, set from the worst
  pinned-stack measurement plus shared-runner headroom. Its only job is catching
  a change that makes stage 0 worse than today.

Collapsing them would let CI go green by moving the target, which is exactly how
a missed budget quietly becomes a met one. The bench prints both verdicts and
its success line reads "no regression is not budget met".

## Objective 3 — DAv2Wrapper returns DepthField (AUTHORIZED, behaviour-neutral)

Audit finding 4 closed at the source; its `known_bug` marker removed in the same
commit so the test guards permanently.

`predict()` returned a bare ndarray that anything could read as metres — and did.
The value flowed through the projector into a Parquet column documented as
"Depth (meters)". DA-V2 emits relative inverse depth with unknown scale *and*
unknown shift, so no constant converts it. Nothing raised; every 3D coordinate
downstream had arbitrary scale while claiming to be metric.

It now returns `DepthField(units="disparity_rel")`, so `unproject()` refuses it
and a caller wanting raw values must ask for `.data` — a visible act rather than
an assumption.

**Behaviour neutrality is asserted, not claimed.** A fixed-input test compares
wrapped values against raw model output with `assert_array_equal` — exact
equality, not a tolerance, since float32→float64 widening is exact and a
tolerance would let a real change hide inside it. A units fix that also moved
the numbers would be two changes wearing one commit.

The one judgement added is the validity mask: non-finite entries are marked
unusable rather than passed through. A NaN depth is not a far one, and one NaN
vector corrupts an entire index.

## Objective 4 — Version-stable fixtures

`real_test_video`'s sha changed when the environment moved from OpenCV 5.0.0 to
the pinned 4.8.1.78, because decode and resize differ between builds. **A
fixture that changes with the decoder is not a fixture**: every reference
embedding built against it silently stops corresponding to it, and nothing
announces that.

The committed `.npz` of *decoded frames* is now the source of truth.
`--regenerate-decoded` is the only way to re-decode, off by default. The
manifest records the decoding stack, which turns "why did this sha change?"
from an investigation into a diff.

The opt-in `@pytest.mark.decoder_dependent` test re-decodes and compares against
committed bytes. It is excluded from CI: a failure means "this machine's OpenCV
differs from the one that made the fixture" — real information, but not a defect
in this repository. Its failure message says explicitly not to regenerate the
fixture to make it pass, since that is the move that would destroy the reference
correspondence.

## Objective 5 — Synthetic indoor set, and the first scorecard

### Why analytic primitives and not Kubric

Kubric 0.1.1 pins the Blender 2.93/3.x-era `bpy` API; the available Blender is
4.5, and the official path is a multi-gigabyte Docker image. Rather than stall
or pretend, scenes are built from geometric primitives with closed-form depth.

This is a genuine advantage for the ground truth — exact depth and analytic
slab-test occlusion, with no z-buffer quantisation and no anti-aliasing
ambiguity at object edges — at the total cost of appearance realism. There is no
global illumination, no material response, no lens model.

### Per-asset clearance record

Synthetic does not mean free: BEDLAM and AGORA are built on SMPL body models
whose commercial use needs a separate Meshcapade license. **The SMPL trap is
avoided here by construction rather than by checking** — every asset is
generated by `scripts/gen_synthetic_indoor.py`, so there is no third-party asset
in the stack at all.

| Asset | Origin | License | SMPL-derived |
| --- | --- | --- | --- |
| room shell (floor, ceiling, four walls) | axis-aligned planes generated by this file | ours — no third-party asset | no |
| furniture (desks, cabinets, partitions) | axis-aligned boxes generated by this file | ours — no third-party asset | no |
| agents (torso box, head sphere, two leg cylinders) | articulated primitives generated by this file | ours — no third-party asset | **NO** — the SMPL trap, avoided by construction. No body model, no scan, no likeness of any person. |
| textures | seeded procedural noise generated by this file | ours — no third-party asset | no |

Registered as `synthetic-indoor-v1`, lane S, and **not conditionally** — unlike
RandPerson/UnrealPerson/ClonedPerson/PersonX, whose lane-S status still depends
on someone verifying their generation stacks.

### THE FIRST SCORECARD

Golden set `v2-indoor`, `set_sha
fa0632eae782d18a1e9b1e1f5e24b5f03601d9f469066cab3f46e7030792905b`, 9 clips.

```
metric                                        value  unit
--------------------------------------------------------------------------
motion_gate.recall                           0.9375  fraction (^ better)
    105 of 112 moving frames woke the next stage.
motion_gate.precision                        1.0000  fraction (^ better)
    105 of 105 wakes were on genuinely moving frames.
motion_gate.f1                               0.9677  fraction (^ better)
motion_gate.false_negatives                  7.0000  frames   (v better)
    The number that matters most: frames with real motion the gate slept through.
motion_gate.false_positives                  0.0000  frames   (v better)
gt.occluded_track_fraction                   0.7778  fraction (v better)
    Difficulty of the set itself, not a model result.
coverage.frames_scored                     126.0000  frames   (^ better)

CAVEATS
  - SYNTHETIC-ONLY. These are GEOMETRY and MOTION numbers, not appearance
    numbers, and they must not be quoted externally. Site Zero footage
    supersedes them.
```

Per clip, worst first:

| clip | frames | tp | **fn** | recall | occluded |
| --- | --- | --- | --- | --- | --- |
| `coverage_gap_3agents__cam_c` | 14 | 8 | **6** | 0.571 | 0.94 |
| `overlap_pair_2agents__cam_b` | 14 | 13 | **1** | 0.929 | 0.54 |
| `blown_window__cam_a` | 14 | 14 | 0 | 1.000 | 0.92 |
| `coverage_gap_3agents__cam_a` | 14 | 14 | 0 | 1.000 | 0.94 |
| `crowded_6agents__cam_a` | 14 | 14 | 0 | 1.000 | 0.90 |
| `far_field_1080p__cam_a_1080` | 14 | 14 | 0 | 1.000 | 0.92 |
| `lights_off_transient__cam_a` | 14 | 14 | 0 | 1.000 | 0.92 |
| `overlap_pair_2agents__cam_a` | 14 | 14 | 0 | 1.000 | 0.92 |
| `near_static__cam_a` | 14 | 0 | 0 | n/a | 0.00 |

**Six of the seven false negatives are in one clip** — the blind-spot-traversal
camera, `coverage_gap_3agents__cam_c`, at recall 0.571. Recorded as measured. It
is a real question whether that is a gate defect or a ground-truth definition
issue: GT motion is defined in *world* space (any agent displaced > 1 cm), while
the gate can only see its own frame. A camera is being scored on motion it may
not be able to observe. **This is not resolved and is not tuned away** — it is
the first item on the Day-7 list.

> **Superseded — see "The scorecard above was measuring the wrong thing" below.**
> It was a ground-truth definition issue. Every number in this section is the
> pre-fix measurement and is kept only as the before-half of the delta.

`near_static__cam_a` scoring `n/a` is correct: no GT motion, no wakes, so recall
is undefined rather than 0 or 1. Reporting `nan` instead of inventing a value is
the honest arithmetic.

### Two defects in the fixture, found by reading the scorecard

Both were in the measuring apparatus, and both would have silently corrupted
every number computed against this set.

**1. The wall was shimmering.** Surface texture was drawn inside the frame loop,
so it resampled every frame. A static scene emitted σ=3 grain on **83% of its
pixels** — background pixels with no agent in either frame changed on 762,168 of
801,206, by up to 20 levels. That turned a declared surface property into
undeclared sensor noise which the motion gate was then scored against, while the
manifest simultaneously claimed "no sensor noise". Texture is now fixed per
camera.

The gate's numbers **did not move** after the fix, which is itself the finding:
its decisions were driven by agent motion, not by grain. Had they moved, the
headline recall would have been measuring the renderer.

**2. `content_sha` did not reproduce across processes.** Seeding used the
builtin `hash()`, which is salted per process via `PYTHONHASHSEED`. Three
interpreters gave `7387`, `954`, `3698` for the same input. Clip bytes — and
therefore every content hash in the golden set — differed on every run,
destroying the immutability guarantee a content-addressed golden set exists for.
Now `blake2s`. A subprocess test runs two interpreters under different hash
salts and requires identical hashes.

My determinism test had passed because both runs shared one process. The
property-based subprocess test replaced a source-scanning test that failed by
matching the word `hash()` inside my own explanatory comment — the same
brittleness class as the Day-1 false-passing normalization detector, so it was
deleted rather than patched.

The golden set was also being populated by hand. It is now minted from the
manifest by `--write-golden`, because **a set of content hashes nobody can
reproduce is not content-addressed**.

### The scorecard above was measuring the wrong thing

The blind-spot question resolved as an eval defect, not a gate defect. GT motion
was `agent_xyz` displacement maximised over *every agent in the scene* — world
space — compared against *one camera's* wake decisions. The camera was charged a
false negative for sleeping through motion outside its frustum, which is the
definition of a frustum rather than a defect.

Projecting motion into each camera and checking visibility fixes that and
produces a second false verdict, so GT motion is now partitioned into three
buckets per camera, not two:

| bucket | meaning | treatment |
| --- | --- | --- |
| not observable | no pixels reach this sensor — outside frustum, or fully occluded by static geometry | excluded from every denominator |
| below envelope | visible and unoccluded, but subtends fewer gate pixels than `min_foreground_fraction` can resolve | excluded from recall, reported as `envelope.limited_misses` |
| above envelope | visible, unoccluded, resolvable | the only honest recall denominator |

The middle bucket is the one that matters. Those are not gate defects; they are
the **capability envelope becoming measurable for the first time**. Folding them
into recall would hide the envelope. Counting them as failures would send
someone tuning `min_foreground_fraction` down until the gate wakes on sensor
noise. On its own line it can do neither, and it becomes a coverage-advisor and
mount-position input instead of a bug.

Observability is resolved from the renderer's own instance masks rather than by
reprojecting agent centroids. The mask is exact, it already accounts for partial
occlusion and frame-edge clipping, and it keeps this module from re-deriving a
projection convention. Silhouette area is scaled into *gate* pixels, so the
1080p rendition and its 720p twin land on the same side of the envelope.

Same set, same `set_sha`, same gate — only the denominator changed:

```
metric                          before      after
motion_gate.recall              0.9375     1.0000
motion_gate.false_negatives          7          0
envelope.limited_misses              —          2
envelope.unobservable_frames         —         58
coverage.frames_scored             126         55
coverage.observable_fraction         —     0.4365
```

**The 1.0 is not the finding. The 0.4365 is.** All seven "false negatives" were
frustum or envelope artefacts — five unobservable, two below envelope, one of
those at 104.0 gate pixels against a 115.2 threshold, which is the envelope
boundary showing up as a single frame. But the corrected denominator also says
**55 of 126 frames carried ground truth this camera could act on**. The set is
far thinner than it looked, and a recall of 1.0 over 41 moving frames is a
statement about the set rather than about the gate. `coverage.observable_fraction`
exists so that cannot be read any other way.

Why so much is unobservable: agents walk behind a furniture slab at 4.00 m and
out through the bottom frame edge. Both are correct rendering — verified frame
by frame, the trajectories are smooth and the projection rejects points behind
the camera — but a set whose agents spend half their frames off-sensor is not
exercising what it claims to. **Re-authoring the scenes so agents stay in
frustum is now the top eval task**, and it is a new versioned set, not an edit
to `v2-indoor`.

Two smaller consequences, both recorded rather than smoothed over:

- `envelope.wakes_outside_envelope` is 64. Excluded frames must not become a
  hiding place for wakes, so they are counted. The stay-awake latch
  (`stay_awake_frames=12`) accounts for all of them — checked per clip.
- The buckets close: 55 scoreable + 13 below envelope + 58 unobservable = 126.
  An accounting that does not close is one where an excluded frame can go
  missing without anyone noticing.

`motion_gate.recall` before and after are **not comparable** — the metric
definition changed, not the system. The pre-fix numbers are kept above as the
before-half of this delta and for no other purpose.

### Coverage

11 of 17 conditions. The 6 uncovered — `bag_carried`, `blinds_drawn`,
`camera_bump`, `clothing_change`, `lens_smudge`, `similar_clothing` — are all
appearance-driven and need real capture. A test asserts they are still reported
missing, so synthetic clips cannot make the set read as done.

## The gitignore defect — six files were never committed

Found while checking why `src/data/scorecard.py` did not appear in `git status`.

In gitignore syntax an unanchored `data/` matches a directory named `data` at
**any** depth. The patterns `data/` and `models/` were therefore also matching
`src/data/` and `src/models/`, silently excluding six source files across three
days:

| File | Day | What it is |
| --- | --- | --- |
| `src/models/preprocess.py` | 3 | `PreprocessSpec`, the preprocessing contract |
| `src/data/golden.py` | 5 | golden sets, `set_sha`, `SiteZeroPlan` |
| `src/data/cvat_ingest.py` | 5 | annotation ingest |
| `src/data/converters/` (3 files) | 5 | dataset converters |

Their tests **are** committed and pass here only because the files exist in this
working tree. A fresh clone would have failed at import, and it would have
surfaced as a broken test suite for whoever cloned first rather than as a
missing-file error anyone could read.

Every artifact pattern is now anchored with a leading slash, with a comment
saying why the slash matters. Verified that `data/`, `models/`, `outputs/` and
`logs/` at the repo root are still ignored.

This is the strongest argument yet for configuring the git remote: six days of
work exist in exactly one directory, and until this commit six of its files were
not even in the local history.

## Objective 6 — ADR 0001, deletable identity adapters

`docs/adr/0001-identity-adapter-architecture.md`. **Status: Accepted.**

Person-specific capability lives in a small adapter over a frozen backbone, and
never in fine-tuned backbone weights.

The argument is erasure. Under a monolithic fine-tune, honouring a withdrawal
means re-running the entire training pipeline to produce a model provably free
of one person's data — weeks, so in practice never, which is precisely why the
consent template currently has to tell participants "we cannot reverse
training". With a frozen backbone, the complete set of parameters derived from
an individual **is** the adapter.

| Tier | What is deleted | Cost |
| --- | --- | --- |
| 1. Gallery eviction | the person's enrolled embeddings | seconds, no retraining — recognition stops at once |
| 2. Adapter retrain | residual influence on the learned metric | hours, CPU |
| 3. Backbone retrain | — | **not applicable by design** |

Tier 3 is the column the decision exists to keep empty.

A second constraint made this easy to decide: all four public gait datasets are
lane R, and the usable re-ID sets carry consent debt or an unverified SMPL
dependency. The trainable identity corpus is Site Zero — people we know by name
who can withdraw in person. An architecture assuming a large anonymous corpus is
mismatched with the only data we may lawfully use.

Costs are recorded, not hidden: an adapter will underperform a full fine-tune,
and the ADR requires that gap to be **measured and put on a scorecard** before
Phase 3 commits. Rejected alternatives include machine unlearning — telling a
participant their data was "approximately removed" is worse than telling them it
was not.

Cross-referenced from the consent template's erasure section with an explicit
instruction **not** to soften the §7 limitation until the architecture exists
and a withdrawal drill has actually been run. `tests/test_identity_adr.py` pins
that. The ADR describes a better position; it does not license claiming it yet.
The counsel question stays open in both documents.

## State

- Full suite: **415 passed, 6 skipped**. Skips are `requires_weights` and the
  excluded `decoder_dependent` marker, each with a stated reason.
- Lint clean (black 23.12.1, flake8 7.0.0); mypy clean on touched modules.
- All work on `foundation/day-6`, seven commits.

## Day 7, in order

1. **Re-author the synthetic scenes so agents stay in frustum.** The blind-spot
   false negatives are resolved — they were an eval defect, and GT motion is now
   partitioned into observable / below-envelope / above-envelope per camera. But
   the corrected denominator exposed the real problem:
   `coverage.observable_fraction` is **0.4365**. Agents spend half their frames
   behind a 4.00 m furniture slab or off the bottom edge, so recall 1.0 rides on
   41 moving frames. This is a new versioned set (`v3-indoor`), never an edit to
   `v2-indoor`, and it is the highest-value eval work available.
2. **Close the cascade budget, or move it deliberately.** 4.57% against 3.00%.
   Either optimise stage 0 with a measured before/after, or change the target
   with a stated hardware justification. Not both, and not silently.
3. **Configure the git remote and push all seven days.** Escalating every day it
   is deferred, and the gitignore finding shows the working directory is not
   even a faithful copy of the history.
4. **Widen the scorecard beyond the motion gate.** Depth and occlusion GT are
   exact and currently unused by any metric. Cheapest large increase in what
   `make eval` actually covers.
5. **MEVA license verification** — still the highest product impact of any
   pending item, and still blocked on a human reading the actual terms.
6. **Counsel review** of the consent template, now with ADR 0001 as the proposed
   architectural answer to its sharpest open question.
7. **Production `vjepa2_vitl_int8.xml`/`.bin`** from whoever ran production, for
   the parked Objective-1 forensic verdict. Unchanged since Day 3.

---

# Day 7

Branch `foundation/day-7`, cut after Objective 0 landed on `foundation/day-6`.

## Objective 0 — Commit and push everything

**The premise was stale, and saying so first.** The six files that "were never
committed" were already recovered on Day 6 by `33cb345`;
`src/models/preprocess.py` is tracked and in history. The real backlog was the
five working-tree files from the Day-6 eval fix. **Files recovered this day: 0.**
Nothing was missing that was not already found.

### Ignore-pattern audit

Every pattern was checked against what it actually matches, not read:

| pattern | anchored | matches |
| --- | --- | --- |
| `/outputs/` `/logs/` `/models/` `/data/` | yes | the artifact dirs only |
| `venv/` `.venv/` `.venv-pinned/` `.vscode/` `.coverage` `coverage.xml` | **was NO** | now anchored |
| `__pycache__/` `*.py[cod]` `*.egg-info/` `.pytest_cache/` `.mypy_cache/` `.hypothesis/` | NO | **deliberately** |

The second group must stay unanchored — those directories occur at every depth,
and anchoring them would stop ignoring caches inside `src/` and fill
`git status` with them. Anchoring "everything unanchored" would have been a
regression, so the file now carries the rule that distinguishes the two cases,
which `33cb345` fixed without writing down.

Confirmed explicitly: **none of the 131 tracked source files is ignored**;
`outputs/`, `data/`, `models/`, `logs/` each remain ignored, checked against the
specific rule that ignores them.

### Fresh-clone verification — and it failed the first time

Clone to a temp dir, venv from `locking-requirements.txt`, `pip install -e .`,
full suite. **Run 1: 1 failed, 422 passed.**

`test_eval_report_scores_the_populated_indoor_set` scores the configured golden
set and never materialised the clips it scores. On this machine they are already
under `data/synthetic/` from the last `make eval`, so it passed; in a fresh clone
`data/` is empty, `eval_report` refuses, and it fails. **It was testing the
developer's working directory rather than the repository** — the same class as
Day 2's vacuous parity claim, and invisible to every run here.

Skipping when the clips are absent would have been worse: that test exists to
catch `make eval` going green-by-abstention, so a version of it that abstains in
CI is the defect it guards against. A session fixture materialises the dataset
instead, with the generator defaults pinned to the ones v2-indoor was minted
from so content hashes still match the manifest. `data/` stays ignored —
committing the clips to make a test pass would be fixing the measurement by
moving the artifact.

Verified this was the only such test: nothing else in `tests/` touches
`resolved_data_dir`, `data/synthetic`, or `eval_report.main([])`.

**Run 2: 423 passed, 1 skipped, 6 deselected.** The skip is
`test_preprocess_spec.py:435`, "no legacy artifacts under outputs; nothing to
refuse" — an explicit reason, not a fake pass.

### Push

**No git remote is configured. All 41 commits across seven branches exist on one
machine and nowhere else.** This is now the seventh day it has been deferred, and
Objective 0 just demonstrated that the working directory is not a faithful copy
of what a clone gets. Exact commands for the human:

```
git remote add origin <url>
git push -u origin main
for b in foundation/day-1 foundation/day-2 foundation/day-3 \
         foundation/day-5 foundation/day-6 foundation/day-7; do
  git push -u origin "$b"
done
```

## Objective 1 — The envelope threshold was excusing a real defect

115.2 gate px decided which misses were gate defects and which were the camera's
physical limit. It appeared nowhere as a literal — it fell out of
`min_foreground_fraction * gate_px` inside the partition — and a Day-6 miss sat
at 104.0 against it.

**The derivation** is now explicit in `MotionGateConfig.envelope_threshold_px`
with the algebra in the docstring. The gate wakes when
`foreground_px / gate_px >= min_foreground_fraction`, so 115.2 is correct
arithmetic about **foreground** area. `gate_pixels()` also takes the source
shape, because a source already below the gate size is not upscaled — the fact
that made Day 2's parity claim vacuous.

**The measurement** (`scripts/measure_envelope.py`) shows it does not predict the
gate's behaviour, because a scorecard knows **silhouette** area:

| displacement (native px/frame) | silhouette area that wakes the gate | vs derived |
| ---: | ---: | ---: |
| 2, 3 | never — absorbed into the background model | — |
| 4 | 250.0 gate px | 2.17× |
| 6 | 132.0 | 1.15× |
| 8 | 110.0 | 0.95× |
| 12, 20 | 108.0 | 0.94× |
| 30 | 92.0 | 0.80× |

**There is no single threshold.** A slow mover is learned as background at any
size, so the envelope is a function of speed and the disagreement runs from 6%
to unbounded. Per the rule, the measurement wins: the envelope is a committed
artifact (`configs/envelope/`), interpolated between samples and clamped outside
them, never extrapolated. Interpolating across a speed that never woke returns
`inf` rather than averaging into a threshold the gate was never observed to
reach.

Result — same golden set, same gate, measured envelope replacing the constant:

```
motion_gate.recall             1.0000 -> 0.9762
motion_gate.false_negatives         0 -> 1
envelope.limited_misses             2 -> 1
coverage.observable_fraction   0.4365 -> 0.4444
```

The 104.0 px miss in `overlap_pair_2agents__cam_b` moved from "below the
envelope" to a **genuine false negative**: at that agent's real image-plane speed
the gate could have woken and did not. **The unargued constant was excusing a
real gate defect** — the failure direction that matters.

Every scorecard now records `envelope_sha` and the measured stack, and
`Scorecard.require_comparable` refuses a delta across differing envelopes or
golden sets. A test caught a real bug in the interpolator: an exact sample whose
slower neighbour never woke returned `inf`, discarding the value measured at that
very speed.

## Objective 2 — Acceptance criterion landed; v3-indoor NOT authored

`mint_golden_set()` refuses a set whose aggregate `observable_fraction` is below
**0.80**, before it can be registered or produce a scorecard. Acceptance runs on
measurements — a clip with no measured fraction is refused outright, because
unmeasured clips are exactly how 0.4444 went unnoticed. The `hard_coverage` tag
keeps a genuine blind-spot scenario out of the aggregate and on its own line, and
two tests hold it honest: it cannot rescue a set whose untagged clips are weak,
and a set of nothing but blind spots is refused.

`v2-indoor`'s `set_sha` is unchanged (`fa0632ea`) — the frozen set stays exactly
as measured.

**Not done: the v3-indoor set itself.** Re-authoring camera placements and agent
paths, ≥20 clips, ≥400 scorable frames, both renditions, mint, register, and
`make eval` remain. The gate that makes that work verifiable is in place; the
authoring is not, and no v3 scorecard exists to report.

## Objective 3 — BLOCKED on real footage

The prerequisite does not exist in this repository. The only real video is
`src/interface/ui/data/raw/test_video.mp4` at **320×176** — the same clip whose
size made Day 2's parity claim vacuous, because it is below the 320×180 gate and
the downscale never runs. A valid parity fixture needs real 720p/1080p footage
that nobody has provided.

**There is still no valid real-footage parity evidence, and none can be produced
from what is in the repo.** The 4.57% vs 3.00% budget miss stands open and
un-profiled. Both are now blocked on a human supplying footage — added to the
blocked list rather than worked around, because synthesising "real" footage to
close this would reproduce the exact defect Day 6 found.

## Objective 4 — Skill rule

`iron-eval-discipline` gained **parameter-scaling bugs**: a bug whose severity
scales with a parameter must be fixed before that parameter is tuned, or the
sweep measures the bug's gradient and it gets read as a property of the system.
Carries the clamp table (25% / 62% / 81% / 95% at T=4/8/16/64), the detection
heuristic — evaluate the defect at the low, middle and high end of the intended
range; if those differ, the sweep is measuring it — and the corollary that the
mildest configuration is the worst place to audit.

## Day 8, in order

1. **Author v3-indoor.** The acceptance floor is in and will refuse anything
   below 0.80. Fix the two traced pathologies: agents exiting the bottom frame
   edge, and the 4.00 m furniture slab occluding the primary sightline. ≥20
   clips, ≥400 scorable frames, both renditions, degenerates retained, at least
   one clip tagged `hard_coverage`.
2. **Configure the git remote and push.** Seventh day deferred. Objective 0
   proved the working directory is not a faithful copy of the history.
3. **Real 720p/1080p footage** for cascade parity and the budget profile.
   Blocked on a human; nothing in Objective 3 can proceed without it.
4. **Widen the scorecard beyond the motion gate.** Depth and occlusion GT are
   exact and still unused by any metric.
5. **Re-measure the envelope on OpenCV 5.0.** The committed model records
   `opencv 4.8.1.78`; MOG2 differs between builds, so the envelope — and
   therefore which misses count as defects — may not transfer.
6. **MEVA license verification** — highest product impact of any pending item,
   still blocked on a human reading the terms.
7. **Counsel review** of the consent template, with ADR 0001 as the proposed
   answer to its sharpest open question.
8. **Production `vjepa2_vitl_int8.xml`/`.bin`** for the parked forensic verdict.
   Unchanged since Day 3.
