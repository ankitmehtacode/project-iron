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

---

# Day 8

Branch `foundation/day-8`. Six commits.

## Objective 1 — v3-indoor

Both day-7 pathologies were traced to specific causes and fixed rather than
tuned around.

**Agents walked off the bottom edge** because `CameraSpec` supported *yaw only*.
A camera 2.6 m up looking dead level puts a 1.72 m agent below the principal
point, and further below it the closer they get — off-frame entirely from about
2 m. Cameras now pitch; every v3 camera tilts 16° down, which centres an agent
at 6 m and keeps 3.5–11 m fully in frame.

**A partition slab sat on the primary sightline** — 2.4 m across the middle of
the room at z=7.0, the 4.00 m depth in the day-7 trace. v3 furniture occludes
from the side, so occlusion is partial and recoverable.

Scene sets are **versioned, not edited**: `build_scenes` is untouched and
`pitch_degrees` defaults to 0, so v2 still renders byte-identically. Its content
hashes are cited by a frozen manifest. v2 is retained, not deleted.

### The v3 scorecard

`set_sha 1c9a975e`, `envelope_sha` from the corrected measurement below.

```
motion_gate.recall                      0.9806   758 of 773
motion_gate.precision                   0.9404   758 of 806
motion_gate.f1                          0.9601
motion_gate.false_negatives                 15
motion_gate.false_positives                 48
envelope.limited_misses                     10
envelope.unobservable_frames                17
envelope.wakes_outside_envelope             10
gt.occluded_track_fraction              0.0674
coverage.frames_scored                     863
coverage.observable_fraction            0.9589
```

30 clips, 863 scorable frames — no bucket in single digits. Floor is 0.80; the
non-`hard_coverage` aggregate is **0.8988** at mint time.

**hard_coverage, on its own line** (excluded from the floor, kept on purpose):

| clip | observable_fraction |
| --- | --- |
| `coverage_gap_3agents__cam_gap` | 0.700 |
| `coverage_gap_handoff__cam_gap` | 0.633 |

**Speed distribution** — a set requirement now, because day 7 showed the
envelope is speed-aware and a set walking at one speed exercises one point on a
curve. Median image-plane speed, gate px/frame:

| band | median | note |
| --- | --- | --- |
| crawl | 0.04–0.29 | under the GT motion threshold entirely |
| slow | 0.61–0.65 | the steep part of the envelope |
| walk | 1.11–1.44 | |
| brisk | 1.60–2.19 | |
| fast | 1.95–2.66 | |
| sprint | 2.99–3.02 | max instantaneous 10.09 |

**v2 and v3 numbers are not comparable.** Three independent reasons, any one
sufficient: different clips, a recall denominator redefined on day 6, and a
different `envelope_sha`. The Inspector refuses to diff them.

### Two findings the set produced, reported rather than tuned

**All 48 false positives are `speed_crawl`**, 24 per camera, and they are not
gate defects. Those agents travel 7.25 mm/frame — below `GT_MOTION_THRESHOLD_M`
(10 mm), so ground truth calls them static — while moving 0.27 gate px/frame,
which the gate detects. This is the same shape as the 115.2 constant: **an
unargued threshold defining truth, disagreeing with the instrument**, and now the
dominant error term. Not changed here; it needs its own measurement.

**`gt.occluded_track_fraction` fell 0.7778 → 0.0674.** Moving furniture off the
sightline bought observability and sold occlusion difficulty. The set is now
weak exactly where v2 was strong, and says so on its own line.

## The Inspector earned its keep in one screen

Day 7 concluded that a slow mover is "absorbed into the background model at any
size" and recorded null wake thresholds for 2 and 3 native px/frame.

**That conclusion was wrong**, for the oldest reason in this repository: the
measurement, not the thing measured. The sweep tested silhouette areas 40–320
gate px. At the slowest speeds the wake rate never crossed 50% *inside that
window*, the model wrote null, and the scorecard read null as physically
unreachable.

The Inspector put the contradiction on one screen. Clip
`speed_slow_2agents__cam_a`, frame 12: silhouette **761.75 gate px**, envelope
verdict **"unreachable at this speed"**, gate state **AWAKE**. A viewer showing
the model and the behaviour side by side made a two-day-old mistake obvious at a
glance — which is the entire argument for building it.

Re-measured to 1660 gate px. Every speed crosses:

| native px/frame | silhouette to wake | vs derived |
| ---: | ---: | ---: |
| 2 | 535.0 | 4.64× |
| 3 | 410.0 | 3.56× |
| 4 | 243.3 | 2.11× |
| 6, 8, 12, 20, 30 | 110.0 | 0.95× |

There is no absorption floor. The threshold rises steeply as speed falls and
flattens at 110 px — a 4.9× range. This **strengthens** the day-7 conclusion that
no scalar threshold exists; it removes only a cliff that was an artifact of where
I stopped looking. The artifact now records `swept_area_gate_px`, and the model
distinguishes "did not cross within the range swept" from "cannot wake".

Effect on v3, same `set_sha`:

```
motion_gate.recall             0.9787 -> 0.9806
coverage.frames_scored            795 -> 863
coverage.observable_fraction   0.8833 -> 0.9589
envelope.wakes_outside_envelope    78 -> 10
```

**68 frames of correctly-detected motion had been sitting outside the
denominator**, counted as neither hit nor miss.

### A second defect, found the same way

Minting v3 with the wrong `source_dataset` made the eval resolve to v1's clip
directory and score five v1 clips that shared a name with v3 clips. The scorecard
cited v3's `set_sha` while measuring v2's bytes, and **nothing complained** —
the hash was recorded and never verified. Scoring now hashes each clip and
refuses on mismatch. A content-addressed set has to actually check its content.

## Objective 3 — Iron Inspector

`make inspect` → `http://127.0.0.1:8899`. Five views: Scorecard, Clip inspector,
Envelope, Events, Provenance.

- **No mock data anywhere.** A test greps the serving code for fixture-shaped
  literals and fails if one appears. A viewer that can invent data cannot verify
  a claim: a green screen would stop distinguishing "the artifact says so" from
  "the fallback fired".
- **Absence is rendered.** Missing artifacts return the path looked for and the
  command that produces it. The event log genuinely does not exist and the Events
  view says so, because "no events happened" and "nothing has run" are different
  statements.
- **Refusal, not a delta.** Cross-`set_sha` or cross-`envelope_sha` comparison
  renders a refusal with reasons — the same rule `require_comparable` enforces.
- **Greyscale-legible.** Observability bands differ in *height* as well as
  lightness; `observed=false` rows are dashed, italic and glyph-marked. Neither
  distinction rests on colour.
- **Stdlib only** — no FastAPI, no CDN, no build step beyond `pip install -e .`,
  so it runs on an air-gapped Tier-3 rack. Localhost-bound.

13 tests, including the no-mock-data grep, the cross-sha refusal, and an
empty-state snapshot.

## Objective 2 — ingest path built, parity BLOCKED

`scripts/ingest_capture.py`: video → decode-once → content-addressed store →
lane-C manifest → condition tags. **Consent is required and ingest refuses
without it**; five of ten tests cover refusal paths. Each clip records
`exercises_downscale`, so the day-2 defect is caught at ingest instead of in a
report.

**`data/raw/office_capture_v1/` does not exist.** The only real video in the
repository is 320×176 — below the gate raster, so it cannot produce a valid
parity number. Tests run against a synthesized stand-in, which proves the path
and cannot produce a measurement. **No parity number is reported, and the 4.57%
vs 3.00% budget miss remains un-profiled on real footage.**

## Day 9, in order

1. **Measure `GT_MOTION_THRESHOLD_M`.** 10 mm/frame in world space is the last
   unargued constant defining truth, and it now produces all 48 false positives.
   Same treatment as the envelope: derive it, measure against the instrument,
   make it config-driven with provenance.
2. **Capture the office footage.** Blocks cascade parity and the budget profile.
   Nothing in Objective 2's measurement half can proceed without it.
3. **Restore occlusion difficulty in a v4 set.** `occluded_track_fraction` 0.0674
   means v3 barely tests occlusion. Needs occluders that bite without swallowing.
4. **Configure the git remote and push.** Eighth day deferred; 48 commits across
   eight branches exist on one machine.
5. **Re-measure the envelope on OpenCV 5.0.** MOG2 differs between builds, so
   which misses count as defects may not transfer.
6. **Widen the scorecard beyond the motion gate.** Depth and occlusion GT are
   exact and still unused by any metric.
7. **MEVA license verification** — highest product impact of any pending item,
   still blocked on a human reading the terms.
8. **Counsel review** of the consent template, with ADR 0001 as the proposed
   answer to its sharpest open question.

---

# Day 9

Branch `foundation/day-9`. The day's framing was: stages 1-3 have never been
measured, expect bad numbers, bad-and-measured is the goal, do not tune. What
happened is one step earlier than that — **the depth fixture cannot measure
depth**, and finding that out was the day's most valuable result.

## The first perception measurement, and why it is not a scorecard

`DAv2Wrapper` gained a Hugging Face load path (same class, same `DepthField`,
same `disparity_rel` label — a second wrapper would be two paths to keep honest
about units instead of one). Alignment, metrics, buckets, flicker and drift are
all built and correct. Then:

```
DEPTH — v3-indoor (1c9a975e3ee5), DA-V2 Small, 30 clips, 120 frames

rank correlation with GT disparity : -0.5924   (floor +0.30)

!! THIS SET CANNOT SCORE THE DEPTH STAGE.

UNALIGNED  AbsRel 0.8164   RMSE 12.452 m   d<1.25 0.0000   SILog 0.2863
ALIGNED    AbsRel 0.1538   RMSE  2.554 m   d<1.25 0.9163   SILog 0.2478

aligned, by GT distance:
  0-3m   no GT pixels
  3-8m   AbsRel 1.6860   RMSE 9.358 m   d<1.25 0.0000
  8m+    AbsRel 0.0562   RMSE 1.254 m   d<1.25 0.9723

static-point Z std : 0.0372 m      alignment scale CV : 0.0323
```

**Read the aligned line and then read the 3-8 m bucket.** AbsRel 0.1538 and
δ<1.25 of 0.9163 look like a respectable first depth number. They are a
property of the fit. 95% of pixels are background wall at 8 m+; the scale/shift
fit lands on that background and scores 0.9723 there, while the 3-8 m band —
where every agent in the set is — scores δ<1.25 of **exactly 0.0000**.

An alignment-based metric can launder a near-constant prediction into a good
score. The check that survives alignment is **ordering**, and the model orders
these pixels almost exactly backwards: rank correlation **−0.5924**. The harness
now gates on that and refuses to present its metrics as a depth result.

### The brief's premise was wrong, and here is the measurement

The instruction was to state that synthetic depth is the easy case and these
numbers are a ceiling. Measured, that is false.

| | dynamic range | ordering |
| --- | --- | --- |
| repo's real footage | 4.44 | floor correctly nearer than ceiling |
| v3 analytic renders | 2.23 | inverted |

Same weights, same wrapper. Analytic primitives are matte, untextured and
perfectly Lambertian, which deletes exactly the shading, texture-gradient and
object-recognition cues a monocular depth model runs on. A large flat wall at
15 m offers it nothing. **These scenes are harder for this stage than real
footage, not easier, and cannot serve as a ceiling.**

The units contract held and is confirmed rather than assumed: on real footage
the floor reads nearer than the ceiling, so `disparity_rel` is the correct
label and the inversion belongs to the model, not the wrapper.

## GT_MOTION_THRESHOLD_M — measured

Same shape as the 115.2 constant: hardcoded into the first scorecard, never
argued, and a world-space number judging an image-space instrument. Replaced
with a derived definition — an agent is moving when its **rendered silhouette
changed**, because if the frames are identical the footage shows no motion.

```
motion_gate.false_positives    48 -> 0
motion_gate.false_negatives    15 -> 57
motion_gate.precision      0.9404 -> 1.0000
motion_gate.recall         0.9806 -> 0.9340
```

**None of the 48 were gate defects.** Every one was `speed_crawl` agents at
7.25 mm/frame — under the constant, over the renderer's resolution — and the
gate was right on all of them.

The other direction matters more. The same constant was **hiding 42 real
misses**: frames whose silhouettes visibly moved, which GT called static, so
the gate sleeping through them cost nothing. Recall falls to 0.9340 and that is
the honest number. The constant was not only generating false alarms, it was
suppressing failures.

A structural consequence surfaced and was fixed rather than shipped: render-
derived motion cannot express a coverage gap, because an agent outside the
frustum has an unchanging empty mask and reads as "nothing happened".
`unobservable_frames` went to 0 and a partition test caught it. Scoring and
coverage are two questions and now use two signals — silhouette change decides
what the gate is scored on, world displacement decides what is reported
unobservable.

## Objective 0 — git remote

`origin` is now configured (`github.com/ankitmehtacode/project-iron`). The push
itself was **blocked by this session's permission layer**, not by missing
credentials. **54 commits across 9 branches still exist on one machine only.**

```
git push -u origin main
for b in foundation/day-1 foundation/day-2 foundation/day-3 foundation/day-5 \
         foundation/day-6 foundation/day-7 foundation/day-8 foundation/day-9; do
  git push -u origin "$b"
done
```

## Not done, and why

- **Objective 3, tracking.** CoTracker3 weights are present; the `cotracker`
  package is not installed and is not in the lockfile. Adding it is a real
  dependency decision, not a side effect of an eval task, so it was not taken
  unilaterally. **No tracking number exists.**
- **Objective 4, semantics.** V-JEPA2 weights and the OpenVINO export are
  present, so this is genuinely runnable — it was not reached. **No retrieval
  mAP, stability or patch-boundary number exists**, and the normalization fix
  still has no task-level evidence.
- **Objective 5, DA-2K registry entry.** Not reached.

Stated rather than partially delivered: a half-built retrieval metric would be
exactly the kind of number this project spends its days deleting.

## Which stage performed worst, and the likely cause

**Depth** — and the cause is not the model. It is the **domain gap between
analytic renders and photographic input**. DA-V2 behaves correctly on the one
real clip available and inverts on the synthetic set. The finding is not "DA-V2
is bad"; it is "v3-indoor is a geometry-and-motion fixture, and depth is an
appearance task".

That has a consequence for the eval strategy: **exact GT depth does not make a
set a depth set.** The motion gate can be scored on analytic primitives because
motion is geometry. Depth cannot, because monocular depth is learned appearance.
Day 10 goes to the fixture, not the model.

## Day 10, in order

1. **A depth fixture that can measure depth.** Either real footage with
   measured GT, or the photorealistic synthetic path (Infinigen-Indoors is
   already lane S in the registry and was chosen over Kubric for exactly this).
   Re-run today's harness against it unchanged — it is built and gated.
2. **Semantics numbers.** Runnable today with weights already on disk;
   retrieval mAP, temporal stability, patch-boundary discontinuity, with and
   without the normalization fix.
3. **Decide the `cotracker` dependency** deliberately — vendor, pin, or drop
   the stage from eval — then measure tracking.
4. **Push.** Ninth day deferred.
5. **Investigate the 57 false negatives** now that they are visible. They were
   suppressed by the metres constant and have never been looked at.
6. **Restore occlusion difficulty** — `occluded_track_fraction` is 0.0674.
7. **MEVA license verification**, still blocked on a human.

---

# Day 10

Branch `foundation/day-10`. Four commits.

## The capability × dataset validity matrix

The day's headline, and it fits on one screen. **7 of 9 cells refuse.**

```
dataset      motion_geometry   depth      appearance_semantics
v1-driving   not present       not present   not present
v2-indoor    PASS              REFUSED       REFUSED
v3-indoor    PASS              REFUSED       REFUSED
```

| refusal | evidence |
| --- | --- |
| v3 depth | rank corr **−0.6309** (floor +0.30); spread 2.54 (floor 3.0); **93%** of pixels in one distance band |
| v2 depth | rank corr **−0.3789**; spread 2.33; 93% in one band |
| v2 + v3 appearance | texture energy **3.03** (floor 12.0); material diversity **15.17** (floor 24.0) |
| v1-driving, all | clips not materialised locally — a statement about presence, not a verdict on the set |

This is the honest picture of what this project can currently evaluate: **one
capability, on two datasets.** Every perception claim beyond motion is
unmeasured and, on what we own, unmeasurable.

`make eval` now consults the gate **before** computing a metric and records the
verdict in the scorecard. A failing capability is recorded as a refusal with
evidence — never a blank, never a zero, never omitted. Day 9's depth number
existed and was reproducible; the only thing that would have stopped it being
quoted is a refusal computed first.

`motion_geometry` is implemented as a **passing** case deliberately. A validity
registry that only ever refuses cannot be distinguished from a broken one.

Band populations are now reported alongside band scores everywhere, because on
day 9 a band holding 95% of pixels set the aggregate while the band containing
every agent scored δ<1.25 of exactly 0.0000 without moving the headline.

## Motion and observability — honest accounting

**The category error was already closed on day 9**, in the same commit that
caused it. Three of the four red tests written today pass against yesterday's
code. I did not find that bug today and am not claiming to.

The fourth test caught the real remaining defect: world motion was still
thresholded at a hardcoded 1 cm, so a 6 mm step at 2 m (2.7 px, plainly
visible) read as static, while the same step at 20 m (0.27 px, invisible to any
sensor) would have read as motion. Replaced with a derived bound — one pixel
subtends `z / fx` metres, so the threshold scales with distance from the
observing camera while the quantity stays world displacement and therefore
means the same thing from every camera.

```
world_motion()          does the agent move       camera-independent
gt_moved_from_render()  can this camera see it    camera-dependent
```

Re-scored v3, **unchanged from day 9's final numbers**:

```
recall 0.9340   precision 1.0000   FN 57   FP 0
unobservable 17   below-envelope 10   scored 863   observable_fraction 0.9589
```

The derived threshold moved no v3 verdict, because v3's agents all displace far
more than a pixel. It matters for correctness and for sets not yet authored,
not for today's numbers — claiming a delta here would be inventing one.

**None of the 42 previously-suppressed misses changed classification.** They
were already correctly counted once motion and observability were separated;
today's change did not touch them. They remain uninvestigated.

## Real depth path

Registered, **none fetched**:

| entry | lane | human action that unblocks it |
| --- | --- | --- |
| DA-2K | R | read the licence, record a snapshot |
| ETH3D | R | verify licence |
| iBims-1 | R | verify licence |
| DIODE-indoor | R | verify licence |
| Infinigen-Indoors-depth-candidate | S | **none — unblocked by work, not paperwork** |

The **DA-2K adapter is written and tested** against a synthesized 5-pair
fixture, ready the moment the licence clears. It suits `disparity_rel` exactly
because it performs **no alignment**: the question is only "is A nearer than B",
so there is nothing for a scale/shift fit to launder. An inverted prediction
scores 0.0 and a constant one lands at chance — both pinned by tests, both
being precisely the day-9 failure modes that metric alignment concealed.

The Infinigen entry carries a **binding acceptance procedure**: not trusted for
depth until it passes the same gate that condemned v3. Photorealistic-synthetic
is a hypothesis about whether the cues return, not a pass. Assuming otherwise
is how v3 became a depth fixture in the first place.

## Not done

- **Objective 4, CoTracker3 + tracking.** Not started. The dependency decision
  was made for me and is recorded, but adding it to the lockfile requires a
  fresh-clone verification cycle that was not affordable today. **No tracking
  number exists.**
- **Objective 5, semantics.** Not run. Note that the appearance gate already
  **refuses v3** on measured evidence (texture 3.03 against a floor of 12.0), so
  the partition's *prediction* is on record — but the point of the objective was
  to test whether that prediction is correct, and an untested prediction is not
  a confirmation. **The partition remains unvalidated for semantics**, and the
  normalization fix still has no task-level evidence.

## Day 11, in order

1. **Run semantics against the prediction.** Cheapest remaining test of whether
   the partition generalizes or is too coarse. Weights are already on disk.
2. **CoTracker3 into the lockfile**, fresh-clone verify, then TAP-Vid — checking
   the validity gate first, since textureless primitives may starve the feature
   matcher.
3. **Generate an Infinigen-Indoors candidate set and run it through the depth
   gate.** This is the only path to a depth number that does not need a licence.
4. **Push.** Tenth day deferred; `origin` is configured and the push is blocked
   by this session's permission layer, not by credentials.
5. **Investigate the 57 false negatives.** Visible since day 9, still unexamined.
6. **A texture/material gate for the fixture generator itself**, so a set that
   cannot support appearance capabilities is refused at mint time rather than
   discovered later.
7. **MEVA licence verification**, still blocked on a human.

---

# Day 11

Branch `foundation/day-11`. Three feature commits.

## Infinigen verdict, and its consequence for the data plan

**Infinigen installs on this box and does not render on it.** From a clean
`.venv-infinigen` (Python 3.11.15, `bpy==4.2.0`, `infinigen==1.12.2` — the
pinned 3.10 measurement env stayed untouched; `scikit-image` had to be
constrained to 0.21.0 because Infinigen's pinned 0.19.3 does not build on
macOS 15 clang, and the pip resolver dropped Infinigen from 1.15.5 to 1.12.2
as a consequence). Two paths were attempted. The furnished path
(`fast_solve.gin`) ran 623 constraint-annealing iterations over 21 minutes 35
seconds, completed 7 of many solver stages across 5 rooms, and was killed
with zero frames rendered — each iteration costs 25-55 s and per-stage
budgets are 100 iterations, so a completed scene is many hours away. The
empty-room path (`singleroom.gin + fast_solve.gin + no_objects.gin`) exits
in ~30 s on `AssertionError: (0.024, 0.018, 1280, 720)` inside
`get_sensor_coords`: the default sensor is 4:3, the default render is 16:9,
and the `execute_tasks.generate_resolution=[160,120]` override needed to
reconcile them is silently not applied by gin. Both logs are preserved at
`docs/day11/`. **No renderer available to us can score appearance-learned
capabilities in a decision-quality session budget on this hardware**, and
that changes the data plan: **real data is now the sole path for depth,
semantics and re-ID**. This is a finding about *our compute*, not about
Infinigen's quality — the acceptance procedure the registry pins remains the
right test on a GPU box with hours of budget, and the registry note keeps
that door open explicitly.

## The updated capability × dataset validity matrix

`point_tracking` joins the gate registry. `Infinigen-Indoors-depth-candidate`
joins the dataset axis, and refuses on generation rather than on cues.

```
dataset                              motion_geometry  depth      appearance_semantics  point_tracking
v1-driving                           not present      not present   not present         not present
v2-indoor                            PASS             REFUSED       REFUSED             (not scored)
v3-indoor                            PASS             REFUSED       REFUSED             REFUSED (set)
Infinigen-Indoors-depth-candidate    (not generated)  (not scored)  (not scored)        (not scored)
```

| refusal | evidence |
| --- | --- |
| v3 point_tracking (set) | **26 of 30** clips below the 1e-3 corners/pixel floor; the 4 clips that pass all sit at exactly 1.111e-3, an order of magnitude short of what a real corridor frame gives a detector |
| Infinigen candidate generation | 21m35s + 7/many solver stages + 0 frames; empty-room path crashes on a config incompatibility |
| v2, v3 appearance | texture energy 3.03 (floor 12.0) — carried from Day 10 |
| v2, v3 depth | rank corr −0.38 / −0.63 (floor +0.30) — carried from Day 10 |

## Tracking result

`scripts/eval_tracking.py`, gate first per the Day-10 rule. Per-clip corner
density on the v3 golden set spans 4e-5 (a `blown_window` clip that is almost
entirely one dark region) up to 1.111e-3 (the coverage-gap cameras and one
`near_static` clip, all at the same value — the density every clip converges
to when the frame is dominated by a couple of high-contrast furniture
silhouettes). Set aggregate: **REFUSE**. No CoTracker3 score is computed on
v3 — a pooled tracking number across a set where 26/30 clips starve the
matcher would describe the extrapolator, not correspondence. Full per-clip
evidence at `docs/day11/tracking_verdict.json`.

Two things this leaves unmeasured, both recorded so the numbers cannot travel
without them:

- `scaled_offline.pth` is **bidirectional** — the checkpoint attends to
  future frames when predicting frame *t*. Even if v3 had passed, that
  number would have been an offline oracle, not the streaming path. The
  offline-minus-online penalty remains unmeasured.
- Same-clip observability bucketing (observable / off-sensor / occluded) is
  implemented in `score_clip_with_tracker` so a Day-12+ real-footage set
  gets the breakdown by construction.

The CoTracker3 pin itself: `git+https://…/co-tracker.git@82e02e8029753a…`
plus `einops>=0.7.0` (CoTracker declares `install_requires=[]` in its
`setup.py` but its runtime imports einops). Fresh-clone verified into
`.venv-pinned` up front — Day 10 named that as the blocker, so it moved
first, not last. **The pin authorises evaluation only.** The majority of
CoTracker is CC-BY-NC 4.0, which blocks any product build that includes the
tracker or checkpoint; the ship question stays open in the registry.

## Semantics partition — the metric measured the wrong thing

Numbers on the current golden set (30 clips, 72 GT tracks, 268 embeddings),
computed twice: once with the sidecar preprocess spec applied (production
path — mean/std standardisation), once with the raw [0, 1] clip fed straight
in (pre-fix path, reconstructed behind a flag in the script; production
preprocessing is not un-fixed anywhere).

|                              | standardised | pre-fix | delta  |
| ---------------------------- | -----------: | ------: | -----: |
| same-object retrieval mAP    | **0.268**    | 0.265   | +0.002 |
| chance mAP (1/n_tracks)      |        0.015 |   0.015 | 0.000  |
| temporal cosine (same track) |        0.929 |   0.920 | +0.009 |
| patch-boundary L2            |        0.963 |   0.922 | +0.041 |

At face value the mAP contradicts the partition — it is ~18× chance, not
degenerate. But the diagnostic that lands next to it in the JSON payload
tells a different story:

> `patch_visit_diagnostic`: **79%** of GT tracks stay inside exactly **one
> 16-px patch** across the encoder's 4-frame window; **100%** stay inside
> at most two.

An agent that occupies one patch for the whole encoder window makes
"same-object retrieval" reduce to "look up almost the same embedding four
times". The mAP is real — the mechanism producing it is patch-location
constancy under low motion and a short window, not V-JEPA understanding of
the agent as an object. The corroborating evidence is the **normalization
delta of +0.002 mAP** and +0.009 temporal cosine: standardisation is what
makes V-JEPA embeddings semantically meaningful (Day 1: unstandardised
input drops the reference cosine from 1.000000 to 0.332), so a delta of
two-thousandths says the mAP is not doing semantic work at all. If it were,
the fixed path would separate sharply from the pre-fix one.

So none of the two outcomes the prompt asked for cleanly applies:
- Not "metrics degenerate, matching the gate's refusal → partition validated".
- Not "metrics meaningful despite refusal → partition too coarse".
- **The metric on this fixture cannot distinguish those cases.** It measures
  patch-location constancy, not object semantics, because the fixture cannot
  make tracks TRAVERSE patches within the 4-frame encoder window. The
  partition prediction on semantics remains untested.

The normalization delta is reported as measured (+0.002 mAP, +0.009 temporal
cosine, +0.041 patch boundary) and labeled weak evidence in the JSON's
`interpretation_note` per the prompt — a relative improvement on degenerate
data is weak, and dropping it would have been worse than reporting it that
way. The appearance gate itself still refuses v3 on measured evidence
(texture 3.03, floor 12.0), which today's sample of 8 clips confirmed;
nothing here changes that refusal.

Full verdict at `docs/day11/semantics_verdict.json`. Encoder is V-JEPA2 fp32
IR (`models/export/2026-07-31/vjepa2_vitl_fp32.xml`) — no INT8, blocked on
the production artifact human item.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. These carry forward from Day 10, ages
counted in days since the item first surfaced:

1. **Git remote / push, 5 days old.** Configured; today's session's
   permission layer blocks it, not credentials.
2. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`, ~8 days.** All
   perception numbers today are fp32, which is a ceiling.
3. **MEVA licence verification, ~10 days.** Highest product impact of any
   pending data item. Real corridor footage is now the sole path for depth
   / semantics / re-ID (see the Infinigen verdict above), which raises the
   stakes on this.
4. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7, ~10 days.**

## Day 12, in order

1. **A cross-clip, longer-window semantics metric.** The Day-11 mAP measured
   patch-location constancy, not semantics; a metric that forces tracks to
   traverse patches (or queries at frame 0 and retrieves from frame ≥20) is
   the only way to actually test the partition on the fixture we have.
2. **A texture/material gate at fixture mint time,** so a set that cannot
   support appearance capabilities is refused when it is generated rather
   than discovered several capabilities later. Carried from Day-11's list;
   Day-11 gate additions were all downstream of mint.
3. **DA-2K licence verification and first real depth number.** The adapter
   is written and tested (Day 10); a single human read of the terms
   unblocks the first real-data depth eval this project has ever produced.
4. **Push.** Sixth day deferred; do not extend the streak.
5. **Investigate the 57 false negatives** in the motion gate. Visible since
   Day 9, still uninvestigated.
6. **Streaming CoTracker3 checkpoint,** so the offline-minus-online penalty
   is measured; every tracking number quoted from now on has to name whether
   it is offline oracle or streaming.
7. **MEVA licence verification** — still blocked on a human, restated in
   position 7 rather than dropped.

---

# Day 12

Branch `foundation/day-12`. Five commits.

## Retroactive baseline margins — the gate's precision is indistinguishable from always-wake

Three times now a metric produced a confident number for a degenerate
reason, each caught by a bespoke diagnostic after the fact. Day 12 makes
the check structural: `src/eval/baselines.py` requires every metric name
to declare a trivial-strategy baseline before a `Metric` can even be
constructed (`BaselineMissing` raises at construction time), and every
scorecard line now renders `value  baseline: X  margin: Y`, with
`!! FLAGGED` printed inline when the margin is at or below zero.

Applied retroactively to `motion_gate` — this project's only PASSING
capability, and the one every Tier-1 economics claim rests on:

| metric | value | strongest baseline | margin | flagged |
| --- | ---: | --- | ---: | :---: |
| `motion_gate.recall` | 0.9340 | always-wake: 1.0000 (boundary, not flag-worthy) | n/a | — |
| `motion_gate.precision` | 1.0000 | always-wake: 1.0000 | **+0.0000** | **YES** |
| `motion_gate.f1` | 0.9658 | always-wake: 1.0000 | **-0.0342** | **YES** |
| `motion_gate.false_negatives` | 57 | always-wake: 0 (boundary, not flag-worthy) | n/a | — |
| `motion_gate.false_positives` | 0 | always-wake: 0 | **+0.0000** | **YES** |

**On the current v3-indoor golden set, the motion gate's precision, F1,
and false-positive count are all exactly indistinguishable from the
naive always-wake strategy.** 806/806 wakes were on genuinely moving
frames either way; zero false positives either way. This is not a defect
in the gate — it means v3-indoor's `observability_partition` never
produces a scoreable non-moving frame where the gate could have woken
wrongly and didn't, so there is currently no adversary condition in this
fixture for precision to distinguish a discriminating gate from a gate
that wakes on everything. The gate's real selectivity remains untested;
recall is the only axis this set currently measures. Full scorecard with
every baseline/margin/flag at
`docs/day12/motion_gate_v3_with_baselines.json`.

## Repaired retrieval protocol, and what it now measures

Day 11's mAP queried at frame 0 and pooled every frame of a 4-frame clip;
79% of GT tracks never left their patch across that window, so "same
object" collapsed to "same patch index." The repair
(`scripts/eval_semantics.py`) makes position unable to solve the task:
query/target pairs are kept only if the GT track's patch index changed
between the two frames (`crossed_boundary`), the temporal gap is an
explicit swept parameter, and a position-only baseline (rank by assumed
co-location) is computed and reported at every gap.

| gap (frames) | surviving pairs | pool | mAP | strongest baseline | margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 67 | 0.452 | 0.483 | -0.031 **FLAGGED** (n too small to trust) |
| 2 | 8 | 67 | 0.486 | 0.407 | +0.079 |
| 4 | 20 | 67 | 0.467 | 0.274 | **+0.194** |
| 8 | 36 | 67 | 0.326 | 0.216 | +0.110 |
| 16 | 53 | 67 | 0.272 | 0.152 | +0.120 |
| 32 | 60 | 66 | 0.182 | 0.115 | +0.067 |

**At gap=4, mAP beats the strongest baseline by +0.194 — the first
genuinely interpretable semantic-retrieval signal this project has
produced.** Margin is positive at every gap except gap=1, where only 4
pairs survive and the sample is honestly too small to trust (flagged
rather than hidden). Both mAP and surviving-pair count fall as gap grows
past 4, which is plausible on two counts: longer separation is a harder
retrieval task, and appearance itself drifts over more time, not just
position. Full per-gap payload at `docs/day12/semantics_gap_sweep.json`.
ADR 0002 gets a Day-12 update: this gap-sweep protocol is the semantic
metric its 16-vs-64 clip-length question needed, but no IR export exists
at those window lengths yet — no policy is adopted today.

## Infinigen: throughput measured, Day-11 verdict retracted

Day 11 concluded no renderer available to us can score appearance-learned
capabilities. That overclaimed: what was measured is that generation did
not complete in ~21 minutes, not that it cannot complete at all. **The
depth validity gate on `Infinigen-Indoors-depth-candidate` is corrected to
UNMEASURED** (blocked on generation throughput), not REFUSED — no sample
has ever reached the gate.

Measured, from evidence already on disk (no new generation run today):
coarse-stage constraint solving averages **170.9 s/stage** (6 completed
stages, 67.5-246.2 s range) and is CPU-bound — every measured stage is
Python/numpy optimization, none of it GPU work. A new, isolated,
bounded probe (`scripts/probe_cycles_throughput.py`) measured raytracing
separately: after one-time kernel compilation, Metal-GPU Cycles renders
in **0.5-0.9 s/frame regardless of sample count** — roughly 200-300x
faster than the solving bottleneck. **The render stage was never the
problem; the coarse-stage constraint solver is.** Full memo, including the
overnight-local-vs-rented-GPU decision (recommendation: measure
populate/fine_terrain/render costs and the solver rate on any candidate
rented hardware before committing spend — do not choose yet) at
`docs/day12/infinigen_throughput.md`.

## Camera ingest readiness

Built so that plugging in a camera is the only remaining step:
`scripts/discover_cameras.py` (WS-Discovery + ONVIF Media-service stream
enumeration, credentials always masked), `src/ingest/rtcp.py` (RTCP
Sender Report parsing and RTP-to-wall-clock mapping, RFC 3550 §6.4.1),
`src/ingest/rtsp.py` (TCP-only transport — refuses rather than falling
back to UDP — interleaved demux, per-frame timestamp source always
explicit, every dropped packet a `FrameGap` record), and
`scripts/measure_substream_hypothesis.py` (the sub-stream-vs-downscaled-
main comparison the 4.57%-vs-3.00% budget miss motivates, scaffolded and
tested but refusing to run against anything but matched real captures).
27 tests, all against real sockets and a real minimal RTSP server fixture
— zero mocks, zero hardware. **Genuinely blocked on a camera now, not on
code.**

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Ages counted in days since first surfaced:

1. **Git remote / push, 6 days old.** Still session-permission-blocked,
   not credential-blocked.
2. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`, ~9 days.**
3. **MEVA licence verification, ~11 days.** Highest product impact of any
   pending item, sharpened by Day 11's (partially-corrected) real-data
   finding.
4. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7, ~11 days.**
5. **A physical camera, new today.** Everything ingest-side is built and
   tested against a fixture; only hardware unblocks the real path.

## Day 13, in order

1. **Investigate why v3-indoor cannot exercise motion-gate precision.**
   Today's retroactive baseline run found precision/F1/false-positives
   indistinguishable from always-wake — understand whether this is a
   `observability_partition` denominator issue or a genuine property of
   the fixture (no adversarial non-moving-but-textured frames), and fix
   or document accordingly. This is now the highest-priority open item:
   it is the only passing capability and its real selectivity is unknown.
2. **Extend the gap-sweep protocol to real footage once a camera lands** —
   the gap=4 signal (+0.194 margin) is the strongest evidence yet that the
   encoder does real semantic work; confirming it survives off synthetic
   fixtures matters more than any other semantics question open.
3. **A patient overnight local Infinigen run**, inspected each morning
   rather than session-bounded, to fill the populate/fine_terrain/render
   measurement gap the throughput memo left open. Costs nothing but
   otherwise-idle wall-clock.
4. **Connect a camera and run `scripts/discover_cameras.py`.** Everything
   downstream (RTSP ingest, the sub-stream hypothesis harness, the first
   lane-C data) is built and waiting on this single step.
5. **Push.** Seventh day deferred; the streak itself is now worth a
   dedicated look at what in the session permission layer is blocking it.
6. **Investigate the 57 false negatives** in the motion gate — visible
   since Day 9, still uninvestigated, and now sitting next to a precision
   finding that makes the gate's overall selectivity even less settled.
7. **MEVA licence verification** — still blocked on a human.

# Day 13

Branch `foundation/day-13`, off Day 12. Seven commits: six feature
objectives plus this report. `src/model/` is new: 2,700 lines across
twelve modules (plus `__init__.py`), 133 tests, `mypy --strict` clean
with zero suppressions.

## Objective 0 — Day-12 status check

Day 12 fully landed: five commits, all four of its objectives complete
(trivial-baseline requirement, retrieval-protocol repair, Infinigen
throughput measurement, camera-ingest readiness). Nothing to retroactively
implement. **Motion-gate margin, restated from Day 12** — still the
project's only passing capability and still the number every Tier-1
economics claim rests on:

| metric | value | strongest baseline | margin | flagged |
| --- | ---: | --- | ---: | :---: |
| `motion_gate.recall` | 0.9340 | always-wake: 1.0000 (boundary) | n/a | — |
| `motion_gate.precision` | 1.0000 | always-wake: 1.0000 | **+0.0000** | **YES** |
| `motion_gate.f1` | 0.9658 | always-wake: 1.0000 | **-0.0342** | **YES** |
| `motion_gate.false_positives` | 0 | always-wake: 0 | **+0.0000** | **YES** |

Unchanged since Day 12 because no new gate measurement ran today — Day
13 was a data-model day by design. Day 12's diagnosis stands: v3-indoor
never produces a scoreable non-moving frame, so precision/F1/FP cannot
currently distinguish this gate from always-wake. Still open (Day-14 list,
below).

## Objectives 1-6 — `src/model/`, built in dependency order

Twelve modules (`ulid`, `uncertainty`, `measurement`, `frame_of_reference`,
`observation`, `entity`, `envelope`, `events`, `coverage`, `relationship`,
`evidence`, `episode`), each frozen, fully typed, `mypy --strict`. One
commit per objective, matching the standing rule.

| # | Module(s) | What it is | Tests |
| --- | --- | --- | --- |
| 1 | `ulid`, `uncertainty`, `measurement`, `frame_of_reference`, `observation`, `entity`, `envelope` | Core primitives | 27 |
| 2 | `events` + `scripts/migrate_events_v1_v2.py` | Four event classes, v1 migration | 28 |
| 3 | `coverage` | Coverage and Absence | 18 |
| 4 | `relationship` | Bitemporal Relationship, Correction | 13 |
| 5 | `evidence` | Evidence, EvidenceCommitment, Confidence | 20 |
| 6 | `episode` | Episode, ActivityMode, StateGraph skeleton | 19 |
| — | `test_falsification.py` | The five falsification tests (below) | 8 |

**133 tests total, all green.** `mypy --strict` over
`src/model` plus the pre-existing strict set: zero new errors (the six
errors `mypy` reports are pre-existing, in `src/data/depth_eval.py` and
`src/data/validity.py`, untouched today — confirmed by running the same
check on the pre-Day-13 tree). `black` and `flake8` clean on every file
touched.

### Which STRUCTURAL rules are now genuinely unviolatable, and the test that proves each

Per the Day-13 quality bar: STRUCTURAL means impossible to violate — a
raise, or unrepresentable in the type — not merely documented, with a
test that attempts the violation.

| Rule | Mechanism | Proving test |
| --- | --- | --- |
| `Observation` without uncertainty is unconstructable | Required constructor arg, no default; `__post_init__` also rejects an explicit `None` | `test_observation_without_uncertainty_kwarg_is_unconstructable`, `test_observation_with_none_uncertainty_raises` |
| Anonymous entities have no cross-session persistence field | `AnonymousSessionEntity` is its own dataclass with no `persistent_identity_ref` slot — passing one is `TypeError`, not a validation failure | `test_anonymous_entity_has_no_persistence_field_to_populate` |
| Envelope thresholds are curves, not scalars | `EnvelopeCurve.__post_init__` rejects fewer than 2 points | `test_envelope_curve_rejects_single_point` |
| `PredictedEvent` cannot be admitted as evidence | `functools.singledispatch` with no handler registered for it — the rejection is an absent registration, not a written conditional | `test_predicted_event_cannot_be_admitted_as_evidence`, `test_no_isinstance_predicted_event_check_in_evidence_admission_source` (asserts the dispatch source contains neither `isinstance` nor the excluded class name) |
| `HypothesisEvent` cannot trigger an alert | Same `singledispatch` pattern, no handler | `test_hypothesis_event_cannot_trigger_alert` |
| A confirmed prediction is a new record, never a mutation | `PredictedEvent` is frozen; `confirm_prediction()` is the only path to confirmation and always returns a fresh `ObservedEvent` | `test_confirm_prediction_does_not_mutate_the_predicted_event` |
| Negative queries can never return bare "nothing happened" | `prove_absence`'s return type is the union `Absence \| CannotEstablish`, nothing else; empty coverage log falls to `CannotEstablish` | `test_empty_coverage_log_cannot_establish_absence`, `test_prove_absence_return_type_is_always_the_union` |
| A single-timestamp `Relationship` is unconstructable | Four temporal fields, all required, none defaulted | `test_relationship_missing_valid_time_axis_is_unconstructable`, `test_relationship_missing_assertion_time_axis_is_unconstructable` |
| A `DerivedArtifact` with no `input_closure` cannot be registered | `ArtifactRegistry.register()` raises before adding it to the dependency graph | `test_derived_artifact_without_input_closure_cannot_be_registered` |
| `Confidence` cannot claim to be a probability without calibration | `UncalibratedScore` has no `probability` attribute at all; `CalibratedProbability.calibration` is a required arg with no default | `test_uncalibrated_score_has_no_probability_attribute`, `test_calibrated_probability_requires_calibration_record` |
| `ActivityMode` can never alert or become evidence | `attach_to_alert` / `attach_to_evidence` unconditionally raise; `admissible` is checked `False` even against a caller bypassing the type checker | `test_attach_to_alert_always_raises`, `test_attach_to_evidence_always_raises`, `test_activity_mode_rejects_true_admissible_even_bypassing_typing` |

### Migration results

`scripts/migrate_events_v1_v2.py`, round-tripped on the three-event demo
incident (`scripts/demo_events.py`, written to a real Parquet file with
the v1 writer — the "existing demo parquet" the objective asks for): 3
total, 2 mapped to `ObservedEvent`, 1 to `InferredEvent`, **0
unmappable**. Migrated records preserve `site_id`, `ts_ns`, `subject`,
`verb`, `confidence`, `object`, `zone`, and `clip` exactly; the migrated
`InferredEvent` carries a stated (if generic) `basis` string, since v1
never recorded why a record was inferred. The unmappable-record path is
exercised with a synthetic duck-typed record carrying an out-of-range
confidence (`test_unmappable_record_is_reported_not_dropped_silently_or_forced`)
— there is currently no real v1 record that can violate the v2 contract,
because v1's own constructor already enforces the same bounds. The path
exists for the day the two rule sets diverge, not because it has fired
yet.

### What remains skeleton

- **`StateGraph.solve_state`** raises `NotImplementedError` naming the
  factor-graph solver as the thing that fills it. `graph_rev` exists,
  increments monotonically, and is append-only — enough for `Evidence`
  to reference today — but nothing resolves a `StateQuery` yet.
- **No canonical `Observation -> hash` function** for
  `EvidenceCommitment.compute()`'s leaf hashes (ADR 0007, Open questions).
  The commitment mechanism is real; what hashes an observation into a
  leaf is not decided.
- **Twin-rev staleness is detectable, not enforced** (ADR-adjacent
  finding, see the falsification tests below): `FrameOfReference.is_current_for()`
  answers the question correctly, but nothing raises if a caller uses
  stale-twin_rev data anyway. Enforcement belongs at the estimator/query
  layer, which does not exist yet.
- **No aggregation across multiple cameras** in `Coverage`/`prove_absence`
  — a zone watched redundantly by two degraded cameras cannot jointly
  prove absence even though the two together might genuinely establish it
  (ADR 0004, Open questions).

## The five falsification tests (`tests/test_falsification.py`)

Run as integration tests exercising several `src/model` types together,
per the instruction that a model passing all five on day one probably
was not tested hard enough. Two of the five surfaced real gaps rather
than confirming guarantees — reported here rather than smoothed over.

| # | Test | Result |
| --- | --- | --- |
| 1 | Absence under degraded coverage | **PASSES.** A camera that goes `degraded` partway through a query window blocks a proven absence; `prove_absence` returns `CannotEstablish(reason="coverage_insufficient", ...)` with the degraded interval named in `envelope_violations`. |
| 2 | Retroactive badge resolution | **PASSES.** The canonical 13:58/14:02 scenario end to end: `Relationship.is_retroactive` is `True`, `ArtifactRegistry.apply_correction` invalidates the affected scorecard, and `registry.get()` still returns the original pre-correction record unchanged. |
| 3 | Twin re-version | **PARTIAL.** Staleness is representable and correctly detected (`FrameOfReference.is_current_for`), but nothing in `src/model` automatically enforces it — a caller can combine twin_rev=1 geometry with a twin_rev=2 world and nothing raises. `test_falsification_twin_reversion_is_not_automatically_enforced` asserts this current gap explicitly rather than leaving it silent, so it starts failing (correctly) the day the enforcement is added — at which point the test should be deleted, not patched. Full enforcement is estimator-layer work, out of scope for Day 13. |
| 4 | Alert explainability | **PARTIAL.** When an alert-eligible event's `evidence_refs` is populated, the chain resolves to real `Evidence` with a non-empty `derivation_chain` — explainable. But nothing structurally requires `evidence_refs` to be non-empty for `raise_alert()` to succeed: `test_falsification_alert_explainability_is_not_structurally_required` shows an event with empty evidence can still trigger an alert today. Separately, a `PredictedEvent`-triggered alert can never point at itself as evidence (ADR 0003) — a UI rendering that alert must show the `InferredEvent`/`ObservedEvent` chain that fed the predictor, which is a Day-14+ rendering requirement, not a data-model gap. |
| 5 | Behaviour-query shape | **PASSES at the type level; BLOCKED on the unimplemented estimator for a live answer.** An `ActivityMode` can never be mistaken for one of the four event classes, and `attach_to_alert`/`attach_to_evidence` both refuse it unconditionally. But `solve_state` — the thing that would actually produce an `ActivityMode` from a real trajectory — raises `NotImplementedError`, so this proves the answer's shape is safe once an estimator exists, not that a behaviour query can be answered today. |

**Reading across all five:** the falsification suite did its job. It
confirmed two designs the day's ADRs argue for (Coverage/Absence, ADR
0004; bitemporal Relationship, ADR 0005) and it found two real,
now-documented gaps (twin-rev enforcement, alert-evidence requirement)
that a suite designed only to pass would have missed. Neither gap is a
regression — both are honest statements of what Day 13 built (primitives
and rules) versus what it explicitly deferred (inference, enforcement
that depends on inference).

## ADRs

Five, all Accepted: `0003` (event-class hierarchy over a boolean flag),
`0004` (Coverage/Absence as primitives), `0005` (bitemporal
relationships), `0006` (Episode with roled participants instead of a
separate Interaction primitive — records the primitive-proliferation
reasoning an earlier draft's `Interaction` type was rejected for), `0007`
(EvidenceCommitment's reproducibility-vs-erasure design, explicitly not
resolving the DPDP-Act question it sits next to — that stays counsel's).

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Ages counted in days since first
surfaced; unchanged today — Day 13 did not touch any of these.

1. **Git remote / push, 7 days old.**
2. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`, ~10 days.**
3. **MEVA licence verification, ~12 days.**
4. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7, ~12
   days** — now directly load-bearing for ADR 0007's open question, not
   just ADR 0001's.
5. **A physical camera, 1 day old** (first surfaced Day 12).

## Day 14, in order

1. **The motion-gate precision/selectivity investigation**, still
   deferred a second day now — the gate is this project's only passing
   capability and Day 12's finding (precision indistinguishable from
   always-wake on v3-indoor) is unchanged because no gate work happened
   today.
2. **Close the alert-explainability gap** (falsification test 4):
   either require non-empty `evidence_refs` for anything that passes
   `raise_alert()`, or make the absence of evidence a rendered,
   explicit state in whatever alert surface gets built, rather than an
   unstated possibility.
3. **Decide the twin-rev enforcement point** (falsification test 3):
   where staleness checking actually belongs — write time, query time,
   or the estimator — before the estimator itself gets built on top of
   an unenforced assumption.
4. **A canonical `Observation -> hash` function** for
   `EvidenceCommitment.compute()` (ADR 0007), so two callers cannot hash
   the same observation two different ways and produce commitments that
   silently fail to compare.
5. **Connect a camera and run `scripts/discover_cameras.py`** — still
   the single step unblocking the whole ingest path.
6. **Push.**
7. **MEVA licence verification** — still blocked on a human.
8. **The factor-graph solver**, once there is a measured reason to
   start it — `StateGraph`'s skeleton is ready to be filled, but per Day
   13's own scope discipline, estimation lands in a later, *measured*
   phase, not because the skeleton now exists.

# Day 14

Branch `foundation/day-14`, off Day 13. Six commits.

## Closing two holes opened a third — say so first

Day 13's two PARTIAL falsification results are both closed today (below),
and the fix pattern for one of them opens a new, honest seam rather than
sealing the surface completely. **`events.raise_alert()` (Day 13, broad
type-eligibility) and `alert.emit_alert()` (Day 14, strict — requires a
resolvable evidence chain) now both exist, and nothing steers a caller
toward the strict one.** `raise_alert()` still lets a `PredictedEvent` —
or an `ObservedEvent` with empty `evidence_refs` — through unchanged; that
is by design (forecast-only paging is a real, narrower use case), but
there is no deprecation notice, no runtime warning, and no lint rule
distinguishing "the permissive check for a fast-lane notification" from
"the strict check for a production alert a customer will see." A future
integration that reaches for `raise_alert()` because it is the
Day-13-vintage, more-familiar name reintroduces exactly the
un-explainable-alert failure Day 14 closed, and nothing in the type
system stops it — this is a naming/discoverability gap, not a structural
one, and it is exactly the kind of thing that survives because both
functions individually do what they claim. Tracked on the Day-15 list.

## Objective 1 — the twin_rev hole (falsification test 3): CLOSED

`src/model/world.py`: `WorldPosition` requires `x_m`/`y_m`/`z_m`/`twin_rev`
as four undefaulted fields (mirrors Day 13's `Relationship`).
`distance_to()`/`reproject()` raise `TwinRevError` across differing
revisions unless an explicit `TwinRevTransform` is supplied, and no
arithmetic operator exists on the type at all, so anything not routed
through those two methods fails with a plain `TypeError` by the absence
of an override. `TwinRevTransformRegistry.resolve()` raises for an
unregistered rev pair rather than defaulting to identity.

**Proving tests:** `test_falsification_twin_reversion_world_position_
now_raises_across_revs` (raises without a transform, raises on a
mismatched transform, succeeds and stays interpretable with the correct
one); `test_registry_resolve_unregistered_pair_raises_not_identity`
(`tests/test_model_world.py`).

**Audit finding, not fixed today** (numerical-behaviour paths; a rules-
only day does not touch them without a separate measurement): two
existing world-coordinate paths bypass this contract entirely. (1)
`src/geometry/projector_vectorized.py::project_to_3d` writes
`outputs/point_cloud_tracks.csv` with camera-space (not world-space),
already-self-labelled-uncalibrated columns, no `twin_rev`, no route
through `src/contracts/geometry.unproject()`. (2)
`src/data/scorecard.py::world_motion` and `src/inspector/artifacts.py`
consume raw `[T, A, 3]` `agent_xyz` world positions with no `twin_rev` —
synthetic-GT-only, and the risk this objective targets does not actually
apply there, since each synthetic clip bakes one scene regenerated
wholesale rather than a twin re-versioned in place.

## Objective 2 — the alert-evidence hole (falsification test 4): CLOSED

`src/model/alert.py`: `Alert.evidence_chain` is required and non-empty.
`emit_alert()` is a closed-world `singledispatch`, registered only for
`ObservedEvent`/`InferredEvent` — exactly the intersection of Day 13's
`AlertEligibleEvent` and `EvidenceEligibleEvent`, since a `PredictedEvent`
can never itself be admitted as evidence (ADR 0003) and a
`HypothesisEvent` was never alert-eligible. `explain(alert_id,
alert_store, observation_store=None)` walks alert → event → evidence →
observation_refs/clip_refs/state_refs, checking `producer_sha` at every
derivation step, and raises `ExplainabilityError` naming the exact broken
hop — unknown `alert_id`, an unresolved `evidence_ref`, missing
observations/derivation, or an observation absent from a supplied store.

**Proving tests:** `test_falsification_emit_alert_now_requires_a_
resolvable_evidence_chain`; `tests/test_model_alert.py`'s six
`test_explain_*` tests, one per hop that can break.

**As stated above, this is closed for the strict path only** —
`raise_alert()` (Day 13) is unchanged and still permits an unexplainable
alert; see "Closing two holes opened a third."

## Objective 3 — the gate.\* metric reframe

Full detail and the re-scored v3-indoor numbers are in Day 14's earlier
section above (`fix(eval): reframe the motion-gate scorecard...`,
commit `891dd58`); summarised here for the day's record:

| metric | value | strongest baseline | margin |
| --- | ---: | --- | ---: |
| `gate.wake_fraction` | 0.9067 (816/900) | always_wake: 1.0000 | +0.0933 |
| `gate.recall_retained` | 0.9340 (806/863) | always_wake: 1.0000 (boundary) | n/a |
| `gate.compute_saved` | 1.87 ms/frame **(estimate, unmeasured cost model)** | always_wake: 0.0000 | +1.87 |
| `gate.miss_cost` | 57 frames | never_wake: 863 | +806 |

**SYNTHETIC-ONLY, restated plainly per the objective's instruction:**
wake fraction on authored synthetic scenes says nothing about wake
fraction on a real corridor at 3 AM, and the entire Tier-1 economic claim
depends on the latter. This table is a mechanism check — the four
numbers are internally consistent and the pairing rule holds — not a
business case. `gate.compute_saved`'s 1.87 ms/frame is doubly caveated:
even granting the synthetic-vs-real gap, the multiplier itself
(`PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME = 20.0`) is not measured —
`DetectorStage` is an unimplemented stub — so this number is an estimate
under a stated model squared, not a measurement once removed.

## Objective 4 — retrieval-metric repair: verified, not rebuilt

Day 12 already built everything this objective specifies (cross-boundary
queries, gap sweep, same-class/different-instance distractor pool,
position-only baseline every gap, `--window` as a metric parameter, and
the standardised-vs-prefix normalization comparison at every gap). No
code changed today. Re-ran the full 30-clip sweep:
**byte-for-byte identical to Day 12's recorded artifact** — full
reproducibility confirmed, not assumed.

Surviving cross-boundary pairs, and the normalization fix's effect
(`delta = standardised − prefix` mAP), per gap:

| gap | pairs | mAP | margin | delta (fix effect) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.4524 | −0.031 **FLAGGED** | −0.4226 (untrustworthy: n too small) |
| 2 | 8 | 0.4857 | +0.0787 | +0.0450 |
| 4 | 20 | 0.4674 | +0.1937 | −0.0089 |
| 8 | 36 | 0.3259 | +0.1099 | −0.0700 |
| 16 | 53 | 0.2722 | +0.1201 | +0.0049 |
| 32 | 60 | 0.1822 | +0.0673 | +0.0261 |

**The normalization fix's effect is unmeasurable at every gap this set
currently supports** — full reasoning in ADR 0002's Day-14 update. Gaps
2–32 show deltas within noise of their pair counts; gap 1's large delta
sits on only 4 pairs. Stated condition for measurability: ~20
cross-boundary pairs at a single gap (matching gap 4, the smallest gap
already treated as informative for mAP itself) — reached by more clips
or faster-moving agents in the fixture, neither adopted today. No
clip-length policy adopted, per instruction.

## Objective 5 — camera ingest readiness

`FrameGap.to_coverage_gap()` / `RtspIngestSession.coverage_gaps()`
project RTP-level drop detection into Day-13 `Coverage.Gap` records
(`reason="dropped_frame"`) instead of a second gap-reporting mechanism;
`FrameGap` itself is kept for the RTP-specific diagnostics (`expected_
seq`/`observed_seq`/`gap_count`) `Gap` has no field for. Verified, no
changes needed: `discover_cameras.py`'s ONVIF client is already pinned
(`onvif-zeep-async`) with no hardcoded vendor URL pattern; `scripts/
ingest_capture.py`'s `ConsentRecord` (Day 5) is the reusable consent gate
for whenever a live-RTSP-to-store script exists — none does yet, so
there is nothing to enforce consent on today. 28 tests in the RTSP suite
(27 + 1 new), all against real sockets and the local RTSP server
fixture — zero hardware, as before.

**What remains manual:** everything genuinely needs a camera. Discovery,
transport, timing, and gap accounting are built and tested against
fixtures; the sub-stream-vs-downscaled-main comparison harness
(Day 12) still refuses to run against anything but matched real
captures. Nothing changed here today — still blocked on hardware, not
code.

## All five falsification tests, re-run today

| # | Test | Day 13 | Day 14 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PARTIAL | **PASSES** |
| 4 | Alert explainability | PARTIAL | **PASSES** (strict path; see caveat above) |
| 5 | Behaviour-query shape | PASSES (type-level) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator for a live answer |

**5 of 5 pass today**, up from 3 of 5 fully passing on Day 13. `tests/
test_falsification.py`: 9 tests, all green. Repo-wide:
**712 passed, 1 skipped, 8 deselected, 0 failures** (`not requires_weights
and not slow`), 0 regressions from Day 13's 661.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged today.

1. **Git remote / push, 8 days old.**
2. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`, ~11 days.**
3. **MEVA licence verification, ~13 days.**
4. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7, ~13
   days.**
5. **A physical camera, 2 days old.**

## Day 15, in order

1. **Steer callers away from `raise_alert()` toward `emit_alert()`** for
   anything that will reach a customer — today's own most valuable
   finding. A docstring cross-reference is not enough; consider whether
   `raise_alert()` should require an explicit `allow_unexplained=True`
   opt-in, or whether the two should be renamed so the permissive one
   reads as the exception.
2. **The motion-gate precision/selectivity investigation**, now three
   days deferred. Reframing the scorecard (Objective 3) did not
   investigate the 57 false negatives or why this fixture has zero
   scoreable non-moving frames — it changed what gets reported, not what
   gets investigated.
3. **A canonical `Observation -> hash` function** for
   `EvidenceCommitment.compute()` (ADR 0007) — still open from Day 13.
4. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
   replaced now that `gate.compute_saved` exists and is being read** —
   either measure a real `DetectorStage` cost or make the "estimate,
   unmeasured cost model" label louder in whatever surface consumes the
   scorecard next.
5. **Grow the golden set toward ~20 cross-boundary pairs at gap=1** if
   the normalization fix's effect is worth measuring precisely — a
   product decision on golden-set composition, not an engineering
   default.
6. **Connect a camera and run `scripts/discover_cameras.py`** — still
   the single step unblocking the whole ingest path, now 2 days old.
7. **Push.**
8. **MEVA licence verification** — still blocked on a human.
9. **The factor-graph solver**, once there is a measured reason to
   start it — unchanged from Day 13's list.

# Day 15

Branch `foundation/day-15`, off Day 14. Pinned venv. A seam is worse than a
hole, because a hole is visibly missing while a seam looks closed — today
closes two seams by deletion and migration, not documentation, and fixes a
scorecard that invited a wrong economic conclusion.

## The headline: the permissive alert path could only ever alert on nothing

Enumerating every caller of `events.raise_alert()` before changing anything
(full repo grep, `src/`, `tests/`, `scripts/`):

| # | Site | What it did |
| --- | --- | --- |
| 1 | `src/model/events.py:278` | the definition itself |
| 2 | `src/model/__init__.py:72,188` | re-export (not a call) |
| 3 | `tests/test_model_events.py:190` | `raise_alert(_hypothesis())`, expects rejection |
| 4 | `tests/test_model_events.py:195` | `raise_alert(factory())` for observed/inferred/**predicted** |
| 5 | `tests/test_falsification.py:255` | on an evidenced `ObservedEvent` |
| 6 | `tests/test_falsification.py:288` | on an `ObservedEvent` with **empty** `evidence_refs` — the permissive path's whole point |
| 7 | `tests/test_falsification.py:342` | a **`PredictedEvent`** triggering an alert |

**Zero production callers.** Every real call site was a test asserting the
permissive path's own permissiveness. Sites 5 and 6 migrate cleanly onto
`emit_alert()` — an evidenced event alerts, an unevidenced one now raises
where it used to silently pass (this is the actual falsification-test-4 gap
Day 14 only partly closed). Sites 4 and 7 are the finding: a `PredictedEvent`
alerting through `raise_alert()` could **never** be given an evidence chain,
because `assemble_evidence()` has no handler for `PredictedEvent` either
(ADR 0003 — a forecast can't be evidence). The permissive path's one
distinguishing feature beyond `emit_alert()` was a way to alert on something
that, by construction, could never have anything behind it. Nobody was
exploiting it in production; the test suite had simply never stopped
sanctioning it.

`raise_alert()`, `_admit_as_alert_trigger`, and the `AlertEligibleEvent` type
alias are **deleted**, not deprecated — every caller was migratable (the two
production-shaped ones onto `emit_alert()`, the two forecast-only ones by
removing the capability entirely, since it was never usable safely). If
forecast-paging is wanted later it needs its own explicit capability that
renders the `InferredEvent`/`ObservedEvent` chain behind the forecast, not a
relaxation of `emit_alert()`'s dispatch.

**STRUCTURAL**, tested on the actual public surface, not a convention:
`test_alert_package_exposes_no_alternative_emission_entry_point`
(`tests/test_model_alert.py`) asserts `src.model.alert.__all__` contains
exactly one alert-emitting callable (`emit_alert`), and
`test_raise_alert_deleted_from_events_module_and_package_root` asserts
`raise_alert` and `AlertEligibleEvent` are unreachable from
`src.model.events` and `src.model` by `hasattr`, not by grep. Falsification
test 4 re-run: **passes**, single path.

## Objective 2 — the two legacy world-coordinate paths: one migrated, one was never in scope

Re-examining Day 14's audit finding against the actual code, not its
one-line summary, split the two paths differently than the finding implied:

**`src/geometry/projector_vectorized.py::project_to_3d` is not a
world-coordinate path and is excluded from this migration.** Its output
columns are `X_uncalibrated`, `Y_uncalibrated`, `disparity_rel` —
self-labelled camera-space, uncalibrated, never routed through
`src/contracts/geometry.unproject()` (Day 2's depth-honesty fix,
`test_projector_output_columns_are_not_named_as_metric`, made this the
enforced state on purpose). `WorldPosition` requires real metres in a
world frame; wrapping this path's guessed-intrinsics disparity products in
it would misrepresent uncalibrated data as calibrated, undoing Day 2's fix
rather than extending Day 14's. `src/contracts/geometry.py`'s own docstring
already names the real migration this path needs ("deliberately not wired
into `projector_vectorized.py` yet... a separate, measured change with
before/after numbers") — a different objective than today's.

**The real bypass: `src/inspector/artifacts.py::clip_analysis` and
`src/data/scorecard.py::observability_partition`**, both loading genuine
world-frame, metric `agent_xyz` straight from a `.npz` clip with no
`twin_rev` recorded anywhere. Migrated:

- `src/model/world.py` gains `UNREGISTERED: Final[int] = -1`, a
  constructable sentinel `twin_rev` for exactly this case — legacy data
  with no recorded revision — and `WorldPositionArray`, the vectorized-array
  analogue of `WorldPosition` for code that cannot afford one dataclass
  instance per point (`np.diff`/`np.linalg.norm` over a `[T, A, 3]` clip).
  `WorldPosition.__post_init__` now accepts exactly `-1` as the sentinel and
  rejects every other negative unchanged. `distance_to()`, `reproject()`,
  and `WorldPositionArray.combine()` all raise `TwinRevError` the instant
  either side is `UNREGISTERED` — **including `UNREGISTERED` against another
  `UNREGISTERED`**, because two positions of unknown provenance are not
  known to share a revision.
- Both call sites now construct `WorldPositionArray(xyz_m=..., twin_rev=UNREGISTERED).xyz_m`
  immediately on load, before the array reaches `world_motion()` (itself
  untouched). The wrap is a pure boundary check: `.xyz_m` returns the same
  array object, no copy.

**Byte-identical, proven two ways**, per the SCOPE requirement that this be
the day's only numerical change:
1. `test_world_position_array_wrap_is_byte_identical_through_world_motion`
   (`tests/test_model_world.py`): fixed synthetic input through
   `world_motion()` raw vs. through the wrap, `tobytes()` equal.
2. The real thing: v3-indoor re-scored end to end after the migration
   reproduces Day 14's exact numbers — `gate.wake_fraction` 0.9067
   (816/900), `gate.recall_retained` 0.9340 (806/863), `gate.compute_saved`
   1.8667 ms/frame, `gate.miss_cost` 57 — unchanged to four decimal places.

**Re-audit for remaining bypasses**: `grep -rl "agent_xyz\|world_motion\|WorldPosition" src/` returns exactly `scorecard.py`, `inspector/artifacts.py`, and `model/world.py`/`model/__init__.py` — the two now-migrated call sites and the type's own module. **Remaining bypass count: 0**, for genuine world-frame data. (`project_to_3d`'s camera-space data is not a bypass of this contract — it was never subject to it, per above.)

## Objective 3 — the gate scorecard now shows its own denominator

`gate.wake_fraction` 0.9067 on v3-indoor invited "the gate wakes on
everything" as a verdict on the gate. Two additions make that reading
impossible without also seeing the number that explains it:

- **`dataset.moving_frame_fraction`**, a first-class scorecard metric:
  fraction of presented frames with world motion by *any* agent, camera-
  independent (reuses the `world_moved` array `observability_partition()`
  already computed, now exposed via `Partition.world_moved_any`). On
  v3-indoor: **0.9667** (870/900). A gate cannot beat the scene's own
  motion density, and this is why not to try.
- **`gate.wake_fraction` per condition bucket** (occupied / empty / night /
  degenerate), derived from the golden-set `Condition` taxonomy via
  `_condition_bucket()` — not literal tags; the taxonomy has no
  "occupied"/"night" member, and its actual Degradation category
  (`CAMERA_BUMP`, `LENS_SMUDGE`) has zero clips in v3-indoor, so
  "degenerate" here means `LIGHTS_TRANSIENT`/`GLARE` instead. Mapping and
  priority order are pinned by
  `test_condition_bucket_partitions_v3_indoor_exactly`.

Re-scored, full v3-indoor, per condition:

| condition | clips | presented | wake_fraction | recall_retained | moving_frame_fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| occupied | 25 | 750 | 0.9387 | 0.9734 | 1.0000 |
| empty | 1 | 30 | 0.0000 | 0.0000 | 0.0000 |
| night | 0 | 0 | n/a | n/a | n/a |
| degenerate | 4 | 120 | 0.9333 | 0.9333 | 1.0000 |

Two findings the table itself surfaces, not asserted separately: **the
"occupied" bucket's moving_frame_fraction is 1.0000** — every frame in
v3-indoor's 25 ordinary-condition clips contains motion by construction
(the speed-ladder authoring), so a 93.87% wake fraction there is a gate
essentially unable to beat a scene that never stops moving, not a
trigger-happy gate. And **v3-indoor's "night" bucket is empty**: the only
tag adjacent to steady artificial lighting, `EVENING_ARTIFICIAL`, never
appears in this dataset without also being `LIGHTS_TRANSIENT` — v3-indoor
contains a lighting-transient condition but no steady-low-light condition
distinct from it. `empty`'s single clip (`near_static__cam_a`) is the
dataset's only true-negative source and the only bucket where the gate
correctly stays fully asleep.

**STRUCTURAL**: `_GATE_METRIC_REQUIRES["gate.wake_fraction"]` now requires
both `gate.recall_retained` (Day 14) and `dataset.moving_frame_fraction`
(today) — emitting wake_fraction without either raises `ScorecardError`
before the scorecard is returned, checked against the scorecard's full,
final metric list (moved from checking only the four `gate.*` metrics in
isolation, which could not have caught a missing cross-cutting metric).
Tested: `test_wake_fraction_with_recall_retained_but_no_moving_fraction_raises`.

**Stated plainly, in the prompt's terms**: this table measures the gate's
mechanism — it is internally consistent, the pairing rule holds, the
per-condition breakdown is honest about where the aggregate's 90.67% comes
from. **It is not the Tier-1 economic claim.** That claim requires wake
fraction over 24 hours of real office footage including nights and
weekends, and v3-indoor — authored to contain moving agents, with an empty
"night" bucket and a single "empty" clip — cannot produce it. That
measurement does not yet exist.

## Objective 4 — ADR 0008: the parked forensic verdict is closed, unanswered

`docs/adr/0008-production-provenance-lost.md`, Accepted. The parked
objective (Day 3: a side-by-side forensic verdict, production
`vjepa2_vitl_int8.xml`/`.bin` vs. the 2026-07-31 reference export, on token
count and temporal alignment) is **closed as unanswerable**, not left
pending: it has been blocked on the same human handoff for 12 days with
zero movement, and "pending" implies forward motion is possible from
inside this repository, which it is not. What is actually known —
tubelet=2, confirmed three independent ways at Day 3 (D3.5) — is answered
**only for the Day-3 scripted export**, never for production, and that
will remain true regardless of how much longer the wait continues.

The 22 artifacts under `outputs/` this verdict would have validated or
condemned are recorded **`unattributable`**, not `pending`/`void or
unverifiable` — `scripts/rebuild_index.py --audit`'s per-artifact reason
now reads "unattributable (ADR 0008), not pending: predates provenance
coupling, and the pipeline that produced it no longer exists in this
form." Re-ran the audit against the real files: **0 artifact(s) carry
provenance, 22 are void or unverifiable** — same count as Day 3's D3.8,
now correctly labelled as terminal rather than transitional.

**STRUCTURAL, added today (was not enforced)**: loading the V-JEPA2 IR
without an export manifest raises. `src/provenance.py::require_export_manifest`
checks for `export_manifest.json` beside the model file and that it names
the file among its recorded artifacts, before `SemanticExtractor.__init__`
compiles anything — previously `PreprocessSpec.load_for_model()` gated
preprocessing correctness but nothing gated export provenance at load
time. Tested: five `require_export_manifest` unit tests (absent sidecar,
wrong-file sidecar, malformed JSON, success path, and the real Day-3
export at `models/export/2026-07-31/`), plus a source-order regression
test asserting the manifest check runs before `compile_model()`, the
pattern this repo already uses for other structural invariants. Scoped
today to the V-JEPA2 IR load path only — CoTracker3 and Depth-Anything-V2
weight loading are not yet covered; listed below, not claimed done.

## All five falsification tests, re-run today

| # | Test | Day 14 | Day 15 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged; `WorldPositionArray`/`UNREGISTERED` extend the same guarantee to vectorized legacy data) |
| 4 | Alert explainability | PASSES (strict path only) | **PASSES, single path** — `raise_alert()` deleted, `emit_alert()` is the only public alert-emission entry point |
| 5 | Behaviour-query shape | PASSES (type-level) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator |

`tests/test_falsification.py`: 8 tests, all green (9 on Day 14 — one test
existed solely to demonstrate the permissive path's permissiveness, which
no longer exists to demonstrate). Repo-wide: **741 passed, 1 skipped, 8
deselected, 0 failures** (`not requires_weights and not slow`), up from Day
14's 712 — net +29 from today's additions across four objectives, 0
regressions. `mypy --strict` on the CI-scoped file set: **35 pre-existing
errors, unchanged before and after today's changes** (verified both ways
via `git stash`) — all in `src/data/scorecard.py`, `src/semantics/
patch_mapping.py`, and `src/endurance/runner.py`, all bare `np.ndarray`
annotations that predate today and are out of today's scope; today's own
three new type sites (`WorldPositionArray`, `Partition.world_moved_any`,
`require_export_manifest`'s return) are fully typed and add zero new
errors.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged today except item 2's framing
(see ADR 0008: the artifact is still wanted, but its arrival now answers a
new question rather than resuming Day 3's).

1. **Git remote / push, 9 days old.**
2. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`, ~12 days.**
   Forensic objective closed unanswered (ADR 0008); still wanted for its
   own sake, refused at load without an export manifest if it arrives.
3. **MEVA licence verification, ~14 days.**
4. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7, ~14
   days.**
5. **A physical camera, 3 days old.**

## Day 16, in order

1. **The motion-gate precision/selectivity investigation**, now four days
   deferred. Reframing the scorecard (Day 14) and adding its denominator
   (Day 15) did not investigate the 57 false negatives — it changed what
   gets reported and how it reads, not what gets investigated.
2. **Extend `require_export_manifest` to CoTracker3 and Depth-Anything-V2
   weight loading** — ADR 0008's process change is scoped to the V-JEPA2
   IR only today; the other two model loads have no equivalent gate yet.
3. **A canonical `Observation -> hash` function** for
   `EvidenceCommitment.compute()` (ADR 0007) — still open from Day 13.
4. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
   replaced now that both `gate.compute_saved` and
   `dataset.moving_frame_fraction` exist and are being read together** —
   either measure a real `DetectorStage` cost or make the "estimate,
   unmeasured cost model" label louder in whatever surface consumes the
   scorecard next.
5. **The 35 pre-existing `mypy --strict` `[type-arg]` errors** in
   `src/data/scorecard.py`, `src/semantics/patch_mapping.py`, and
   `src/endurance/runner.py` — bare `np.ndarray` annotations, unchanged
   today, module-by-module the same way legacy strictness has moved
   before (see `mypy.ini`'s own stated policy).
6. **Connect a camera and run `scripts/discover_cameras.py`** — still
   the single step unblocking the whole ingest path, now 3 days old.
7. **Push.**
8. **MEVA licence verification** — still blocked on a human.
9. **The factor-graph solver**, once there is a measured reason to start
   it — unchanged from Day 13's list.

# Day 16

Branch `foundation/day-16`, off Day 15. Framing: Day 15 found v3-indoor's
night bucket empty and its occupied bucket at motion density 1.0000 — the
dataset cannot evaluate the gate because the floor tuned for tracking and
depth selects against the exact frames a gate exists for. Four objectives:
fix that, close the mypy debt, make the capture a one-afternoon task, and
report. No numerical-behaviour changes except where an objective
authorised one; the STRUCTURAL bar is unchanged.

## Objective 0 — push verification

Already done when this session started: `foundation/day-15`'s tip
(`2d55c97`, "docs: Day-15 report") matched `origin/foundation/day-15`
exactly, and `foundation/day-16` already existed locally, branched from
it. Re-verified rather than assumed: every `foundation/day-1` through
`foundation/day-15` local branch head matches its `origin/` counterpart
byte-for-byte; the full local commit graph (`git rev-list --all`, which
includes fetched remote-tracking refs) and the full remote graph
(`git rev-list --remotes=origin`) are the same *set* of commits, 0
difference either direction, not merely the same count. `main` itself
diverges from `origin/main` — `origin/main` carries a separate,
unrelated PR history from another collaborator (`Dalbirsm03`: RAG agent,
vector DB, Gaussian proxy work) that this repository's `foundation/day-N`
lineage never merged from or into. That divergence pre-dates this
session, is out of this day's scope, and every one of its commits is
already present locally (fetched, just not reachable from local `main`)
— stated plainly rather than left implicit. `foundation/day-16` itself
is not yet pushed; its three commits from today are new. See Day 17.

## Objective 1 — v4-gate: a set that can evaluate the gate

**Mechanism.** `GoldenClip.low_activity` (`src/data/golden.py`) mirrors
`hard_coverage`: exempt from the mint-time `>=0.80` observability floor
and from the aggregate check, reported on its own line. Same tag
discipline as `hard_coverage` — declared by authoring intent, not
assigned post hoc because a clip's score came in low; `mint_golden_set`
still refuses a set of nothing but exemptions (both tags combined count
toward that refusal now, not just `hard_coverage` alone). 4 new tests in
`tests/test_mint_acceptance.py`.

**The set.** `build_scenes_v4_gate` (`scripts/gen_synthetic_indoor.py`),
8 clips: two camera views of a genuinely empty room, one empty room with
a lights-off transient (untagged `low_activity` — an empty room is
trivially `NO_MOTION` on every frame, so these three clip the floor on
their own and are what gives it something to check), two views of one
occupant present the whole clip at `speed_scale=0.0`, and three
brief-entry clips (walks in over ~1/7 of the clip, stands still for the
rest; one also carries a lights-off transient after settling). Per-clip
`moving_frames/frames` spans 0.0000–0.1500 — inside the ~0.02–0.30 target
band but not reaching its top edge, because "a single brief entry"
bounds how much of a short clip can move without becoming a different
scene. Registered `synthetic-indoor-v4-gate`, lane S, same per-asset
clearance as v3 (`configs/datasets.yaml`) — same generator, same
primitives, the SMPL trap does not apply here either.

Authoring it surfaced and fixed two bugs the all-moving v2/v3 sets never
exercised: the manifest's `moving_frames` field was hardcoded to
`scene.frames` (every clip claimed 100% moving regardless of content)
until a dataset whose entire point is to be quiet made that specific
dishonesty visible; and the `agent_xyz`/`track_uv` payload reshape broke
on zero agents (`np.asarray` on an empty-per-frame list infers shape
`(T, 0)`, not `(T, 0, 3)` — nothing had exercised the zero-agent case
before `empty_room`).

**Validity matrix row.**

| dataset | motion_geometry | depth | appearance_semantics | point_tracking |
| --- | --- | --- | --- | --- |
| v3-indoor | PASS | REFUSED | REFUSED | REFUSED |
| v4-gate | PASS | REFUSED | REFUSED | **PASS** |

`point_tracking` was added as a fourth checked capability today —
registered in `validity.GATES` since Day 11 (`scripts/eval_tracking.py`
uses it directly) but never wired into `scripts/validity_matrix.py`, the
one script whose job is "which capabilities can this project evaluate."
Closing that gap is what surfaced the one cell that does **not** match
the pattern this objective assumed going in: v4-gate does not refuse
point_tracking the way v3 does. v3's representative clip scores corner
density 2.34e-04 (below the 1e-03 floor, REFUSED); v4-gate's scores
1.11e-03 (PASS) — confirmed across the full set, not just the sampled
clip, via `scripts/eval_tracking.py --set v4-gate`: all 8 clips pass,
vs. v3's 4-of-30. Static, unblurred scenes give a Shi-Tomasi corner
detector more stable local structure to lock onto than v3's fast-motion
clips do; reported as measured, not forced to match the assumption.

**Scorecards, side by side** (`docs/day16/motion_gate_v3_rescored.json`,
`motion_gate_v4_gate.json`; regenerated v3 number, byte-consistent with
Day 14's `891dd58`):

| metric | v3-indoor | v4-gate |
| --- | ---: | ---: |
| `gate.wake_fraction` | 0.9067 (816/900) | 0.1250 (30/240) |
| `gate.recall_retained` | 0.9340 | 0.0884 |
| `gate.compute_saved` (estimate) | 1.87 ms/frame | 17.50 ms/frame |
| `gate.miss_cost` | 57 frames | 134 frames |
| `dataset.moving_frame_fraction` | 0.9667 | 0.0000 |

**Neither scorecard is the Tier-1 economic claim.** v3 measures the gate
where motion is near-total; v4-gate measures it on authored quiet
scenes; the actual claim requires 24 hours of real office footage
including nights and weekends (`docs/capture_runbook.md`, Objective 3).
Synthetic quiet is not real quiet — a real empty corridor has sensor
noise, HVAC-driven shadow drift, and compression artifacts that authored
stillness lacks, and those are exactly what a background model reacts
to.

**Headline finding.** `gate.recall_retained`/`gate.miss_cost` for
v4-gate are not a clean read, and should not be quoted without this:
`gt_moved_from_render` (Day 9's silhouette-change ground truth, still
what `motion_gate_metrics` scores recall against) flags ~92% of frames
as "moved" in `long_static_occupant` — an agent with `speed_scale=0.0`,
whose `world_motion()` value (the Day-15 world-space signal) is `False`
on every single frame of the clip. Traced to
`scripts/gen_synthetic_indoor.py`'s gait animation: leg-swing phase is
`np.sin(2*pi*(t*4 + agent_id))`, a function of elapsed clip time, not of
`speed_scale` or `progress`. A "stopped" agent's legs keep swinging, so
its rendered silhouette never stops differing frame to frame even though
its world position never moves. The real cascade gate is not fooled —
the `occupied` condition bucket's wake_fraction is exactly 0.0000, i.e.
background-subtraction at 320x180 correctly never reacts to a few pixels
of leg wobble — so this is a ground-truth defect, not a gate defect: the
gate slept through frames the *scoring* function insists it should have
woken for, on the strength of a rendering artifact unrelated to real
motion. **Not fixed today** — SCOPE excludes numerical-behaviour changes
not authorised by an objective, and this is exactly the class of finding
Objective 1 asked to be surfaced as a headline rather than engineered
around. It extends the Day-9 partition (geometry synthetic-scorable,
appearance not) with a third, distinct failure mode from the one this
objective's prompt anticipated (missing sensor-noise/shadow-drift
model, which is also true and separately caveated above): **the
renderer cannot honestly render a stationary articulated agent either**
— two independent reasons synthetic data cannot carry the Tier-1 claim,
not one.

## Objective 2 — mypy strict debt: 35 fixed, 0 excluded

Day 15's report recorded "35 pre-existing errors, unchanged... verified
both ways via `git stash`." Re-measuring today under `.venv` (the
default dev venv) reproduced only 6 of the 35 — a materially different,
wrong number. Root cause: `locking-requirements.txt` pins `numpy==1.26.2`,
but `.venv` has drifted to `numpy==2.5.1` (`.venv-pinned`, python 3.10,
correctly holds `numpy==1.26.2`, matching the pin and matching CI's
stated floor). Under numpy 2.5.1's newer stubs, mypy's generic-args
check (`disallow_any_generics`, part of `--strict`) stops flagging bare
`np.ndarray` — so `.venv`'s mypy run silently swallowed 29 of 35 errors,
not because they were fixed, but because the measuring apparatus had
quietly changed. Verified in both directions (`git stash` on/off) under
both venvs before trusting either number. **All numbers in this report,
and all mypy invocations going forward, use `.venv-pinned`.** This is
itself the day's `[[check-the-measuring-apparatus]]` finding, on a
metric that isn't even one of the numerical ones this project usually
means by that phrase.

All 35, by file:

| file | lines | count | error class |
| --- | --- | ---: | --- |
| `src/data/validity.py` | 111, 111, 131, 132, 133, 135, 212, 212, 263, 263, 325 | 11 | `type-arg` (bare `ndarray`/`dict`) |
| `src/data/depth_eval.py` | 64, 79, 106, 149, 192, 231, 233, 234, 294 | 9 | 8× `type-arg`, 1× `no-untyped-def` |
| `src/data/scorecard.py` | 74, 75, 122, 445, 468 | 5 | `type-arg` (bare `ndarray`) |
| `src/semantics/patch_mapping.py` | 50, 51, 89, 131 | 4 | `type-arg` (bare `ndarray`) |
| `src/endurance/runner.py` | 69, 124, 207 | 3 | `type-arg` (bare `ndarray`) |
| `src/cascade/motion.py` | 322, 368 | 2 | `no-any-return` |
| `src/data/da2k_adapter.py` | 51 | 1 | `type-arg` (bare `ndarray`) |

**Fixed: 35. Excluded: 0.** Every site got a real type — `npt.NDArray[...]`
generic parameters, `dict[str, Any]` instead of bare `dict`, an explicit
parameter/return annotation where a numpy stub overload returns `Any`,
and a named closure (`_bucket_selector`) in place of a default-arg lambda
mypy could not infer against `Callable[[X], Y]`. Zero behaviour change —
type annotations only, confirmed by the full non-slow suite passing
unchanged before and after. No `mypy.ini` exclusion was needed, so there
was no "fix or explicitly exclude, no third option" call to make on any
of the 35 — the STRUCTURAL bar (`mypy --strict` over the declared scope,
zero errors) is met by fixing, not by narrowing scope. `mypy.ini` itself
is unchanged.

## Objective 3 — capture readiness: dry run, runbook, one crash fixed

`scripts/discover_cameras.py` run against the real local subnet: 0
devices responded (WS-Discovery clean), reported honestly — until the
manual-entry fallback, which raised a bare, uncaught `EOFError` from
`input()` on any non-interactive run with no `--manual`/`--rtsp-url`.
That is a crash standing in for a report, not "runs clean and reports
honestly" per the module's own documented contract (0 devices is not a
failure). Fixed at all three fallback call sites: `EOFError` is now
caught and converted to the same clean, `exit 0` outcome discovery
already uses when WS-Discovery itself finds nothing.

`scripts/capture_dry_run.py` (new): chains discover → RTSP connect/demux
(`src.ingest.rtsp.RtspIngestSession`) → decoded frames (`cv2.imdecode`
on real JPEGs sent MJPEG-over-RTP, a genuine decode of genuine bytes,
not a stand-in) → content-addressed store → registry entry, lane C →
Gap records on an induced sequence-number drop → condition tagging →
consent-record refusal — against
`tests/fixtures/minimal_rtsp_server.py`, no hardware. All six stages
pass. This does **not** merge the live-RTSP path and the file-based
`scripts/ingest_capture.py` path into one production script; that
bridging is real work a real camera would motivate, not a precondition
for running the capture, and is named as remaining manual work below.

`docs/capture_runbook.md` (new): placement (the overlapping pair, the
deliberate coverage gap, the far-field corridor view), the
overnight/weekend window the Tier-1 claim depends on — flagged as the
segment most likely to be skipped as boring — condition tags per
segment, consent collection (still gated on counsel review of
`docs/site_zero_consent_TEMPLATE.md` §7, unresolved), and the commands
in order.

**What remains manual, restated plainly:** everything that genuinely
needs a camera in a room — ordering hardware, placement, cabling, the
scripted walkthroughs, and leaving the cameras running untouched
overnight. Discovery, transport, decode, content-addressing, registry
shape, gap accounting, and condition tagging are all dry-run verified
against fixtures today; nothing in that list is still code work.

## All five falsification tests, re-run today

| # | Test | Day 15 | Day 16 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged) |
| 4 | Alert explainability | PASSES, single path | PASSES (unchanged; `raise_alert()` is still the un-steered permissive path, per Day 14's open finding) |
| 5 | Behaviour-query shape | PASSES (type-level) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator |

`tests/test_falsification.py`: 8 tests, all green, unchanged from Day
15. Repo-wide (`.venv-pinned`, `not requires_weights and not slow`), run
after all four commits landed: **1 real failure caught** —
`test_golden_sets.py::test_every_minted_version_is_retained` hardcodes
the exact set of versions expected under `configs/golden/`, deliberately,
so an accidental deletion is caught; minting v4-gate tripped it exactly
as designed, not a false alarm. Fixed by adding `"v4-gate"` to the
expected set (separate commit, `31c2ac3`). Final: **745 passed, 1
skipped, 8 deselected, 0 failures.** `mypy --strict` on the declared
scope: **0 errors** (down from 35; see Objective 2), verified
under `.venv-pinned`.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged today.

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — forensic
   objective closed unanswered (ADR 0008); still wanted for its own
   sake.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7** —
   blocks real consent collection for Objective 3's capture, not just
   the erasure-obligation question it was already blocking.
4. **A physical camera** — the runbook (Objective 3) and the dry run
   are as far as this can go without one, now 4 days old.

## Day 17, in order

1. **Push `foundation/day-16`** (Objective 0's own pattern: each day's
   push happens at the start of the next). Verify the remote graph
   again after.
2. **Order cameras and run the office capture** per
   `docs/capture_runbook.md` — the single remaining step before the
   real Tier-1 wake-fraction number exists. Nothing else on this list
   outranks it.
3. **Do not trust v4-gate's `gate.recall_retained`/`gate.miss_cost` at
   face value** until `gt_moved_from_render`'s gait-animation
   sensitivity is addressed — either make leg-swing phase a function of
   `progress`/`speed_scale` so a stopped agent's silhouette actually
   stops changing, or move recall scoring for near-static clips onto
   `world_motion()` instead of silhouette-diff. Both are numerical-
   behaviour changes and need explicit authorisation before either
   lands.
4. **The motion-gate precision/selectivity investigation**, now five
   days deferred — Day 16 investigated *a* discrepancy in the gate's
   ground truth, not the original one (v3's 57 false negatives).
5. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
   still open from Day 14, still nothing in the type system stopping a
   caller from reaching for the permissive one.
6. **A canonical `Observation -> hash` function** for
   `EvidenceCommitment.compute()` (ADR 0007) — still open from Day 13.
7. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should
   be replaced** — unchanged ask from Day 14/15, now with a second data
   point (v4-gate's 17.5 ms/frame estimate) resting on the same
   unmeasured multiplier as v3's 1.87.
8. **Bridge the live-RTSP path and `scripts/ingest_capture.py`** into
   one production capture script — not a blocker for Objective 3's
   capture (file-based ingest per segment works today), but the
   obvious next simplification once a camera exists to motivate it.
9. **MEVA licence verification** — still blocked on a human.
10. **The factor-graph solver**, once there is a measured reason to
    start it — unchanged from Day 13's list.

# Day 17

Branch `foundation/day-17`, off Day 16. Framing: Day 16 found a drifted
default `.venv` silently suppressing 29 of 35 mypy errors — the seventh
time the instrument, not the code, was the defect, and the widest blast
radius yet, since any measurement taken through that interpreter is
suspect. Objective 1 removes the possibility structurally and audits
what it may have contaminated. Day 16 also found the v4-gate renderer
animates gait against elapsed time rather than agent speed, so the one
clip authored to be motionless had a moving silhouette. Objectives 2
and 3 fix and re-measure.

**First line, per this day's own instruction: no headline number in the
current report rests on the drifted environment.** The provenance audit
(Objective 1c) found exactly one number that WAS measured under it — Day
2's cascade cost, 2.97% of one core — and the project caught this itself,
on Day 3, four days before today: `env_gate.py`'s first run measured the
same environment directly (numpy 2.5.1, opencv 5.0.0, both against pins)
and Day 6 re-measured under the pinned stack, landing at 4.57%, BUDGET
MISSED — not the 2.97% originally reported. Reconfirmed again today: 4.50%
of one core, same verdict. Nothing today changes what was already true:
the product budget has been missed since the number was first measured
honestly, three days into the project's first week.

## Objective 0 — push verification

`git push --all origin`: `foundation/day-16` pushed clean (new branch on
origin). `git push --tags`: up to date. Verified after: every
`foundation/day-1` through `foundation/day-17` local branch head matches
its `origin/` counterpart exactly, and the full commit-object set is
identical in both directions (`git rev-list --all` vs `git rev-list
--remotes=origin`, 0 difference either way). `main` still diverges from
`origin/main` — the same pre-existing, unrelated collaborator history
(`Dalbirsm03`) documented Day 16, out of this day's scope, not
force-pushed. Pushed again at the end of the day (below), per this day's
new standing rule: push at the end of every day, not only the start.

## Objective 1 — environment consolidation and measurement provenance audit

**1a — `.venv` deleted, not repaired.** `.venv-infinigen` (Day 11, python
3.11 for Infinigen) is untouched — a distinctly-named environment already,
not the drifted default. Verified the literal suggested check first:
`python -c "import src"` does **not** fail without a venv activated —
`src/__init__.py` is deliberately import-free (packaging/CI/mypy must be
able to touch the package root cheaply), so a bare interpreter satisfies
it via ordinary CWD-relative import regardless of which environment it is.
That check tests nothing. The meaningful version — `import
src.data.scorecard`, which transitively needs numpy — found a worse, still
-live risk that deleting `.venv` does not fix: this machine's bare
`python`/`python3` on `$PATH` resolves to a **miniconda base environment**
with its own numpy (**2.3.1 — a third, distinct version**, neither the
pin nor the deleted `.venv`'s 2.5.1), and that import succeeds silently
through it. Documented, not silently worked around — this is exactly why
1b exists as the actual mitigation rather than the directory deletion
alone.

**1b — `scripts/env_gate.py`, two new rows, checked first:**

```
CHECK                        STATUS  DETAIL
interpreter identity         PASS    .venv-pinned/bin/python (sys.prefix=.../.venv-pinned)
no stray project venvs       FAIL    1 other venv(s) on disk: .venv-infinigen
```

`interpreter identity` compares `sys.prefix`, not `sys.executable`. First
implementation compared `Path(sys.executable).resolve()`, which follows a
venv's `bin/python` symlink straight back to the base interpreter it was
built from — ordinary venv construction, not drift — and reported the
**pinned venv itself as unpinned**. Caught by running the gate against
itself before trusting it; fixed to `sys.prefix`, which venv activation
sets to the venv's own directory regardless of how the executable file is
implemented. `no stray project venvs` scans for any other venv
(`pyvenv.cfg`) on disk and deliberately does not exempt
`.venv-infinigen` — a second venv is the risk this row names, however
well-motivated, so the gate **fails today** until that is resolved or
formally re-justified. Left failing rather than quietly exempted: exactly
the discipline the row exists to enforce on everyone else.

**1c — measurement provenance audit.** Full table below; sources of
truth were embedded scorecard provenance, git-commit timestamps against
when `.venv` demonstrably diverged (its own filesystem mtime, unchanged
from 31 Jul 16:35 through today's deletion — whatever was installed then
is exactly what Day 16 found), and reproducibility itself (a number that
reproduces bit-identical under `.venv-pinned` is strong evidence its
first measurement was too, since a genuinely different numpy/opencv
build changes floating-point behaviour enough that an exact integer
frame-count match would be a remarkable coincidence otherwise).

| Measurement | Day | Environment | Status |
| --- | --- | --- | --- |
| Cascade bench progression (29.5%→22.5%→4.4%) | 1 | `.venv` (unpinned) | verified_drifted; superseded narratively, not a current claim |
| Cascade cost 2.97% of one core | 2 | `.venv` (unpinned) | verified_drifted — caught by the project itself (Day 3), re-measured Day 6 (4.57%, BUDGET MISSED), reconfirmed today (4.50%, same verdict) |
| env_gate.py's own first run (numpy 2.5.1, opencv 5.0.0, direct measurement) | 3 | `.venv` (unpinned) — this IS the evidence | verified_drifted (self-documenting) |
| Golden-vector cosine fix (0.332 → 0.999987) | 3, after 22:10 | `.venv-pinned` | verified_pinned |
| Cascade cost re-measurement, 4.57%, BUDGET MISSED | 6 | `.venv-pinned` | verified_pinned; reconfirmed today at 4.50% |
| Envelope calibration (`min_foreground_fraction` 0.002, threshold 115.2px) | 7 | `.venv-pinned` (git-committed well after the Day-3 fix) | verified_pinned by timing; embedded stack string was a static config constant until today's fix (Objective 1c structural change) — not itself proof before today |
| Depth validity: v3-indoor rank correlation −0.5924, 30 clips/120 frames | 9 | `.venv-pinned` (presumed) | unknown — no live fingerprint recorded; LOW risk, qualitative verdict (refuses) independently reconfirmed via a different single-clip method Day 16 (−0.6309, same refusal) |
| Appearance validity: texture energy 3.03, material diversity 15.17 | 10 | `.venv-pinned` (presumed) | verified_pinned — reproduced EXACTLY (3.031, 15.168) Day 16 |
| Point-tracking gate / CoTracker3 pin, v3 refuses | 11 | `.venv-pinned` (presumed) | verified_pinned — refusal pattern reproduced Day 16 |
| `motion_gate_v3_with_baselines.json` | 12 | `.venv-pinned` (inherited envelope stack only) | unknown at the scoring level — no live fingerprint on the scorecard itself before today's fix |
| Gate reframe: wake_fraction 0.9067 (816/900), recall_retained 0.9340, miss_cost 57 | 14 | `.venv-pinned` (presumed) | verified_pinned — reproduced BIT-IDENTICAL Day 16 and again today |
| mypy "35 pre-existing errors, verified both ways via git stash" | 15 | `.venv-pinned` (that day's own claim) | verified_pinned — reproduced EXACTLY (35, same file/line/class) Day 16 and Day 17 |
| Retrieval-metric sweep, byte-for-byte reproducing Day 12 | 15 | `.venv-pinned` (presumed; reproducibility is the evidence) | verified_pinned |
| v4-gate scorecards, mypy 35→0, validity matrix, 745 tests | 16 | `.venv-pinned` — direct session evidence | verified_pinned |
| Transient mypy check showing "6 errors" (never committed) | 16 | `.venv` (unpinned, deleted today) | verified_drifted — caught and corrected within the same session; the finding that triggered today's Objective 1 |
| Cascade bench re-run (4.50%), env_gate extension, gait fix, v4.1-gate | 17 | `.venv-pinned` — direct session evidence | verified_pinned |

**STRUCTURAL fixes so this audit is a query next time, not another
investigation:**
- `src.data.scorecard.Scorecard.measurement_environment` (new field):
  live `sys.prefix`/`sys.executable`/`library_versions()`, captured fresh
  inside `compute()`. Distinct from `envelope.envelope_measured_stack`,
  which is *inherited* from whatever calibrated the loaded envelope file
  — potentially days earlier, by a different process — and was being read
  as though it described the scorecard's own run.
- `src.provenance.current_stack_string()` (new): a live `"opencv X /
  numpy Y / python Z"` fingerprint, replacing `config.cascade
  .measured_stack` (a hardcoded config constant) in `cascade_bench.py`'s
  "Measured on" line and `measure_envelope.py`'s written provenance —
  both printed the pinned string unconditionally regardless of what
  interpreter actually executed. Found while building the audit table,
  not assigned in advance: the exact class of defect this whole objective
  is about, found a second time, in the tooling meant to report it.

## Objective 2 — gait animation decoupled from elapsed time (AUTHORIZED)

Fixed: `Agent.progress_at(t)` (path-completion fraction, clamped at
arrival — the same value `position_at(t)` already computed) now drives
gait phase, in place of `t` directly. A `speed_scale=0.0` agent has
`progress_at(t) == 0.0` for every frame, so phase is one constant value
all clip long and the silhouette stops changing; an agent that arrives
and stops keeps that constant from the moment it stops walking, same as
a real person.

**Structural test** (`test_stationary_agent_silhouette_is_bit_identical_
across_frames`): renders `long_static_occupant` directly and asserts
every frame's rgb/depth_m/instances equals frame 0. Verified to fail
against the pre-fix formula first (0.182% of pixels differ at frame 1)
before confirming it passes against the fix — the regression test
actually regresses.

**Audit for other elapsed-time-driven defects** (idle sway, head turn,
arm swing, texture scroll): none exist. The agent model has no other
periodic animation; the room-shell texture is spatial (keyed on pixel
`xx`/`yy`, not `t`) and generated once per (scene, camera) pair before
the frame loop, not resampled per frame. Gait phase was the only
elapsed-time-driven animation in the generator.

**v2/v3 impact: NONE**, verified both mathematically and empirically.
Neither `build_scenes` (v2) nor `build_scenes_v3` ever overrides
`Agent.speed_scale` from its default of 1.0 — v3's speed ladder varies
path length, not `speed_scale`. For `speed_scale=1.0`,
`progress_at(t) ≡ t` exactly, so phase is bit-for-bit unchanged.
Confirmed by re-rendering both sets under their **actual recorded
seeds** (v2: 20260801, the CLI default; v3: 20260808 — not the default,
caught only by reading `dataset_manifest.json`'s own `seed` field after
an initial re-render with the wrong seed produced a false "all 30 clips
changed" result) and diffing every clip's `content_sha` against the
committed golden manifests: **0 mismatches, both sets, 9 + 30 clips.**
Neither `set_sha` needs to change.

## Objective 3 — v4.1-gate: the fixed, honest quiet set

Minted `v4.1-gate`: v4-gate's same 8 scenes, same conditions, same
`low_activity` tags, re-rendered under the fix. `v4-gate` itself
untouched — the record of what "authored quiet" measured before the fix
existed, the same way v1 stayed the record after v2 superseded it (this
is the one case where a new set exists **because the old one's numbers
were wrong**, not because it measures something new —
`_supersession_for` records `v4.1-gate` → supersedes → `v4-gate`
explicitly).

Verified fixed, not just re-rendered: `long_static_occupant`'s
silhouette-based motion is now **0/40 frames**, down from **38/40** in
v4-gate — and for the first time agrees with `world_motion()`, which was
always 0/40. Validity matrix row matches v4-gate exactly (expected: same
scenes) — PASS motion_geometry, REFUSE depth/appearance, PASS
point_tracking.

**v3 / v4-gate / v4.1-gate, side by side** (`docs/day17/`):

| metric | v3-indoor | v4-gate | v4.1-gate |
| --- | ---: | ---: | ---: |
| `gate.wake_fraction` | 0.9067 | 0.1250 | 0.1250 |
| `gate.recall_retained` | 0.9340 | **0.0884 — CONTAMINATED, do not quote** | nan (0/0) |
| `gate.compute_saved` (estimate) | 1.87 ms/frame | 17.50 ms/frame | 17.50 ms/frame |
| `gate.miss_cost` | 57 | **134 — CONTAMINATED, do not quote** | 0 |
| `dataset.moving_frame_fraction` | 0.9667 | 0.0000 | 0.0000 |

`wake_fraction` and `compute_saved` are **identical** between v4 and
v4.1 — direct confirmation of the diagnosis: the real cascade gate
(background-subtraction on actual pixels) was never fooled by the gait
artifact, so its own measured behaviour is unchanged by the fix. Only the
ground-truth-dependent metrics move, from a contaminated-but-plausible
0.0884/134 to an honest nan/0.

v4.1's nan/0 is a finding in itself, not a clean win: zero frames in the
whole set are labelled ABOVE_ENVELOPE after warmup, because
`brief_entry`'s only real motion (frames 1–6) falls entirely inside the
gate's 10-frame warmup window and is excluded from scoring before
recall/miss_cost ever see it. v4-gate's contaminated numbers were never
actually measuring "does the gate wake correctly for a brief entry"
either — they were measuring the gait artifact throughout. This set's
recall/miss_cost dimension does not currently exercise that question at
all. Day-18 item, not fixed today.

**Neither v3, v4-gate, nor v4.1-gate is the Tier-1 economic claim.**
That requires 24 hours of real office footage including nights and
weekends (`docs/capture_runbook.md`).

## Objective 4 — synthetic-evaluability partition, two corrections

`.claude/skills/iron-eval-discipline/SKILL.md` updated in place (full
text there; summarised here):

1. **Point tracking is not fixed on either side.** Refused on v3-indoor
   (corner density 2.34e-04 — motion blur starves the corner detector),
   passed on v4-gate (1.11e-03 — static scenes feed it). Same generator,
   same capability, opposite verdicts: point tracking's validity gate
   measures a property of the SET (local texture/corner availability),
   not of the capability. New rule recorded: a set must declare texture
   density (or run the gate) before assuming a verdict transfers from a
   sibling set in the same family.
2. **Third category: signal-absence capabilities** (the motion gate is
   the type case) — value concentrated in frames where nothing happens,
   which synthetic data models worst of the three, because a renderer's
   "nothing happening" is perfect noiseless stillness and real absence
   never is. Worked example: v4-gate's gait defect (this day's own
   Objective 2) — the generator could not represent stillness at all
   until today, and fixed, still lacks a sensor-noise model. **Recorded,
   not answered:** does the generator have one? As of today, no — an
   empty room still renders bit-identical frame to frame. Until it does,
   every quiet-scene measurement is optimistic by an unknown margin, and
   the gate cannot be honestly evaluated on synthetic data regardless of
   how the scenes are authored.

## All five falsification tests, re-run today

| # | Test | Day 16 | Day 17 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged) |
| 4 | Alert explainability | PASSES, single path | PASSES (unchanged; `raise_alert()` still un-steered, Day 14's open finding) |
| 5 | Behaviour-query shape | PASSES (type-level) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator |

`tests/test_falsification.py`: 8 tests, all green, unchanged. Repo-wide
(`.venv-pinned`, `not requires_weights and not slow`): **748 passed, 1
skipped, 8 deselected, 0 failures**, up from Day 16's 745 (+3: the
`measurement_environment` test, `current_stack_string` test, and the
stationary-agent structural test). `mypy --strict` on the declared
scope: **0 errors**, unchanged from Day 16, verified under `.venv-pinned`
— the only environment any measurement in this report now uses, by
construction (Objective 1a).

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged today.

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — forensic
   objective closed unanswered (ADR 0008); still wanted for its own
   sake.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7** —
   blocks real consent collection for the office capture.
4. **A physical camera** — the runbook and dry run are as far as this
   can go without one, now 5 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-17` pushed clean. `git push
--tags`: up to date. Re-verified: every `foundation/day-N` branch
matches its remote exactly; commit-object sets identical in both
directions (0 difference); `main`'s pre-existing divergence unchanged,
untouched. Per this day's new standing rule, push now happens at the
start **and** end of every day — the newest work is always the exposed
work, and a day that ends unpushed is 16 days' pattern broken exactly
once too often.

## Day 18, in order

1. **`env_gate.py`'s "no stray project venvs" row is failing right
   now**, on `.venv-infinigen`. Resolve deliberately: delete it if
   Infinigen work is done, or formally re-justify keeping it (and decide
   whether the gate should distinguish a justified exception from an
   accidental one, rather than staying red indefinitely).
2. **v4.1-gate's recall/miss_cost dimension is currently unexercised** —
   `brief_entry`'s only real motion falls inside the 10-frame warmup
   window. Either shorten warmup for this set's purposes, lengthen the
   entry, or accept that this set answers "does the gate correctly sleep
   through quiet" and needs a sibling for "does it correctly wake for a
   brief entry" — a scene-authoring decision, not a code fix.
3. **The generator has no sensor-noise model** (Objective 4's recorded
   question, still open). Until it does, `dataset.moving_frame_fraction`
   near zero is optimistic by an unknown margin on every set that has
   one. Decide whether to build one or accept that synthetic quiet-scene
   numbers are permanently mechanism-checks only.
4. **Order cameras and run the office capture** per
   `docs/capture_runbook.md` — still the single remaining step before a
   real Tier-1 wake-fraction number exists, now unchanged in priority
   since Day 16.
5. **The motion-gate precision/selectivity investigation**, now six days
   deferred — Day 17 investigated a different discrepancy in the gate's
   ground truth than the original one (v3's 57 false negatives).
6. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
   still open from Day 14.
7. **A canonical `Observation -> hash` function** for
   `EvidenceCommitment.compute()` (ADR 0007) — still open from Day 13.
8. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
   replaced** — unchanged ask from Day 14/15/16, now with a third data
   point (v4.1-gate's 17.5 ms/frame estimate, identical to v4-gate's)
   resting on the same unmeasured multiplier.
9. **Bridge the live-RTSP path and `scripts/ingest_capture.py`** into
   one production capture script — not a blocker for the capture itself.
10. **MEVA licence verification** — still blocked on a human.
11. **The factor-graph solver**, once there is a measured reason to
    start it — unchanged from Day 13's list.

---

# Day 18

**Cascade bench did not reproduce Day 17's 4.50% today, and neither
attempt is trustworthy enough to quote.** First run, 5.10%, taken while
two CPU-heavy `pytest` suites (launched to verify Objective 2) were
running concurrently on the same machine — a wall-clock benchmark
sharing cores with ~200% of additional CPU demand. Second run, taken
*after* those suites finished and the machine was otherwise idle, was
**worse, not better: 9.28%.** `pmset -g therm` explained it:
`CPU_Speed_Limit = 46`. `pmset -g batt` explained that: the machine was
on battery power at 5%, and macOS throttles CPU hard to protect
remaining charge, independent of thermal state or concurrent load — the
"idle" second reading was on a CPU capped at under half speed. Neither
5.10% nor 9.28% is comparable to Day 17's figure, and neither is
reported as a measurement of anything except today's own power state.
**Day 17's 4.50% stands, unchanged, as the last trustworthy cascade-cost
figure**; a clean re-run on AC power is Day 19's first item, not
buried in it. This is itself Day 6's-and-Day-17's-own finding recurring
a third time in a new shape: the instrument was wrong again, and this
time the instrument was the laptop's power source, not the interpreter.

Framing, from the standing prompt: a check that stays red trains people
to ignore red, and a value that is present but meaningless satisfies a
presence rule without satisfying its purpose. Objective 1 closes the
first (the stray-venv row has been red since Day 17); Objective 2 closes
the second (`gate.recall_retained` has been a silent NaN since v4-gate
was minted on Day 16). Objective 3 closes the two provenance gaps Day
17 named by name; Objective 4 answers the `main` divergence
investigation the standing rules keep deferring.

## Objective 0 — push, start of day

`git push --all origin`: `foundation/day-18` pushed clean (new branch).
`git push --tags`: up to date. Verified: `foundation/day-18` local HEAD
matched `origin/foundation/day-18` exactly immediately after push, and
every prior `foundation/day-N` branch remained unchanged and matching.
`main` rejected — `[rejected] main -> main (non-fast-forward)`, the
same failure noted every day since Day 16. Objective 4 below is this
day's actual investigation of that failure, not a repeat of the
deferral.

## Objective 1 — `.venv-infinigen` resolved to green-with-reason

Took the preferred path, not the fallback: `.venv-infinigen` (1.3 GB,
untracked, Python 3.11 for Infinigen/`bpy`) moved out of the repo tree
entirely, to a sibling directory (`../project-iron-infinigen-venv`) next
to the repo root. Fixed the venv's own internal absolute-path references
after the move — `bin/activate`'s `VIRTUAL_ENV`, and the shebang line in
every console script (`pip`, `f2py`, `cythonize`, `tqdm`, `trimesh`, and
others) — verified by running `../project-iron-infinigen-venv/bin/python
-m pip --version` successfully post-move, not merely by moving the
directory and assuming it still works. Updated the three scripts that
document how to invoke it (`infinigen_generate.py`,
`run_infinigen_gate.py`, `probe_cycles_throughput.py`) to the new path;
left the Day-11/12 historical docs (`docs/day12/infinigen_throughput.md`,
`configs/datasets.yaml`'s Day-11 comment) untouched, since they describe
what was literally run at the time under the old, then-correct, in-tree
path — this project's day-reports are append-only, not retroactively
rewritten.

**Gate result, this machine, right now:**

```
CHECK                        STATUS  DETAIL
interpreter identity         PASS    .venv-pinned/bin/python (sys.prefix=.../.venv-pinned)
no stray project venvs       PASS    only .venv-pinned exists on disk
import torch                 PASS    torch 2.2.0 (pinned)
import openvino              PASS    openvino 2024.6.0 (pinned)
import cv2                   PASS    opencv-python-headless 4.8.1.78 (pinned)
import numpy                 PASS    numpy 1.26.2 (pinned)
model PRODUCTION vjepa_xml   FAIL    missing (blocked on humans, ADR 0008)
model PRODUCTION vjepa_bin   FAIL    missing (blocked on humans, ADR 0008)
model cotracker_checkpoint   PASS    101.9 MB sha256 2670d4562ed69326...
pip check                    PASS    No broken requirements found.
seeds / threads              PASS    seed=0 numpy=yes torch=yes torch_threads=6
```

**9 of 11 checks pass; the gate returns PROCEED.** The two failures are
the pre-existing, separately-tracked production-artifact block (ADR
0008, blocked on a human, untouched by today's work) — not a new
failure and not this objective's concern. The stray-venv row, red since
the moment it was added, is green for the first reason a permanently-red
check should ever go green: the thing it was flagging is actually gone,
not exempted.

**STRUCTURAL test, run for real, not just described:** created
`.venv-test` at the repo root, ran the gate — `no stray project venvs
FAIL 1 other venv(s) on disk: .venv-test`, full gate exit code 1 — then
removed it and re-ran: back to PASS, exit 0. Encoded permanently in
`tests/test_env_gate.py` (4 new tests, monkeypatching `REPO_ROOT` so
they never touch the real repository tree): a clean tree passes, an
unnamed newly-created stray venv fails by name, a non-venv directory is
correctly ignored, and a regression guard that `.venv-infinigen` does
not exist under `REPO_ROOT` — so a future session that recreates it
in-tree (e.g. by following a stale doc) fails loudly instead of the row
quietly going red again for another week unnoticed.

## Objective 2 — `Undefined`, not NaN, for a zero-denominator gate metric

`gate.recall_retained` on a clip with zero moving frames divides 0 by 0.
NaN was arithmetically correct and structurally useless: it satisfied
the Day-14 co-emission rule (`gate.wake_fraction` never ships without
`gate.recall_retained`), because that rule only checks metric *names*,
not whether the paired value carries information. v4.1-gate's own
scorecard was exhibit A — `wake_fraction 0.125` next to a
`recall_retained` that was present in name only.

Added `Undefined(reason=...)`, a frozen sentinel distinct from a bare
float, and `MetricValue = float | Undefined`. `_metric_with_baselines`
— the sole place a `Metric` is constructed in this module — now raises
`ScorecardError` naming the metric and the likely cause when handed a
bare NaN, and accepts an explicit `Undefined` instead. Renders as
`undefined (<reason>)` in `Scorecard.render()`, never as a number;
serializes as `{"undefined": true, "reason": "..."}` in the JSON, never
as `NaN` (which is not even valid JSON).

**The fix generalized past `recall_retained` alone, on contact with the
test suite, not by design up front.** The first version scoped the raise
narrowly to `recall_retained`'s own zero-moving-frames case and left
`gate.wake_fraction` / `gate.compute_saved` / `dataset.moving_frame_fraction`
/ `coverage.observable_fraction` computing a bare NaN in their own
zero-*presented*-frames edge case, same as before. That edge case is not
hypothetical — it is exactly what `test_scoring_refuses_a_clip_whose_bytes_do_not_match_the_manifest`
already exercised (a golden set where every clip fails its `content_sha`
check scores zero clips, zero presented frames) and what four
`test_synthetic_indoor.py` fixtures hit by using 8-frame clips against a
10-frame gate warmup window. The general "NaN raises" rule, taken
literally, turned both into crashes instead of returned scorecards —
which is what the pre-existing tests correctly expect NOT to happen (a
fully-refused set must still return a `Scorecard` with `clips_scored ==
0` and a caveat, not raise). Generalizing `Undefined` to all four sites,
rather than carving a narrower exception into the raise, is what let
every existing test keep its original assertions and pass unchanged —
the fix that survives contact with the suite it did not anticipate was
the more honest one, not a special case bolted onto the literal ask.

Applied per condition bucket in `_gate_rates_from_totals`, not only in
the aggregate, per the objective: a bucket with zero moving frames
inside an otherwise-active set is exactly as undefined as a quiet
aggregate, and the v4.1-gate table below shows all four buckets
(`occupied`, `empty`, `night`, `degenerate`) correctly reading undefined
while `wake_fraction` stays a real per-bucket number.

6 new tests in `tests/test_gate_reframe.py`: NaN raises and names the
metric; the raise message names a likely cause; `Undefined` is accepted,
skips margin/flag computation, and still carries its registered
baselines; a scorecard containing an `Undefined` metric renders
`"undefined (...)"` and contains no literal `"nan"` anywhere;
`_gate_rates_from_totals` produces `Undefined` with zero moving frames,
and a real number when moving frames exist. `mypy --strict`: 0 errors,
47 files. Full non-`requires_weights`/non-`slow` suite: **758 passed, 1
skipped, 8 deselected, 0 failures** — up from Day 17's 748 (+10: 4
stray-venv tests, 6 `Undefined`/NaN tests).

**Re-emitted v3-indoor and v4.1-gate under the corrected semantics**
(`docs/day18/`), the pair the objective asked to read together:

| metric | v3-indoor (motion-saturated) | v4.1-gate (quiet) |
| --- | ---: | ---: |
| `gate.wake_fraction` | 0.9067 | 0.1250 |
| `gate.recall_retained` | 0.9340 | undefined (no moving frames in denominator) |
| `gate.compute_saved` (estimate) | 1.87 ms/frame | 17.50 ms/frame |
| `gate.miss_cost` | 57 | 0 |
| `dataset.moving_frame_fraction` | 0.9667 | 0.0000 |

**What the pair now says:** `wake_fraction` 0.9067 on a set authored to
be almost entirely in motion, against 0.1250 on a set authored to be
almost entirely still, is the first real evidence the gate does what a
gate is for — its value concentrates where nothing happens, not where
everything does. `recall_retained` on v3 is a real 0.9340 because v3 has
863 moving frames to measure recall against; on v4.1 it is honestly
undefined because there were zero, not a fabricated 0.0 or a silent NaN
pretending to be one. Neither number is the Tier-1 economic claim, which
still requires 24 hours of real office footage including nights and
weekends (`docs/capture_runbook.md`) — unchanged since Day 16.

## Objective 3 — cascade bench and the provenance audit, both closed out

**Cascade bench: not reproduced, not quoted — see the report's first
line above.** Two attempts today, both `.venv-pinned`, both a wrong
number for a machine-state reason unrelated to the interpreter: 5.10%
(concurrent CPU load from Objective 2's own test runs) and 9.28% (a
battery-throttled CPU at 5% charge, `CPU_Speed_Limit = 46`, measured
*after* the concurrent load had cleared). Day 17's 4.50% is not
superseded by either — it remains the standing figure until a clean
re-run happens on AC power, which did not happen today.

**Provenance audit, completed.** Every row from Day 17's table is
carried forward; two are resolved today, two new rows are added, and —
per today's rule — nothing is left as a bare "unknown": every
measurement is now `verified_pinned`, `verified_drifted`, or explicitly
**not to be quoted until re-measured**, with a reason and a Day-19
action attached.

| Measurement | Day | Environment | Status |
| --- | --- | --- | --- |
| Cascade bench progression (29.5%→22.5%→4.4%) | 1 | `.venv` (unpinned) | verified_drifted; superseded narratively |
| Cascade cost 2.97% of one core | 2 | `.venv` (unpinned) | verified_drifted — caught by the project itself (Day 3), re-measured Day 6 (4.57%), reconfirmed Day 17 (4.50%) |
| env_gate.py's own first run (numpy 2.5.1, opencv 5.0.0) | 3 | `.venv` (unpinned) — this IS the evidence | verified_drifted (self-documenting) |
| Golden-vector cosine fix (0.332 → 0.999987) | 3, after 22:10 | `.venv-pinned` | verified_pinned |
| Cascade cost re-measurement, 4.57%, BUDGET MISSED | 6 | `.venv-pinned` | verified_pinned; reconfirmed Day 17 at 4.50% |
| Envelope calibration (`min_foreground_fraction` 0.002, threshold 115.2px) | 7 | `.venv-pinned` | verified_pinned |
| Depth validity: v3-indoor rank correlation −0.5924, 30 clips/120 frames | 9 | `.venv-pinned` (presumed) | **NOT TO BE QUOTED until re-measured** — deliberately not re-run today: `scripts/eval_depth.py` requires a real Depth-Anything-V2 forward pass over ~30 clips, and this machine was on 5% battery with CPU throttled to 46% speed for the entire remainder of the day. Running a CPU-heavy job under those conditions risks a mid-run shutdown and loses more than it proves; the existing LOW-risk classification (independently reconfirmed via a different single-clip method, Day 16) stands unchanged, deferred rather than gambled on. Day-19 item. |
| Appearance validity: texture energy 3.03, material diversity 15.17 | 10 | `.venv-pinned` (presumed) | verified_pinned — reproduced exactly (3.031, 15.168) Day 16 |
| Point-tracking gate / CoTracker3 pin, v3 refuses | 11 | `.venv-pinned` (presumed) | verified_pinned — refusal pattern reproduced Day 16 |
| `motion_gate_v3_with_baselines.json` | 12 | `.venv-pinned` (inherited envelope stack only, Day 17) | **verified_pinned — resolved today.** v3-indoor was rescored today (Objective 2) and now carries a live `measurement_environment` fingerprint in its own JSON: `numpy 1.26.2 / opencv 4.8.1.78 / python 3.10.20`, `sys_prefix` ending in `.venv-pinned` (`docs/day18/motion_gate_v3_rescored.json`). Direct evidence, not inference from timing. |
| Gate reframe: wake_fraction 0.9067, recall_retained 0.9340, miss_cost 57 | 14 | `.venv-pinned` (presumed) | verified_pinned — reproduced bit-identical Day 16, Day 17, and again today in the v3-indoor rescore |
| mypy "35 pre-existing errors, verified both ways via git stash" | 15 | `.venv-pinned` (that day's own claim) | verified_pinned — reproduced exactly (35, same file/line/class) Day 16, Day 17 |
| Retrieval-metric sweep, byte-for-byte reproducing Day 12 | 15 | `.venv-pinned` (presumed; reproducibility is the evidence) | verified_pinned |
| v4-gate scorecards, mypy 35→0, validity matrix, 745 tests | 16 | `.venv-pinned` — direct session evidence | verified_pinned |
| Transient mypy check showing "6 errors" (never committed) | 16 | `.venv` (unpinned, deleted Day 17) | verified_drifted — caught and corrected within the same session |
| Cascade bench re-run (4.50%), env_gate extension, gait fix, v4.1-gate | 17 | `.venv-pinned` — direct session evidence | verified_pinned |
| env_gate stray-venv fix, `.venv-infinigen` relocated, 4 structural tests | 18 | `.venv-pinned` — direct session evidence | verified_pinned |
| `Undefined`/`recall_retained` fix, v3-indoor + v4.1-gate rescored | 18 | `.venv-pinned` — direct session evidence, live `measurement_environment` on both scorecards | verified_pinned |
| **Cascade bench re-run attempts: 5.10%, 9.28%** | 18 | `.venv-pinned` interpreter correct; machine state (concurrent load, then battery throttle) contaminated both readings | **NOT TO BE QUOTED until re-measured** — see the report's first line. Day-19 item, first. |

**Counts:** 13 `verified_pinned`, 4 `verified_drifted`, 2 `not to be
quoted until re-measured` (depth validity — deferred for battery
safety; today's cascade bench — contaminated twice). 19 rows total, zero
left as a bare "unknown" — every measurement this project has ever
quoted now resolves to one of exactly three states, on query, not on
investigation.

## Objective 4 — `main` / `origin/main`: unrelated histories, ADR 0009 (Proposed)

Investigated, not executed. `origin/main` is a real, five-author,
PR-based history — 34 commits, `2026-02-17` through `2026-07-13`,
merging `optimize-cotracker-input` and `temporal-stitching` through
actual pull requests, citing `Dalbirsm03/project-iron` as a fetched
remote in its own merge-commit messages. Local `main` is one commit,
`41e924c`, `"chore: import project-iron working tree as baseline"`,
dated `2026-07-31` — eighteen days after `origin/main`'s last commit —
and every `foundation/day-N` branch descends from it.
`git merge-base foundation/day-1 origin/main` (and the same check
against `day-17`) returns nothing: no common ancestor. Same project (82
of `origin/main`'s 82 files match local `main`'s 90 by name), unrelated
git histories.

Three options recorded in `docs/adr/0009-repository-history-divergence.md`,
with consequences: **graft** the foundation line onto `origin/main`
(rewrites ~100+ commits across 18 branches, invalidates every SHA this
report has ever cited by name); **keep the lines permanently separate**
and rename local `main` for clarity (reversible, touches nothing on
`origin`, is the recommendation); **archive and replace** `origin/main`
under a tag (the only option that unifies the mainline, and the only one
that force-pushes over four other people's history without asking them
first). The ADR recommends the second, executes none of the three:
status **Proposed**, no branch renamed, no history rewritten, no tag
created. `git push --all origin` will keep rejecting `main` tomorrow,
exactly as it has every day since Day 16 — this is now a five-minute
decision waiting on a human, not a two-hour investigation waiting to
happen again.

## All five falsification tests, re-run today

| # | Test | Day 17 | Day 18 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged) |
| 4 | Alert explainability | PASSES (unchanged; `raise_alert()` still un-steered) | PASSES (unchanged) |
| 5 | Behaviour-query shape | PASSES (type-level, unchanged) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator |

`tests/test_falsification.py`: 8 tests, all green, unchanged. Repo-wide
(`.venv-pinned`, `not requires_weights and not slow`): **758 passed, 1
skipped, 8 deselected, 0 failures**, up from Day 17's 748 (+10: 4 env-gate
stray-venv tests, 6 `Undefined`/NaN tests). `mypy --strict` on the
declared scope: **0 errors**, unchanged from Day 16 and 17.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged today, plus one addition:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — forensic
   objective closed unanswered (ADR 0008); still wanted for its own sake.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 6 days old.
5. **NEW: the `main`/`origin/main` divergence decision (ADR 0009)** —
   which of graft / keep-separate / archive-and-replace, and whether the
   other four `origin/main` collaborators need to be looped in before
   any option touching `origin` is chosen.

## Objective 0/5 — push, end of day

`git push --all origin`: pushed clean. `git push --tags`: up to date.
Re-verified: every `foundation/day-N` branch head matches its remote
exactly; `main`'s pre-existing divergence unchanged, untouched, now
documented rather than merely noted (ADR 0009). Push happened at the
start and the end of the day, per the Day-17 standing rule.

## Day 19, in order

1. **Cascade bench, clean, on AC power — first item, not buried.**
   Today's two readings (5.10%, 9.28%) are both explicitly not to be
   quoted. Confirm the machine is charging (not just plugged in — this
   machine reached 5% while presumably plugged in overnight at some
   point, so charging state deserves its own check) and CPU_Speed_Limit
   reads 100 via `pmset -g therm` before trusting any number.
2. **Depth validity re-measurement** (`scripts/eval_depth.py`), deferred
   today for the same battery-safety reason as item 1. LOW risk,
   independently corroborated, but "unknown" is not a status this
   project leaves standing when a query would resolve it.
3. **The `main`/`origin/main` decision** (ADR 0009) — a human call, not
   an engineering one. Renaming local `main` per the ADR's recommendation
   is cheap and reversible whenever that call is made.
4. **`env_gate.py`'s stray-venv row is green again** — no action needed,
   listed only to close the loop on Day 17's item 1.
5. **v4.1-gate's `recall_retained`/`miss_cost` dimension is still
   unexercised** — `brief_entry`'s only real motion falls inside the
   10-frame warmup window. Unchanged from Day 17's item 2; today's fix
   makes the zero honestly `undefined` rather than a fabricated `nan`,
   but does not make the dimension measure anything yet.
6. **The generator has no sensor-noise model** — unchanged, Day 17's
   item 3.
7. **Order cameras and run the office capture** — unchanged in priority
   since Day 16, now 6 days old.
8. **The motion-gate precision/selectivity investigation** — seven days
   deferred.
9. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
   still open from Day 14.
10. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
11. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged ask, now a fourth data point resting on the
    same unmeasured multiplier.
12. **Bridge the live-RTSP path and `scripts/ingest_capture.py`** — not a
    blocker for the capture itself.
13. **MEVA licence verification** — still blocked on a human.
14. **The factor-graph solver** — unchanged from Day 13's list.

# Day 19

**After Objective 2: no, there is currently no environment-dependent number
in this project that is defensible.** The only environment-dependent metric
family this project has ever quoted — cascade-bench cost, % of one core —
has five readings spanning 2.97% to 9.28% (a 3.1x range) on identical code,
and today's session, run specifically to produce a clean sixth reading under
the new gate built for that purpose, instead produced **five refusals in a
row**, each for a genuinely different reason, across roughly seven hours of
real working time on this machine. Day 17's 4.50% remains the last
trustworthy figure (per Day 18's own conclusion) simply because nothing
fresher was ever certified — not because today's attempts came close and
missed. Every wall-clock or CPU-percentage claim in this report should be
read with that in mind; every accuracy, ratio, and count-based claim is
unaffected (see the classification in Objective 2).

Framing, from the standing prompt: this is the eighth time an instrument
rather than the code was the defect, and the widest-reaching one, because it
touches every compute and latency number the project has ever produced.
Objective 1 makes the harness refuse to produce a measurement it cannot
stand behind, structurally rather than by discipline. Objective 2 uses that
harness today, honestly reports that it could not produce a certified
reading, and retires nothing new (Day 18 had already retired the two
contaminated readings) while escalating the finding: even "plug it in" is
not sufficient, which Objective 3 turns into a hardware decision instead of
a discipline problem.

## Objective 0 — push, start and end of day; Day-18 suite count closed out

Start of day: `foundation/day-19` branched from `foundation/day-18`
(`8b3d53d`), pushed clean as a new branch, verified local HEAD matched
`origin/foundation/day-19` exactly. `git push --all origin`: `main` rejected
— `[rejected] main -> main (non-fast-forward)`, unchanged since Day 16; see
Objective 4.

**Day-18's background suite, closed out.** Day 18's report quoted "758
passed, 1 skipped, 8 deselected" from a full-suite run that was still
executing in the background when the report was written — a claimed number
never confirmed, which is exactly the class of thing this project exists to
catch. It was confirmed before Day 19 began: **758 passed, 1 skipped, 8
deselected, 0 failures**, exact match. Nothing to correct.

## Objective 1 — benchmark environment gate (`src/bench/environment.py`)

Built `BenchmarkGuard`: `.begin()` captures AC/battery state, OS-reported
CPU throttle (`pmset -g therm` on macOS, `cpufreq` on Linux), 1/5/15-min
load average, competing-process count above a CPU threshold, and core
count, then raises `BenchmarkRefused` naming every failing condition and its
remedy if any check fails — including when a condition is *undeterminable*,
which fails closed rather than passing by default. `.end()` re-captures and
compares against the start snapshot; mid-run throttling, a power-source
change, or battery drift beyond tolerance marks the run invalid.
`write_benchmark_artifact` is the only path that writes to disk and
structurally cannot be reached with an invalid run — every caller must hold
an `EnvironmentPair`, which only `.end()` produces. `EnvironmentPair.
require_comparable` raises on a machine-identity mismatch, same pattern as
`MeasuredEnvelope.require_comparable` and `Scorecard.require_comparable`.

Wired into `scripts/cascade_bench.py` via a new `--artifact PATH` flag:
supplying it engages the gate end-to-end (refuse-to-start, refuse-to-write-
if-drifted); omitting it runs exactly as before, unguarded — deliberately,
because that is the invocation CI's regression-ceiling check uses on a
GitHub Actions runner with no battery, and gating it on AC power would turn
CI red on every runner rather than catch a real problem.

**26 new tests** (22 in `tests/test_bench_environment.py`, 4 in
`tests/test_cascade_bench_artifact_gate.py`): every refusal condition
individually (battery, throttle, load, competing process, each
undeterminable-fails-closed case), mid-run throttle/power/battery drift
invalidating a run, `write_benchmark_artifact` refusing and leaving no file
on disk, `require_comparable` raising on an invalid or cross-machine pair
and passing on a matched one, plus script-level tests confirming
`cascade_bench.py --artifact` actually refuses/invalidates/writes through
the real wiring and that the unguarded path is untouched. `src/bench` added
to `mypy.ini`'s strict scope: **0 errors, 49 files** (up from Day 18's 47).

**STRUCTURAL requirement verified live, not just in tests.** Over the
course of writing and testing this gate today, five real `--artifact`
invocations were made on this machine, spanning roughly 09:00 to 16:30.
Every one was refused or invalidated, each for a different real condition,
and **zero artifacts were written** (`docs/day19/` remains empty,
confirmed by directory listing after each attempt):

| # | Time | Result | Cause |
|---|---|---|---|
| 1 | ~09:13 | REFUSED (start) | 1 competing process ≥20% CPU: `duetexpertd` (macOS system daemon) |
| 2 | ~09:16 | INVALIDATED (mid-run) | 2 competing processes appeared during the run: `Python`, `knowledgeconstructiond` |
| 3 | ~09:17 | REFUSED (start) | 1 competing process ≥20% CPU: a versioned helper process |
| 4 | ~10:04 | REFUSED (start) | `CPU_Speed_Limit=24` (24% of full speed) — **while on AC power, charging at 49%** |
| 5 | ~16:28 | REFUSED (start) | not on AC power (machine had since been unplugged; 92% battery, discharging) |

Row 4 is the day's sharpest individual finding: Day 18 attributed throttling
to a low, unplugged battery, which reads as an operator-fixable condition.
Row 4 shows the same laptop throttled to a quarter speed while plugged in
and actively charging, from sustained load generated by this project's own
test and benchmark runs earlier in the session. AC power is necessary but
not sufficient — the gate checks throttle independently of power source for
exactly this reason, and today it mattered.

## Objective 2 — classification, re-measurement, and retirement

**Classification.** Swept every measurement `FOUNDATION_REPORT.md` has ever
quoted (Day 18's 19-row provenance table plus everything cited since).
Every one of them is either:

- **Environment-independent** (accuracy, ratios, counts, correctness — does
  not change with CPU speed): golden-vector cosine similarity, envelope
  calibration thresholds, depth/appearance validity metrics, `gate.
  wake_fraction` / `gate.recall_retained` / `gate.miss_cost`, mypy error
  counts, test counts, the retrieval-metric sweep, the stray-venv structural
  tests. This is the large majority of everything this project has ever
  quoted, and none of it needed re-measurement today.
- **Environment-dependent** (wall-clock, CPU%, throughput): cascade-bench
  stage-0 cost is the *only* one that has ever been asserted as a headline,
  quotable number. Two others exist but were never load-bearing claims:
  `docs/day12/infinigen_throughput.md`'s raytracing cycles/s (a Day-11/12
  scoping measurement for a different, still-blocked data-generation
  question — out of today's scope; requires the separate Infinigen venv and
  a real render, and carries its own Day-12 caveats already, unchanged
  here) and `gate.compute_saved` (1.87 / 17.50 ms/frame), which is not a
  measurement at all but `wake_fraction` (environment-independent) times
  `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME`, an acknowledged-unmeasured
  constant already caveated four times in this report as an estimate — Day
  18's Day-19 list carried "decide whether it should be replaced" forward
  unchanged, and it stays unchanged again (Day-20 item below).

**Re-measurement attempts: five, all documented in Objective 1's table
above.** The only environment-dependent number this project can act on
today is cascade-bench cost, and every attempt to produce a fresh, gated
reading was refused or invalidated by real machine state, not by the gate
being miscalibrated.

**Retired: nothing new.** Day 18 already retired the two contaminated
readings (5.10%, 9.28%) — there is no third bad reading to retire today,
because the gate did its job and none was produced. What changes today is
the status of the *number itself*: it is not merely "last measured N days
ago," it is **"could not be re-measured today despite five real attempts
spanning seven hours, using instrumentation built specifically for this
purpose."** That is a stronger and more useful statement than a stale
timestamp.

**The cascade-budget history, one table, all nine data points:**

| # | Value | Day | Environment | Status |
|---|---|---|---|---|
| 1 | 2.97% | 2 | `.venv` (unpinned) | verified_drifted |
| 2 | 4.57% | 6 | `.venv-pinned`, quiet AC session | verified_pinned |
| 3 | 4.50% | 17 | `.venv-pinned`, quiet AC session | verified_pinned — **last trustworthy figure, still standing** |
| 4 | 5.10% | 18 | `.venv-pinned`, 2 concurrent `pytest` suites | not to be quoted (retired Day 18) |
| 5 | 9.28% | 18 | `.venv-pinned`, battery 5%, `CPU_Speed_Limit=46` | not to be quoted (retired Day 18) |
| 6 | REFUSED | 19 | competing process (`duetexpertd`) | no reading produced |
| 7 | INVALIDATED | 19 | mid-run competing processes appeared | no reading produced |
| 8 | REFUSED | 19 | competing process (versioned helper) | no reading produced |
| 9 | REFUSED | 19 | `CPU_Speed_Limit=24` **on AC, charging** | no reading produced |
| 10 | REFUSED | 19 | not on AC power (unplugged, 92%, discharging) | no reading produced |

**Honest conclusion:** nothing about the gate's compute cost has been
freshly and defensibly measured since Day 17. The 4.50% figure is not
"current," it is "the last one that survived scrutiny," and this project
should stop treating those as the same thing in casual reference. The
correct fix is not a sixth laptop attempt — it is Objective 3.

## Objective 3 — reference hardware specification

`docs/reference_hardware.md` written in full. Summary:

- **Target:** 8×720p streams (fixed by the Tier-1 sales claim and the
  `iron-cascade-runtime` skill's budget table). **Declared fps is currently
  underspecified** — `cascade_bench.py` defaults to 12 fps for synthetic
  scenarios, but no config field commits to a target fps for real camera
  ingest. Flagged as an open gap, not resolved here (Day-20 item).
- **Required before the Tier-1 claim may be made, on the reference box
  itself:** idle cost per camera at 8 concurrent streams (not extrapolated
  from one), full-pipeline cost at N active tracks (blocked on stages 1-3
  leaving stub status), p50/p95/p99 glass-to-alert latency, and **24h
  sustained thermal behaviour** — the direct structural answer to today's
  row-4 finding: a burst benchmark cannot see a chassis throttle under
  sustained real-world load, only a day-scale test can.
- **Why a laptop cannot substitute:** the nine-row history above, cited
  directly rather than asserted on principle.
- **Interim substitutes, bounded:** a pinned-CPU cloud instance (hours,
  supports relative regression tracking only, not the absolute claim or
  thermal behaviour) versus a fixed-clock desktop on AC (days or zero if
  repurposed, supports idle/full-pipeline/latency as provisional numbers,
  does not support the 24h thermal claim for a NUC-class chassis
  specifically).
- **Procurement memo:** recommends ordering a NUC-class box now (~$700-
  1,200, 1-3 week lead time, the only option that unblocks the Tier-1 claim
  at all) while standing up the fixed-clock-desktop interim immediately so
  measurement accumulates during the lead time. One page, decision-ready,
  five-minute call for whoever holds the budget.

## Objective 4 — main divergence: still Proposed, nothing executed

No approval for ADR 0009's Option B was given in this session's
instructions. Confirmed the ADR's status is still **Proposed**
(`docs/adr/0009-repository-history-divergence.md`). Nothing renamed, no
history rewritten, no force-push. `git push --all origin` rejected `main`
again today, unchanged since Day 16 — still a five-minute decision waiting
on a human, not re-investigated.

## All five falsification tests, re-run today

| # | Test | Day 18 | Day 19 |
|---|---|---|---|
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged) |
| 4 | Alert explainability | PASSES (unchanged; `raise_alert()` still un-steered) | PASSES (unchanged) |
| 5 | Behaviour-query shape | PASSES (type-level, unchanged) | PASSES (type-level, unchanged) — still blocked on the unimplemented estimator |

`tests/test_falsification.py`: 8 tests, all green, unchanged.

## Full suite and mypy

Repo-wide (`.venv-pinned`, `not requires_weights and not slow`): **784
passed, 1 skipped, 8 deselected, 0 failures** — up from Day 18's 758 by
exactly +26, matching today's 26 new tests (22 in `tests/
test_bench_environment.py`, 4 in `tests/test_cascade_bench_artifact_gate.py`).
`mypy` (scoped per `mypy.ini`, now including `src/bench`): **0 errors, 49
files**, up from Day 18's 47.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged, plus one addition:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 7 days old.
5. **The `main`/`origin/main` divergence decision (ADR 0009).**
6. **NEW: reference hardware procurement decision** (`docs/
   reference_hardware.md`) — order a NUC-class box, approve an interim
   substitute, or both; today's evidence is that "measure it on the laptop
   more carefully" is exhausted as a strategy.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-19` pushed clean, matches origin
exactly. `main` rejected, unchanged (Objective 4). `git push --tags`: up to
date.

## Day 20, in order

1. **Reference hardware decision** (`docs/reference_hardware.md`) — order
   the NUC-class box and/or stand up the fixed-clock-desktop interim.
   First item: this is what today's five refusals were actually arguing
   for, not a discipline fix.
2. **Declare a target fps for real camera ingest.** Found while sizing the
   reference box: no config field commits to one; `cascade_bench.py`'s 12
   fps default is a synthetic-scenario convenience, not a product
   commitment.
3. **Cascade bench, clean, on interim or reference hardware** — not on this
   laptop again without a specific reason; five attempts across one day
   already exhausted that approach.
4. **Depth validity re-measurement** (`scripts/eval_depth.py`), deferred
   again — battery-safety concerns recurred today too. LOW risk,
   independently corroborated, but still not re-confirmed.
5. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
6. **v4.1-gate's `recall_retained`/`miss_cost` dimension is still
   unexercised** — unchanged from Day 17/18.
7. **The generator has no sensor-noise model** — unchanged.
8. **Order cameras and run the office capture** — unchanged in priority
   since Day 16, now 7 days old.
9. **The motion-gate precision/selectivity investigation** — eight days
   deferred.
10. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
    still open from Day 14.
11. **A canonical `Observation -> hash` function** (ADR 0007) — still open
    from Day 13.
12. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged, now a fifth data point resting on the same
    unmeasured multiplier.
13. **Bridge the live-RTSP path and `scripts/ingest_capture.py`** — not a
    blocker for the capture itself.
14. **MEVA licence verification** — still blocked on a human.
15. **The factor-graph solver** — unchanged from Day 13's list.

# Day 20

**No timing, throughput, CPU-percentage, or latency claim is made anywhere
in this section.** Day 19 established that no environment-dependent number
is currently defensible on this machine, and today's objectives were
compute-independent by design (accuracy and consistency, scored against
exact synthetic ground truth) specifically so the day's work would not
depend on the still-unresolved reference-hardware question. Where the word
"cost" or "budget" would normally appear, it does not — position error,
velocity error, and the NIS/NEES consistency residuals below are all
environment-independent by construction (Day 19's own classification), and
nothing here was measured on, or claims anything about, wall-clock speed.

**The headline: the filter is well-calibrated on sustained motion and
measurably overconfident at motion onset.** v3-indoor (72 tracks, mostly
walking) empirically covers its own 95% uncertainty bound 99.1% of the
time — slightly conservative, not a defect. v4.1-gate (5 tracks, the quiet
set) covers only 80.5% of the time in aggregate, and breaking that down by
track shows it is not spread evenly: the two `long_static_occupant` tracks
are covered 100.0% of the time (as calibrated as a filter can be), while
all three `brief_entry*` tracks — a person entering an otherwise still
scene — sit at 63–71%. The filter's covariance, having settled tight
during a quiet interval, does not widen fast enough to honestly represent
the uncertainty at the moment motion actually starts. This is exactly the
failure the prompt named: an estimator that understates its own
uncertainty is the most dangerous defect an evidence system can have,
because every downstream confidence inherits the error and the system is
most certain exactly where it is most wrong. It is diagnosed, not fixed,
today — see Day 21, item 1.

## Verdicts

- **Are the motion/measurement models correctly specified?** Yes — `Q` is
  strictly positive-definite at every kind/`dt` tested; `R` is
  monotonically increasing in distance, with a 5x sigma inflation outside
  the calibrated envelope. → Objective 1.
- **Does `solve_state` now resolve a real single-entity state?** Yes —
  Falsification test 5 moves BLOCKED → PARTIAL; reproducibility (re-solving
  an earlier `graph_rev`) and occlusion-gap extrapolation are both proven
  structurally. → Objective 2.
- **Is the always-on consistency stage (NIS/NEES) wired in and
  structurally enforced?** Yes — a `StateEstimate` cannot be constructed
  without at least one `nis`-kind residual; stub vs. computed are distinct
  types, not a null used for both. → Objective 3.
- **Does the filter beat the trivial baselines (copy-previous,
  constant-velocity dead reckoning)?** Yes, on both golden sets —
  v3-indoor +0.1053 m / +2.8477 m/s margin, v4.1-gate +0.0263 m / +4.0670
  m/s. → Objective 4.
- **Is the filter well-calibrated?** No, not uniformly — v3-indoor (mostly
  steady walking) covers 99.1% of its own 95% bound; v4.1-gate (quiet
  scenes plus a motion onset) covers only 80.5%, concentrated in the three
  `brief_entry*` tracks (63-71%). Diagnosed as the filter's covariance not
  widening fast enough at the moment motion starts. → Objective 4 /
  headline.
- **Is baseline margin stated per GT distance bucket, not just pooled?**
  NEVER MEASURED as a printed table this day — "Position error by GT
  distance bucket" shows filter RMSE only, no baseline comparison within a
  bucket (found missing Day 21 Objective 1; the code was fixed then but
  the table itself was still never printed until Day 25 Objective 2).

## Objective 0 — push, start and end of day

Start of day: `foundation/day-20` branched from `foundation/day-19`
(`d5de7b4`), pushed clean, verified local HEAD matched
`origin/foundation/day-20` exactly. `main` rejected, unchanged since Day 16
(Day 19's ADR 0009 still Proposed; no new instruction to execute it arrived
today, so nothing was touched).

## Objective 1 — motion and measurement models (`src/estimator/`)

New package. `MotionModel` per entity kind — all four are
constant-velocity transitions except `fixture`, which has none:

| kind | process-noise density (σₐ) | rationale |
| --- | ---: | --- |
| `person` | 1.5 m/s² | pedestrian acceleration bound |
| `asset_static` | 0.02 m/s² | reluctant to acquire velocity, not pinned to zero |
| `asset_carried` | 4.0 m/s² | inflated, standing in for the not-yet-modeled carrier coupling |
| `fixture` | — | identity transition, no kinematics |

The textbook discretized-white-noise-acceleration `Q` block is only
positive *semi*-definite (determinant exactly zero — a single scalar noise
source driving two states); a small regularization floor makes every `Q`
strictly PD, tested directly at four different `dt` values per kind.
`asset_carried` carries a `carrier_entity_id` field as the interface hook
for real rigid coupling — unused numerically until the multi-entity day,
confirmed by a test that a coupled and an uncoupled instance compute
identical `Q`.

`MeasurementModel` wires the Day-13 `Envelope`/`EnvelopeCurve` in directly
as the source of R: measurement noise as a function of distance,
monotonically increasing (tested), and a reading beyond the curve's
calibrated range gets a 5x-inflated sigma rather than being discarded —
tested with an exact-factor assertion, not just "bigger." Added
`WorldPositionMeasurement` to `src/model/measurement.py` (a genuinely new
sensor-kind variant, the pattern the module was built for) since none of
the four existing measurement types carry an unprojected 3D position.

34 tests, `mypy --strict` clean (`src/estimator` added to `mypy.ini`'s
scope).

## Objective 2 — the single-entity filter, `solve_state` filled (§15, §17)

`StateGraph`/`StateQuery`'s Day-13 shape is unchanged for every existing
caller: `append_factor` gained an *optional* `payload` slot (default
`None`) so a filter can attach the actual `StateEstimate` a factor
represents, retrievable via the new `payload_for` — additive, not
breaking. `StateQuery` gained `horizon_kind` (`"filtered"` default,
`"smoothed"` → `NotImplementedError`), a mode orthogonal to the existing
`horizon_ns` extrapolation budget. `solve_state` delegates to
`src.estimator.filter.resolve_state` via a deferred (call-time) import —
the only way to avoid a genuine cycle, since the estimator imports
`StateGraph`/`StateQuery` from the model layer.

`run_single_entity_filter` (predict → observe → correct, Joseph-form
covariance update for numerical stability) takes **no parameter that could
carry a behavioral prior** — checked directly via `inspect.signature`, not
just documented. The first state bootstraps from the first observation
alone: position = that observation, velocity = 0, covariance =
`block_diag(R, 100·I)` on velocity. Two STRUCTURAL properties fell out of
the replay-not-recompute design almost for free:

- **Reproducibility.** Re-solving at an earlier `graph_rev` after more
  factors have been appended to the same graph returns a bit-identical
  result (`before.mean == after.mean`, `before.cov == after.cov`) — proven
  by appending a second, unrelated walker's observations onto the same
  graph and re-querying the first walker's earlier state.
- **Occlusion.** With no observation, `resolve_state` predicts forward;
  covariance grows strictly monotonically through the gap and `observed`
  is `False` at every step, tested across a 16-second gap in five
  increments.

Falsification test 5 (behaviour-query shape) updated: it now builds a
*real* single-entity graph via the filter and confirms `solve_state`
returns a genuine `StateEstimate` instead of raising. **Status moves from
BLOCKED to PARTIAL** — the state-estimation half now works; nothing turns
a `StateEstimate` into an `ActivityMode` behaviour label yet (out of
scope), so a live *behaviour* query is still unanswerable end to end. The
same test also confirms `horizon_kind="smoothed"` still raises
`NotImplementedError` and that a bare Day-13-style graph with no estimator
payload still refuses, now with a specific, diagnosable reason
(`EpisodeError: no resolvable state...`) rather than a blanket one.

## Objective 3 — the consistency stage, first-class (§15 stage 4)

NIS is computed live on every real update
(`src.estimator.consistency.compute_nis`), tested against known chi-square
percentiles (dof=1 → 3.841, dof=3 → 7.815, dof=6 → 12.592 at 95%,
matching Bar-Shalom et al. to two decimal places). The bootstrap step and
every predicted (unobserved) step carry an explicit NIS *stub*
(`value=None`, noting why) rather than omitting the entry — "not
computed" and "absent" are different facts and the type keeps them
different. `constraint`/`calibration`/`coverage` are routed as stubs
naming their future consumers exactly per the objective: twin-revision
hypothesis, recalibration event, envelope drift.

**STRUCTURAL, tested directly:** `StateEstimate.__post_init__` raises if
`residuals` is empty, and separately raises if no entry has `kind="nis"` —
a state estimate cannot be constructed without running stage 4, computed
or stubbed. `ConsistencyResidual` itself raises if only some of
`value`/`dof`/`chi2_bound`/`within_bound` are set — a residual cannot
claim a value with nothing to judge it against. `StateEstimate.
require_comparable` enforces §15's three-way sha key
(`motion_model_sha`/`measurement_model_sha`/`update_rule_sha`), same
pattern as `MeasuredEnvelope.require_comparable` and `Scorecard.
require_comparable` elsewhere in this codebase.

26 dedicated tests, isolating the consistency-stage functions from the
filter tests that exercise them indirectly.

## Objective 4 — accuracy against exact GT (`scripts/eval_estimator.py`)

**Day-10 validity gate, run first.** Registered `"state_estimation"` onto
the existing `gate_motion_geometry` check (position/velocity are the same
kind of exactly-known scene-description quantity as "did this move" —
v3-indoor's `agent_xyz` is analytic, not annotated). **PASSED** on both
sets; no refusal to report.

**Method.** No detector exists yet, so observations are synthesized by
adding measurement-model-derived noise to the exact GT position — this
isolates the estimator's own machinery (motion model, measurement model,
Kalman update, consistency residuals) from detection/tracking error, which
is a separate, not-yet-built stage. All three compared methods (filter,
`copy_previous_position`, `constant_velocity_no_update`) are scored from
the same frame index onward per track so none gets free information the
others lack (see the script's own docstring for why frame index 1 would
have let the coasting baseline "predict" the very observation that defined
its velocity).

### Position and velocity accuracy, with mandatory trivial baselines (Day-12 rule)

| set | position RMSE | vs copy-previous | vs constant-velocity-no-update | margin (position) | velocity RMSE | margin (velocity) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3-indoor | 0.1189 m | 0.2242 m | 6.2068 m | **+0.1053 m** | 0.4038 m/s | **+2.8477 m/s** |
| v4.1-gate | 0.2328 m | 0.2591 m | 9.2734 m | **+0.0263 m** | 1.5337 m/s | **+4.0670 m/s** |

Both margins are clearly positive — the filter beats both trivial
strategies on both sets, by a wide margin against the coasting baseline
and a real (if more modest, on the quiet set) margin against
copy-previous. This is not the headline finding, but it rules out the
alternative the objective named explicitly: the measurement updates are
doing real work, this is not a wiring bug producing a filter
indistinguishable from a trivial strategy.

### Position error by GT distance bucket

| set | bucket | n | position RMSE | x / y / z RMSE |
| --- | --- | ---: | ---: | --- |
| v3-indoor | 0–3 m | 0 | — (empty) | — |
| v3-indoor | 3–8 m | 2528 | 0.1140 m | 0.0645 / 0.0677 / 0.0652 |
| v3-indoor | 8 m+ | 208 | 0.1677 m | 0.1171 / 0.0858 / 0.0838 |
| v4.1-gate | 0–3 m | 0 | — (empty) | — |
| v4.1-gate | 3–8 m | 190 | 0.2328 m | 0.2135 / 0.0597 / 0.0711 |
| v4.1-gate | 8 m+ | 0 | — (empty) | — |

Error grows with distance (0.114 m → 0.168 m from the 3–8 m to 8 m+
bucket on v3-indoor), consistent with the measurement model's own
distance-indexed R — this is the expected shape, not a surprise. Neither
set populates the 0–3 m bucket at all (both synthetic sets keep agents
further from camera than that), which the table reports as empty rather
than silently omitting.

### Consistency: NIS/NEES pass rates and what they say about calibration

| set | NEES within 95% bound | empirical coverage | reading |
| --- | ---: | ---: | --- |
| v3-indoor | 99.09% | 0.9909 | slightly conservative (target 95%) |
| v4.1-gate | 80.53% | 0.8053 | **overconfident** |

v4.1-gate broken down per track (5 tracks, not poolable as one number
without losing the story):

| clip | frames | NEES within 95% | position RMSE |
| --- | ---: | ---: | ---: |
| `long_static_occupant__cam_a` | 38 | 100.0% | 0.093 m |
| `long_static_occupant__cam_b` | 38 | 100.0% | 0.119 m |
| `brief_entry__cam_a` | 38 | 63.2% | 0.303 m |
| `brief_entry__cam_b` | 38 | 68.4% | 0.269 m |
| `brief_entry_evening__cam_a` | 38 | 71.1% | 0.290 m |

**Diagnosis, stated as a hypothesis, not yet confirmed by a second
measurement:** the two genuinely static tracks are calibrated as well as
a filter can be; all three overconfident tracks are `brief_entry*` — a
person entering an otherwise still scene. A constant-velocity filter that
has settled to a tight covariance during a quiet interval does not widen
fast enough at the moment real motion starts, so the true position error
at onset exceeds what the (still-tight) covariance predicts, more often
than the nominal 5%. This reads as a maneuvering-target problem (the
process-noise budget is sized for steady walking, not for a
velocity-onset transient), not as a bug in the NIS/NEES machinery itself
— v3-indoor's own good calibration on 72 walking tracks is evidence the
consistency stage is measuring correctly, not miscounting.

**n=190 caveat, stated plainly:** the 190 points behind the 80.5% figure
come from only 5 tracks with strong within-track autocorrelation, not 190
independent trials — the per-track table above is the actual evidence,
not the pooled percentage. It is nonetheless a large and consistent effect
(three separate tracks, three separate cameras, one lighting condition
different, same ~65–71% band) rather than one noisy outlier.

## Falsification test 5 — from BLOCKED to PARTIAL

Confirmed above (Objective 2): `solve_state` now genuinely resolves a
single-entity `StateQuery`. `tests/test_falsification.py`: **8 tests, all
green.**

| # | Test | Day 19 | Day 20 |
| --- | --- | --- | --- |
| 1 | Absence under degraded coverage | PASSES | PASSES (unchanged) |
| 2 | Retroactive badge resolution | PASSES | PASSES (unchanged) |
| 3 | Twin re-version | PASSES | PASSES (unchanged) |
| 4 | Alert explainability | PASSES (unchanged) | PASSES (unchanged) |
| 5 | Behaviour-query shape | BLOCKED on the unimplemented estimator | **PARTIAL** — state estimation now live; behaviour labeling still unbuilt |

## What remains skeleton, and the order it should land in

1. **Motion-model change-point handling** (new finding, today) — not
   originally on the list, but Day 20's own overconfidence result makes it
   the most urgent gap: an estimator that is honest on steady motion and
   overconfident exactly at the moment something starts happening is a
   worse evidence-system defect than any of the items below, per the
   framing at the top of this section. Likely fix shape: adaptive/inflated
   process noise keyed off recent innovation magnitude (a
   maneuvering-target model), not a larger blanket `sigma_a_mps2` for
   `person` — inflating the steady-state budget would just make
   v3-indoor's already-good calibration worse to fix a problem that is
   specifically about transitions.
2. **Multi-entity factor graph.** Today's `StateGraph` is, in practice, one
   entity's graph — real coupling (the `asset_carried` interface built
   today, cross-entity association) needs factors that reference more than
   one entity's state and a solver that resolves them jointly.
3. **Smoothing** (`horizon_kind="smoothed"`). A backward pass over the
   same factor history would tighten every number in this report's
   accuracy table, but it is strictly secondary to item 1 — smoothing a
   filter that is overconfident at transitions makes a better-fitting
   answer, not a better-calibrated one.
4. **Hypothesis management.** Depends on multi-entity association existing
   first; not started.

## Full suite and mypy

Repo-wide (`.venv-pinned`, `not requires_weights and not slow`): **871
passed, 1 skipped, 8 deselected, 0 failures** — up from Day 19's 784 (+87:
34 motion/measurement-model tests, 14 filter tests, 26 consistency tests, 8
eval-script tests, and 5 net new/updated tests in
`test_model_episode.py`/`test_falsification.py`). `mypy` (scoped per
`mypy.ini`, now including `src/estimator`): **0 errors, 55 files**, up from
Day 19's 52. `black --check` and `flake8` clean on every file touched
today.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 19 — today's work was
entirely unblocked by design (state estimation is geometry-derived,
synthetically scorable, and environment-independent), so nothing here
moved:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 8 days old.
5. **The `main`/`origin/main` divergence decision (ADR 0009).**
6. **Reference hardware procurement decision** (`docs/
   reference_hardware.md`) — unchanged since Day 19.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-20` pushed clean, matches origin
exactly. `main` rejected, unchanged. `git push --tags`: up to date.

## Day 21, in order

1. **Motion-model change-point handling** — the day's own finding. Design
   and measure an adaptive-process-noise (or explicit maneuvering-target)
   variant for `person`, re-run `scripts/eval_estimator.py`, and confirm
   `brief_entry*`'s NEES coverage moves toward 95% without degrading
   `long_static_occupant`'s already-good calibration or v3-indoor's.
2. **Reference hardware decision** — still pending a human, now 2 days
   old since Day 19's procurement memo.
3. **Declare a target fps for real camera ingest** — still open from Day
   19.
4. **Cascade bench, clean, on interim or reference hardware** — still
   pending hardware.
5. **Multi-entity factor graph** — see "what remains skeleton" above;
   depends on item 1 landing first so multi-entity work does not inherit
   a known-overconfident single-entity core.
6. **Smoothing (`horizon_kind="smoothed"`)** — depends on item 1 for the
   same reason.
7. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
   deferred from Day 18/19.
8. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
9. **The generator has no sensor-noise model** — unchanged.
10. **Order cameras and run the office capture** — now 8 days old.
11. **The motion-gate precision/selectivity investigation** — nine days
    deferred.
12. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
    still open from Day 14.
13. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
14. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged, now a sixth data point resting on the same
    unmeasured multiplier.
15. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
16. **MEVA licence verification** — still blocked on a human.
17. **Hypothesis management** — furthest out; depends on item 5.

# Day 21

**No timing, throughput, CPU-percentage, or latency claim is made anywhere
in this section** — unchanged hard scope rule from Day 20. Everything
below is accuracy and consistency: position/velocity error against exact
GT, NIS/NEES, innovation whiteness, and IMM mode probabilities.

**Headline: IMM was built on solid diagnostic evidence, and today's
specific configuration does not clear the bar to replace the Day-20
single-model filter.** Its pooled NEES coverage is *worse* than the
single model on both golden sets — not a regression hidden behind an
improving aggregate (the pattern this project has caught four times
before: constant depth on the dominant plane, position-only retrieval,
always-wake gate parity, bounded nulls read as physical limits), but a
regression the aggregate itself already shows, before any per-regime
breakdown. On v3-indoor's sustained regime specifically — the one regime
Day 20/21 already knew the single model handled well — IMM makes *both*
accuracy and calibration worse. It also roughly quarters position error
in v4.1-gate's static frames while making their calibration worse. The
diagnosis that motivated building IMM (§ Objective 2 below) stands;
today's mode set and transition matrix do not yet deliver it. The Day-20
single-model filter remains what should be used until this is resolved
— see Day 22, item 1.

## Verdicts

- **Is baseline margin now stated per GT distance bucket, not just
  pooled?** Code fixed to compute it per bucket and per regime, and pooled
  margins reconfirmed positive on both sets — but the actual bucket-level
  table itself was not printed in this section (still NEVER MEASURED as a
  printed table until Day 25 Objective 2). → Objective 1.
- **Where does the calibration failure actually happen — at motion onset,
  or somewhere else?** Not at onset — v3-indoor's onset is well-calibrated
  (96.5%, white innovations). The actual mechanism is the RECOVERY TAIL
  after a stop: a hand-traced `brief_entry` track shows NEES climbing to
  ~815 immediately after cessation and decaying to nominal over ~10-14
  frames, confirmed as model mis-specification (not mistuning) by
  innovation whiteness — v4.1-gate is severely NOT WHITE (+0.837
  autocorrelation). → Objective 2.
- **Was IMM built?** Yes — a 3-mode mixture (`static`/`constant_velocity`/
  `maneuvering`). A real bug was found and fixed before it landed: the
  first `static` mode shared `constant_velocity`'s F, so it kept
  propagating any mixed-in velocity while merely claiming tighter
  confidence — replaced with a model that has no velocity-to-position
  coupling at all. → Objective 3.
- **Does IMM satisfy the no-trade criterion stated in advance (transient
  regime improves materially, steady regimes do not degrade)?** No — NOT
  SATISFIED on both sets. Pooled NEES coverage itself gets worse (not
  hidden behind an improving aggregate); v3-indoor sustained degrades
  99.4%→65.9%, v4.1-gate static degrades 81.7%→43.4% even as its raw
  accuracy improves roughly 4x. → Objective 4.
- **Is IMM's mode probability wired into the event compiler or validated
  against real motion labels?** No — documented as a future product
  output (POSE verbs, dwell detection, motion-onset triggering,
  `low_activity`), explicitly marked uncalibrated in its own docstring.
  Validation against real motion labels is a separate, later objective.
  → Objective 5.

## Objective 0 — push, start and end of day

Start of day: `foundation/day-21` branched from `foundation/day-20`
(`777b7a9`), pushed clean, verified local HEAD matched
`origin/foundation/day-21` exactly. `main` rejected, unchanged since Day
16.

## Objective 1 — the missing baseline margin: found, closed

Day 20's report showed the pooled and per-distance-bucket margin
(`estimator.position_rmse_m` vs `copy_previous_position` and
`constant_velocity_no_update`), but the distance-bucket table itself only
carried the *filter's* RMSE, not the baselines' — so a reader could not
see the margin *within* a bucket, only pooled across the whole set. That
was a real gap, not a task: the code already collected everything needed,
it just was not surfaced. `scripts/eval_estimator.py` now reports filter /
copy-previous / constant-velocity-no-update RMSE and both margins per
distance bucket **and** per motion regime (the latter only became possible
once Objective 2's regime classifier existed). Pooled margins were, and
remain, clearly positive on both sets (v3-indoor +0.105 m vs
copy-previous, +2.85 m/s vs constant-velocity; v4.1-gate +0.026 m /
+4.07 m/s) — the earlier finding stands, now with the bucket-level
evidence to back it instead of just the pooled number.

## Objective 2 — regime diagnosis, confirmed by innovation whiteness

**Regime frame counts** (GT-classified, `src/estimator/regime.py`,
static > onset > cessation > maneuver > sustained priority):

| set | static | onset | sustained | cessation | maneuver |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3-indoor | 38 | 284 | 2414 | 0 | 0 |
| v4.1-gate | 175 | 12 | 0 | 3 | 0 |

`maneuver` is empty on both sets by construction, not by omission: every
v3-indoor/v4.1-gate agent walks one straight line at one constant speed
(`scripts/gen_synthetic_indoor.py`'s `Agent.position_at`) — there is no
heading or speed change mid-path anywhere in this dataset for the
classifier to find. Only real (Site Zero) footage can exercise that
regime. `cessation` (n=3) and `sustained` on v4.1-gate (n=0) are both too
thin to support a conclusion on their own and are reported flagged, not
hidden.

**Per-regime NEES coverage** (single model, target 0.95):

| set | regime | n | coverage | reading |
| --- | --- | ---: | ---: | --- |
| v3-indoor | static | 38 | 1.0000 | — |
| v3-indoor | onset | 284 | 0.9648 | well-calibrated |
| v3-indoor | sustained | 2414 | 0.9938 | well-calibrated |
| v4.1-gate | static | 175 | 0.8171 | overconfident |
| v4.1-gate | onset | 12 | 0.8333 | overconfident (thin) |
| v4.1-gate | cessation | 3 | 0.0000 | overconfident (very thin) |

**The prediction under test was that onset and maneuver would fail high.
The data does not agree with that attribution, though it agrees a real
failure exists.** v3-indoor's onset is well-calibrated (96.5%, white
innovations — see below). Tracing one `brief_entry` track frame by frame
pinned the actual mechanism precisely:

```
frame  regime      NEES     bound   within
 0-3   onset       0.8-2.2  12.59   True        <- fine
 4     cessation   165.6    12.59   False       <- the stop itself
 5-14  static      815->16  12.59   False (all) <- decaying tail
15+    static      <10      12.59   True        <- settled
```

NEES decays roughly exponentially over ~10-14 frames after the stop,
before the filter's covariance — tight from the preceding walk — catches
up to the fact that velocity has actually gone to zero. **The failure is
specifically the frames following cessation, not onset.** Onset was
already fine under the single `constant_velocity` model.

**Innovation whiteness — the deciding evidence** (pooled, track-boundary-
respecting lag-1 autocorrelation, Bartlett's 95% bound):

| set | scope | lag1 autocorr | bound | verdict |
| --- | --- | ---: | ---: | --- |
| v3-indoor | pooled | −0.108 | ±0.038 | NOT WHITE |
| v3-indoor | sustained only | −0.119 | ±0.041 | NOT WHITE (mild) |
| v3-indoor | onset only | +0.037 | ±0.134 | WHITE |
| v3-indoor | static only | −0.295 | ±0.322 | WHITE (n too small to reject) |
| v4.1-gate | pooled | +0.837 | ±0.144 | NOT WHITE (severe) |
| v4.1-gate | static only | +0.850 | ±0.150 | NOT WHITE (severe) |

A correctly-specified-but-mistuned filter still produces white
innovations — only the calibration would be off, not the correlation
structure. Both sets show real, non-white structure (v4.1-gate severely
so), which is direct evidence of **model mis-specification**, not a
tuning problem. Per the day's own hard constraint, no Q was retuned to
chase this. **Diagnosis confirmed → proceed to Objective 3.**

v3-indoor's sustained regime showing mild negative autocorrelation is
itself informative and is not the failure being chased here: v3-indoor's
walkers move in a mathematically exact straight line at exact constant
velocity (zero true process noise), while the single model's `person`
`Q` assumes nonzero acceleration noise — a small, expected mismatch
between an idealized synthetic track and a Q sized for a real pedestrian,
not the maneuvering-target problem this day is about.

## Objective 3 — IMM: built

`src/estimator/imm.py`. Three modes — `static`, `constant_velocity`
(Day 20's `person` model, unchanged), `maneuvering` (new, high-Q) — sized
to what was actually measured: onset needed nothing (already fine), so a
fourth mode for it would have solved a problem the data did not show.
Standard IMM cycle (mixing → mode-matched filtering → mode-probability
update → combination), config-driven and versioned transition matrix
(`ImmConfig.sha`). `StateEstimate` gained `imm_config_sha` and
`mode_probabilities` (both optional, backward compatible;
`require_comparable` extended to a four-way key when set). §17's prior
firewall re-tested on the new code path (`inspect.signature`, no
parameter carries a prior; mode probabilities derive only from
observation likelihoods, starting uniform at bootstrap).

**A real bug found and fixed before this landed.** The first `static`
mode reused `asset_static`'s tight-Q *constant-velocity* model — which
shares the identical velocity-to-position `F` with every other mode, so
any velocity mixed in from another mode kept propagating just as well as
under `constant_velocity`. It only claimed a *tighter* covariance while
doing so, which let it win the mode-probability contest on covariance
width alone, including during a clean, noiseless, pure 0.5 m/s walk
(99.97% "static" by the last frame — a filter this cheap to fool would
say more about the mode contest than the target). Fixed with
`NearlyConstantPositionMotionModel`: no velocity-to-position coupling in
`F` at all — position genuinely does not move in this mode, however
confident or not that claim is. This is exactly the kind of thing a
"measure before rebuilding" day is supposed to catch before it ships, not
after.

## Objective 4 — re-evaluated per regime: no-trade criterion NOT satisfied

Same observations for both filters (identical seed, identical draw
order) so the comparison is about the filter, not about which noise draw
each one got.

**Pooled NEES coverage, single-model vs IMM:**

| set | single-model | IMM | reading |
| --- | ---: | ---: | --- |
| v3-indoor | 0.9909 | 0.6886 | **worse** |
| v4.1-gate | 0.8053 | 0.4632 | **worse** |

This is stated first because it means the per-regime table below is not
a case of "the aggregate looks fine but hides a regression" — the
aggregate itself already fails on both sets.

**Per regime, single-model → IMM:**

| set | regime | n | coverage: single → IMM | position RMSE: single → IMM |
| --- | --- | ---: | --- | --- |
| v3-indoor | static | 38 | 1.0000 → 1.0000 | 0.085 m → 0.044 m |
| v3-indoor | onset | 284 | 0.9648 → 0.9014 | 0.156 m → 0.147 m |
| v3-indoor | sustained | 2414 | **0.9938 → 0.6587** | **0.114 m → 0.205 m** |
| v4.1-gate | static | 175 | **0.8171 → 0.4343** | 0.237 m → 0.060 m |
| v4.1-gate | onset | 12 | 0.8333 → 0.9167 | 0.190 m → 0.194 m |
| v4.1-gate | cessation | 3 (thin) | 0.0000 → 0.3333 | 0.159 m → 0.143 m |

**Acceptance criterion, stated in advance: onset/maneuver coverage moves
materially toward nominal AND static/sustained do not degrade. Evaluated
explicitly, both sets: NOT SATISFIED.**

- v3-indoor: no transient regime improved materially (onset actually
  moved *away* from nominal, 96.5%→90.1%), and sustained — a steady
  regime — degraded severely (99.4%→65.9% coverage, RMSE nearly doubled).
- v4.1-gate: static — a steady regime — degraded severely (81.7%→43.4%),
  even though its accuracy improved a great deal (0.237m→0.060m).

**The v4.1-gate static result is the sharpest, most important number
today and deserves its own sentence: IMM cut position error to roughly a
quarter while making the filter's own stated confidence in that estimate
measurably less honest.** The mechanism is legible: `NearlyConstantPositionMotionModel`'s
process noise is tight enough to genuinely improve raw accuracy once IMM
locks onto the static mode, but the combined covariance shrinks faster
than the actual residual distribution's tails, so the reported
uncertainty band no longer covers the real error as often as it claims
to. Accuracy and calibration are not the same axis, and this is a clean
demonstration that they can move in opposite directions from the same
change.

**Why sustained got worse under IMM (v3-indoor) is the more concerning
result, because that regime needed no fix.** The most likely mechanism:
`ImmConfig`'s transition matrix keeps re-mixing a small, constant share
(2.5% each, at `persistence_probability=0.95`) of the `static` and
`maneuvering` modes' less-accurate predictions into the combined output
on *every* step, even once mode probability has converged strongly
toward `constant_velocity` — a standing tax on a regime that already had
none of the disease. This is stated as the likely mechanism, not
confirmed by a second measurement — an actual test of it (e.g. a stickier
matrix, or a probability floor before blending) is Day 22's first item,
not something to chase today per the day's own instruction against
iterating a model to force a target number.

**Day-10 validity gate:** re-run, **PASSED** on both sets (unchanged —
gating logic does not depend on which filter scored the set).
**Falsification test 5:** re-run, still **PASSES/PARTIAL** exactly as Day
20 left it (`tests/test_falsification.py`, 8/8 green) — it exercises the
single-model path, which is unaffected by anything built today.

## Objective 5 — mode probability: documented as a product output, not wired in

`StateEstimate.mode_probabilities` (built in Objective 3) is the IMM's
directly-useful output, not an internal quantity — documented, along with
a new `dominant_mode()` convenience accessor, with its actual intended
consumers named directly in the docstring: POSE verbs ("sat", "stood")
and dwell detection keyed off a sustained `static`-family mode; motion-
onset event triggering keyed off a shift away from one; and the motion
gate scorecard's existing `low_activity` classification
(`src/data/golden.py`, Day 15/16), which already tries to characterize
the same thing from outside the estimator's own view of it.

**Not wired into the event compiler today, and stated as uncalibrated in
the docstring itself** — Objective 4 just measured that the *combined
estimate's* covariance is overconfident in some regimes, and mode
probability has never been checked against real motion labels at all.
Validating it is its own future objective, not assumed by exposing the
field.

## What remains skeleton, and the order it should land in

1. **IMM's mixing overhead on steady regimes** (new finding, today) — the
   most urgent gap, ahead of everything below: shipping IMM without
   understanding why it degrades an already-good regime would repeat
   Day 20's own mistake in a new shape.
2. **Multi-entity factor graph.** Unchanged from Day 20 — still needed
   before `asset_carried`'s rigid-coupling interface does anything.
3. **Smoothing** (`horizon_kind="smoothed"`). Still secondary — smoothing
   a filter whose calibration is not yet trustworthy tightens a number
   that is not yet honest, rather than making it more honest.
4. **Hypothesis management.** Furthest out; depends on multi-entity
   association existing first.
5. **Mode-probability validation against real motion labels** (new, from
   Objective 5) — required before any of Objective 5's documented
   consumers (POSE verbs, dwell detection, motion-onset triggering,
   `low_activity`) may actually be wired up.

## Full suite and mypy

`mypy` (scoped per `mypy.ini`, unchanged file list — `src/estimator`
already covered every new Day-21 module): **0 errors, 58 files**, up from
Day 20's 55. Repo-wide (`.venv-pinned`, `not requires_weights and not
slow`): **918 passed, 1 skipped, 8 deselected, 0 failures** — up from Day
20's 871 (+47: regime classification, innovation diagnostics, IMM
structural/behavioural tests, the eval-script comparison harness, and the
`dominant_mode()` accessor). `black --check .` / `flake8 .` show
pre-existing formatting/lint drift in ~25 files from earlier days,
unrelated to anything touched today (confirmed by diff — none are files
this session modified); every file touched today is clean under both.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 20 — today's work was
entirely unblocked by design, same as Day 20:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 9 days old.
5. **The `main`/`origin/main` divergence decision (ADR 0009).**
6. **Reference hardware procurement decision** (`docs/reference_hardware.md`)
   — unchanged since Day 19.

## Objective 0/6 — push, end of day

`git push --all origin`: `foundation/day-21` pushed clean, matches origin
exactly. `main` rejected, unchanged. `git push --tags`: up to date.

## Day 22, in order

1. **Investigate IMM's mixing overhead on steady regimes** — why does a
   3% (approx.) standing mixture of `static`/`maneuvering` modes measurably
   degrade `constant_velocity`'s already-good calibration on v3-indoor's
   sustained frames? Candidate fixes to *measure*, not assume: a stickier
   transition matrix (higher `persistence_probability`), a minimum
   mode-probability floor before a mode contributes to the combination,
   or reconsidering whether `maneuvering`'s Q is simply too wide relative
   to what a converged `constant_velocity` estimate needs protecting
   from. Re-run `scripts/eval_estimator.py --filter both` against the
   same acceptance criterion — do not consider IMM ready until it passes.
2. **Investigate v4.1-gate static's accuracy/calibration trade** —
   confirm the "covariance shrinks faster than the tail" mechanism with a
   second measurement (e.g. the empirical error distribution's kurtosis
   in that regime) before deciding whether `NearlyConstantPositionMotionModel`'s
   process noise needs a floor.
3. **Reference hardware decision** — still pending a human, now 3 days
   old.
4. **Declare a target fps for real camera ingest** — still open from Day
   19.
5. **Cascade bench, clean, on interim or reference hardware** — still
   pending hardware.
6. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
   deferred from Day 18/19.
7. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
8. **Multi-entity factor graph** — depends on item 1 landing first.
9. **Smoothing (`horizon_kind="smoothed"`)** — depends on item 1.
10. **Mode-probability validation against real motion labels** — depends
    on item 1/2; needed before any Objective-5 consumer is wired up.
11. **The generator has no sensor-noise model** — unchanged.
12. **Order cameras and run the office capture** — now 9 days old.
13. **The motion-gate precision/selectivity investigation** — ten days
    deferred.
14. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
    still open from Day 14.
15. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
16. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged.
17. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
18. **MEVA licence verification** — still blocked on a human.
19. **Hypothesis management** — furthest out; depends on item 8.

# Day 22

**No timing, throughput, CPU-percentage, or latency claim is made anywhere
in this section** — unchanged hard scope rule from Day 20/21. Everything
below is accuracy and consistency: mixture-aware NEES/coverage, a
physically-derived covariance floor, and per-regime position/velocity
error against exact GT.

**Headline: neither of today's two candidate fixes changed the estimator
decision, and both failures are informative rather than inconclusive.**
The metric-validity question (was Day 21's pooled NEES even a fair test of
IMM?) was a real but PARTIAL red herring — mixture-valid metrics move the
exact magnitude of IMM's degradation but not its direction; steady regimes
are still measurably worse under IMM than under the single model. The
physically-derived velocity covariance floor — a two-line constraint,
derived once from a constant this project already declared on Day 20, no
tuning — turned out to be numerically INERT on both golden sets: config B
(single model + floor) is bit-for-bit identical to config A everywhere,
verified against raw per-frame covariance traces (not a wiring bug — the
natural Kalman steady state at this project's 12fps already sits well
above the floor). And underneath both questions sits a third, more basic
one this day surfaced rather than resolved: the no-trade criterion itself
cannot be evaluated on either golden set, because cessation — the regime
Day 21 diagnosed as the actual problem — has 0 frames on v3-indoor and 3
on v4.1-gate, both below this project's own 10-frame floor for a
conclusion. Config A (Day 20's single model, unchanged) remains in use.
See ADR 0010 for the full decision record.

## Verdicts

- **Is pooled NEES a valid metric for judging IMM (a mixture posterior)?
  Does using a mixture-valid metric rescue IMM?** Partially invalid —
  NEES assumes a single Gaussian; two mixture-valid alternatives were
  built (`compute_mixture_nees`, `empirical_coverage_by_sampling`). It
  does NOT rescue IMM: only the magnitude of the degradation moves
  (v3-indoor sustained softens from a ~33pt to a ~20pt regression under
  the sampling metric; v4.1-gate static is identical, 43.4%, under every
  metric). → Objective 1.
- **Does the physically-derived velocity covariance floor fix the
  cessation problem?** Measured as numerically INERT on both golden sets
  at this project's 12fps — floor (~0.0156 (m/s)²) sits 4-6x looser than
  natural steady-state convergence (~0.065-0.09 (m/s)²); config B is
  bit-for-bit identical to config A everywhere. (Later found Day 24 to be
  a per-timestep derivation bug, not a physical result about this filter
  — see Day 24 Objective 2 and Day 25 Objective 1.) → Objective 2.
- **Does any config (B/C/D) clear the no-trade criterion against config
  A?** UNSCOREABLE on every set/candidate — cessation has 0 (v3-indoor) or
  3 (v4.1-gate) frames, both below the 10-frame floor for a conclusion.
  Independent of that, static/sustained measurably regressed under C/D
  regardless (v3-indoor sustained −0.110, v4.1-gate static −0.383). →
  Objective 3.
- **Which configuration is adopted, and why?** Config A (Day 20's single
  model), unchanged — the blocker is identified as a DATA gap (no
  cessation volume to score against), not an unbuilt estimator. →
  Objective 4 / ADR 0010.

## Objective 0 — push, start and end of day

Start of day: `foundation/day-22` branched from `foundation/day-21`
(`d0694b9`), pushed clean, verified local HEAD matched
`origin/foundation/day-22` exactly. `main` unchanged.

## Objective 1 — is pooled NEES valid for an IMM? Partially, and it doesn't rescue IMM

NEES assumes a single-Gaussian posterior; IMM's combined estimate is a
Gaussian mixture collapsed to one `(mean, cov)`. `src/estimator/consistency.py`
now makes this a STRUCTURAL rule rather than an implicit assumption:
`compute_nees` takes a `posterior_family` argument and raises
`PosteriorFamilyError` on anything but `"gaussian"` — a single-Gaussian
NEES applied to a mixture no longer silently returns a number. The old
Day-21 number is kept, not deleted, via
`compute_collapsed_gaussian_nees_diagnostic`, whose own `consumer` field
names it invalid for judging IMM's consistency — reported "alongside" the
new metrics per the objective, not presented as validated. Two
mixture-valid alternatives were added: `compute_mixture_nees`
(probability-weighted per-mode NEES, each mode scored against its own
`(mean, cov)`) and `empirical_coverage_by_sampling` (nonparametric
highest-posterior-density coverage via Monte Carlo — no Gaussianity
assumed for the mixture's overall shape, only for each component).
`StateEstimate` gained `mode_states` (per-mode mean/cov, alongside the
existing `mode_probabilities`) so these can be computed without reaching
into IMM's private internals.

Re-evaluated per regime, all three metrics side by side:

| set | regime | n | single-model | IMM collapsed (diagnostic) | IMM mixture-weighted NEES coverage | IMM sampling coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v3-indoor | onset | 284 | 0.9648 | 0.9014 | 0.8732 | 0.9613 |
| v3-indoor | sustained | 2414 | 0.9938 | 0.6587 | 0.5058 | **0.7962** |
| v4.1-gate | static | 175 | 0.8171 | 0.4343 | 0.4343 | 0.4343 |

**Guarding against motivated reasoning, stated plainly: the mixture-valid
metrics do not rescue IMM.** v3-indoor sustained moves from a ~33-point
apparent regression (collapsed) to a real ~20-point regression (sampling)
relative to the single model's 99.4% — smaller, but still a clear,
directional degradation of a regime that needed no fix. v4.1-gate static
is identical under every metric (43.4%) because IMM's mode probability
there is concentrated enough on one mode that collapsing the mixture
changes almost nothing — a useful negative control: the metric-validity
question can only ever matter where the mixture is genuinely spread across
modes, and it changes the story there (sustained) without changing the
verdict. **Conclusion: the metric question was a real, partial red
herring** — it was worth asking, it changed a number materially, and it
did not change what that number means for whether IMM is ready.

## Objective 2 — the velocity covariance floor: physically sound, measurably inert

`pedestrian_velocity_covariance_floor_mps2(dt_s)` derives a floor on a
person-kind mode's posterior velocity variance from `PERSON_SIGMA_A_MPS2`
(1.5 m/s², already declared Day 20 as this project's pedestrian
acceleration bound) — the same quantity `Q`'s own velocity-block entry
already injects every single predict step (`(sigma_a * dt)^2`), so a
posterior claiming tighter velocity confidence than what one step's own
process noise already asserts is possible is physically incoherent — which
is exactly Day 21's diagnosed cessation mechanism. `apply_velocity_covariance_floor`
clamps the velocity diagonal up to this floor, proven PD-preserving by
construction (raising one diagonal entry of a PD matrix is a rank-1 PSD
perturbation). Opt-in per motion-model instance
(`velocity_covariance_floor=True`), valid only for `person` and
`asset_carried` (which inherits `person`'s bound, not its own inflated
`sigma_a_mps2` — the floor represents the carrier's body, not the carried
object's process noise); rejected outright, not silently ignored, for
`asset_static`/`fixture`. The derivation is reproduced by a unit test from
`PERSON_SIGMA_A_MPS2` directly, not asserted against a hardcoded literal.

**Measured result: the floor changes nothing on either golden set.**
Config B (single model + floor) is numerically identical to config A —
every RMSE, every coverage figure, to displayed precision — on both
v3-indoor and v4.1-gate, at every regime. This was checked directly
against raw per-frame velocity covariance on the actual scored
`brief_entry` cessation track (not just aggregate RMSE, to rule out a bug
hidden by rounding): at this project's 12fps (`dt_s ≈ 0.083`), the floor
evaluates to `~0.0156 (m/s)^2`; this filter's natural steady-state
velocity variance, even after only 6 onset frames — short of full
convergence — is already `~0.065-0.09 (m/s)^2`, four to six times looser
than the floor. **The floor never binds, so it changes nothing, on this
data.** Per the day's own instruction: this is reported as the finding it
is, not tuned to force a different answer. It was not swept; the same
derivation that produced the ineffective value is the one shipped, opt-in
and undefaulted.

## Objective 3 — four-way head-to-head, per regime

`scripts/eval_estimator.py --config A --config B --config C --config D`
(default: all four). Position RMSE / coverage (mixture-valid sampling for
C/D, NEES-based for A/B) / frame count, per regime, both sets:

| set | regime | n | A | B | C | D |
| --- | --- | ---: | --- | --- | --- | --- |
| v3-indoor | static | 38 | 0.085m / 1.000 | = A | 0.044m / 1.000 | = C |
| v3-indoor | onset | 284 | 0.156m / 0.965 | = A | 0.147m / 0.961 | = C |
| v3-indoor | sustained | 2414 | 0.114m / 0.994 | = A | 0.205m / **0.796** | = C |
| v3-indoor | cessation | 0 | empty | empty | empty | empty |
| v3-indoor | maneuver | 0 | empty | empty | empty | empty |
| v4.1-gate | static | 175 | 0.237m / 0.817 | = A | 0.060m / **0.434** | = C |
| v4.1-gate | onset | 12 | 0.190m / 0.833 | = A | 0.194m / 0.917 | = C |
| v4.1-gate | sustained | 0 | empty | empty | empty | empty |
| v4.1-gate | cessation | 3 (THIN) | 0.159m / 0.000 | = A | 0.143m / 0.333 | = C |
| v4.1-gate | maneuver | 0 | empty | empty | empty | empty |

`maneuver` is empty on both sets by construction (every synthetic agent
walks one straight line at constant speed — Day 21). Trivial-baseline
margins (copy-previous, constant-velocity dead reckoning) stay clearly
positive for every config in every non-empty regime, unchanged from
Day 20/21's finding — not reproduced above for space, printed in full by
the script.

The no-trade criterion (restated around cessation specifically, since
Day 21/22 pinned the failure there, not at onset or maneuver — Day 21's
"any transient regime" framing would have let an unrelated onset wiggle
satisfy a criterion meant to certify a cessation fix) was applied to every
candidate against config A:

| set | A → | cessation n | status |
| --- | --- | ---: | --- |
| v3-indoor | B | 0 | **UNSCOREABLE** |
| v3-indoor | C | 0 | **UNSCOREABLE** |
| v3-indoor | D | 0 | **UNSCOREABLE** |
| v4.1-gate | B | 3 | **UNSCOREABLE** |
| v4.1-gate | C | 3 | **UNSCOREABLE** |
| v4.1-gate | D | 3 | **UNSCOREABLE** |

Both below `MIN_REGIME_FRAMES_FOR_A_CONCLUSION` (10). This is independent
of every number in the table above: even v4.1-gate cessation's NEES pass
rate moving 0.000→0.333 under IMM, or RMSE improving 0.159m→0.143m, are
real, correctly-computed numbers from 3 frames — not evidence at the
confidence this project requires before calling a regime fixed. Static and
sustained REGRESSED under C/D in both sets regardless (v3-indoor sustained
−0.110, v4.1-gate static −0.383, both flagged `** REGRESSION **`), which
alone would fail the criterion even if cessation were scoreable.

## Objective 4 — ADR 0010: config A remains in use

`docs/adr/0010-estimator-configuration.md`, status Accepted. Records: the
cessation diagnosis and mechanism (Day 21, restated); the pooled-NEES
metric verdict (Objective 1 — partially invalid, does not rescue IMM); the
four-way results and no-trade verdicts (Objectives 2-3); the adopted
configuration (A) and why B/C/D are each rejected on their own,
independent grounds (B: inert, not wrong; C/D: measurably degrade steady
regimes with no offsetting cessation evidence because none exists in this
data). Explicitly: **the blocker is data, not an unbuilt estimator** —
neither golden set contains enough cessation frames to certify any fix,
whatever that fix turns out to be.

## What remains skeleton, and the order it should land in

1. **Cessation frame volume in the synthetic golden sets** — the single
   highest-leverage item surfaced today: until v3-indoor/v4.1-gate (or a
   new set) contain ≥10 cessation frames, no estimator change touching
   cessation can be certified against the no-trade criterion, regardless
   of how it is built. Unblocks re-evaluating every configuration in
   ADR 0010.
2. **A differently-derived velocity floor**, if revisited — sized to an
   asserted stopping-relevant variance rather than one predict step's own
   Q injection, since today's derivation (correct as far as it went) is
   looser than this filter's own convergence at 12fps. Needs its own
   independent physical basis stated before measurement, not chosen to
   move the number.
3. **IMM's steady-regime "mixing overhead" hypothesis** (Day 21, restated
   Day 22 item 3 originally) — still unconfirmed by a second measurement,
   still open, independent of cessation.
4. **Multi-entity factor graph.** Unchanged — still needed before
   `asset_carried`'s rigid-coupling interface does anything, and still
   inherits config A's known overconfidence on v4.1-gate static.
5. **Smoothing** (`horizon_kind="smoothed"`). Still secondary for the same
   reason Day 21 gave: smoothing a filter whose calibration is not yet
   trustworthy tightens a number that is not yet honest.
6. **Mode-probability validation against real motion labels.** Unchanged
   from Day 21 — still required before any of Objective 5's (Day 21)
   documented consumers may be wired up.
7. **Hypothesis management.** Furthest out; depends on item 4.

## Full suite and mypy

`mypy` (scoped per `mypy.ini`): **0 errors, 58 files**, unchanged file
count from Day 21 — today's work extended existing estimator modules
rather than adding new ones. Repo-wide (`.venv-pinned`, `not
requires_weights and not slow`): **966 passed, 1 skipped, 8 deselected, 0
failures** — up from Day 21's 918 (+48: posterior-family guard, mixture
NEES/coverage, the velocity floor's derivation and binding/non-binding
behavior, the four-way eval harness and generalized no-trade verdict). `black --check .` / `flake8 .`
show the same pre-existing formatting/lint drift in ~25 files noted
Day 21, unrelated to anything touched today (confirmed absent from the
list); every file touched today is clean under both.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 21 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification.**
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 10 days old.
5. **The `main`/`origin/main` divergence decision (ADR 0009).**
6. **Reference hardware procurement decision** (`docs/reference_hardware.md`)
   — unchanged since Day 19.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-22` pushed clean, matches origin
exactly. `main` unchanged. `git push --tags`: up to date.

# Day 23

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-22.
Everything below is accuracy, consistency, and data volume: exact-GT
regime classification, per-regime position/velocity error, mixture-aware
NEES/coverage, and a controlled frame-rate sweep of a closed-form
covariance floor against the filter's own measured convergence.

**Headline: Day 22's cessation finding was under-labeled, not
under-measured — relabeled and given real volume, it survives at full
strength and gets more decisive, not less.** Objective 1 found the
apparent 0-3-frame cessation regime was a labeling defect: the recovery
tail Day 21 traced by hand (NEES ~800 decaying to nominal over 10-14
frames) had been silently absorbed into `static` the instant GT speed hit
zero. Fixed, it raised v4.1-gate's cessation count to 39 — real, but
still only ~2 underlying trajectories. Objective 2 supplied the volume
that fix needed: a new `synthetic-indoor-v5-cessation` set, 373 cessation
frames from 19 diverse stop events, minted only after clearing its own
volume gate. Objective 3 used it: the no-trade criterion is now scoreable
on two of three sets and NOT_SATISFIED for every candidate — the velocity
floor still never binds (confirmed, and now shown to never bind at ANY
frame rate from 1 to 1000 fps), and IMM makes cessation worse while
regressing static harder than any set measured to date. Config A remains
adopted, now correct-by-measurement rather than correct-by-elimination.
Objective 4 built the legal office-data path in parallel: 12 research
datasets registered with validity-matrix cells, an ordered
licence-verification list (confirms MEVA, then DA-2K), and a new
`C_pending_consent` lane closing a real DPDP gap around the company's
existing CCTV archive.

## Verdicts

- **How many frames actually support Day 21's NEES-815 diagnosis?** More
  than believed at Day 22, but still layered: a labeling defect (not a
  data gap) had absorbed the entire post-stop recovery tail into `static`;
  fixed, v4.1-gate's cessation count rises 3→39 (12x) — still only ~2
  underlying trajectories, highly autocorrelated, not 39 independent
  samples. v3-indoor remains genuinely 0 (no stop events exist in that
  set). (Day 24 Objective 1 later found the "815" figure specifically
  traces to n=1 hand-traced track, distinct from this n=39 aggregate —
  see Day 24/25.) → Objective 1.
- **Can the no-trade criterion be scored with real statistical power?**
  Yes — new `synthetic-indoor-v5-cessation` golden set, 373 cessation
  frames from 19 diverse stop events, minted only after clearing its own
  volume gate (≥200 cessation, ≥30 every other non-exempt regime),
  enforced at mint time by `RegimeVolumeError`. → Objective 2.
- **Does any config clear the no-trade bar on v5-cessation? Does the
  velocity floor bind at any frame rate from 1-1000fps?** No config
  clears the bar (NOT_SATISFIED for B/C/D on both scoreable sets); IMM
  makes cessation coverage worse (0.5013→0.3727) while regressing static
  harder than any set measured to date (0.9962→0.0808). The floor never
  binds at ANY tested frame rate. (Both the "never binds" finding and the
  no-trade verdicts were later found Day 24/25 to rest on a mis-derived,
  per-timestep floor formula — corrected, the floor binds and B's verdict
  changes; see Day 24 Objective 2 and Day 25 Objectives 1 and 3.) Config A
  remains adopted today, now correct-by-measurement rather than
  correct-by-elimination. → Objective 3.
- **What is the state of the legal/licensing data path?** 12 research
  datasets registered with validity-matrix cells; MEVA ranked #1 for
  licence verification (best structural fit, most permissive-terms
  hypothesis), DA-2K #2 (adapter ready, only path to a real, non-refused
  `depth` number). New `C_pending_consent` lane refuses both training and
  eval use of non-consented footage (DPDP-grounded), covering the
  company's own existing CCTV archive. → Objective 4.

## Objective 0 — push, start of day

`foundation/day-23` branched from `foundation/day-22` (`4066e4f`), pushed
clean before any Day-23 commit landed — verified local matched
`origin/foundation/day-23` exactly. `main` unchanged. (End of day: see
the closing Objective 0/5 section below.)

## Objective 1 — how many frames support NEES 815? A labeling defect, now fixed — the finding survives

Day 22's diagnosis rested on cessation frame counts of 0 (v3-indoor) and
3 (v4.1-gate) — hard to reconcile with Day 21's own headline: NEES
climbing to ~815 and decaying back to nominal over 10-14 frames. Both
turned out true, for a reconcilable reason: `classify_track`
(`src/estimator/regime.py`) labeled only the *anticipatory* window before
a stop as `cessation`; the instant GT speed reached zero, every following
frame — including the entire 10-14 frame recovery tail Day 21 traced by
hand — was immediately `static`. Reproduced directly against the real
`brief_entry` track: NEES 759.9→9.1 over frames confirmed to have been
labeled `static` throughout (Day 21's own reported 815→16 differs only by
minor codebase drift since — same phenomenon, same order of magnitude).

Fixed with a second labeling pass: a frame the first pass calls `static`
is reclassified `cessation` if it falls inside a recovery window of the
moving-to-static transition that produced it (a static run beginning at a
track's own start is excluded — no GT evidence it ever stopped). Window
length reuses `PEDESTRIAN_STOP_DURATION_S` (Day 22's already-declared
~1s pedestrian-settling bound, not fitted to this decay curve) — 12
frames at 12fps, independently close to Day 21's empirical 10-14.
Measured effect: **v4.1-gate cessation count 3→39 (12x)**, now above
`MIN_REGIME_FRAMES_FOR_A_CONCLUSION` and genuinely scoreable; config A's
cessation coverage measures 12.8% under the corrected label — real,
decisive overconfidence, not the noise a 3-frame estimate could produce.
v3-indoor stays at 0 — a genuine data gap (that set has no stop events at
all), not a labeling artifact.

**Caveat stated plainly, not resolved by the label fix alone:** even at
n=39, v4.1-gate's cessation frames come from only 2 distinct underlying
stop trajectories, tripled by camera/lighting variants and expanded by
the recovery window itself — highly autocorrelated, not 39 independent
samples. The label fix makes the phenomenon measurable; it does not
supply behavioral diversity. That gap is what Objective 2 closes.
(`bc6afdb`; `tests/test_estimator_regime.py` — reproduces the recovery
window, its eventual settling, and its track-start exclusion directly.)

## Objective 2 — v5-cessation: a set that can score the criterion

`synthetic-indoor-v5-cessation`: 19 clips built on a new
`MultiSegmentAgent`/`PathSegment` (piecewise-linear walk/pause paths,
abrupt or `ease_out`-gradual deceleration) that generalizes the existing
`Agent` — a single-segment `MultiSegmentAgent` reduces to a plain `Agent`
exactly, verified by test. 8 radial (toward/away from camera, so R varies
through the walk) and 8 lateral (crossing at fixed depth, R roughly
constant) stop events spanning slow/medium/fast approach × near/far
distance × abrupt/gradual deceleration × a long-hold variant, plus 2
stop-then-restart clips and 1 double-stop clip.

Volume measured directly via `classify_track` over every scene's raw GT
track, not asserted: **1059 total frames — cessation 373, static 260,
sustained 282, onset 129, maneuver 15** (declared out of scope,
`V5_CESSATION_EXEMPT_REGIMES` — this generator's straight-leg paths
cannot genuinely exercise heading-change maneuvers without either dozens
of hand-authored turns or a curved-path model neither exists; the 15 are
incidental `ease_out` artifacts, measured and reported, not hidden). The
acceptance criterion (≥200 cessation frames, ≥30 for every other
non-exempt regime) is enforced AT MINT TIME by
`enforce_regime_volume`/`RegimeVolumeError` — the same "a set that cannot
score the criterion it exists for must not mint" mechanism as the
observability floor and `low_activity` exemption — tested both to pass on
v5-cessation and to refuse a set with no cessation (v3). Stopped-agent
bit-identity (Day-17 discipline) reproduced on this set's own pause
segments: rgb/depth/instances bit-identical across every held frame. v3
and v4.1-gate were never touched.

Registered lane S with full per-asset clearance in `configs/datasets.yaml`
(same primitives as every other synthetic-indoor set — SMPL trap does not
apply). Ran the Day-10 validity gates (`scripts/validity_matrix.py`):
**PASS `motion_geometry`, PASS `state_estimation`, REFUSE `depth`
(rank correlation −0.767, the worst of any set measured, 100% of pixels
in the 8m+ band), REFUSE `appearance_semantics`, REFUSE `point_tracking`**
— geometry-yes, appearance-no, exactly as this set's own purpose
predicts. Found and fixed a second instance of Day 16's "gate registered
but never wired into the matrix" defect while doing this:
`state_estimation` has been in `src/data/validity.py`'s `GATES` since
Day 20 and was never added to `validity_matrix.py`'s `CAPABILITIES`
tuple — silently missing from the one script whose job is to report every
registered capability, for three days, until this objective needed to
read a row that didn't exist. Fixed the same way Day 16 fixed it for
`point_tracking`. (`b6f6ed6`.)

## Objective 3 — four-way re-eval on v5, floor frame-rate dependence, ADR 0010 revised

Re-ran `scripts/eval_estimator.py`'s four-way A/B/C/D comparison against
v5-cessation and against v4.1-gate under Objective 1's corrected label.
Full per-regime table, v5-cessation:

| regime | n | A/B RMSE | A/B coverage¹ | C/D RMSE | C/D coverage² | margin(copy-prev), A |
| --- | ---: | --- | --- | --- | --- | --- |
| static | 260 | 0.1381m | 0.9962 | 0.0832m | 0.0808 | +0.1338m |
| onset (scored)³ | 91 | 0.1991m | 0.8132 | 0.2686m | 0.9011 | +0.1413m |
| sustained | 282 | 0.1470m | 0.9574 | 0.2067m | 0.9504 | +0.1789m |
| cessation | 373 | 0.2499m | 0.5013 | 0.1432m | 0.3727 | +0.0368m |
| maneuver (exempt) | 15 | 0.1789m | 0.4000 | 0.2608m | 0.8000 | +0.2855m |

¹NEES pass rate (single-Gaussian, valid for A/B). ²Mixture-valid
sampling-HPD coverage, not the collapsed-Gaussian diagnostic (Day 22's
`PosteriorFamilyError` discipline). ³91, not v5-cessation's own raw 129 —
`eval_estimator.py` drops each track's first 2 frames as filter warm-up
(`FIRST_COMPARABLE_INDEX`); all 38 dropped frames are onset-labeled
because every v5-cessation track starts walking. B=A and D=C everywhere
(floor still inert). Every config beats `constant_velocity_no_update` by
large margins throughout; omitted as uninformative, consistent with
Day 22.

No-trade verdict, now scoreable on two of three sets:

| version | A→B | A→C | A→D |
| --- | --- | --- | --- |
| v3-indoor | UNSCOREABLE (n=0) | UNSCOREABLE (n=0) | UNSCOREABLE (n=0) |
| v4.1-gate (n=39) | NOT_SATISFIED | NOT_SATISFIED (static −0.3604 **REGRESSION**) | NOT_SATISFIED |
| v5-cessation (n=373) | NOT_SATISFIED | NOT_SATISFIED (static −0.8231 **REGRESSION**) | NOT_SATISFIED |

No config clears the bar on either set with cessation frames. IMM does
not merely fail to improve cessation — it makes coverage worse
(0.5013→0.3727) while cratering static calibration harder than any set
measured to date (0.9962→0.0808). v3-indoor, untouched by today's work,
stays unscoreable — a genuine data gap specific to that set, not
something v5-cessation was ever meant to fix. **Magnitude nuance,
reported plainly:** config A's cessation coverage was 0.1282 on
v4.1-gate's thin n=39 (~2 underlying trajectories) and measures 0.5013 on
v5-cessation's properly-powered n=373 — both decisively below the 0.95
target, direction unchanged, but the earlier thin-sample magnitude was
itself partly a small-n artifact, exactly the caveat Objective 1 flagged
in advance.

**Frame-rate dependence of the velocity floor: it never binds.** New
`scripts/velocity_floor_frame_rate_sweep.py` runs config A (floor
disabled) over a controlled constant-velocity walk at a swept 1-1000 fps
range and compares the filter's own converged posterior velocity
variance against the floor's closed form at the same `dt_s`:

| fps | natural (m/s)² | floor (m/s)² | floor/natural |
| ---: | ---: | ---: | ---: |
| 1 | 3.8641 | 2.2500 | 0.5823 |
| 2 | 0.6570 | 0.5625 | **0.8562 (closest approach)** |
| 12 (this project) | 0.0650 | 0.0156 | 0.2402 |
| 90 | 0.0103 | 0.0003 | 0.0270 |
| 120 | 0.0113 | 0.0002 | 0.0138 |
| 1000 | 0.5480 | 0.0000 | 0.0000 |

Never binds anywhere tested. Least-obvious finding: natural convergence
is **not monotonic** in frame rate — worst at very low fps, minimized
around fps≈90-120, worse again at very high fps (differencing positions
close together in time, against fixed measurement noise, amplifies
velocity noise). The floor shrinks monotonically as `dt_s²`, so it gets
MORE inert as fps rises past 12, not less — "higher fps converges
tighter, so the floor binds sooner" is backwards. Closest approach: ~86%
of natural, at 2fps — already below any rate this product would run at.
**Retired as a live finding, not carried as dead code** — kept as
tested, opt-in machinery per ADR 0010's original Decision 6, since a
sound constraint that doesn't bind is real information, not a bug; no
longer an open research thread, because no frame rate this product could
plausibly run at makes it relevant. (Day 22's OTHER open question — a
floor derived from the actual stopping deceleration profile, a different
physical basis — is untouched and remains open.)

`docs/adr/0010-estimator-configuration.md` revised: original Day-22
decision and evidence left untouched and visible; new "Day 23 revision"
section carries all of the above. **Decision unchanged: config A remains
adopted** — Day 22 adopted it because no alternative could be shown to
clear the bar and the bar itself couldn't be evaluated; Day 23 evaluated
it, on 373 real cessation frames, and no alternative clears it.
Correct-by-elimination becomes correct-by-measurement. (`f349b02`.)

## Objective 4 — the legal office-data path

Registered 12 research datasets (OA18 new; MEVA, Charades, NTU-RGBD-120,
Toyota-Smarthome, InHARD, MECCANO, MMPTRACK, DA-2K, ETH3D, iBims-1,
DIODE-indoor backfilled) with lane R, `license_snapshot: null`,
`hypothesis_class`, and a new `validity_matrix_cell` one-liner each.
MMPTRACK's cell is recorded as a genuine gap (no registered validity gate
covers cross-camera multi-person tracking), not forced into a poor fit.

**Ordered licence-verification list, by product value unblocked:**

1. **MEVA** — best structural match to the product's own deployment
   shape (overlapping multi-camera indoor facility footage,
   surveillance-style activity), and the only activity candidate with a
   "historically unusually permissive terms" hypothesis — highest chance
   verification actually unlocks broad use, not just confirms
   research-only status.
2. **DA-2K** — the only candidate with a ready, TESTED adapter (zero
   engineering lag once cleared); pairwise relative depth with no
   scale/shift laundering, for a capability (`depth`) that has ZERO real
   coverage across every dataset this project owns today — confirmed by
   this same day's validity-matrix run, every set REFUSES `depth`.
3. **ETH3D** — real laser-scanned metric GT; fills the metric-AbsRel/RMSE
   cell DA-2K's rank-correlation-only eval structurally cannot.
4. **iBims-1** — closest public domain match (indoor-specific), planarity/
   boundary error metrics complementary to ETH3D, not redundant.
5. **OA18** — closest activity-taxonomy match to this product's own
   office-monitoring verbs, but a weaker (unhinted) license hypothesis
   than MEVA and no adapter readiness — verify after the higher-leverage
   structural gap MEVA covers.
6. **Charades** — verb-generalization breadth; `hypothesis_class` itself
   says "verify carefully", the weakest-stated confidence of the activity
   set.
7. **DIODE-indoor** — dense long-range GT for the coverage/far-field gap
   Day 15 found; overlaps ETH3D/iBims-1 at typical range, incremental
   value is specifically long range.
8. **NTU-RGBD-120** — narrower use case (skeleton pose-verb benchmarking,
   lab-only conditions).
9. **Toyota-Smarthome** — hardest realistic eval available, but
   untrimmed/long-duration is a harder integration lift than its
   marginal value over OA18/Charades justifies verifying first.
10. **InHARD / MECCANO** — explicitly Tier-2, not indoor-surveillance-
    relevant today, per their own registry notes.
11. **MMPTRACK** — a genuine capability gap, not just a licensing one: no
    validity-matrix gate exists for cross-camera tracking yet, so
    verifying its license alone would not unlock a scoreable capability.

**Confirms the stated expectation — MEVA first, then DA-2K** — both for
reasons already on record in the registry (structural fit + permissive
hypothesis for MEVA; adapter readiness + a zero-coverage capability for
DA-2K), not because either was assumed correct going in.

`C_pending_consent`: self-collected footage with no consent record for
the purpose at hand, rejected by BOTH `open_for_training` and
`open_for_eval` — unlike lane R, non-consented footage of real people has
no basis for even an internal eval number. The refusal names the DPDP
Act, 2023 purpose-change reasoning explicitly. Registered
`thinkwill-cctv-archive` under it: the company's existing
premises-security CCTV archive, recorded for a different purpose than AI
development — no footage fetched, copied, or processed by anything in
this repository. **Refusal tested three ways** — training refusal, eval
refusal, and that the refusal message names DPDP specifically (all
passing, `tests/test_data_registry.py`).

`scripts/ingest_capture.py`: `--source-kind` is now required, no default
(`fresh`/`archive`). `fresh` unchanged (lane C, `--consent` required,
refuses without it). `archive` lands in `C_pending_consent`
UNCONDITIONALLY — even if `--consent` is supplied, since an archive's
original consent basis does not automatically cover a new purpose.
Archived footage can never reach lane C by omission or by reusing old
paperwork; tested directly
(`test_cli_archive_source_kind_writes_c_pending_consent_manifest`).

`docs/capture_runbook.md` gained a walk-then-stop scripting block (vary
approach speed, abruptness, stop duration, radial/lateral direction,
include stop-then-restart) — the same appearance/micro-motion gap
v5-cessation's own manifest declares synthetic data cannot fill (gait
dissipation, balance micro-motion at a real stop are appearance-learned
signals no analytic primitive renders). (`545b60c`.)

## What remains skeleton, and the order it should land in

1. **Multi-entity factor graph and smoothing** — Day 22 gated both on
   "cessation data volume lands, or a new floor basis is found." The
   first half just landed (v5-cessation). Both move up: no longer
   blocked, next in line.
2. **IMM's steady-regime mixing-overhead hypothesis** — still
   unconfirmed by a second measurement; v5-cessation's sharper static
   regression under IMM (−0.8231, worse than any set measured to date)
   is one more data point consistent with something systematic, not a
   confirmation of the specific mechanism.
3. **A velocity floor derived from the actual stopping deceleration
   profile** (a different physical basis than the one just retired) —
   the only live floor-shaped avenue left open, per ADR 0010.
4. **`ConsentRecord` for `thinkwill-cctv-archive`**, if the company
   decides to pursue the archive at all — a DPDP notice-and-consent
   process, not a coding task; `C_pending_consent` will keep refusing
   every loader until one is attached.
5. **`tests/test_synthetic_indoor.py`'s rendering tests are not marked
   `@pytest.mark.slow`**, despite matching that marker's own definition
   ("renders the full synthetic set... costing ~90s" — v5-cessation's own
   12 new tests alone measured at 533s). Found incidentally while
   preparing this report's full-suite run; `-m 'not slow'` does not
   actually exclude them, contrary to what the marker promises. Not
   fixed today (out of scope for Day 23's objectives) — flagged so it
   does not go unnoticed.

## Full suite and mypy

`mypy` (scoped per `mypy.ini`): **0 errors, 58 files** — unchanged file
count from Day 22 (today's estimator/data work extended existing modules;
`scripts/` is outside `mypy.ini`'s scope). `black --check .` / `flake8 .`
show pre-existing formatting/lint drift in 23 files, all pre-existing and
unrelated to today (confirmed absent from the list, cross-checked
individually); every file touched today is clean under both.

Targeted (every file touched or plausibly affected by today's changes,
run individually rather than trusted from a broader sweep): estimator
regime + registry (68), the 12 new v5-cessation/`MultiSegmentAgent` tests,
filter/`eval_estimator`/frame-rate-sweep (34), `fetch_dataset`/
`calibration_set` (19, checked because `configs/datasets.yaml` changed),
point-tracking validity (4), `test_golden_sets.py` (19, after the fix
below) — **156 tests, 0 failures.**

Repo-wide (`.venv-pinned`, `not requires_weights and not slow`, 1009
collected, 8 deselected — same deselection count as Day 22): found and
fixed one real failure while running this — `tests/test_golden_sets.py`'s
`test_every_minted_version_is_retained` hardcodes the set of golden
versions that should exist, the same fixture Day 17 updated for v4-gate
(`31c2ac3`); v5-cessation is a real, retained version, so it belongs in
the enumeration (`ec47d8d`). Not a defect in today's other work — this
fixture doing exactly its job the moment it went stale. The repo-wide run
reached >92% (through the point immediately before
`tests/test_synthetic_indoor.py`'s legacy v1-v4.1 rendering tests, this
report's own item 5 above) with the fix applied and zero further
failures observed; those remaining tests are unmodified by anything Day
23 touched (diff-reviewed: today's changes to that file are purely
additive — new classes/functions plus one cosmetic reformat of an
existing assert) and were not waited on to completion given their
independently-flagged, pre-existing, multi-minute-per-file cost (item 5
above). Confidence in "0 failures repo-wide" rests on the targeted 156
plus this partial-but-unbroken repo-wide pass plus the diff review, not
on watching the last 2% finish.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 22 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — now explicitly ranked #1 in
   Objective 4's ordered list (highest product impact of any pending
   data item, confirmed by reasoning today rather than merely assumed).
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 12 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 5 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-23` pushed clean, matches origin
exactly. `main` unchanged. `git push --tags`: up to date.

## Day 24, in order

1. **Multi-entity factor graph** — no longer blocked (v5-cessation
   supplied the cessation data volume Day 22 was waiting on); the
   highest-leverage next build.
2. **Smoothing** (`horizon_kind="smoothed"`) — same unblock as item 1,
   still secondary per Day 21's own reasoning (smoothing a filter whose
   calibration is not yet trustworthy tightens a number that is not yet
   honest).
3. **A velocity floor derived from the actual stopping deceleration
   profile** — the one live floor-shaped avenue left (Day 23 retired the
   current derivation as never-binding); needs its own independent
   physical basis stated before measurement, not chosen to move a number.
4. **IMM's steady-regime mixing-overhead hypothesis** — unconfirmed, now
   three days deferred; v5-cessation's sharper static regression is one
   more consistent data point, not a mechanism confirmation.
5. **Mode-probability validation against real motion labels** — depends
   on item 1 landing.
6. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; still blocked on a
   human.
7. **DA-2K licence verification** — ranked #2; the adapter is built and
   tested, zero engineering lag once cleared, and unlocks this project's
   first real (non-refused) `depth` number.
8. **Reference hardware procurement decision** — now 5 days old.
9. **Declare a target fps for real camera ingest** — still open from
   Day 19.
10. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
11. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; also now gates on item 7 for a real depth number to
    re-measure against.
12. **The `main`/`origin/main` decision** (ADR 0009) — a human call,
    still Proposed since Day 18.
13. **The generator has no sensor-noise model** — unchanged.
14. **Order cameras and run the office capture** — now 12 days old;
    `docs/capture_runbook.md` now includes the Day-23 walk-then-stop
    block, ready the moment hardware arrives.
15. **The motion-gate precision/selectivity investigation** — still
    deferred.
16. **Steer callers away from `raise_alert()` toward `emit_alert()`** —
    still open from Day 14.
17. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
18. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged.
19. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
20. **Mark `tests/test_synthetic_indoor.py`'s rendering tests
    `@pytest.mark.slow`** — found Day 23, not fixed; the marker's own
    definition already covers exactly this case.
21. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a DPDP
    process decision, not a coding task.
22. **Hypothesis management** — furthest out; depends on item 1.

# Day 24

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-23.
Everything below is accuracy, consistency, and test/measurement hygiene:
frame-support auditing, a corrected covariance-floor derivation and its
re-evaluation, a registry-derived validity matrix, and repo-wide test
timing.

**Headline, leading with the biggest reversal: the velocity-covariance
floor Day 22-23 reported as permanently, physically inert was
mis-derived, not inert. It scaled with the current predict step's own
`dt_s` instead of the absolute physical stop duration it was meant to
bound, so it shrank exactly as frame rate rose — which is why Day 23's
1-1000fps sweep found it never binding anywhere. Corrected to an
absolute bound, it now binds at every practical frame rate, and config B
(single model + floor) clears the no-trade criterion outright on
v4.1-gate and comes within one safe-direction deviation of clearing it
on v5-cessation. Day 23's four-way evaluation was comparing three real
configs and one no-op; today's is the first time all four have actually
been different filters.** Config A stays adopted today regardless — see
Objective 2 below for why re-deciding in the same session that found
this would repeat exactly the mistake Day 21 was built to avoid — but
the evidentiary picture ADR 0010 rests on is materially stronger for B
than at any point since Day 22 first proposed it.

Objective 1 separately found that Day 21's founding NEES-815 number —
the one that motivated IMM, the floor, and three days of work — was
itself under-supported as originally reported (n=1 hand-traced track).
The underlying phenomenon is real and is now independently established
at n=373 (Day 23), but the two questions are distinct: whether the
diagnosis was well-supported when made, and whether the thing it
diagnosed turned out to be true. The first answer is no; the second is
yes.

## Verdicts

- **How many frames actually supported Day 21's NEES-815 number
  originally?** Under-supported as originally reported: n=1 hand-traced
  track, not an aggregate, and not even the same frames Day 21's own
  printed `cessation` row (n=3) referred to. The underlying phenomenon is
  now independently established at n=373 (Day 23) — Days 21-22 are better
  read as "established the problem existed" than "measured how bad it
  was." → Objective 1.
- **Is the velocity floor's per-timestep derivation the bug behind its
  measured inertness?** Yes — the floor used the CURRENT PREDICT STEP's
  own `dt_s` instead of the absolute physical stop duration it was meant
  to bound. Corrected to an absolute form, it binds at every practical
  frame rate; config B clears the no-trade bar outright on v4.1-gate and
  is blocked on v5-cessation only by `sustained` moving in the SAFE
  (underconfident) direction under the (then-)symmetric scoring. Config A
  remains adopted today, pending the scoring-asymmetry question. (Resolved
  Day 25: the criterion was made directional and config B is adopted —
  see Day 25 Objectives 1 and 3.) → Objective 2.
- **Is the validity matrix generated from the registry, or hand-copied?**
  Was hand-copied — the same defect caught Day 16 for `point_tracking`.
  Now derived live from `GATES` at runtime, with a structural test.
  Parallel-maintenance audit found and fixed three more instances
  (`LANE_DESCRIPTIONS`, `CONFIG_DESCRIPTIONS`/`CONFIG_SPECS`,
  `_REQUIRED_PARAMS`); four other candidates checked and explicitly not
  flagged, with reasons on record. → Objective 3.
- **Are the genuinely expensive tests in `test_synthetic_indoor.py`
  marked `@pytest.mark.slow`, and does the full suite now run in bounded
  time?** Yes — 13 tests marked (every one calling `gen.generate()`);
  this file's `not slow` subset drops 957.71s→6.93s. Repo-wide `not
  requires_weights and not slow`: 1001 passed, 1 skipped, 21 deselected, 0
  failures — closing Day 23's open caveat. A repo-wide duration-threshold
  gate (`tests/conftest.py`, 15s) now fails any future unmarked test that
  exceeds it. → Objective 4.

## Objective 0 — push, start of day

`foundation/day-24` branched from `foundation/day-23` (`82742d3`), pushed
clean before any Day-24 commit landed — verified local matched
`origin/foundation/day-24` exactly. `main` unchanged since Day 16 (ADR
0009 still Proposed). (End of day: see the closing Objective 0/5 section
below.)

## Objective 1 — how many frames actually supported NEES 815?

Full derivation in `docs/adr/0010-estimator-configuration.md`'s "Day 24
revision (Objective 1)" section (`bcb5763`); summarized here.

**Verdict, one sentence: Day 21's NEES-815 diagnosis was under-supported
as originally reported — n=1 hand-traced track, not an aggregate, and
not even the same frames as the `cessation` row (n=3) Day 21's own report
printed alongside it — but the phenomenon it pointed at is now
independently established at real power (373 frames, 19 stop events,
Day 23), so Days 21-22 are better read as "established that the problem
existed" than "measured how bad it was."**

Peak NEES and decay curve (Day 21's `brief_entry` trace, reproduced
verbatim, the only source for the "815" figure):

```
frame  regime      NEES     bound   within
 0-3   onset       0.8-2.2  12.59   True
 4     cessation   165.6    12.59   False
 5-14  static      815->16  12.59   False (all)
15+    static      <10      12.59   True
```

Per-regime NEES coverage, Day 21 (single model, target 0.95), for
reference against the trace above:

| set | regime | n | coverage |
| --- | --- | ---: | ---: |
| v3-indoor | static | 38 | 1.0000 |
| v3-indoor | onset | 284 | 0.9648 |
| v3-indoor | sustained | 2414 | 0.9938 |
| v4.1-gate | static | 175 | 0.8171 |
| v4.1-gate | onset | 12 | 0.8333 |
| v4.1-gate | cessation | 3 | 0.0000 |

Under the pre-Day-23 regime label, frames 5-14 of the trace above were
classified `static`, not `cessation` — they are folded into v4.1-gate's
`static` row (diluted by ~165 genuinely-converged frames), not visible
in the `cessation` row printed next to the trace. **Day 21's own
aggregate table and its own headline number described two different
slices of the data that happened to share a name**, which is why neither
looking at the trace nor looking at the table alone would have surfaced
this — only comparing them does.

Regime definition, then vs now: **then**, `classify_track`
(`src/estimator/regime.py`, pre-Day-23) assigned `cessation` only to the
anticipatory window immediately before a stop; every frame from the stop
onward fell to `static`, including the entire recovery tail. **Now**
(Day 23 Objective 1, unchanged today), a second labeling pass
reclassifies a `static` run's opening frames back to `cessation` inside
a `PEDESTRIAN_STOP_DURATION_S` recovery window (12 frames at 12fps)
following a moving-to-static transition — the fix that raised
v4.1-gate's cessation count 3→39 and is the label v5-cessation's 373
frames were classified under from the start.

**Process note — the second instance of the same omission pattern.**
Day 20's per-distance-bucket margin existed in the underlying data but
was not surfaced in the report's own table until Day 21 went looking.
Here, the frame-support answer existed in Day 23's own Objective 1 body
(the `brief_entry` track is named directly) but was never assembled into
an explicit statement of the number's own evidentiary weight — Day 23's
headline described the fix and the outcome, not the support behind the
number that motivated it. Two instances is a pattern: a number can be
technically present in a report and still functionally missing if no
sentence states what it does or doesn't support. Applied going forward:
any report section that opens a multi-day investigation on the strength
of one measured number should state that number's own sample size in the
same paragraph it is first reported, not leave it inferable from a trace
printed for a different purpose.

## Objective 2 — the velocity floor: re-derived as an absolute bound

Full derivation, units at every step, and both re-evaluated golden sets
are in ADR 0010's "Day 24 revision (Objective 2)" section (`38fa5bb`);
summarized here.

**The bug.** `pedestrian_velocity_covariance_floor_mps2` computed
`sigma_v_floor^2 = (PERSON_SIGMA_A_MPS2 * dt_s)^2` — units
`(m/s^2 * s)^2 = (m/s)^2`, dimensionally a velocity variance, which is
why it read as correct. But `dt_s` there was the CURRENT PREDICT STEP's
own timestep, so the formula answers "how much velocity uncertainty does
one sample interval's own process noise inject" — a quantity that
shrinks as the sample interval shrinks, i.e. as frame rate rises. That
is not the physical question the floor exists to bound: how much could a
person's velocity have changed since the filter last had strong evidence
pinning it down, given a person can go from walking to at-rest in about
`PEDESTRIAN_STOP_DURATION_S` (~1s) — a fact about the world, independent
of how often a camera samples it.

**The fix.** Anchor to the absolute duration instead of the current
step's interval:

```
sigma_v_floor^2 = (PERSON_SIGMA_A_MPS2 * PEDESTRIAN_STOP_DURATION_S)^2
                 = (1.5 m/s^2 * 1.0 s)^2 = (1.5 m/s)^2 = 2.25 (m/s)^2
```

constant at every fps. The two formulas coincide, by construction, at
exactly `dt_s == PEDESTRIAN_STOP_DURATION_S` (1 fps) — which is why Day
23's own sweep table already contained the value `2.2500` in its fps=1
row without anyone noticing it was a coincidence, not a data point.

**Floor vs converged σ_v on v5, at this project's 12fps:** natural
Kalman convergence measures **0.0650 (m/s)²** (unchanged, re-measured);
the corrected floor is **2.2500 (m/s)²** — ~34.6x larger, i.e. it binds,
hard. Re-running `scripts/velocity_floor_frame_rate_sweep.py` across the
full 1-1000fps range: the floor now binds everywhere except fps=1 (the
coincidence point above), the exact reversal of Day 23's "never binds
anywhere" finding — confirming the objective's own hypothesis ("a floor
that never binds anywhere from 1-1000fps is more likely mis-derived than
physically irrelevant").

**Re-evaluated: `scripts/eval_estimator.py`'s four-way A/B/C/D
comparison, both golden sets with real cessation volume** (config B/D
now genuinely differ from A/C for the first time):

v5-cessation (n=373):

| regime | n | A cov | B cov | C cov | D cov |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 260 | 0.9962 | 0.9923 | 0.0808 | 0.7154 |
| onset (scored) | 91 | 0.8132 | 0.9890 | 0.9011 | 0.9011 |
| sustained | 282 | 0.9574 | 0.9929 | 0.9504 | 0.9787 |
| cessation | 373 | **0.5013** | **0.9946** | 0.3727 | 0.8606 |

v4.1-gate (n=39):

| regime | n | A cov | B cov | C cov | D cov |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 139 | 0.9928 | 1.0000 | 0.5468 | 0.5468 |
| onset | 12 | 0.8333 | 0.9167 | 0.9167 | 0.9167 |
| cessation | 39 | **0.1282** | **1.0000** | 0.0256 | 0.1282 |

No-trade verdicts (A→B):

| version | cessation Δ | steady-regime Δ | verdict |
| --- | --- | --- | --- |
| v4.1-gate (n=39) | +0.7718 (IMPROVED) | static −0.0072 | **SATISFIED** |
| v5-cessation (n=373) | +0.4040 (IMPROVED) | sustained −0.0355 (REGRESSION by the criterion's symmetric scoring) | NOT_SATISFIED |

**Config B clears the bar outright on v4.1-gate.** It misses it on
v5-cessation purely because `sustained` moved from 0.9574 to 0.9929 —
i.e. from slightly underconfident to more underconfident, the SAFE
direction of miscalibration, not the overconfident direction Day 21
diagnosed as dangerous. The no-trade criterion currently scores
`|coverage − 0.95|` growing as a regression regardless of which
direction it grows in, so this counts against B exactly as hard as
moving toward overconfidence would.

**Decision: config A remains adopted today, unchanged — but not because
B was re-measured and found wanting.** B's own defect is that no-trade's
sustained-regime scoring cannot currently tell "the filter got 4 points
more conservative" from "the filter got 4 points more overconfident,"
and changing that scoring specifically because it would flip B's verdict
this session is the exact "iterate a metric until it passes" pattern
this project's rules prohibit — however defensible the argument sounds
in isolation. Day 25's first item, stated before any re-scoring: decide
whether the no-trade criterion should weight over- and under-confidence
deviations asymmetrically, on its own methodological merits, and only
then revisit B's adoption.

## Objective 3 — the validity matrix is now derived, not copied

`scripts/validity_matrix.py`'s `CAPABILITIES` tuple was a hand-maintained
copy of `src.data.validity.GATES`'s keys — the exact defect caught and
hand-patched on Day 16 (`point_tracking`) and Day 23
(`state_estimation`), both times by adding to the tuple rather than
removing it. Replaced with `registered_capabilities()`, which reads
`GATES` live on every call — there is no second list to fall out of
sync a third time. `tests/test_validity_matrix.py` (new): registers a
dummy gate directly in `GATES` and confirms it appears with zero edits
to the script, plus a grep-verify that no other hand-maintained copy of
the capability list exists in `src/`, `scripts/`, `tests/`, or `configs/`.

**Parallel-maintenance audit, as asked.** Same question asked of the
rest of the codebase: where else is a registry kept in sync with a
consumer by memory instead of by construction? Found and fixed three
more, all the same latent shape, none yet drifted:

- `LANE_DESCRIPTIONS` (`src/data/registry.py`) — a dict keyed by the
  `Lane` Literal with no completeness check. Added
  `test_lane_descriptions_covers_every_lane`.
- `CONFIG_DESCRIPTIONS` / `CONFIG_SPECS` (`scripts/eval_estimator.py`) —
  two separately hand-maintained dicts keyed by the same four config
  labels. Added `test_config_descriptions_covers_every_config_spec`.
- `_REQUIRED_PARAMS` (`src/model/uncertainty.py`) — keyed by the
  `UncertaintyKind` Literal, no completeness check against
  `ALL_UNCERTAINTY_KINDS`. Added
  `test_required_params_covers_every_uncertainty_kind`.

Checked and explicitly NOT flagged: `tests/test_golden_sets.py`'s
hardcoded retained-version set (a deliberate append-only ledger — an
auto-derived version would make the test check disk against itself,
which is tautological, not a fix); `ACTION_TO_VERB`
(`src/data/converters/ntu_skeleton.py`, documented as a deliberately
partial/conservative mapping); `Envelope.capability`
(`src/model/envelope.py`, a free string field, not an enum with a
companion list); `DATASET_NAMES`
(`scripts/gen_synthetic_indoor.py`, single consumer, no parallel list).
(`d36e3d7`.)

## Objective 4 — test hygiene, and an apparatus finding along the way

Measured with `--durations=0` rather than guessed, per Day 23's own
flag that the `slow` marker's definition ("renders the full synthetic
set... costing ~90s") already covered these tests without excluding
them: `tests/test_synthetic_indoor.py`, unmarked, ran 957.71s across 29
tests. **13 tests genuinely qualify** — every one that calls
`gen.generate()` directly or via a subprocess, ranging 16.12s-421.13s.
The other 16 top out at 5.39s; two of those render individual frames via
`gen.render_frame()` (not the full set) and correctly do not match the
marker's own definition. Marked exactly the 13. Re-measured: this file's
`-m "not slow"` subset drops **957.71s → 6.93s**.

**Repo-wide, closing Day 23's open caveat** (`-m "not requires_weights
and not slow"`, the number Day 23 could not confirm because it stopped
at >92% on this file's then-unmarked tests): **1001 passed, 1 skipped,
21 deselected, 0 failures.**

**An apparatus finding surfaced while confirming that number, reported
rather than the clean re-run being quietly kept.** The first run of the
command above measured **969.07s**. Re-running the identical command
immediately after — same flags, same machine, no code change in
between — measured **58.42s**, with byte-identical pass/skip/deselect
counts. A `--durations=25` breakdown of the fast run tops out at 6.68s
for any single test and sums to 34.1s across all 25 slowest — nowhere
near enough to explain a 969s total either way. This is a measurement-
apparatus artifact, not a code or marking defect: the 969s run
immediately followed this same session's 16-minute, CPU-saturating
`--durations=0` run of `test_synthetic_indoor.py` alone, and this
machine has twice before (Day 18, Day 19) been caught throttling under
sustained load, including once while on AC power. `pmset -g therm`
after the fact showed `CPU_Speed_Limit 100` (not currently throttled)
with the battery at 16% and discharging — consistent with, but not
direct confirmation of, throttling during the slow run specifically,
since thermal state was not checked live at the time (unlike Day 19's
own catch, which was). Both numbers are reported for that reason: the
969s figure is not disowned, it is explained as far as the evidence
available actually supports, which is short of certain. The trustworthy
number for "does the not-slow suite run in bounded time" is the
uncontended one (58.42s, corroborated by a third run at 72.74s with the
new duration hook active below) — bounded, and consistent with what the
per-file numbers already predicted.

**STRUCTURAL, and cheap: a repo-wide duration-threshold gate.**
`tests/conftest.py` gained a `pytest_runtest_makereport` hookwrapper that
fails any test exceeding `SLOW_THRESHOLD_S` (15s — above the clean run's
observed 6.68s maximum for an unmarked test, below the 16s+ tier that
should already be marked) without `slow`/`requires_weights`/
`decoder_dependent` set, so the next expensive test marks itself rather
than being found by someone watching a terminal. Verified against an
isolated scratch fixture before wiring in (lowered threshold; one
intentionally-slow unmarked test fails with the expected message; one
marked-slow test of the same duration passes); the full repo-wide suite
still passes clean with the hook active (1001 passed, 1 skipped, 21
deselected, 72.74s). (`0abd8c7`.)

## Full suite and mypy

`mypy` (scoped per `mypy.ini`, unchanged file list — today's estimator/
validity/model work extended existing modules; `scripts/` and `tests/`
are outside scope): **0 errors, 58 files** — unchanged from Day 23.
`black --check` / `flake8` clean on every file touched today
(`src/estimator/motion_model.py`, `scripts/validity_matrix.py`,
`scripts/velocity_floor_frame_rate_sweep.py`, `tests/conftest.py`,
`tests/test_synthetic_indoor.py`, `tests/test_estimator_models.py`,
`tests/test_velocity_floor_frame_rate_sweep.py`,
`tests/test_validity_matrix.py`, `tests/test_data_registry.py`,
`tests/test_eval_estimator.py`, `tests/test_model_primitives.py`,
`docs/adr/0010-estimator-configuration.md`).

Repo-wide `-m "not requires_weights and not slow"`: **1001 passed, 1
skipped, 21 deselected, 0 failures** — see Objective 4 for the two
measured runtimes and why both are reported. This is the number Day 23
flagged as unconfirmed; it is now confirmed, not carried forward as an
open caveat.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 23 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; now 1 day older.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 13 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 6 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-24` pushed clean, matches origin
exactly. `main` unchanged. `git push --tags`: up to date.

## Day 25, in order

1. **Decide the no-trade criterion's directionality question, before
   re-scoring anything** — should `static`/`sustained` degradation be
   scored symmetrically around 0.95, or should moving toward
   underconfidence count less than moving toward overconfidence? Decide
   on the methodological merits first; only then revisit config B's
   adoption (Objective 2 today found B clears the bar on v4.1-gate
   outright and misses v5-cessation only via a safe-direction deviation
   under the current, symmetric scoring).
2. **Multi-entity factor graph** — no longer blocked since Day 23's
   v5-cessation landed; the highest-leverage next build regardless of
   item 1's outcome.
3. **Smoothing** (`horizon_kind="smoothed"`) — same unblock, still
   secondary per Day 21's reasoning.
4. **A velocity floor derived from the actual stopping deceleration
   profile** — Day 22's other open question, untouched by today's fix
   (today corrected the SAME physical basis's derivation; a genuinely
   different basis is still untried) — lower priority now that the
   corrected absolute floor already performs well, but not yet closed.
5. **IMM's steady-regime mixing-overhead hypothesis** — unconfirmed, now
   4 days deferred.
6. **Mode-probability validation against real motion labels** — depends
   on item 2.
7. **MEVA licence verification** — still blocked on a human, highest
   product impact of any pending data item.
8. **DA-2K licence verification** — adapter built and tested, zero
   engineering lag once cleared.
9. **Reference hardware procurement decision** — now 6 days old.
10. **Declare a target fps for real camera ingest** — still open from
    Day 19.
11. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
12. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; gates on item 8 for a real depth number.
13. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
14. **The generator has no sensor-noise model** — unchanged.
15. **Order cameras and run the office capture** — now 13 days old.
16. **The motion-gate precision/selectivity investigation** — still
    deferred.
17. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
18. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged.
19. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
20. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a DPDP
    process decision, not a coding task.
21. **Hypothesis management** — furthest out; depends on item 2.

Dropped from this list today: "steer callers away from `raise_alert()`
toward `emit_alert()`" — carried forward unchanged since Day 14 despite
`raise_alert()` having been deleted entirely on Day 15
(`tests/test_model_alert.py::test_raise_alert_deleted_from_events_module_and_package_root`
asserts its absence). Nine days of carrying an already-resolved item is
the same near-identical-recurring-boilerplate-line risk this list has
already shown once (Day 20's append-order slip) — checked directly
against the filesystem before dropping, not assumed stale.

# Day 25

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-24.
Everything below is accuracy, consistency, and report/process hygiene: a
real-run confirmation of the velocity floor's binding, a directional
no-trade criterion, and a structural fix for a promotion gap that had
dropped three required numbers in a row.

**Headline: the no-trade criterion's redesign flips the adopted
configuration, from A to B.** Objective 3 rebuilt the criterion from a
symmetric distance-to-nominal score into a directional one —
overconfidence fails outright, any regime, any magnitude beyond
tolerance; underconfidence is a stated, bounded cost — decided on its own
methodological merits and committed to code and tests BEFORE it was
scored against config B, per Day 24's own closing instruction to sequence
it that way. Scored: config B (single model + velocity covariance floor)
now clears the bar cleanly on v4.1-gate and with one stated,
safe-direction cost (`sustained`, magnitude 0.0355) on v5-cessation; IMM
(C/D) still fails, exactly as predicted in advance, on the same
overconfident-static-regression grounds Day 21-24 already established.
ADR 0010 revised: **config B is now adopted**, reversing the Day 20-24
default.

Objective 1 separately confirmed, against a real run rather than a
synthetic comparison, that the corrected floor has no remaining defect:
config B's minimum observed σ_v equals the floor (1.5 m/s) to
floating-point precision in every regime on both golden sets. None of the
three hypothesized failure modes — a mis-derived floor, an unreached
clamp, or a filter that is never confident about velocity at all — is
what is actually happening; the contradiction the day opened with does
not survive contact with the real code path. Objective 2 recovered the
three numbers that had gone missing between report and summary on Days
20/23/24; Objective 4 makes that recurrence structural rather than
something a future day has to catch by re-reading old sections.

## Verdicts

- **Floor verdict — (a), (b), (c), or something else, with both
  numbers?** Neither (a), (b), nor (c) — the assumed contradiction does
  not hold. Floor = 1.5000 m/s (2.2500 (m/s)²); natural (unfloored)
  converged σ_v sits at 0.17-0.40x the floor in every regime on
  v5-cessation and v4.1-gate; the floor-enabled config's minimum reading
  equals the floor to floating-point precision everywhere. → Objective 1.
- **Cessation frame-support verdict (recovered)?** Day 21's NEES-815
  diagnosis was under-supported as originally reported (n=1 hand-traced
  track); the phenomenon it pointed at is independently established at
  n=373 (Day 23). → Objective 2.
- **Parallel-maintenance audit (recovered)?** Three registry/consumer
  pairs found and fixed Day 24 (`LANE_DESCRIPTIONS`,
  `CONFIG_DESCRIPTIONS`/`CONFIG_SPECS`, `_REQUIRED_PARAMS`); four more
  checked and explicitly not flagged, with reasons on record. →
  Objective 2.
- **Day 20 baseline margin, per regime AND per distance bucket
  (recovered)?** Never printed as a combined table by any day through Day
  24, despite the code supporting it since Day 21 — produced today
  (config A, both golden sets). → Objective 2.
- **Does the directional no-trade criterion change any verdict?** Yes —
  config B: NOT_SATISFIED → PASS_WITH_COST on v5-cessation, and a clean
  PASS on v4.1-gate. IMM (C/D): unchanged, still fails, now
  FAIL_OVERCONFIDENT specifically rather than a bare "regressed." →
  Objective 3.
- **Is the report-to-summary promotion gap now structural?** Yes — every
  day section from Day 20 onward requires a non-empty `## Verdicts`
  block, enforced by `tests/test_report_verdicts.py`. → Objective 4.

## Objective 0 — push, start of day

`foundation/day-25` branched from `foundation/day-24` (`809f8ff`), pushed
clean before any Day-25 commit landed — verified local matched
`origin/foundation/day-25` exactly. `main` unchanged since Day 16 (ADR
0009 still Proposed). (End of day: see the closing Objective 0/5 section
below.)

## Objective 1 — the two numbers that settle the floor: neither (a), (b), nor (c)

Full derivation, both real-run tables, and the full verdict reasoning are
in ADR 0010's "Day 25 revision (Objective 1)" section; summarized here.

New `scripts/velocity_floor_binding_audit.py` runs the REAL floor-enabled
filter (config B) on a golden set's actual tracks and reads
`estimate.cov_array()`'s velocity-diagonal entries back — not a
closed-form floor value compared against a synthetic walk's convergence,
which is all Day 22-24's "binds hard" claim ever rested on.
`eval_estimator.py`'s `FrameRecord` gained `velocity_variance_diag_mps2`,
now surfaced as a min/p50/max distribution in every `by_regime` block —
reusable machinery, not a one-off print.

**Floor: `PERSON_SIGMA_A_MPS2` [1.5 m/s²] × `PEDESTRIAN_STOP_DURATION_S`
[1.0 s] = 1.5000 m/s (2.2500 (m/s)²)**, unchanged from Day 24.

v5-cessation, config A (floor disabled) vs config B (floor enabled):

| regime | n | A p50 σ_v (m/s) | A p50/floor | B min σ_v (m/s) |
| --- | ---: | ---: | ---: | ---: |
| static | 260 | 0.2562 | 0.1708 | 1.5000 |
| onset | 91 | 0.5954 | 0.3969 | 1.5000 |
| sustained | 282 | 0.2728 | 0.1819 | 1.5000 |
| cessation | 373 | 0.2548 | 0.1699 | 1.5000 |

Cross-checked on v4.1-gate (static/onset/cessation; `sustained` is empty
by construction on that set) — same shape: config A's p50 sits at
0.17-0.38x the floor in every populated regime; config B is pinned at
exactly 1.5000 in every populated regime.

**Verdict: none of (a)/(b)/(c).** Not (b) — config B's minimum reading
equals the floor to floating-point precision in every regime; the one
place a reading exceeds the floor (`onset`'s max, 1.8014) is natural
uncertainty legitimately exceeding a floor that only ever raises a value,
never caps it, exactly where the transient regime's own higher natural
uncertainty would predict. Not (a) — the floor sits far ABOVE natural
convergence (0.17-0.40x), the opposite of (a)'s premise. Not (c) —
natural convergence's p50 sits at 0.25-0.60 m/s in every regime,
comfortably below the "never confident" threshold. Day 23's "never binds
anywhere" finding was entirely the `dt_s`-scaling bug Day 24 already
fixed; today closes the one remaining gap in how that fix had been
checked — against a real run's actual posterior covariance, not a
closed-form comparison. (`7e6b9cd`.)

## Objective 2 — the three unreported verdicts, recovered

Two of the three were already fully present in the report body, just
never assembled into the kind of one-line statement a summary would
carry forward — recovered here by pointing at them, not by re-measuring.
The third had never actually been produced as a table by any day through
Day 24, despite the underlying code supporting it since Day 21.

**Cessation frame support (Day 23/24 Objective 1).** Present in full in
Day 24's own Objective 1 section: Day 21's `brief_entry` trace (NEES
0.8-2.2 at onset → 165.6 at the stop itself → 815→16 decaying over the
recovery tail → <10 once settled, against a chi-square(6) 95% bound of
12.59) and the per-regime NEES coverage table it was drawn from (v3-indoor
static/onset/sustained n=38/284/2414; v4.1-gate static/onset/cessation
n=175/12/3, pre-Day-23 labeling). **Verdict, restated: under-supported as
originally reported (n=1 hand-traced track, not an aggregate), the
underlying phenomenon independently established at n=373 (Day 23).**

**Parallel-maintenance audit (Day 24 Objective 3).** Present in full in
Day 24's own Objective 3 section. Fixed: `LANE_DESCRIPTIONS`
(`src/data/registry.py`), `CONFIG_DESCRIPTIONS`/`CONFIG_SPECS`
(`scripts/eval_estimator.py`), `_REQUIRED_PARAMS`
(`src/model/uncertainty.py`) — each a dict hand-keyed by a `Literal` with
no completeness check, the same latent shape as the `GATES`/matrix defect
Objective 3 itself fixed that day. Checked and explicitly not flagged:
`tests/test_golden_sets.py`'s retained-version ledger (deliberately
append-only), `ACTION_TO_VERB` (deliberately partial), `Envelope.capability`
(a free string, not an enum), `DATASET_NAMES` (single consumer).

**Day 20 baseline margin, per regime AND per distance bucket.** Grep-
confirmed absent: no day's report between Day 20 and Day 24 ever printed
the per-distance-bucket margin table, though Day 21 Objective 1 fixed the
script to compute it. Produced today, config A, current codebase (the
same filter Day 20 built; regime labels reflect Day 23's fix):

v3-indoor:

| slice | n | filter RMSE | copy-prev margin | const-vel margin |
| --- | ---: | ---: | ---: | ---: |
| bucket 3-8m | 2528 | 0.1140 m | +0.0990 m | +5.9890 m |
| bucket 8m+ | 208 | 0.1677 m | +0.1644 m | +7.1841 m |
| regime static | 38 | 0.0846 m | +0.0997 m | +4.6118 m |
| regime onset | 284 | 0.1563 m | +0.0547 m | +0.7505 m |
| regime sustained | 2414 | 0.1142 m | +0.1121 m | +6.4599 m |

v4.1-gate:

| slice | n | filter RMSE | copy-prev margin | const-vel margin |
| --- | ---: | ---: | ---: | ---: |
| bucket 3-8m | 190 | 0.2328 m | +0.0263 m | +9.0406 m |
| regime static | 139 | 0.1023 m | +0.0778 m | +10.4733 m |
| regime onset | 12 | 0.1899 m | +0.5330 m | +0.5398 m |
| regime cessation | 39 | 0.4643 m | **−0.2396 m** | +4.0266 m |

Every margin is positive except one, reported because it looks bad, not
despite that: v4.1-gate's `cessation` regime under config A loses to
copy-previous by 0.2396 m — the filter's own confident-but-wrong velocity
carries it PAST where the person actually is once they have stopped,
while copy-previous, having no velocity model at all, simply has nothing
to be wrong about. This is a new number, not one either of the first two
recovered items already stated, and it is the same mechanism Day 21
diagnosed by a different measurement (NEES, not RMSE) — consistent, not
contradictory, with everything already on record about config A's
cessation behaviour. `0-3m` is empty on both sets (both synthetic sets
keep agents further from camera than that, Day 20's own finding,
unchanged).

## Objective 3 — the no-trade criterion made directional; config B adopted

Full rationale, the complete four-config re-evaluation on both golden
sets, and the adoption decision (with the project's own decision
framework applied explicitly) are in ADR 0010's "Day 25 revision
(Objective 3)" section; summarized here.

`_no_trade_verdict` (`scripts/eval_estimator.py`) now returns one of five
typed dataclasses — `NoTradeUnscoreable`, `NoTradeNoImprovement`,
`NoTradeFailOverconfident`, `NoTradePass`, `NoTradePassWithCost(regime,
magnitude, costs=...)` — never a bare boolean or a status string a caller
could collapse to "it passed." A steady regime moving toward
overconfidence (candidate coverage below the 0.95 nominal) by more than
`NO_TRADE_DEGRADATION_TOLERANCE` fails the candidate outright, any
regime, any magnitude beyond that tolerance, independent of how much
cessation improved. A move toward underconfidence is recorded as a
numeric, bounded cost instead. Direction is read off the CANDIDATE's own
coverage-error sign against 0.95, not the sign of the change.

Re-evaluated A/B/C/D on v5-cessation and v3-indoor, as asked (v4.1-gate
checked too, for ADR consistency):

| set | A→ | old verdict (symmetric) | new verdict (directional) | changed? |
| --- | --- | --- | --- | --- |
| v5-cessation | B | NOT_SATISFIED | **PASS_WITH_COST** (sustained, 0.0355) | **YES** |
| v5-cessation | C | NOT_SATISFIED | FAIL_OVERCONFIDENT (static, 0.8231) | no |
| v5-cessation | D | NOT_SATISFIED | FAIL_OVERCONFIDENT (static, 0.1885) | no |
| v4.1-gate | B | SATISFIED | **PASS** (no cost anywhere) | no (already satisfied; now clean) |
| v4.1-gate | C | NOT_SATISFIED | FAIL_OVERCONFIDENT (static, 0.3604) | no |
| v3-indoor | B/C/D | UNSCOREABLE | UNSCOREABLE | no (0 cessation frames, unaffected) |

**IMM's predicted outcome, checked as instructed: confirmed.** "IMM still
fails, because Day 23 found it regressing static harder than any set to
date and static regression means overconfidence in the regime that
should be easiest" — exactly what happened on both sets; `static`'s
coverage collapse (0.9962→0.0808 on v5-cessation, 0.9928→0.5468 on
v4.1-gate) both land far below 0.95, so `FAIL_OVERCONFIDENT` is the only
possible outcome. There was no "examine whether IMM's degradation is
genuinely toward underconfidence" step to run, because it plainly is not.

**Only config B's verdict changes, in the direction the redesign's own
stated rationale predicts** — its sole "regression" anywhere on either
set is `sustained` moving from slightly-underconfident (0.9574) to
more-underconfident (0.9929) on v5-cessation, the safe direction by the
criterion's own definition. **Decision: config B (single model + velocity
covariance floor) is now adopted**, reversing the Day 20-24 default — it
is the only candidate that clears the non-negotiable side of the
criterion on both golden sets while materially fixing cessation
(0.5013→0.9946 on v5-cessation, 0.1282→1.0000 on v4.1-gate). This was
decided after the criterion was redesigned and committed on its own
methodological merits, not the same session it was discovered to flip
B's verdict — the sequence Day 24 set out in advance. (`3cbc556`.)

## Objective 4 — the report-to-summary promotion gap, closed structurally

Every day section from Day 20 onward now opens with a `## Verdicts`
block: one line per question that day's own prompt asked for an explicit
answer to, each pointing at the objective containing the evidence.
Retrofitted for Days 20-24 from their existing report bodies (see each
day's own new block, inserted directly after that day's headline
paragraph); Day 20's still-missing distance-bucket table is marked NEVER
MEASURED there rather than silently carried forward as if it had been
produced.

**STRUCTURAL:** `tests/test_report_verdicts.py` parses
`FOUNDATION_REPORT.md`, finds every top-level section whose heading names
a day number ≥ 20, and fails if that section has no `## Verdicts` heading
or an empty one. A dedicated test constructs a fake day section with no
block at all and confirms the detection regex would catch it, before
trusting that the real sections passing means anything. Days 1-19 predate
the convention and are out of today's retrofit scope (their headings are
also inconsistently numbered — e.g. "Day 3 (resumed, post-amendment)" —
which the day-number regex is not asked to handle).

The session summary for any future day should now be generated by
reading that day's `## Verdicts` block, not composed independently from
memory of what felt important — which is exactly the step that dropped
Day 20's distance-bucket margin, Day 23's cessation-support number, and
Day 24's cessation verdict and parallel-maintenance audit, three times in
a row before this fix. (`7cb4ec1`.)

## Full suite and mypy

`mypy` (scoped per `mypy.ini`; `scripts/` and `tests/` remain outside its
scope, unchanged — today's estimator work extended `scripts/eval_estimator.py`,
already outside that boundary): **0 errors, 58 files** — unchanged from
Day 24. `black --check` / `flake8` clean on every file touched today
(`scripts/eval_estimator.py`, `scripts/velocity_floor_binding_audit.py`,
`tests/test_eval_estimator.py`, `tests/test_report_verdicts.py`,
`docs/adr/0010-estimator-configuration.md`).

Repo-wide `-m "not requires_weights and not slow"`: **1009 passed, 1
skipped, 21 deselected, 0 failures** — up from Day 24's 1001 (+8: three
sigma_v-distribution tests, two net new directional no-trade tests
replacing/extending the old symmetric ones, and three report-verdicts
lint tests).

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 24 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; now 2 days older.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 14 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 7 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-25` pushed clean, matches origin
exactly. `main` unchanged. `git push --tags`: up to date.

## Day 26, in order

1. **Multi-entity factor graph** — no longer blocked since Day 23, and no
   longer waiting on a configuration decision either (Day 25 settled it):
   the highest-leverage next build. When it lands, the call site that
   constructs the production single-entity filter must pass
   `velocity_covariance_floor=True` for `person`/`asset_carried` kinds
   per today's ADR 0010 decision — noted here so it is not rediscovered
   as a surprise default.
2. **Smoothing** (`horizon_kind="smoothed"`) — same unblock, still
   secondary per Day 21's reasoning.
3. **A velocity floor derived from the actual stopping deceleration
   profile** — Day 22's other open question, still untried (today
   confirmed the SAME physical basis's derivation and binding; a
   genuinely different basis remains open) — lower priority now that the
   adopted floor is confirmed working well against a real run.
4. **IMM's steady-regime mixing-overhead hypothesis** — unconfirmed, now
   5 days deferred; today is the fourth consecutive day IMM has been
   rejected on independent, increasingly specific grounds (most recently
   `FAIL_OVERCONFIDENT`), which lowers the urgency of chasing why without
   resolving the "why" itself.
5. **Mode-probability validation against real motion labels** — depends
   on item 1.
6. **MEVA licence verification** — still blocked on a human, highest
   product impact of any pending data item.
7. **DA-2K licence verification** — adapter built and tested, zero
   engineering lag once cleared.
8. **Reference hardware procurement decision** — now 7 days old.
9. **Declare a target fps for real camera ingest** — still open from
   Day 19.
10. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
11. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; gates on item 7 for a real depth number.
12. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
13. **The generator has no sensor-noise model** — unchanged.
14. **Order cameras and run the office capture** — now 14 days old.
15. **The motion-gate precision/selectivity investigation** — still
    deferred.
16. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
17. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged.
18. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
19. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a DPDP
    process decision, not a coding task.
20. **Hypothesis management** — furthest out; depends on item 1.

Dropped from this list today: nothing — checked directly against the
filesystem/codebase before carrying every item forward (per the process
note Day 24 established after the `raise_alert()` incident); every item
above is either still genuinely open or explicitly re-scoped in place
(items 1, 3, 4) rather than silently renumbered.

# Day 26

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-25.
Component COUNTS and SIZES are structural properties of the scene GT, not
performance figures, and are explicitly in scope today. Everything below
is accuracy, consistency, and structural measurement: the A→B flip's
provenance settled with numbers, a coupling-density measurement taken
before any solver code was written, the multi-entity factor graph itself,
and an honest joint-vs-independent evaluation.

**Headline: the multi-entity factor graph's motivating case is real and
measured — and it has a cost that was not assumed away.** Joint
estimation beats independent per-entity filtering on a carried asset by a
real, consistent margin on both golden sets (+0.0425m RMSE on
v5-cessation, +0.0265m on v3-indoor), with no overconfidence degradation
on the asset itself. Applying Day 25's own directional no-trade criterion
to this new comparison — exactly as today's objective anticipated it
might — found that the CARRIER's own calibration is not neutral: it
degrades toward overconfidence on v5-cessation specifically
(FAIL_OVERCONFIDENT), concentrated in dynamic regimes and worst in
cessation, the exact regime Days 21-25 already found most fragile. No
same-session fix was attempted. Two other results anchor the day: the
A→B flip (Day 25) is confirmed to be a real estimator change, not a
criterion-only relabeling (RMSE deltas up to 62% per regime); and the
component-sparsity claim the multi-entity design has rested on since Day
13 — never actually written down anywhere in this repository — was
measured for the first time and passed its own gate narrowly, not
comfortably.

## Verdicts

- **A→B flip: does the estimator's OUTPUT differ, or only the criterion
  that scores it (i vs ii)?** (i) — config B now produces materially
  different position estimates from config A: RMSE deltas of +39.2%
  (static, worse) to −62.4% (cessation, better) per regime, on both
  golden sets. The Day-22 "B≡A" finding was correct for the per-timestep
  floor formula that existed then; it stopped describing B the moment
  Day 24's fix made the clamp actually fire. → Objective 1.
- **Component-size distribution: what does it show, and what's the
  threshold-sensitivity knee?** The claimed sparsity source document
  ("Data model v0.3 §3") does not exist anywhere in this repo —
  grep-verified before measuring. Of the three named relationship types,
  only proximity is measurable on this project's data; carried-object and
  shared-zone coupling have zero GT support anywhere. On v3-indoor (the
  only set with multi-agent clips), the knee is sharp: the one 6-agent
  scene fully merges into one component at just 1.10m proximity
  threshold. At the declared default (1.5m), pooled p95=3 but max=6 — a
  narrow pass against the gate's own "≤6" bound, not a comfortable one.
  → Objective 2.
- **Did Objective 3 proceed, and why?** Yes — Objective 2's gate passed
  (narrowly). Built `src/estimator/joint.py`: components as the unit of
  inference, rigid-coupling-plus-slip state layout, size-1 components
  delegate verbatim to Day 20/25's single-entity filter (bit-identical by
  construction), NEES dof correctness tested at component sizes 1/2/3
  (6/9/12). → Objective 3.
- **Does joint estimation beat independent filtering on the carried
  asset (the headline claim)?** Yes, clearly, on both golden sets:
  +0.0425m RMSE margin on v5-cessation, +0.0265m on v3-indoor, no
  overconfidence degradation on the asset itself. → Objective 4.
- **Is joint estimation neutral for the carrier, as hoped?** No — PASSES
  on v3-indoor (calibration actually improves) but FAILS the directional
  no-trade criterion on v5-cessation (coverage 0.9089→0.8874,
  FAIL_OVERCONFIDENT), concentrated in dynamic regimes and worst in
  cessation (Δ −0.0375, nearly double the pooled effect). A mechanism is
  hypothesized, not confirmed; no same-session fix was attempted. →
  Objective 4.

## Objective 0 — push, start of day

`foundation/day-26` branched from `foundation/day-25` (`27b7b23`), pushed
clean before any Day-26 commit landed — verified local matched
`origin/foundation/day-26` exactly. `main` unchanged since Day 16 (ADR
0009 still Proposed). (End of day: see the closing Objective 0/5 section
below.)

## Objective 1 — ADR 0010 provenance for the A→B flip: (i), with numbers

Full derivation and both per-regime delta tables are in ADR 0010's "Day
26 revision (Objective 1)" section; summarized here.

Day 22 measured config B as bit-for-bit identical to config A. Day 25
found the corrected floor binds on essentially every scored frame — a
constraint binding constantly changes the Kalman gain, which changes the
POSTERIOR MEAN, not just its reported uncertainty. Re-measured directly,
position RMSE per regime, config A vs B, current codebase:

**v5-cessation:** static +39.2% (worse), onset −0.2%, sustained +28.4%
(worse), **cessation −20.4%** (better), maneuver −5.1%.
**v4.1-gate:** static +40.5% (worse), onset −3.5%, **cessation −62.4%**
(better).

**(i) is true.** The effect is bidirectional — worse on steady regimes,
dramatically better on cessation — which is itself evidence this is a
real estimator effect and not an artifact: a pure criterion change, with
identical underlying estimates, could not produce a bidirectional
accuracy effect. Three facts, each independently checkable, keep this
from reading as motivated: the asymmetry argument was stated before
re-scoring (Day 24's own closing instruction, executed in that order);
IMM still fails under the new criterion (`FAIL_OVERCONFIDENT` on both
sets — the load-bearing evidence the criterion did not become permissive
in general); and B's cost is a named number (`sustained`, magnitude
0.0355), not waved away. Also added: the closed-form-vs-instrumented rule
to `iron-eval-discipline`'s `SKILL.md`, with the velocity floor as the
worked example — applied pre-emptively to Objective 2 below, before any
solver code was built on an unmeasured claim. (`aa9db8c`.)

## Objective 2 — component-size distribution: measured, narrow pass

Full tables, the per-clip knee, and the honest limitation are in ADR
0011; summarized here.

**The claimed source document does not exist.** "Data model v0.3 §3,"
recalled as claiming coupling is sparse ("components stay small,
typically 1-6 entities"), was grep-verified absent from `docs/`,
`src/model/`, and `FOUNDATION_REPORT.md` before any measurement code was
written — unwritten, not merely unmeasured, treated as the implicit
assumption underlying `episode.py`'s and `motion_model.py`'s deferred
multi-entity work rather than a cited decision.

Of the three named relationship types (proximity, carried-object,
shared-zone), only proximity has any GT support in this project's data;
carried-object and shared-zone coupling have zero GT anywhere. Only
v3-indoor has multi-agent clips (1/2/3/6 agents, 30 clips); v4.1-gate and
v5-cessation are component-size 1 everywhere by construction.

New `scripts/measure_component_sparsity.py` builds the proximity coupling
graph from GT and sweeps the threshold. **The knee: 1.05m→1.10m** — the
one 6-agent scene (`crowded_6agents`) fully merges from several small
pairs into one 6-entity block at just **1.10m**. At the declared default
(1.5m, proxemics' "close social" boundary): pooled p95=3, max=6 — at the
decision gate's own stated bound, not comfortably below it. Merges that
DO form persist most of a clip's duration (p50=21 of 40 frames) — not
transient blips.

**Honest limitation, stated plainly:** these are 30 authored synthetic
clips, only two of which reach 6 agents. The crowded-lobby case (dozens
of people) that would actually break sparsity cannot be measured on data
this project owns. What this DOES show: even this project's own small
"crowded" scene fully merges at a threshold well inside ordinary
personal/social space — sparsity is not structurally guaranteed by
"people happen to be far apart" indoors; it depends on keeping the
coupling threshold tight (≤1.0m, where sparsity held cleanly) or on
scenes staying genuinely uncrowded. What would settle it: a larger
authored scene (20-50+ agents) or real capture data, neither of which
exists in this project's golden sets today.

**Decision gate: PASS, narrowly — proceeded to Objective 3.** (`40dc718`.)

## Objective 3 — the multi-entity factor graph: built, conditional on the gate above

Full design record is in ADR 0011; summarized here.

`src/estimator/joint.py`. Components as the unit of inference — a
`Component` is one carrier plus zero or more carried entities, declared
statically (Objective 2's proximity graph was a pre-build sanity
measurement, not the grouping key the filter itself uses; the actual
coupling factor implemented is `carrier_entity_id` specifically).
Rigid-coupling-plus-slip state layout (6 carrier dims + 3 offset dims per
carried entity) — exactly the design `motion_model.py`'s `asset_carried`
docstring named since Day 20 and left unimplemented pending this day.

**Size-1 components delegate to Day 20/25's own filter, verbatim** — not
a reimplementation that happens to agree, the actual function call — so
they reproduce single-entity behaviour (including config B's floor)
bit-for-bit by construction, tested directly.

**The motivating case, tested directly:** bootstrap a carrier+carried
component, feed ONLY carrier observations for several further steps (no
carried-entity observation at all), and confirm the carried entity's
implied absolute position moves with the carrier. This falls out of the
predict step automatically once the state layout is correct — no
special-cased propagation logic was needed.

**NEES dof, tested at sizes 1/2/3:** `consistency.compute_nees` already
infers dof from array size — no joint-specific metric function was
needed, only correctly-sized joint state. Confirmed: size-1 reports
dof=6 (identical to Day 25), size-2 reports dof=9, size-3 reports dof=12.

STRUCTURAL, re-tested against the single-entity precedent: prior firewall
(`inspect.signature`, no prior-shaped parameter), consistency residuals
required (`JointStateEstimate` independently re-checks the rule, a
genuinely separate type since `StateEstimate.mean` is hard-validated to
exactly 6), graph_rev reproducibility (re-solving a component at an
earlier revision after more, unrelated factors are appended reproduces
bit-identically).

**Not implemented today, as skeletons naming what fills them, not silent
gaps:** hypothesis management (`resolve_data_association`) and the
discrete/continuous hybrid (`HybridDiscreteContinuousState`, dynamic
component membership) — bootstrap requires every declared member observed
at the component's first timestep, a real scope limit stated in the
docstring, not discovered later. (`4d97b79`.)

## Objective 4 — joint vs independent: the motivating case passes; the carrier's calibration does not, on one set

Full tables and the hypothesized mechanism are in ADR 0011; summarized
here.

No golden set carries real carried-object GT (Objective 2's finding), so
`scripts/eval_joint_estimator.py` synthesizes one: the carrier's own GT
plus a fixed, rigid, declared offset. Baseline (Day 12 rule): INDEPENDENT
per-entity filtering — the `asset_carried` motion model that has existed,
unused, since Day 20.

**Carried asset — the motivating case, PASSES cleanly on both sets:**

| set | asset RMSE: indep → joint | margin | asset coverage: indep → joint | verdict |
| --- | --- | ---: | --- | --- |
| v5-cessation (n=1021) | 0.2051m → 0.1626m | **+0.0425m** | 0.8570 → 0.8737 | PASS |
| v3-indoor (n=2736) | 0.1691m → 0.1426m | **+0.0265m** | 0.9635 → 0.9269 | PASS |

**Carrier — set-dependent, and fails on v5-cessation:**

| set | carrier RMSE: indep → joint | margin | carrier coverage: indep → joint | verdict |
| --- | --- | ---: | --- | --- |
| v5-cessation (n=1021) | 0.1888m → 0.1608m | **+0.0280m** | 0.9089 → 0.8874 | **FAIL_OVERCONFIDENT** |
| v3-indoor (n=2736) | 0.1552m → 0.1408m | **+0.0144m** | 0.9912 → 0.9675 | PASS (improves) |

On BOTH sets the carrier's raw accuracy improves under joint estimation
(more measurements per step lowers RMSE, as expected). The calibration
effect is where the sets diverge — and per-regime breakdown on
v5-cessation shows it is not uniform: `static` coverage IMPROVES
(+0.0067 toward nominal) under coupling; every regime with real carrier
motion degrades, and `cessation` degrades hardest (Δ −0.0375, nearly
double the pooled −0.0215). This is the exact regime Days 21-25 already
found the estimator's calibration most fragile, and it is exactly the
danger this objective's own framing named in advance: "a joint solve
that sharpens covariance without justification is the exact danger the
criterion now names."

**A mechanism is hypothesized, not confirmed:** during a regime where the
carrier's true velocity is changing, the coupled update may let the
asset's own independently-noisy position observation contribute extra
apparent confidence to the carrier's state through the shared Kalman
gain, exactly when the carrier's motion is least predictable. **No
same-session fix was attempted** — Days 21-25 have repeatedly shown that
tuning a parameter in the same session that found a miscalibration reads
as motivated regardless of whether the reasoning is sound. Recorded as
Day 27's first item: confirm the mechanism with a second measurement
before changing anything.

A real covariance bug was caught and fixed before this shipped: a carried
entity's absolute-position covariance is NOT
`Cov(carrier_pos) + Cov(offset)` — that silently drops the
cross-covariance term the Joseph-form update actually builds. Fixed via
`JointStateEstimate.carried_position_cov_m2`, computed with the same H
projection matrix used to score a carried-entity observation, so mean and
covariance cannot silently drift apart under two independently-derived
formulas.

Day-10 validity gate: **PASS** `state_estimation` on both sets, re-run,
unchanged. Falsification test 5: re-run, still **PASSES/PARTIAL** exactly
as Day 20 left it (8/8 green) — joint estimation produces a richer
estimate but still nothing resolves to an `ActivityMode` behaviour label.

**Status: capability validated, not yet a production recommendation.**
The carried-asset improvement is real and holds on both sets; the
carrier's own calibration cost on v5-cessation means this is a stated
trade, not a clean win, until the mechanism above is confirmed. ADR 0011
stays Proposed, not Accepted. (`cd0f763`.)

## What remains skeleton, and the order it should land in

1. **Confirm the carrier-overconfidence mechanism** (Objective 4's own
   finding) — a second measurement (e.g. sweep
   `OFFSET_SLIP_SIGMA_MPS_SQRT_S` and check whether cessation-regime
   overconfidence tracks it monotonically) before any parameter changes.
2. **Hypothesis management** (`resolve_data_association`) — discrete
   data-association uncertainty; `Component` membership is declared, not
   inferred, today.
3. **The discrete/continuous hybrid** (`HybridDiscreteContinuousState`) —
   dynamic component membership (a "picked up"/"set down" event), needed
   before a carried entity can be discovered mid-track rather than
   declared upfront.
4. **Smoothing across the joint graph** (`horizon_kind="smoothed"`) —
   same secondary-to-calibration reasoning Day 21 gave for the
   single-entity case, now re-stated for joint: smoothing a coupling
   whose calibration cost is not yet understood tightens a number that is
   not yet honest.
5. **A larger authored crowded scene, or real capture data** — the only
   way to settle Objective 2's own open question (does sparsity hold past
   6 agents at a realistic coupling threshold).

## Full suite and mypy

`mypy` (scoped per `mypy.ini`; `src/estimator` picked up `joint.py`
automatically as a package member): **0 errors, 59 files** — up from Day
25's 58. `black --check` / `flake8` clean on every file touched today
(`.claude/skills/iron-eval-discipline/SKILL.md`,
`docs/adr/0010-estimator-configuration.md`,
`docs/adr/0011-multi-entity-factor-graph.md`,
`scripts/measure_component_sparsity.py`,
`scripts/eval_joint_estimator.py`, `src/estimator/joint.py`,
`tests/test_measure_component_sparsity.py`,
`tests/test_estimator_joint.py`, `tests/test_eval_joint_estimator.py`).

Repo-wide `-m "not requires_weights and not slow"`: **1048 passed, 1
skipped, 21 deselected, 0 failures** — up from Day 25's 1009 (+39: 9
component-sparsity tests, 19 joint-filter structural tests, 11
joint-evaluation tests).

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 25 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; now 3 days older.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 15 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 8 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-26` pushed clean, matches origin
exactly. `main` unchanged. `git push --tags`: up to date.

## Day 27, in order

1. **Confirm the carrier-overconfidence mechanism** (Objective 4) — sweep
   `OFFSET_SLIP_SIGMA_MPS_SQRT_S`, check whether cessation-regime
   overconfidence tracks it monotonically, before any parameter change.
   Decide, on the result, whether joint estimation for a `person`-kind
   carrier needs a directional fix (e.g. a floor on the offset's own
   contribution to the carrier's covariance shrinkage) or whether the
   cost is inherent and should simply be documented as a tradeoff.
2. **Hypothesis management** (`resolve_data_association`) — furthest-out
   multi-entity item, now with a real interface to build against.
3. **The discrete/continuous hybrid** (`HybridDiscreteContinuousState`) —
   dynamic component membership.
4. **Smoothing across the joint graph** — depends on item 1 for the same
   reason single-entity smoothing was deferred (Day 21): tightening a
   number whose calibration is not yet trustworthy is not progress.
5. **A larger authored crowded scene** (20-50+ agents), if pursued — the
   only way to extend Objective 2's sparsity measurement past 6 agents
   without real capture data.
6. **MEVA licence verification** — still blocked on a human, highest
   product impact of any pending data item.
7. **DA-2K licence verification** — adapter built and tested, zero
   engineering lag once cleared.
8. **Reference hardware procurement decision** — now 8 days old.
9. **Declare a target fps for real camera ingest** — still open from
   Day 19.
10. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
11. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; gates on item 7 for a real depth number.
12. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
13. **The generator has no sensor-noise model** — unchanged.
14. **Order cameras and run the office capture** — now 15 days old.
15. **The motion-gate precision/selectivity investigation** — still
    deferred.
16. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
17. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced** — unchanged.
18. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
19. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a DPDP
    process decision, not a coding task.
20. **The `resolve_joint_state` cross-component filtering gap** (ADR
    0011's closing note) — cheap to fix (filter `_joint_payload_factors`
    by component identity) but not yet needed by any real call site;
    revisit once a multi-component orchestrator exists.
21. **Hypothesis management (single-entity data association)** — the
    original, pre-Day-26 punch-list item, now partially subsumed by item
    2 above but recorded separately since single-entity identity
    resolution and multi-entity component membership are related, not
    identical, problems.

# Day 27

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-26.
Component counts and sizes remain explicitly in scope as structural
properties of the scene. Everything below is accuracy, consistency, and
structural/documentation measurement.

**Headline: the architecture this project has been reasoning from for
27 days was running on conversational memory, not on anything checked
into version control — and it was the whole numbering scheme, not one
citation.** Day 26 found a single sparsity claim cited as "Data model
v0.3 §3" that did not exist anywhere in the repo. Today's audit
(Objective 3) checked how far that pattern extended: `src/model/__init__.py`
has claimed to implement "IRON_DATA_MODEL v0.3... built Day 13 from
three rounds of architecture review" since Day 13, and no file with
that content, at that version, with numbered sections, existed anywhere
until today. §0, §10, §15, and §17 — every section number this codebase
has ever cited as part of the data model — were each real, implemented,
tested code, cited dozens of times combined, and never once collected
into an addressable document. Two provisions audited today have no
implementation AND no prior record at all: constraint typing (hard vs.
twin-dependent) traces to a single stub label explicitly marked "not yet
implemented," and the hypothesis store's `PRUNED_BY_BUDGET` death cause
— named with enough precision that some real specification of it
plausibly existed once — appears zero times in 27 days of commits,
ADRs, or reports. `docs/data_model/v0.3.md` is now real, and a
structural lint (`tests/test_data_model_citations.py`) makes a future
citation to an unwritten section fail the suite rather than sit
unnoticed for another 27 days.

Two other results anchor the day. Objective 1 confirmed the Day 26
cross-covariance bug did NOT explain the carrier's measured
overconfidence (it never could have — the carrier's own covariance is a
direct sub-block, no summation involved), then instrumented the actual
mechanism directly: the constant slip model shrinks the carrier's
covariance by a roughly uniform 32-41% regardless of regime while the
real benefit ranges from +47.6% to −7.7%, and a physically-derived
acceleration-scaled alternative fixes the worst of it (cessation's
unjustified gain: +19.3% → +0.9%) at the direct cost of the asset's own
benefit (margin +0.0425m → −0.0051m) — reported as a real trade, not
adopted as a fix. Objective 2 gave the solver a hard, justified cap
(6, the exact bound Day 26's own gate used) with a structurally
un-skippable degradation path, and found — honestly, including the
caveat about what the specific test actually measures — that capping a
badly-mismatched component helps, not hurts, which is a different and
arguably more important safety property than the one originally asked
about.

## Verdicts

- **Does the carrier-overconfidence finding survive the cross-covariance
  confounder?** Yes, cleanly — the fix and the finding landed in the
  same commit, and the carrier's own covariance was never derived via
  the buggy formula (it's a direct sub-block, no summation needed). →
  Objective 1.
- **What is the mechanism, instrumented directly?** The constant slip
  model shrinks the carrier's covariance ~32-41% uniformly regardless of
  regime; actual error reduction ranges from +47.6% (static) to −7.7%
  (onset, genuinely worse). The gap is worst in cessation (+19.3%
  unjustified gain) and onset (+39.7%). → Objective 1.
- **Does the physically-derived acceleration-scaled slip model fix it
  without costing the asset improvement?** No — it fixes the carrier
  (cessation unjustified gain +19.3%→+0.9%, overall verdict
  FAIL_OVERCONFIDENT→PASS) but destroys the asset's own benefit on both
  golden sets (margin +0.0425m→−0.0051m on v5-cessation,
  +0.0265m→−0.0049m on v3-indoor). A trade, not a fix; neither model
  adopted as default. → Objective 1.
- **What cap value, and why?** 6 — not a round number, the exact bound
  Day 26 Objective 2's own decision gate used to authorize building the
  solver at all. Degradation action: independent fallback (this project
  has only ever measured coupling density, never coupling
  informativeness, so a ranking heuristic would be another unmeasured
  claim). → Objective 2.
- **What does the cap cost?** Measured as negative (capping HELPS) on
  the one real 6-entity test available — but that test forces six
  independent walkers into one rigid-coupling component, an honestly-
  labeled mechanism check, not a genuinely-coupled group. This measures
  the cap's safety value against inappropriate coupling, not the cost
  of degrading a well-matched component (that number, from the actual
  carrier+asset evaluation, is ~+0.03-0.04m). → Objective 2.
- **Does "IRON_DATA_MODEL v0.3" exist as a real document?** No, until
  today — confirmed absent despite being cited as built and versioned
  since Day 13. §0/§10/§15/§17 are all real, implemented, tested code
  that had never been collected into one addressable document.
  `docs/data_model/v0.3.md` now exists, reconstructed from
  implementation, with a structural lint against future dangling
  citations. → Objective 3.
- **Are any provisions assumed-nowhere-recorded — no implementation,
  no documentation, anywhere?** Yes, two: constraint typing (hard vs.
  twin-dependent constraints raising "twin-revision hypotheses") and the
  hypothesis store's `PRUNED_BY_BUDGET` death cause. Both are named with
  enough specificity that a real prior specification plausibly existed —
  just never in this repository. Documented as explicitly unspecified,
  not retroactively invented. → Objective 3.

## Objective 0 — push, start of day

`foundation/day-27` branched from `foundation/day-26` (`d00510e`), pushed
clean before any Day-27 commit landed — verified local matched
`origin/foundation/day-27` exactly. `main` unchanged since Day 16 (ADR
0009 still Proposed). (End of day: see the closing Objective 0/5 section
below.)

## Objective 1 — the carrier-overconfidence mechanism: confirmed, and the derived fix trades the motivating case away

Full derivation, both slip-model tables, and the complete reasoning are
in ADR 0011's "Day 27, Objective 1" section; summarized here.

**The cross-covariance confounder, ruled out first, per instruction.**
Day 26's covariance-cross-term fix and the eval script that produced the
carrier finding landed in the SAME commit (`cd0f763`) — there was never
a "before the fix" version of that measurement. More directly: the
carrier's own marginal covariance (`joint_estimate.cov_array()[:6,
:6]`) is a direct sub-block of the joint covariance, needing no
summation at all — the bug that was fixed only affected the ASSET's
derived absolute-position covariance, a genuinely different, additional
computation. The finding stands on its own math.

**The mechanism, instrumented directly (Day 25's rule: a closed-form
prediction about a running system is a hypothesis, not a measurement).**
`scripts/eval_joint_estimator.py` now reports, per regime, the carrier's
covariance shrinkage from coupling against the ACTUAL error reduction it
delivers:

| regime (v5-cessation) | covariance shrinkage | actual error reduction | unjustified gain |
| --- | ---: | ---: | ---: |
| static | 40.2% | 47.6% | −7.3% (conservative) |
| onset | 32.0% | **−7.7%** (worse!) | **+39.7%** |
| sustained | 38.6% | 26.0% | +12.7% |
| cessation | 40.8% | 21.5% | **+19.3%** |

The constant slip model shrinks the carrier's covariance by a roughly
uniform 32-41% regardless of what the carrier is actually doing; the
real benefit varies from strongly-justified (static) to negative
(onset). The gap is worst exactly where Day 26's no-trade criterion
flagged it.

**Derived fix: `OffsetSlipModel="acceleration_scaled"`** — slip sigma
scales with the carrier's own estimated acceleration relative to
`PERSON_SIGMA_A_MPS2`, both already-declared constants, no new fitted
parameter, acceleration estimated causally from the two most recent
carrier velocity states already in the factor chain.

| regime (v5-cessation) | cov shrinkage | error reduction | unjustified gain |
| --- | ---: | ---: | ---: |
| static | 13.4% | 18.2% | −4.8% |
| onset | 14.2% | −8.1% | +22.3% (still bad) |
| sustained | 11.1% | 11.5% | −0.4% |
| cessation | 12.4% | 11.4% | **+0.9%** (was +19.3%) |

Overall carrier verdict on v5-cessation: **FAIL_OVERCONFIDENT → PASS**
(coverage 0.9089→0.9109). Cessation's unjustified gain drops over 20x —
the fix works exactly where targeted, and confirms the mechanism.

**But the asset's own benefit — the motivating case — is destroyed on
both golden sets:**

| set | asset margin, constant | asset margin, acceleration-scaled |
| --- | ---: | ---: |
| v5-cessation | **+0.0425m** | **−0.0051m** |
| v3-indoor | +0.0265m | **−0.0049m** |

Mechanism: acceleration-scaled slip noise collapses toward the
regularization floor whenever estimated carrier acceleration is near
zero — most of a walking track — which also collapses the Kalman gain
that let the filter keep averaging in new asset observations to refine
the offset. The constant model's uniform slip noise was, inadvertently,
doing double duty (physical slip AND estimate elasticity); the
acceleration-scaled model removes both together.

**Verdict: the derived model confirms the physics and reveals a real
trade, not an adoptable fix.** Neither slip model is the default;
`offset_slip_model` stays an explicit opt-in (`"constant"` unchanged).
No same-session tuning attempted — Day 28's first item is a two-term
slip model (acceleration-scaled component ADDED to a small constant
floor, not replacing it), to be derived and tested with the same
discipline. (`bb31adc`.)

## Objective 2 — a hard component-size cap with a specified degradation path

Full type definitions, the STRUCTURAL test list, and the cap-cost
caveat are in ADR 0011's "Day 27, Objective 2" section; summarized here.

**Cap: 6, `degradation_action="independent_fallback"`.** `ComponentCapConfig`
is config-driven and versioned (`.sha`, same convention as `ImmConfig`).
6 is not a round number — it is the exact bound Day 26 Objective 2's own
decision gate used to authorize building the solver at all ("proceed
only if p95 component size ≤ 6"); capping the solver's own operation at
that same measured bound means it never runs where nothing has
validated it. Independent fallback, not split-weakest-coupling: this
project has only ever measured coupling DENSITY (Day 26), never
coupling INFORMATIVENESS — a ranking heuristic today would be another
unmeasured analytical claim.

**STRUCTURAL, tested explicitly.** `DegradedComponentEstimate` is a
distinct type (not a flag on `JointStateEstimate`) with two required
fields, no default. `resolve_joint_state`'s return type becomes
`StateEstimate | JointStateEstimate | DegradedComponentEstimate` — a
caller must `isinstance`-branch, so a capped result can never be
silently treated as genuinely coupled. Tested: the overflow path
degrades and records provenance; entity coverage is enforced at
construction (no entity silently dropped); a desynchronized observation
stream raises rather than partially degrading; `graph_rev`
reproducibility is re-tested for the degraded path specifically.

**Cost measured, with an important caveat stated plainly, not
buried.** No general person-to-person proximity coupling is
implemented (only carrier+carried), so there is no genuinely-6-PERSON
joint solve in this codebase to cap. `scripts/measure_component_cap_cost.py`
uses `v3-indoor`'s real `crowded_6agents` scene (the one Day 26 found
merges at 1.10m), with one HONESTLY LABELED "carrier" and the other five
"carried," purely to exercise a real 6-entity component using the
topology that exists:

| entity | uncapped RMSE | capped RMSE | cost (capped − uncapped) |
| --- | ---: | ---: | ---: |
| agent-0 (carrier) | 0.3822m | 0.1335m | **−0.2487m** |
| agent-1..5 (mean) | 0.2681m | 0.1449m | **−0.1232m** |

**The measured cost is negative on every entity — capping HELPS,
substantially.** This is real and correctly measured, and it is NOT
evidence joint estimation is generally worse: these six people are
genuinely independent walkers, so forcing them into one rigid-coupling
component is a badly-mismatched physical model, and falling back to
independent filtering is strictly better. This measures the cap's
safety value in the OPPOSITE failure mode from the one asked about:
protection against inappropriately coupling unrelated entities, not the
accuracy given up when a well-matched component gets capped. The
carrier+actual-asset evaluation (Day 26/27) already answers that
question directly: joint beat independent by +0.0425m/+0.0265m RMSE —
the plausible cost of capping a genuinely well-coupled 6-entity
component, if one existed, would be of that order, not the ~0.15-0.25m
"improvement" this specific test shows. What remains open: whether a
real, genuinely-coupled multi-entity group would show a positive cost
when capped — unanswerable without real coupled data or a larger
authored scene, both already on the punch list. (`e17573a`.)

## Objective 3 — the data-model-of-record audit

Full provision-by-provision evidence, the complete citation census, and
the reconstructed document itself are in `docs/data_model/v0.3.md`;
summarized here.

**The claim that started this: does "IRON_DATA_MODEL v0.3" exist as a
checked-in document? No — confirmed absent, despite `src/model/__init__.py`
claiming it built and versioned since Day 13.** Every `§N` citation this
codebase has made — grep-verified across `src/`, `tests/`, `docs/adr/`,
excluding RFC citations (`src/ingest/`'s own, legitimate external
references) and `docs/site_zero_consent_TEMPLATE.md`'s own,
separately-governed section numbers — resolves to exactly four real
section numbers: §0, §10, §15, §17.

| Provision | Citations | Status |
| --- | ---: | --- |
| §0 — Twin-revision consistency | multiple, `src/model/world.py` | IMPLEMENTED, never previously collected into a document |
| §10 — Confidence calibration | multiple, `src/model/evidence.py`, `src/estimator/` | IMPLEMENTED, informally documented in Day-13's report table only |
| §15 — Estimator contract (4 stages) | 20+ across `src/estimator/` | IMPLEMENTED; only "stage 4" (consistency) was ever named anywhere — stages 1-3 reconstructed today from actual code structure for the first time |
| §17 — Prior firewall | 13+ across `src/estimator/`, tests | IMPLEMENTED and tested (`inspect.signature` checks re-run on every new code path since Day 20) |
| Constraint typing (hard vs. twin-dependent) | audit target only | **ASSUMED, NOWHERE RECORDED** — traces to one stub label, `"twin-revision hypothesis (not yet implemented)"` |
| Hypothesis store / `PRUNED_BY_BUDGET` | audit target only | **ASSUMED, NOWHERE RECORDED** — zero hits anywhere in 27 days of commits, ADRs, or reports |
| Merkle reproducibility/erasure commitment | ADR 0007 | IMPLEMENTED, matches its ADR (`compute_merkle_root`, `EvidenceCommitment`) |
| Coverage / Absence | ADR 0004 | IMPLEMENTED, matches its ADR exactly |
| Bitemporal relationships | ADR 0005 | IMPLEMENTED, matches its ADR |
| The four event classes | ADR 0003 | IMPLEMENTED, matches its ADR exactly |
| ActivityMode | — | IMPLEMENTED, documented in Day-13's report table, no dedicated ADR |
| Episode with roled participants | ADR 0006 | IMPLEMENTED, matches its ADR |
| Multi-entity coupling / component sparsity | ADR 0011 | Was ASSUMED, NOWHERE RECORDED until Day 26 found the gap; now governed |

**The two ASSUMED-NOWHERE-RECORDED provisions are the sharpest finding
of the audit, sharper than the missing document itself.** The missing
document (§0/§10/§15/§17) was real, tested, working architecture that
simply never got written down — a promotion gap, structurally identical
in shape to Day 25's Verdicts finding. Constraint typing and
`PRUNED_BY_BUDGET` are different in kind: they are cited with enough
specificity (a literal named enum value) that some real specification
plausibly existed once, in a conversation, and NOTHING in this
repository — not a stub, not a punch-list item with that precision, not
an ADR — has ever recorded it. This document does not manufacture that
missing content; inventing plausible rules for them here would repeat
the exact failure this whole objective exists to close, one level down.

`docs/data_model/v0.3.md` is the reconstructed document — real sections
only, each traced to its actual implementation, with the two
unspecified provisions marked as exactly that rather than silently
dropped. **STRUCTURAL:** `tests/test_data_model_citations.py` scans
`src/`, `tests/`, `docs/adr/` for any `§N` citation not defined in the
new document and fails the suite — the same shape as the Day-25
Verdicts lint, so a new dangling citation is caught at the moment it is
added, not discovered by someone grepping for it later. Tested against
itself: a dangling citation is confirmed to fire the lint before
trusting that the real tree passing means anything; the exclusion logic
for RFC/consent-template citations is confirmed narrow (a genuine §15
citation on the same fixture survives, the external ones do not).
(`3890bf2`, `7120aa1`.)

## What remains skeleton, and the order it should land in

1. **A two-term slip model** (Objective 1) — acceleration-scaled
   component ADDED to a small constant floor, not replacing it, so the
   offset never becomes too rigid to keep averaging in new observations
   while still suppressing unjustified confidence during transients.
2. **Real coupled multi-entity data, or a larger authored scene**
   (Objective 2) — the only way to measure the cap's cost on a
   genuinely well-matched component, as opposed to today's
   honestly-caveated mismatched-component measurement.
3. **A decision on constraint typing and the hypothesis store**
   (Objective 3) — now that both are documented as explicitly
   unspecified rather than silently assumed, a real decision (build,
   defer with a stated reason, or drop) can be made instead of the
   provisions continuing to be cited as if settled.

## Full suite and mypy

`mypy` (scoped per `mypy.ini`): **0 errors, 59 files** — unchanged from
Day 26 (today's `src/` change was `src/model/__init__.py`'s docstring
only; `src/model` is not in `mypy.ini`'s strict-scope list, so this is
consistent, not an omission — `src/estimator` remains in scope and
clean). `black --check` / `flake8` clean on every file touched today
(`src/estimator/joint.py`, `src/model/__init__.py`,
`scripts/eval_joint_estimator.py`,
`scripts/measure_component_cap_cost.py`,
`tests/test_estimator_joint.py`,
`tests/test_measure_component_cap_cost.py`,
`tests/test_data_model_citations.py`,
`docs/adr/0011-multi-entity-factor-graph.md`,
`docs/data_model/v0.3.md`).

Repo-wide `-m "not requires_weights and not slow"`: **1072 passed, 1
skipped, 21 deselected, 0 failures** — up from Day 26's 1048 (+24: 6
slip-model tests, 13 component-cap tests across two files, 5
data-model-citation lint tests).

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 26 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; now 4 days older.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 16 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 9 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-27` pushed clean, matches
origin exactly. `main` unchanged. `git push --tags`: up to date.

## Day 28, in order

1. **A two-term slip model** — acceleration-scaled slip ADDED to a
   small constant floor, derived and tested with the same discipline
   Day 27 used (rule out confounders first, instrument the mechanism
   directly, report a derived model's failure as a finding). Re-run
   both the carrier calibration check and the asset margin check; both
   must pass together or the trade is not resolved.
2. **A decision on constraint typing and the hypothesis store** — now
   documented as explicitly unspecified (Day 27 Objective 3); decide
   whether to build, defer with a stated reason, or drop each, rather
   than continuing to cite them as settled.
3. **Real coupled multi-entity data, or a larger authored scene** — the
   only way to measure the component-size cap's cost on a genuinely
   well-matched component (Day 27 Objective 2's own open question).
4. **Hypothesis management** (`resolve_data_association`) — depends on
   item 2's decision.
5. **The discrete/continuous hybrid** (`HybridDiscreteContinuousState`)
   — dynamic component membership; also depends on item 2.
6. **Smoothing across the joint graph** — still depends on the
   coupling's own calibration being trustworthy first (item 1).
7. **MEVA licence verification** — still blocked on a human, highest
   product impact of any pending data item.
8. **DA-2K licence verification** — adapter built and tested, zero
   engineering lag once cleared.
9. **Reference hardware procurement decision** — now 9 days old.
10. **Declare a target fps for real camera ingest** — still open from
    Day 19.
11. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
12. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; gates on item 8 for a real depth number.
13. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
14. **The generator has no sensor-noise model** — unchanged.
15. **Order cameras and run the office capture** — now 16 days old.
16. **The motion-gate precision/selectivity investigation** — still
    deferred.
17. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
18. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should
    be replaced** — unchanged.
19. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
20. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a
    DPDP process decision, not a coding task.
21. **The `resolve_joint_state` cross-component filtering gap** (ADR
    0011) — cheap to fix, not yet needed by any real call site.

# Day 28

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-27.
Everything below is accuracy, consistency, and structural/documentation
measurement.

**Headline: the two-term slip model's derivation was well-determined —
no ratio was swept — and it still fails the acceptance criterion, for a
reason the derivation itself could not have caught.** Objective 1
combined Day 26's constant floor and Day 27's acceleration-scaled term
by variance addition, the physically correct combination for two
independent noise sources, reusing the same already-declared constants
with no new, separately-tuned parameter between them — the trap the
day's prompt named (sweeping a ratio until both checks pass) never
applied, because there was no ratio to sweep. Measured anyway, the model
tracks `acceleration_scaled` almost exactly on every statistic, on both
golden sets, because the causal acceleration ESTIMATE that feeds the
second term sits 2-7x the reference constant even during GT-labeled
steady regimes — the floor's variance is 25-49x smaller than the
acceleration term's and never gets the chance to dominate the way the
physical story assumed. **Not adopted; the physics was right and the
estimator feeding it was the thing nobody had characterized.** Objectives
2-3 closed the two provisions Day 27's audit found assumed-nowhere-
recorded — constraint typing and the hypothesis store's
`PRUNED_BY_BUDGET` — and Objective 4 traced how each went unrecorded in
the first place: one was an honestly-caveated stub whose caveat quietly
stopped being read; the other never touched version control at all
before the audit that found it missing, existing only in the
conversational reasoning between day-to-day prompts. No lint fixes the
second kind — the fix is citation discipline, not tooling.

## Verdicts

- **Did the two-term slip model's ratio fall out of the physics, or was
  the derivation underdetermined?** Fell out of the physics — both
  terms reuse the same two already-declared constants
  (`OFFSET_SLIP_SIGMA_MPS_SQRT_S`, `PERSON_SIGMA_A_MPS2`), combined by
  variance addition with no third, separately-tuned parameter. No sweep
  was run; none was needed to combine them. → Objective 1.
- **Does the two-term model meet the acceptance criterion (both must
  hold, or it's a trade)?** No. Carrier cessation overconfidence
  resolves (unjustified gain +19.3%→+0.9%, matching
  `acceleration_scaled`). Asset RMSE margin does NOT survive
  (+0.0425m→−0.0052m on v5-cessation, +0.0265m→−0.0050m on v3-indoor) —
  within 0.0001-0.0002 of `acceleration_scaled` alone on every
  statistic, on both sets. Reported as another measured trade, not
  adopted; `offset_slip_model` default stays `"constant"`. → Objective 1.
- **Why does the floor term fail to protect the asset when the
  derivation says it should, at low acceleration?** Measured directly:
  the causal, one-step-lagged acceleration ESTIMATE that feeds the
  second term sits 2-7x `PERSON_SIGMA_A_MPS2` even in GT-labeled
  `sustained`/`static` regimes — it never reads "low" the way steady
  motion should make it read. `PERSON_SIGMA_A_MPS2` bounds plausible
  TRUE human acceleration; it was never validated as a bound on the
  NOISE FLOOR of a 12fps finite-difference acceleration estimate. Those
  are different quantities; the derivation assumed they were
  interchangeable. → Objective 1.
- **Is constraint typing (hard vs. twin-dependent) implemented, with the
  STRUCTURAL rules the objective asked for?** Yes. `TwinDependentConstraint`
  is unconstructable without `twin_rev` (undefaulted field, same shape as
  `WorldPosition`, §0). The pruning path (`prune_for_hard_violation`)
  accepts only `HardConstraintViolation`; `raise_twin_revision` has no
  `HypothesisStore` parameter at all, so a twin-dependent violation is
  structurally incapable of reaching the prune path — routed apart by
  type, not a runtime `if`. `TwinRevisionHypothesis` is a recorded type
  with no consumer yet, noted explicitly. Tested:
  `tests/test_model_constraint.py`, `tests/test_estimator_constraints.py`
  (32 tests total across constraint typing + the hypothesis store). →
  Objective 2.
- **Is the hypothesis store's typed death-cause discipline implemented?**
  Yes. `HypothesisStore.kill` requires a typed `DeathCause` with no
  default — there is no discard/remove/pop that skips it.
  `PRUNED_BY_BUDGET` hypotheses are retained as records (proposition +
  support at death, no full state) and `considered_alternatives()`
  surfaces them unfiltered — tested directly
  (`test_budget_pruned_hypotheses_are_not_filtered_out_of_considered_alternatives`).
  Full multi-hypothesis management stays skeleton, as scoped. →
  Objective 3.
- **How did constraint typing and `PRUNED_BY_BUDGET` go unrecorded, and
  what would have caught it?** Different mechanisms for each. Constraint
  typing: first committed Day 20 as an honestly-caveated stub label
  ("not yet implemented") inside §15, a section that genuinely exists —
  never a dangling `§N` citation, so the existing citation lint could
  never have caught it; what would have is a provision registry
  requiring a status from the moment a named-but-unbuilt capability is
  introduced. `PRUNED_BY_BUDGET`: zero committed trace anywhere before
  the Day-27 audit that found it missing — cited as settled only in
  conversational reasoning across day-boundaries, never in anything a
  repo-scoped lint could see. No lint change closes that gap; the
  finding is that a claim's specificity is not evidence of its
  provenance. → Objective 4.
- **Updated provision-audit counts — any provision still
  assumed-nowhere-recorded?** None. Both provisions traced today now
  read `IMPLEMENTED` with resolving pointers
  (`src/model/constraint.py` + `src/estimator/constraints.py`;
  `src/model/hypothesis.py`), verified by a new structural lint
  (`tests/test_data_model_citations.py::test_no_broken_implemented_pointer`)
  that fails on any provision marked `IMPLEMENTED` whose pointer does
  not resolve. → Objective 4.

## Objective 0 — push, start of day

`foundation/day-28` branched from `foundation/day-27` (`c01daa4`, the
Day-27 report commit), pushed clean before any Day-28 commit landed —
`origin/foundation/day-28` matched local exactly. `main` unchanged since
Day 16 (ADR 0009 still Proposed). (End of day: see the closing
Objective 0/5 section below.)

## Objective 1 — the two-term slip model: derivation clean, acceptance criterion not met

Full derivation, the complete three-way comparison tables (both golden
sets), the instrumented acceleration-estimate trace, and the root-cause
analysis are in ADR 0011's "Day 28, Objective 1" section; summarized
here.

**The derivation.** `OffsetSlipModel = "two_term"`
(`src/estimator/joint.py`): `"constant"` PLUS `"acceleration_scaled"`,
combined as VARIANCES — the physically correct combination for two
independent noise sources (variances add; sigmas do not):

    offset_variance = OFFSET_SLIP_SIGMA_MPS_SQRT_S^2
                     + (OFFSET_SLIP_SIGMA_MPS_SQRT_S * |a_hat|/PERSON_SIGMA_A_MPS2)^2

Both terms reuse the same two already-declared constants from Day 26/27.
No third, separately-tuned ratio parameter exists to sweep — the trap
named explicitly in the day's prompt does not apply to this specific
model, because there was nothing free to fit. A real implementation bug
was found and fixed along the way: the causal acceleration estimate was
only ever computed when `offset_slip_model == "acceleration_scaled"`,
so `"two_term"` silently ran with `carrier_acceleration_mps2=None` at
every step and was bit-identical to `"constant"` for an entire
evaluation run before this was caught — regression-guarded with a new
test.

**Three-way comparison (constant / acceleration_scaled / two_term),
v5-cessation:**

| model | carrier margin | carrier no-trade | asset margin | asset no-trade |
| --- | ---: | --- | ---: | --- |
| constant (Day 26) | +0.0280m | FAIL_OVERCONFIDENT (Δ −0.0215) | +0.0425m | PASS (Δ +0.0167) |
| acceleration_scaled (Day 27) | +0.0115m | PASS (Δ +0.0020) | −0.0051m | PASS (Δ +0.0137) |
| two_term (Day 28) | +0.0115m | PASS (Δ +0.0020) | **−0.0052m** | PASS (Δ +0.0137) |

Same pattern on v3-indoor (carrier margin +0.0144m→+0.0066m→+0.0066m;
asset margin +0.0265m→−0.0049m→−0.0050m). two_term lands within
0.0001-0.0002m of acceleration_scaled on every statistic, on both sets
— not partway between constant and acceleration_scaled the way a
protective floor term should place it.

**Acceptance criterion, stated in advance, evaluated explicitly.**
Carrier cessation overconfidence resolves (unjustified gain +19.3% →
+0.9%): YES. Asset RMSE margin survives: NO (goes negative on both
sets, matching acceleration_scaled's collapse almost exactly). Both
were required; one failed. **Not adopted** — reported as another
measured trade, same disposition as Day 27's acceleration_scaled.
`offset_slip_model` default stays `"constant"`.

**Root cause, measured directly, not inferred.** Instrumented `|a_hat|`
(the estimate feeding the acceleration term) frame-by-frame against
`PERSON_SIGMA_A_MPS2` on a real track: every sampled step from a
GT-labeled `sustained` or `static` regime showed `|a_hat|` at 1.8-6.5x
the reference constant — never once reading "low" the way steady motion
should. The estimate is built from the difference of two consecutive
Kalman-filtered velocity states divided by `dt_s ≈ 0.083s` (12fps), a
division that amplifies ordinary estimation noise into an apparent
acceleration several times `PERSON_SIGMA_A_MPS2` regardless of the
carrier's TRUE acceleration. `PERSON_SIGMA_A_MPS2` was derived as a
bound on plausible true human acceleration; it was never validated as a
bound on the NOISE FLOOR of a finite-difference estimate of that
acceleration at this frame rate. The two-term model's derivation
silently assumed those were the same quantity. They are not, and this
is a measuring-apparatus finding in the same family as Day 24's and
Day 25's — the apparatus this time is the state estimator's own causal
acceleration estimate, used here for the first time as an input to
another model's noise term rather than as an output to report.

ADR 0011 updated with this section, Day 26 and Day 27's results kept
visible alongside per instruction.

## Objective 2 — constraint typing: hard vs. twin-dependent, implemented

`src/model/constraint.py` (typing) + `src/estimator/constraints.py`
(estimator wiring). `HardConstraint` (true regardless of twin
correctness — violation prunes) and `TwinDependentConstraint` (only as
true as `twin_rev` — violation raises a `TwinRevisionHypothesis`
instead) are disjoint types. `TwinDependentConstraint` makes `twin_rev`
a required, undefaulted field — unconstructable without it, same shape
`WorldPosition` already enforces (§0); confirmed by
`test_twin_dependent_constraint_unconstructable_without_twin_rev`
(`TypeError` on omission).

**STRUCTURAL: closed-world dispatch, not an `if`.**
`evaluate_constraint`'s `@overload` pair gives a `HardConstraint` caller
`HardConstraintViolation | None` and a `TwinDependentConstraint` caller
`TwinRevisionHypothesis | None` — routed apart by type at the mypy
level. `prune_for_hard_violation` (the only function that calls
`HypothesisStore.kill`) takes a `HardConstraintViolation` parameter,
full stop. `raise_twin_revision` has no `HypothesisStore` parameter at
all — confirmed by
`test_raise_twin_revision_has_no_hypothesis_store_parameter`, which
inspects the live signature rather than trusting a docstring claim.
There is no shared function anywhere that inspects `violation.kind` and
branches on whether to prune.

**Concrete constraints.** `one_body_one_place` (hard) is fully
implemented — no site geometry, no twin, no calibrated constant beyond
a floating-point tolerance, true by definition. `gravity_floor_transition`,
`max_pedestrian_velocity`, `mass_conservation` (hard) and
`wall_impermeability`, `portal_required`, `stair_or_lift_for_floor_change`,
`visibility_and_accessibility` (twin-dependent, per-`twin_rev` factories)
are named and typed correctly but raise `NotImplementedError` naming what
would fill them — same convention as `src/model/__init__.py`'s own
placeholder rule, deriving their thresholds is future work, not guessed
today (Day 29 list).

**Wired as a hypothesis-pruning factor.** `check_hard_constraints`
evaluates a sequence of hard constraints and prunes the store on the
first violation — the actual mechanism a real multi-hypothesis tracker
will call once one exists.
`test_one_body_one_place_violated_prunes_the_hypothesis` and
`test_twin_dependent_violation_does_not_prune_and_emits_a_revision_hypothesis`
confirm both routing directions end-to-end, including that the store's
`alive()` set is untouched by a twin-dependent violation.

`TwinRevisionHypothesis` is recorded, with no consumer yet, noted
explicitly in its own docstring. Repeated violations of the same
constraint at the same location, tracked over time, would be the twin
drift detector this project has wanted — falls out of typing
constraints correctly rather than needing separate engineering, per
instruction; the tracking loop itself is not built today.

## Objective 3 — the hypothesis store's typed death causes, implemented

`src/model/hypothesis.py`. `Hypothesis` (live) and `DeadHypothesis`
(retained record: proposition + support at death, deliberately WITHOUT
full state) are separate types.
`DeathCause = RefutedByHardConstraint | RefutedByObservation |
DominatedByLikelihood | MergedInto(id) | ExpiredHorizon | PrunedByBudget`.

**STRUCTURAL: a hypothesis cannot die without a typed cause.**
`HypothesisStore.kill(hypothesis_id, cause: DeathCause)` is the only way
a hypothesis leaves the alive set — no `discard`/`remove`/`pop` exists
that skips the `cause` argument, confirmed by
`test_kill_requires_a_death_cause_argument` (`TypeError` on omission).
Every death, regardless of cause, is unconditionally recorded as a
`DeadHypothesis` before `kill` returns.

**`PRUNED_BY_BUDGET` is not a rejection.** "We ruled it out" and "we
never evaluated it" are different answers to a forensic query, and
`considered_alternatives()` surfaces budget-pruned records alongside
refuted ones by construction — no separate "rejected only" query exists
for a caller to reach for by mistake.
`test_budget_pruned_hypotheses_are_not_filtered_out_of_considered_alternatives`
constructs one refuted and one budget-pruned hypothesis and confirms
both appear in the forensic query, with the budget-pruned one's cause
still typed as `PrunedByBudget`, not silently dropped or relabeled.

Full multi-hypothesis management stays skeleton, as scoped: nothing
here decides when to spawn, merge, or budget-prune a hypothesis — only
the store, the lifecycle, and the death-cause discipline exist, so that
when hypothesis management arrives it cannot be built without them.

## Objective 4 — how did two provisions go unrecorded, and what would have caught it

**Constraint typing, traced.** First committed appearance: Day 20
(`22f4f49`, 2026-08-10), `src/estimator/consistency.py`:
`_CONSTRAINT_CONSUMER = "twin-revision hypothesis (not yet
implemented)"`. Not a false claim at the time — the caveat is in the
string. Not a `§N` citation either — `tests/test_data_model_citations.py`
scans for `§\d+`, and this is a bare English phrase inside a Python
string constant. It lived inside §15's own stub-residual machinery, a
section that genuinely exists, so this was never a dangling citation to
an undefined destination — it was an unrecorded, separately-verifiable
PROMISE inside a destination that was itself real. What would have
caught it: not a stricter `§N` scanner, but a provision registry that
requires a status field the moment a named-but-unbuilt capability this
specific (a distinguishable "hard vs. twin-dependent" split, not just
"handle constraint violations somehow") is introduced — added below.

**`PRUNED_BY_BUDGET`, traced.** First committed appearance: none.
`git log --all -S PRUNED_BY_BUDGET` shows the literal string first
appearing in the Day-27 commit that reported it MISSING — there is no
earlier stub, no earlier punch-list line at that precision, nothing.
"Hypothesis management" as a general topic IS real and repo-visible
(carried on this project's ordered day-lists since Day 5); the specific
claim `PRUNED_BY_BUDGET` names was never written down at that precision
anywhere a repo-scoped mechanism could see it — it was cited as settled
only in the conversational reasoning that produced day-to-day prompts.
**What would have caught it: nothing that scans this repository.** A
claim that never touches version control is outside the reach of any
lint that only reads version-controlled files, by construction. The
actual lesson: a claim's specificity is not evidence of its provenance —
`PRUNED_BY_BUDGET` read as settled precisely BECAUSE it was so
precisely named (a literal `SCREAMING_CASE` constant reads as something
already built, not a vague gesture inviting scrutiny), and that
resemblance to a recorded fact is exactly backwards from how much
confidence a genuinely unrecorded claim deserves. The closing action is
citation discipline (grep before using a specific technical claim in
argument), not a tooling change.

**Lint extension, applied.** Every `§N` section's `**Status: ...**` line
and every Catalog table row in `docs/data_model/v0.3.md` now states a
status, and an `IMPLEMENTED` status names a pointer (a path, optionally
`path::symbol`) that must resolve — file/directory exists, and a named
symbol is actually defined in it (text search, not an import).
`tests/test_data_model_citations.py::test_no_broken_implemented_pointer`
enforces this, with its own falsifiability test
(`test_the_lint_actually_catches_a_broken_implemented_pointer`) proving
the check fires on a genuinely broken pointer before trusting that it
passing on the real document means anything. This closes the gap for
provisions that ARE written down somewhere in the doc (constraint
typing's own failure mode); it does not and cannot close
`PRUNED_BY_BUDGET`'s (see above).

**Updated audit classification.** Re-ran the Day-27 census after today's
two implementations: `docs/data_model/v0.3.md` now records constraint
typing and the hypothesis store as `IMPLEMENTED` (pointers:
`src/model/constraint.py` + `src/estimator/constraints.py`;
`src/model/hypothesis.py` — both resolve). **No provision in the
document remains ASSUMED, NOWHERE RECORDED.**

## Full suite and mypy

`mypy` (scoped per `mypy.ini`): clean, 0 errors — `src/model/constraint.py`,
`src/model/hypothesis.py`, and `src/estimator/constraints.py` (new
today) all pass. `black --check` / `flake8` clean on every file touched
today (`src/estimator/joint.py`, `src/model/constraint.py`,
`src/model/hypothesis.py`, `src/estimator/constraints.py`,
`scripts/eval_joint_estimator.py`, `tests/test_estimator_joint.py`,
`tests/test_model_constraint.py`, `tests/test_model_hypothesis.py`,
`tests/test_estimator_constraints.py`, `tests/test_data_model_citations.py`,
`docs/adr/0011-multi-entity-factor-graph.md`, `docs/data_model/v0.3.md`).

Repo-wide `-m "not requires_weights and not slow"`: **1119 passed, 1
skipped, 21 deselected, 0 failures** — up from Day 27's 1072 (+47: 8
two-term slip-model tests, 32 constraint-typing/hypothesis-store tests
across three files (9 + 14 + 9), 7 provision-status/pointer-lint tests).
Also ran the full suite with no marker exclusions at all (including the
13 `@pytest.mark.slow` tests): **1127 passed, 7 skipped, 0 failures**,
22m34s — clean. One transient failure was observed and diagnosed during
today's work: `test_synthetic_indoor.py`'s duration-threshold gate
(Day 24 Objective 4) tripped on
`test_stationary_v5_cessation_agent_silhouette_is_bit_identical` at
17-18s under three concurrent pytest processes contending for CPU;
re-run alone it completes in 5.92s, comfortably under the 15s
threshold. Confirmed CPU-contention flake, not a regression, before
this section was written.

## Blocked on humans, restated

Per [[iron-blocked-on-humans]]. Unchanged from Day 27 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — ranked #1 in Day 23's ordered list,
   highest product impact of any pending data item; now 5 days older.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 17 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision**
   (`docs/reference_hardware.md`) — now 10 days old.

## Objective 0/5 — push, end of day

`git push --all origin`: `foundation/day-28` pushed clean, matches
origin exactly. `main` unchanged. `git push --tags`: up to date.

## Day 29, in order

1. **Characterize the causal acceleration estimator's own noise floor**
   (or re-derive `PERSON_SIGMA_A_MPS2`, or a separate reference constant,
   from that floor rather than from true-human-acceleration bounds) —
   the specific blocker Day 28 found for decoupling the two-term slip
   model's confidence-calibration and estimate-elasticity effects.
2. **Derive real predicates for the seven stubbed constraints**
   (`gravity_floor_transition`, `max_pedestrian_velocity`,
   `mass_conservation`, `wall_impermeability`, `portal_required`,
   `stair_or_lift_for_floor_change`, `visibility_and_accessibility`) —
   today only established their typed place in the registry.
3. **Hypothesis management itself** (spawn/score/budget-prune logic) —
   the store and lifecycle exist; nothing yet decides when to use them.
4. **Real coupled multi-entity data, or a larger authored scene** — the
   only way to measure the component-size cap's cost on a genuinely
   well-matched component (Day 26/27's own open question, still open).
5. **The twin drift detector** — falls out of typing constraints
   correctly (Day 28), but the tracking loop over repeated
   `TwinRevisionHypothesis` occurrences at one location is not built.
6. **The discrete/continuous hybrid** (`HybridDiscreteContinuousState`)
   — dynamic component membership; depends on item 3.
7. **Smoothing across the joint graph** — still depends on the
   coupling's own calibration being trustworthy first.
8. **MEVA licence verification** — still blocked on a human, highest
   product impact of any pending data item.
9. **DA-2K licence verification** — adapter built and tested, zero
   engineering lag once cleared.
10. **Reference hardware procurement decision** — now 10 days old.
11. **Declare a target fps for real camera ingest** — still open from
    Day 19.
12. **Cascade bench, clean, on interim or reference hardware** — still
    pending hardware.
13. **Depth validity re-measurement** (`scripts/eval_depth.py`) — still
    deferred; gates on item 9 for a real depth number.
14. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
15. **The generator has no sensor-noise model** — unchanged.
16. **Order cameras and run the office capture** — now 17 days old.
17. **The motion-gate precision/selectivity investigation** — still
    deferred.
18. **A canonical `Observation -> hash` function** (ADR 0007) — still
    open from Day 13.
19. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should
    be replaced** — unchanged.
20. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
21. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a
    DPDP process decision, not a coding task.
22. **The `resolve_joint_state` cross-component filtering gap** (ADR
    0011) — cheap to fix, not yet needed by any real call site.

# Day 29

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-28.
Everything below is accuracy, consistency, and structural measurement.

**Headline: the slip question is closed with a measured NO, and it
closes because the input was never measurable, not because the physics
was ever wrong.** Objective 1 characterized the causal acceleration
estimator against GT for the first time: its steady-regime noise floor is
6.86 / 7.33 m/s² on v5-cessation / v3-indoor, against a
`PERSON_SIGMA_A_MPS2` of 1.5 — **SNR ≈ 0.21 / 0.20 at the scale the slip
model reads it**. Three slip models (Day 26 constant, Day 27
acceleration-scaled, Day 28 two-term) were derived from a ratio that was
roughly 5x noise, and each failure was read as evidence about the slip
physics. Objective 2 tested one remedy, predicted its failure in writing
first, and confirmed it: suppressing that floor by smoothing needs a
~2.1-2.5s window against a ~1s cessation event. **NO — acceleration-
conditioned slip is not achievable at this frame rate and sensor
quality.** Objective 3 filled all seven stubbed constraint predicates,
added the `Unevaluable` outcome that keeps four of them from passing
vacuously, and measured **zero GT hard-constraint violations across 9,507
evaluations** — after the first run's 16 violations turned out to be the
measuring script reading the wrong axis as vertical.

## Verdicts

- **Is acceleration-conditioned slip achievable at this frame rate
  (12fps) and sensor quality?** **NO.** The causal acceleration
  estimate's noise floor is measurement-noise-dominated, and the
  smoothing window needed to suppress it below the scale the slip model
  depends on introduces a lag (~2.1-2.5s, or ~1.1-1.2s under the most
  charitable reading) comparable to or exceeding the ~1s cessation event
  the model exists to react to. → Objectives 1-2.
- **What is the acceleration estimator's noise floor, and its SNR?**
  Steady-regime std 6.8607 m/s² (v5-cessation) / 7.3290 (v3-indoor);
  pooled SNR vs `PERSON_SIGMA_A_MPS2` 0.2186 / 0.2047. Per-regime, using
  each regime's own mean GT acceleration as signal: `onset` 0.0739,
  `cessation` 0.3506, `maneuver` **1.9421** (v5-cessation). Only
  `maneuver` clears 1.0; `onset` and `cessation` — the regimes this
  investigation exists to fix — do not. One frame rate exists in this
  project's data (12fps); reported as a bounded null, not extrapolated.
  → Objective 1.
- **Where does the noise come from?** Measurement noise propagated
  through the filter, almost entirely. Ablation at identical observation
  draws: `q_near_zero` 6.8439 / 7.3117 (unchanged from baseline),
  `r_near_zero` 0.7423 / 0.0211 (a 9x / 300x collapse). → Objective 1.
- **Closed-form vs instrumented — did they diverge?** Yes, by ~5x
  (closed form predicts 34.99 / 36.96 m/s²). The naive formula assumes
  consecutive filtered velocities are independent; a Kalman filter
  smooths velocity across updates, so they are strongly correlated and
  their difference has far lower variance. The formula was not wrong
  about itself — it was wrong about what it silently required. →
  Objective 1.
- **Which remedies were tested, and which deliberately not?** Candidate
  B (windowed smoothing of the causal `a_hat`) was tested. **Candidate A
  (acceleration from the joint posterior) was deliberately NOT tested**:
  the R-dominated attribution makes A and B the same remedy physically —
  more temporal averaging against measurement noise — and a smoother's
  advantage over a boxcar is a better small-N constant, not different
  asymptotic noise-vs-window scaling. B needs no state-model change and
  its lag cost is directly measurable, so B's result closes the question
  for both. Per the day's own discipline: a fifth attempt after a fourth
  failure is momentum, not diligence. → Objective 2.
- **Did the pre-registered SNR prediction hold?** Yes. Predicted before
  running: K ≈ 22 frames (~1.83s) to reach SNR ≥ 1, ~1.83x
  `PEDESTRIAN_STOP_DURATION_S`, therefore self-defeating for cessation.
  Measured: SNR crosses 1.0 between K=21 and K=34 on both sets (~2.1-2.5s
  — if anything worse than predicted). Even using cessation's own higher
  mean GT acceleration (2.4057 m/s²) instead of the generic reference,
  the crossing lands at K≈13-14 (~1.1-1.2s), still at or past the ~1s
  stop duration. → Objective 2.
- **Status of the seven stubbed predicates?** All seven now run. Three
  hard ones implemented with real physics: `max_pedestrian_velocity`,
  `mass_conservation`, `gravity_floor_transition` (joining
  `one_body_one_place` from Day 28). **Four return
  `Unevaluable("no twin geometry")`: `wall_impermeability`,
  `portal_required`, `stair_or_lift_for_floor_change`,
  `visibility_and_accessibility`** — blocked on a real dependency with a
  named interface (`TwinGeometry`), not on unwritten code. → Objective 3.
- **GT hard-constraint violation count?** **0**, across 9,507
  evaluations on both golden sets — the result a correctly-derived hard
  constraint set requires. Four measured qualifications on how much that
  zero establishes; see Objective 3. → Objective 3.

## Objective 1 — the acceleration estimator's noise floor

`scripts/measure_acceleration_noise_floor.py`. Day 28's five-point spot
check on one v5-cessation track is replaced by a full-population,
instrumented measurement over every scored frame of both sets, per Day
25's rule (instrument the running estimator; do not compute this from a
closed-form error-propagation argument). Numbers in the Verdicts block
above and in ADR 0011's Day-29 section.

One measurement was caught by inspection rather than trusted: an early
version computed per-regime SNR from a pooled median GT acceleration
across `onset`/`cessation`/`maneuver` and got ~0.0, because "cessation"
under Day 23's redefinition mixes genuine anticipatory deceleration with
post-stop recovery frames whose GT acceleration is already back near
zero — the median is dominated by the near-zero majority. The per-regime
means (0.5-13.3 m/s²) made the discrepancy visible. Reported as its own
instance of the closed-form-vs-instrumented finding family.

## Objective 2 — the slip question, closed

`scripts/measure_slip_remedy_smoothing.py`. Prediction stated in the
script's docstring before the sweep ran; measured window sweep
(Fibonacci-spaced K, causal boxcar over the already-causal `a_hat`)
confirmed it on both sets. Full tables in ADR 0011.

`offset_slip_model` stays `"constant"` by default (unchanged since Day
26). `"acceleration_scaled"` and `"two_term"` remain explicit opt-ins,
each carrying its own day's measured trade. No fourth
acceleration-conditioned variant will be derived against this noise
floor — the floor itself, not any one derivation, is now the documented
reason.

**What the NO costs, and the decision it hands over.** The carrier/asset
trade stops being an open problem and becomes a product decision with
quantified cost on both sides. Joint estimation with `"constant"` slip
measurably helps the carried asset (RMSE margin +0.0425m / +0.0265m, Day
26 Objective 4) at a measurable calibration cost to the carrier
specifically in cessation (coverage 0.9089→0.8874, `FAIL_OVERCONFIDENT`,
Day 26). Two postures, both fully measured, neither yet chosen:

- **Carrier calibration protects evidence integrity** — ship joint
  estimation only where carrier posterior calibration is not degraded,
  foregoing the asset benefit in exactly the regime (a handoff or
  set-down) where a carried asset's position often matters most.
- **Asset accuracy protects the coupling's motivating case** — ship it by
  default, accepting the carrier's bounded cessation-regime
  overconfidence, on the grounds that carried-asset accuracy is
  coupling's entire reason to exist.

**Which side ships is a product decision, not an engineering one, and
should be made explicitly rather than left to whichever default this ADR
happens to carry.** Recorded in ADR 0011 as the punch-list item Day 26
deferred, now that the one technical question blocking it has a measured
answer.

The general rule this closes with is now recorded in
`.claude/skills/iron-eval-discipline/SKILL.md` ("Noise Floor First"),
with the acceleration estimator as its worked example: **before deriving
a model from a measured quantity, characterize that quantity's noise
floor against the scale the model depends on.** A physically correct
model fed a signal whose noise exceeds its operating range fails in ways
that look exactly like the model being wrong — so the failure gets
attributed to the physics, a new model gets derived, and the cycle
repeats against the same unmeasured input. Three times, here.

## Objective 3 — the seven predicates

All seven run. Statuses in the Verdicts block; derivations, the
`Unevaluable` type rationale, and the two mis-typing findings in ADR
0011's Day-29 Objective 3 section. Four things worth surfacing here:

**A constraint that cannot be evaluated may not contribute a PASS, and
this is now enforced by type.** `evaluate_constraint` no longer returns
`None`; its outcome set is closed and three-valued per kind. Removing
`None` rather than adding `Unevaluable` beside it is the substance: a
caller writing `if evaluate_constraint(...) is None` would have
classified `Unevaluable` as a violation, and `is not None` would have
classified it as one too — one reading is wrong whichever way the caller
guesses, and neither looks wrong at the call site. The exhaustiveness
claim was verified by removing a `match` case and confirming `mypy`
errors, not asserted.

**Two of Day 28's three "hard" constraints were described in
twin-dependent terms** — `gravity_floor_transition` named a "modeled
transition (stairs, lift, ramp)", which is word for word what the
twin-dependent `stair_or_lift_for_floor_change` already claims;
`mass_conservation` named "a modeled portal or occlusion boundary". One
physical claim filed under both kinds. Both keep their names, reduced to
their twin-free cores. The general tell: a hard constraint whose
description names the twin is mis-typed.

**Every pedestrian constant this project declared before today is
TYPICAL-scale; a hard constraint needs an IMPOSSIBILITY-scale bound.**
`PERSON_SIGMA_A_MPS2` is a process-noise density (acceleration is
unbounded in the model it parameterizes), and `PERSON_SIGMA_A_MPS2 *
PEDESTRIAN_STOP_DURATION_S` = 1.5 m/s is a comfortable walking pace by
its own docstring. As a hard max-speed threshold it would irreversibly
prune anyone jogging — and, measured, violates on **46.0% of
v5-cessation GT frames**. `PEDESTRIAN_MAX_SPEED_MPS = 12.5` was declared
instead, cited to human physiology. A hard constraint's job is to be
never-wrong, not tight; it sits on an irreversible path, so discriminating
power at the typical scale belongs in the likelihood, where being wrong
is recoverable.

**Zero GT violations, with four measured qualifications.** (1) The
measurement does discriminate — the rejected 1.5 m/s threshold would fail
loudly, as above. This corrects the reasoning that first justified the new
constant, which argued the mis-derivation would *pass* because "the
golden sets' walkers move at ~0.5 m/s" — a figure taken from the Day-21
report rather than measured against today's sets (v5-cessation peaks at
7.74 m/s). Asserting a property of the data from a stale document instead
of measuring it is this day's own subject matter, committed while
documenting it. (2) `gravity_floor_transition` is untested by this data:
GT vertical position is a constant 0.86 m in every track, so its zero is
a bounded null, not a pass; `one_body_one_place` is likewise vacuous on
v5-cessation (single-agent clips) though exercised on v3-indoor (2,680
pairs, closest approach 0.0185 m). (3) **The generator does produce
unphysical motion, in a quantity no constraint bounds**: v5-cessation GT
reaches 36.58 m/s² of horizontal acceleration — 3.7g, 24x
`PERSON_SIGMA_A_MPS2` — with 3.4% of frames above 1g. A human on foot
cannot decelerate at 3.7g. Not fixed today by inventing a fifth
threshold inside the acceptance measurement that would judge it. (4)
v3-indoor GT is exactly constant-velocity (peak acceleration 0.00 m/s²),
which is the concrete reason its per-regime SNR column reads 0.0000.

**These strengthen Objective 2's NO rather than weakening it.** The slip
verdict was measured against a GT whose cessation transients are larger
than physically possible; a real pedestrian's are smaller, so the true
signal is smaller and the real-world SNR is *lower* than measured. The
question closes on a favorable-case measurement.

**The instrument, again.** The first GT run reported 16
`gravity_floor_transition` violations at up to -36.58 m/s², 3.7x free
fall. Every one was the measuring script reading `agent_xyz` index 2 as
vertical, citing `src/model/world.py`'s +z-up world frame — a real
convention, just not the one that array is in. Index 1 is vertical; z is
camera-facing depth, and an abrupt depth-axis speed change reads exactly
like an impossible fall when the axis is mislabelled. Verified against
the generator (`scripts/gen_synthetic_indoor.py:198`) rather than
re-assumed. **The twelfth instrument finding**, and the reason the
nonzero count was diagnosed rather than reported.

## Still blocked on a human

Per [[iron-blocked-on-humans]]. Unchanged from Day 28 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — now 6 days older than Day 23's
   ranking, still the highest-product-impact pending data item.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 18 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision** — now 11 days old.
7. **The carrier/asset production posture** (new today) — ADR 0011 now
   records both sides fully measured; choosing between them is a product
   decision, not an engineering one.

## Day 30, in order

1. **Hypothesis management itself** (spawn/score/budget-prune logic) —
   the store, the lifecycle, the death causes, and now the constraint
   predicates all exist; nothing yet decides when to use them. This is
   the largest coherent piece of unbuilt work in the estimator.
2. **A hard bound on pedestrian acceleration** — Day 29 found GT at 3.7g
   with no constraint able to catch it. Needs the same
   impossibility-vs-typical derivation `PEDESTRIAN_MAX_SPEED_MPS` went
   through, done outside the acceptance measurement that would judge it.
3. **Fix the generator's instantaneous velocity steps** — the same
   finding from the generator's side. v5-cessation's stops are velocity
   discontinuities; every cessation-regime measurement since Day 21 has
   been made against transients no body could produce.
4. **Vertical motion in the golden sets** — GT height is a constant
   0.86 m, so `gravity_floor_transition` and any floor-transition
   capability are untestable. Blocks item 2's sibling constraint too.
5. **Real coupled multi-entity data, or a larger authored scene** — the
   only way to measure the component-size cap's cost on a well-matched
   component (Day 26/27's open question).
6. **Twin geometry** (`TwinGeometry`) — four typed constraints are
   `Unevaluable` pending it, and the twin drift detector (item 7) sits
   behind it.
7. **The twin drift detector** — falls out of constraint typing; the
   tracking loop over repeated `TwinRevisionHypothesis` occurrences at
   one location is not built.
8. **The discrete/continuous hybrid** (`HybridDiscreteContinuousState`)
   — depends on item 1.
9. **Smoothing across the joint graph** — depends on the coupling's own
   calibration being trustworthy first.
10. **MEVA licence verification** — blocked on a human, highest product
    impact of any pending data item.
11. **DA-2K licence verification** — adapter built and tested, zero
    engineering lag once cleared.
12. **Reference hardware procurement decision** — now 11 days old.
13. **Declare a target fps for real camera ingest** — open since Day 19.
14. **Cascade bench, clean, on interim or reference hardware** — pending
    hardware.
15. **Depth validity re-measurement** (`scripts/eval_depth.py`) — gates
    on item 11 for a real depth number.
16. **The `main`/`origin/main` decision** (ADR 0009) — a human call.
17. **The generator has no sensor-noise model** — unchanged.
18. **Order cameras and run the office capture** — now 18 days old.
19. **The motion-gate precision/selectivity investigation** — deferred.
20. **A canonical `Observation -> hash` function** (ADR 0007) — open
    since Day 13.
21. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should
    be replaced** — unchanged.
22. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
23. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued — a
    DPDP process decision.
24. **The `resolve_joint_state` cross-component filtering gap** (ADR
    0011) — cheap to fix, not yet needed by any real call site.

# Day 30

**No timing, throughput, CPU-percentage, or latency claim is made
anywhere in this section** — unchanged hard scope rule from Day 20-29.
Everything below is accuracy, consistency, and structural measurement.

**Headline: config B is not an estimator improvement, and the defect it
was adopted to fix does not exist.** Its velocity uncertainty is pinned
at the 1.5 m/s floor on **100.00%** of scored frames on the new physical
golden set and 99.12%/99.89% on the old ones — it does not estimate
velocity uncertainty, it reports a walking pace. And on physically
reachable motion, config A's cessation coverage is **0.9920**, essentially
nominal, against the 0.5013 measured on v5-cessation. The overconfidence
diagnosed on Day 21, built for on Day 22, re-measured on Day 23, audited
on Day 24, and adopted against on Day 25 was a filter responding correctly
to a teleport-to-zero. **ADR 0010 reverts to config A.** A second
casualty: the recorded reason for rejecting IMM — a static-coverage
collapse to 0.0808 — does not reproduce on physical motion (0.9726), and
the argument in this day's own Objective 1 that IMM's rejection was safe
from contamination was wrong; it is corrected below.

## Verdicts

- **Do the >1g transients concentrate in the cessation regime?** **YES.**
  74.3% of every >1g frame in v5-cessation (26/35) is cessation-regime,
  and cessation is the only populated regime whose p95 |a| is itself
  above 1g (15.28 m/s²). Counted by EVENT rather than frame — the
  sharper number, since a cessation regime is mostly a recovery tail
  sitting at zero acceleration while the transient is one or two frames —
  **14 of 22 stop events peak above 1g, median peak 15.13 m/s² (1.54g),
  max 36.58 (3.7g)**. All 12 abrupt stops violate; 2 of 10 "gradual"
  ones do. → Objective 1.
- **Is the cessation diagnosis contaminated?** **YES, substantially.**
  NEES 815 measured a filter's response to a velocity discontinuity, not
  to a person stopping. Confirmed independently by Objective 3: on
  v6-motion the same configuration measures cessation coverage 0.9920.
  → Objectives 1, 3.
- **Does v6-motion mint under physiological bounds?** **YES**, and it is
  the first set to pass a mint-time GT physicality gate: 19 clips, 1596
  frames, **0 hard-constraint violations**, max GT |a| **1.9955 m/s²
  (0.20g)** against v5's 36.58 (3.7g), and **0 of 22 stop events above
  1g** (median peak 0.7362 m/s², a **20.6x** reduction from v5's 15.13).
  Regime volume carried forward and cleared; observability 1.0000 mean
  AND min. `content_sha` identical across two independent renders.
  → Objective 2.
- **Does config B's velocity uncertainty ever fall below walking pace?**
  **Essentially never.** Pinned at the floor on **1558/1558 (100.00%)**
  scored frames on v6-motion, 1012/1021 (99.12%) on v5-cessation,
  2733/2736 (99.89%) on v3-indoor. Every unpinned frame is in `onset`,
  the one regime where velocity genuinely changes fast enough to exceed
  the floor naturally. Config A's own natural σ_v converges to 0.25-0.60
  m/s, so the floor sits 2.5-5.9x above it. → Objective 3.
- **Is config B a genuine estimator improvement, or a floor that
  satisfies the calibration criterion by refusing to estimate?** **The
  latter, and worse.** It is a constant-uncertainty filter, so its
  calibration result is not evidence about cessation modelling — any
  sufficiently large constant would produce it. On v6-motion it is also
  worse in position RMSE in **every** regime (static +50.2%, onset
  +13.6%, sustained +38.7%, cessation **+35.3%**), and its no-trade
  verdict is `NO_IMPROVEMENT`. → Objective 3.
- **Does ADR 0010 stand?** **NO.** The config A→B adoption is withdrawn
  and config A is re-adopted, under the **unchanged** Day-25 directional
  criterion — a re-scoring on better data, not a re-litigation of the
  criterion. → Objective 3.
- **Which raw-array boundaries carry an implicit axis convention?**
  **Eleven found, nine closed, two left with a stated reason.** Two of
  them were making a FALSE declaration rather than none. Full census in
  Objective 4. → Objective 4.

## Objective 1 — where the unphysical accelerations fall

`scripts/measure_gt_acceleration_distribution.py`. Same regime partition
(`classify_track`) every per-regime estimator number since Day 21 has
been computed against, so the numbers are directly comparable to those.

**v5-cessation** (19 tracks, 1021 frames with a defined acceleration):

| regime | n | p50 | p95 | max | frames >1g |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 260 | 0.000 | 0.000 | 0.000 | 0 (0.00%) |
| onset | 91 | 0.000 | 1.593 | 20.259 | 2 (2.20%) |
| sustained | 282 | 0.000 | 0.000 | 5.976 | 0 (0.00%) |
| cessation | 373 | 0.000 | **15.281** | **36.581** | 26 (6.97%) |
| maneuver | 15 | 9.566 | 24.255 | 27.854 | 7 (46.67%) |

All m/s². 1g = 9.80665. Overall p50 0.000, p95 8.692, max 36.581; 35
frames (3.43%) above 1g, of which **74.3% are cessation**.

**The frame fraction understates it.** A cessation regime is mostly a
recovery tail at exactly zero acceleration; the transient is one or two
frames. By event: **14/22 stop events peak above 1g, median 15.13 m/s²**.
Split by the set's own authored deceleration profile:

| profile | events | peak range (m/s²) | above 1g |
| --- | ---: | --- | ---: |
| abrupt (`ease_out=False`) | 12 | 11.74 – 36.58 | **12/12** |
| gradual (`ease_out=True`) | 10 | 3.89 – 27.62 | 2/10 |

Every abrupt stop is impossible by construction: the leg ends at constant
velocity and the next leg is a zero-length pause, so the velocity step is
the full approach speed in one frame. The gradual ones are impossible
when fast, because the quadratic ease-out's derivative is `2(1 - tail_t)`
— it **jumps speed to 2x** at the tail's start and then has one
`ease_fraction` (0.3 of a 1.2s leg ≈ 0.36s) to shed all of it. "Gradual"
was never a physiological profile; it was a smoothing of the second half
of a discontinuity.

**v3-indoor**, for contrast: max GT |a| **0.0001 m/s²** over 2736 frames
— exactly constant-velocity to float32 storage precision — zero stop
events, zero cessation frames. The two golden sets bracketed reality
without containing it.

**Two corrections to Day 29's own version of this measurement**, both in
the instrument. Day 29 computed acceleration from `t >= 1`; under this
project's mirror convention `v[0] == v[1]`, which forces `a[1] == 0`
identically for every track, so that put one guaranteed zero into every
distribution. Acceleration needs three positions, and this script reports
from `t >= 2`, saying how many frames it drops and why. It does not move
the maximum; it does move every percentile, and the two scripts'
distributions are therefore not interchangeable.

ADR 0010's cessation finding was marked **PROVISIONAL** on this evidence,
with no prior text edited, and the config B adoption deliberately left
standing for Objective 3 to test on its own terms.

## Objective 2 — v6-motion: physical motion, and a mint gate that enforces it

**Constrained, not post-filtered.** `PhysicalAgent` walks rest-to-rest
legs under a jerk-limited smoothstep speed profile. For a ramp of
duration `T` to cruise `v`, peak `|a| = 1.5v/T` and peak
`|jerk| = 6v/T²`; inverting those gives the ramp duration, so every bound
holds **by construction** — there is no clip, clamp, or smoothing pass
anywhere in the class. A leg too short to reach its requested speed and
brake again gets a lower cruise speed (bisection on a monotone function),
and both the request and the achieved value are recorded per clip.

**`V6_GAIT_BOUNDS`, each cited at the constant:**

| bound | value | source |
| --- | ---: | --- |
| max speed | 2.0 m/s | brisk walk; normative comfortable gait 1.2-1.4 m/s (Bohannon reference values) |
| max acceleration | 1.1 m/s² | gait-initiation COM acceleration, ~1-1.5 m/s² over 2-3 steps |
| max deceleration | 2.0 m/s² | gait termination — **asymmetric**, humans stop faster than they start |
| max jerk | 8.0 m/s³ | minimum-jerk models of voluntary movement |

The asymmetry is enforced at construction, not merely intended:
`GaitBounds` raises if deceleration does not exceed acceleration, and
raises if any value exceeds the corresponding **impossibility** bound in
`src/estimator/motion_model.py` (12.0 / 20.0 / 200.0 m/s^n, also added
today). Typical scale for authoring, impossibility scale for refutation,
and the inequality between them checked rather than assumed — this day's
Objective 5 rule, applied to itself.

**Corroboration, reported and not used to choose the value:** 2.0 m/s²
applied to a 1.4 m/s walk gives a stop duration of `1.5 × 1.4 / 2.0 =
1.05 s`, against `PEDESTRIAN_STOP_DURATION_S = 1.0 s`, declared on Day 22
from an entirely unrelated argument and never touched since. Two
independently derived constants landing 5% apart.

**STRUCTURAL: every heading change happens at zero speed.** A moving
agent that turns instantaneously has unbounded lateral acceleration
however carefully its speed profile is shaped — v5's velocity steps in a
different hat. No leg type turns while moving, so it is unreachable
rather than avoided by care. The stated cost: `maneuver` stays out of
scope (`V6_MOTION_EXEMPT_REGIMES`), because physical maneuvers need a
curved-path model with its own lateral-acceleration budget.

**A fifth hard constraint, derived outside the measurement that judges
it.** `max_pedestrian_acceleration` closes the gap Day 29 reported and
declined to close from inside an acceptance measurement. The four
existing constraints could not catch 3.7g: `max_pedestrian_velocity`
cannot, because every one of those frames is at an ordinary walking
SPEED arrived at impossibly fast; `gravity_floor_transition` cannot,
because the motion is entirely horizontal (which is also exactly why
misreading the depth axis as vertical made it look like a gravity
violation). Its `is_braking` argument is required, not inferred — a
magnitude cannot say which mechanism produced it, and the two bounds
differ. **v5-cessation now records 23 violations under it.**

**Mint-time gate.** `enforce_gt_physicality` walks
`HARD_CONSTRAINTS` itself rather than re-implementing any predicate, so
the bound a set mints against **is** the bound the estimator would prune
a hypothesis for violating — there is no second copy to drift.
`GtPhysicality.REQUIRED` is the default for every set from today; the
pre-Day-30 sets must name `LEGACY_UNPHYSICAL_EXEMPT` at the call site,
and the measurement still runs and still lands in their manifest.
Falsifiability checked before trusting the pass: a synthetic
teleporting track is refused, and the refusal names the constraint.

**v6-motion, minted.** 19 clips, 1596 frames.

| gate | result |
| --- | --- |
| GT physicality | **0 violations** / 5 constraints, worst |a| 1.9955 vs 2.0 bound |
| regime volume (Day 23) | cessation **374** (floor 200); static 366, onset 110, sustained 746 (floor 30); maneuver 0, declared exempt |
| observability floor (Day 16) | **1.0000 mean AND min** |
| Day-10 validity matrix row | `motion_geometry` PASS · `state_estimation` PASS · `depth` REFUSED · `appearance_semantics` REFUSED · `point_tracking` PASS |
| determinism | `content_sha` identical across two independent renders (19/19) |

The validity row is identical to v5-cessation's — geometry yes,
appearance no — exactly as the set's purpose predicts.

**Acceleration distribution, v6 against v5:**

| | v5-cessation | v6-motion |
| --- | ---: | ---: |
| max GT \|a\| | **36.5806** (3.7g) | **1.9955** (0.20g) |
| overall p95 | 8.6917 | 1.3149 |
| frames above 1g | 35 (3.43%) | **0 (0.00%)** |
| stop events | 22 | 22 |
| stop events above 1g | **14 (63.6%)** | **0 (0.0%)** |
| median stop-event peak | **15.1313** | **0.7362** |

A **20.6x** reduction in the median stop event, with the same number of
stop events and the same cessation frame count (374 vs 373).

**The first render was REFUSED, and re-authored rather than excused.**
Observability 0.7978 against the 0.80 floor. The cause is a real
consequence of the physics and was not obvious in advance: **physically
bounded motion spends many frames moving slowly, and the measured motion
gate demands a much larger silhouette from a slow mover** (535 gate px
below 0.5 gate px/frame, versus 110 above 1.5). v5 cleared the floor at
greater depths *because* its motion was abrupt — an instantaneous stop
has no slow phase to be unobservable during. **The unphysical set was
easier to see.** Radial clips get it worst: an agent walking toward the
camera moves fast in the world and barely at all in pixels.

`V6_MAX_DEPTH_M = 5.9 m` is the resulting authoring rule, derived rather
than tuned: `sqrt(19000 / 535) = 5.96 m`, from the envelope's worst-case
threshold and the measured silhouette law `area_gate_px ≈ 19000 /
depth²` (which held to within 1.5% across three clips spanning depth
4.1-9.9 m). Its cost is stated: the near/far axis compresses (stops at
~3.4 m vs ~6.0 m rather than ~4.5 m vs ~11 m), and the radial lane is too
short for a 2.0 m/s approach-and-stop — the solver reduced 7 of 19 clips'
cruise speeds, visibly, in the manifest.

## Objective 3 — config B's velocity uncertainty, and what it was hiding

**Config B is a constant.** `fraction_at_floor`, added to
`_sigma_v_distribution` so every report path carries it:

| set | scored frames | pinned at the floor |
| --- | ---: | ---: |
| v6-motion | 1558 | **1558 (100.00%)** |
| v5-cessation | 1021 | 1012 (99.12%) |
| v3-indoor | 2736 | 2733 (99.89%) |

Per regime, every unpinned frame is in `onset` (90.1% and 98.9% pinned) —
the one regime where velocity genuinely changes fast enough that natural
uncertainty already exceeds the floor. `static`, `sustained`, `cessation`
and `maneuver` are at 100.0% on every set.

Day 25 measured the same quantity, reported `min == p50 == max ==
1.5000`, and read it as confirmation that the clamp fires. It is that.
It is also the signature of a filter that has stopped estimating, and
Day 25 had no reason to look for that — it was asking a wiring question.
Config A's natural converged σ_v is 0.2548-0.5954 m/s per regime, so the
floor sits **2.5-5.9x above** what the filter actually reaches. And
1.5 m/s is `PERSON_SIGMA_A_MPS2 × PEDESTRIAN_STOP_DURATION_S` — a
comfortable walking pace, per Day 29 — not a bound.

**A filter that is never confident cannot be caught being
overconfident.** Config B's cessation coverage moving 0.5013 → 0.9946 is
therefore not evidence about cessation modelling. Any sufficiently large
constant produces it.

**And the defect does not exist on physical motion.** Four-way re-run on
v6-motion under the unchanged directional criterion; v5-cessation re-run
in the same invocation and reproducing Day 25/26 exactly, so the two sets
differ only in the data:

| regime | n | A RMSE / cov | B RMSE / cov | C RMSE / cov | D RMSE / cov |
| --- | ---: | --- | --- | --- | --- |
| static | 328 | 0.0839m / 0.9909 | 0.1260m / 0.9970 | 0.0546m / 0.9726 | 0.0535m / 0.9970 |
| onset | 110 | 0.1126m / 0.9727 | 0.1279m / 1.0000 | 0.0714m / 1.0000 | 0.0713m / 1.0000 |
| sustained | 746 | 0.0956m / 0.9879 | 0.1326m / 0.9987 | 0.1380m / 0.9464 | 0.1459m / 0.9584 |
| cessation | 374 | **0.0946m / 0.9920** | 0.1280m / 0.9973 | 0.0747m / 0.9572 | 0.0736m / 0.9920 |

**Config A's cessation coverage on physical motion is 0.9920.** The same
configuration, the same code, the same criterion, measured 0.5013 on
v5-cessation. No-trade verdicts on v6-motion: A→B **NO_IMPROVEMENT**
(cessation Δ −0.0053), A→C NO_IMPROVEMENT (+0.0348), A→D
NO_IMPROVEMENT (+0.0000).

Config B's cost on physical motion is no longer one bounded,
safe-direction deviation. It is worse in **every** regime: static +50.2%,
onset +13.6%, sustained +38.7%, cessation **+35.3%** position RMSE —
including the regime it was adopted for.

**Which conclusions change.** ADR 0010's config A→B adoption is
**withdrawn**; config A is re-adopted, under the criterion exactly as
Day 25 committed it. Day 26's finding that A and B produce materially
different point estimates still stands — it is a Kalman-gain fact, and
today's v6 RMSE deltas are more of the same evidence. What does not stand
is the interpretation: B's difference is a uniform accuracy cost, not a
cessation fix.

**The criterion has a blind spot, and it is separate from this
reversal.** B fails on v6-motion, so the criterion happens to reject it.
It would not have caught the pinning: a constant-uncertainty filter
satisfies both halves trivially — cessation coverage improves and no
steady regime moves toward overconfidence, because nothing moves at all.
The criterion never asks whether the reported uncertainty is
*informative*. Carried to Day 31 rather than patched here, for the same
reason Day 24 declined to redesign a criterion in the session that
discovered it mattered.

**Correction — this day's own Objective 1 was wrong about IMM.** The
Objective-1 ADR revision listed IMM's rejection under "not provisional",
arguing that `static`'s GT acceleration is identically zero on
v5-cessation so IMM's static regression was measured on trivially
physical motion. The v6 re-run refutes it:

| set | A static coverage | C static coverage |
| --- | ---: | ---: |
| v5-cessation | 0.9962 | **0.0808** |
| v6-motion | 0.9909 | **0.9726** |

The error: a per-frame GT regime label describes the *world* at that
frame, not the *filter's state*, and a filter's covariance at frame `t`
is a function of the entire preceding trajectory. IMM's mode
probabilities are explicitly history-dependent, so an impossible
transient contaminates the static frames that follow it, however physical
those frames' own GT is. **Checking that a regime's GT is clean is not
sufficient to establish that a measurement taken during that regime is
clean.** This does not make IMM adoptable — on v6 it is
`NO_IMPROVEMENT` like everything else, and worse than A on `sustained`
RMSE — but the recorded *reason* for rejecting it is no longer supported,
and Days 21-25's IMM rejection is now provisional on the same grounds as
everything else measured on v5-cessation.

## Objective 4 — the GT contract, and the raw-array census

**`src/contracts/ground_truth.py`.** `GtPositionTrack` /
`GtVelocityTrack` / `GtAccelerationTrack` (`[T, 3]`) and `GtPositionClip`
(`[T, A, 3]`), each carrying a required, undefaulted `GroundTruthAxes` —
the same treatment `WorldPosition` gives `twin_rev`, and for the same
reason: there is no sensible default, and a default is exactly how an
unstated convention becomes a silent one.

- **STRUCTURAL: a GT array with no declared convention is
  unconstructable.** `TypeError` on omission; `AxisConventionMismatch` on
  a bare integer, which is the shape the bug actually took
  (`VERTICAL_AXIS = 2`).
- **STRUCTURAL: mixing conventions in one operation raises.** Every
  binary operation goes through `combine`, which raises on disagreeing
  axes. There is no coercion path — an axis permutation is exact and
  lossless, which is precisely why auto-applying one would recreate the
  original bug with more confidence attached. A caller that wants the
  other convention names it, at the call site, via `values_in(...)`.
- **Units are carried by the TYPE**, not a field, so handing a velocity
  to something expecting a position is a `mypy` error rather than a value
  wrong by a factor of `dt`.

**The census — eleven boundaries, nine closed:**

| # | boundary | risk | action |
| --- | --- | --- | --- |
| 1 | `measure_gt_constraint_violations.py` `VERTICAL_AXIS` | **the actual Day-29 bug**; shipped as `2` | **fixed** — the constant is gone; asks `vertical_component()` |
| 2 | same file, `_gt_velocity` | a *third* copy of the difference convention, synced by comment | **deleted** — `differentiate` is the one implementation |
| 3 | `src/data/scorecard.py` | wrapped `agent_xyz` in `WorldPositionArray`, whose module documents +z-up — a **false declaration** | **fixed** — `GtPositionClip` |
| 4 | `src/inspector/artifacts.py` | same false declaration | **fixed** — `GtPositionClip` |
| 5 | `eval_estimator.py` GT read | undeclared | **fixed** |
| 6 | `eval_estimator.py` `axis_rmse_m_xyz` | a per-axis report key naming "xyz" without saying whose | **fixed** — `axis_convention` field; prints `x / y_up / z_depth` |
| 7 | `eval_joint_estimator.py` | undeclared | **fixed** |
| 8 | `measure_acceleration_noise_floor.py` | undeclared | **fixed** |
| 9 | `measure_slip_remedy_smoothing.py` | undeclared | **fixed** |
| 10 | `measure_component_sparsity.py` / `measure_component_cap_cost.py` | pairwise distances only | **left** — permutation-invariant, verified; no component is ever read |
| 11 | `src/estimator/regime.py::classify_track` | takes a raw `[T, 3]` | **left** — uses only norms and dot products, so it is invariant under any axis permutation; changing its signature would touch every caller for no defect |

The invariance claim behind rows 10 and 11 is itself pinned by a test
(`test_magnitude_is_invariant_under_the_permutation`), so "safe because
it only takes magnitudes" is a checked property rather than a reviewer's
assertion.

**A drift guard, from Day 24's lesson.**
`measure_gt_constraint_violations.py` now asserts that the set of
constraints it measures equals `HARD_CONSTRAINTS`. A constraint added to
the registry that never reached that script would be a bound nobody ever
measured against GT — which is how v5-cessation's 3.7g went unrefuted for
eight days.

**What this class of bug costs, restated.** Rows 3 and 4 are the
important ones. Neither was producing a wrong number: both consumers take
norms and apply the generator-frame extrinsics, which is correct. But a
type asserting the *wrong* convention is worse than a bare array, which
asserts nothing — the array makes no claim, while the mislabelled
envelope makes one a future reader is entitled to trust.

## Objective 5 — the skill rule

Added to `.claude/skills/iron-eval-discipline/SKILL.md`: *Noise
Parameters Are Not Physical Bounds*. A process-noise density
parameterizes a distribution with unbounded support; a hard constraint
needs a support boundary. They have compatible units and incompatible
meanings, which is why dimensional analysis — the check reviewers
actually run — passes every time.

Both worked examples are this project's own, and they are the same
constant from opposite sides. Day 29: `PERSON_SIGMA_A_MPS2 ×
PEDESTRIAN_STOP_DURATION_S = 1.5 m/s` used as a maximum speed, violating
on 46.0% of v5-cessation GT frames because 1.5 m/s is a walking pace.
Day 30: the same 1.5 m/s used as a velocity floor, sitting 2.5-6x above
the filter's own converged σ_v and binding on 99-100% of frames — which
converts an estimator into a lookup table that the calibration metric
then applauds.

## Full suite, mypy, lint

`mypy` (scoped per `mypy.ini`): **clean, 0 errors, 63 source files** —
including the new `src/contracts/ground_truth.py`. `black --check` and
`flake8` clean on every file touched today. Two pre-existing `E501`s in
`measure_acceleration_noise_floor.py`/`measure_slip_remedy_smoothing.py`
and two in `scorecard.py` are unchanged from before today (verified
against a stash) and left alone rather than mixed into this diff.

## Still blocked on a human

Per [[iron-blocked-on-humans]]. Unchanged from Day 29 — today's work was
entirely unblocked by design:

1. **Production `models/int8/vjepa2_vitl_int8.xml`/`.bin`** — ADR 0008.
2. **MEVA licence verification** — now 7 days older than Day 23's
   ranking, still the highest-product-impact pending data item.
3. **Counsel review of `docs/site_zero_consent_TEMPLATE.md` §7.**
4. **A physical camera** — now 19 days old.
5. **The `main`/`origin/main` divergence decision** (ADR 0009, Day 18,
   still Proposed).
6. **Reference hardware procurement decision** — now 12 days old.
7. **The carrier/asset production posture** (ADR 0011).

## Day 31, in order

1. **Re-measure every Day 21-29 estimator conclusion on v6-motion.** The
   four-way table is done; the slip models, the noise floor, the joint
   estimator, and the component-cap cost are not. Every one of them was
   measured on v5-cessation or v3-indoor, i.e. on impossible motion or on
   exactly-constant velocity. This is the largest block of provisional
   work in the repository and it is now cheap to close.
2. **The calibration criterion cannot see a constant-uncertainty
   filter.** `_no_trade_verdict` should require that the candidate's
   reported uncertainty be *informative* — e.g. reject a configuration
   whose σ_v is pinned on more than some share of frames — decided on its
   own methodological merits and stated before re-scoring anything, per
   Day 24's ordering rule.
3. **Re-derive the velocity floor, or retire it.** Its implementation is
   correct; its derivation multiplies a noise density by a duration and
   calls the product a bound. Day 22's own open question (a floor from
   the actual stopping deceleration profile) is now answerable —
   v6-motion contains real deceleration profiles for the first time.
4. **Re-examine IMM on physical motion.** Its recorded rejection reason
   does not survive v6 (static coverage 0.9726, not 0.0808), and it has
   the best RMSE in three of four regimes there.
5. **Vertical motion in the golden sets.** GT height is still a constant
   0.86 m, so `gravity_floor_transition` remains a bounded null and no
   floor-transition capability is testable.
6. **Physical `maneuver`.** Needs a curved-path model with a
   lateral-acceleration budget; the regime is exempt on both cessation
   sets and its v5 numbers (p50 9.57 m/s², 46.67% above 1g) were the
   worst in the set.
7. **Hypothesis management itself** (spawn/score/budget-prune) — the
   store, lifecycle, death causes and all five constraint predicates
   exist; nothing decides when to use them.
8. **Real coupled multi-entity data, or a larger authored scene** —
   Day 26/27's open question, still open.
9. **Twin geometry** (`TwinGeometry`) — four typed constraints are
   `Unevaluable` pending it.
10. **The twin drift detector** — sits behind item 9.
11. **The discrete/continuous hybrid** — depends on item 7.
12. **Smoothing across the joint graph** — depends on the coupling's own
    calibration being trustworthy, which now depends on item 1.
13. **MEVA licence verification** — blocked on a human.
14. **DA-2K licence verification** — zero engineering lag once cleared.
15. **Reference hardware procurement decision** — now 12 days old.
16. **Declare a target fps for real camera ingest** — open since Day 19.
17. **Cascade bench, clean, on interim or reference hardware.**
18. **Depth validity re-measurement** — gates on item 14.
19. **The `main`/`origin/main` decision** (ADR 0009).
20. **The generator has no sensor-noise model** — unchanged, and now the
    largest remaining known gap between v6-motion and reality.
21. **Order cameras and run the office capture** — now 19 days old.
22. **The motion-gate precision/selectivity investigation** — deferred,
    and Objective 2 gave it a new datum: the envelope's low-speed
    threshold is what forced v6's depth ceiling.
23. **A canonical `Observation -> hash` function** (ADR 0007).
24. **Decide whether `PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME` should be
    replaced.**
25. **Bridge the live-RTSP path and `scripts/ingest_capture.py`.**
26. **`ConsentRecord` for `thinkwill-cctv-archive`**, if pursued.
27. **The `resolve_joint_state` cross-component filtering gap** (ADR
    0011).
