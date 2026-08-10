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
