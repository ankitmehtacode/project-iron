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
