# CHIRLA published baselines — Day 38, Objective 4

## EXTERNAL PUBLISHED RESULTS — not measured on this system, cited for calibrating expectations only

**Nothing on this page was run by this project.** Every number below is
copied from CHIRLA's own paper, not produced by `src/identity/bakeoff.py`
or any other code in this repository. It is a citable prior for how hard
long-term, multi-camera, real-clothing-change re-ID actually is —
independent of whether CHIRLA's own data ever becomes usable here (that
question is still open; see `docs/blocker_ledger.yaml`'s
`chirla-deployment-scope-legal-review` and `chirla-license-verification-
confirm` entries).

**Citation:** Domínguez-Dager et al., *Scientific Data*, DOI
`10.1038/s41597-025-06425-3` (accessed via PMC12852133). The paper reports
CMC (Cumulative Matching Characteristic, rank-1 = "CMC@1") and mAP (mean
Average Precision) for roughly 30 re-ID backbones across CHIRLA's
benchmark scenarios.

## Why this table exists, separately from Day 36's bake-off self-test output

`src/identity/bakeoff.py` (Day 36, Objective 4) is this project's OWN
harness, and everything it has ever produced is a self-test —
`SELF_TEST_LABEL`-stamped output against no real data, because CHIRLA has
no `LicenseSnapshot` yet. Mixing a self-test artifact with a table of
externally published numbers would let one be mistaken for the other the
moment either is read out of context — the same discipline this project
already applies to `PromotionResult`'s `scenario_realism` field (Day 36)
and the Coverage/Absence view's `scenario_realism` labelling (Day 37,
Objective 4). This table lives in `docs/`, as prose, never inside
`src/identity/bakeoff.py`'s output or any artifact that harness writes.

## The headline numbers

| finding | value | model | source |
|---|---|---|---|
| Long-term re-ID, best model tested — CMC@1 | **18.81%** | ResNet101 | paper, long-term scenario table |
| Long-term re-ID, best model tested — mAP | **23.24%** | ResNet101 | paper, long-term scenario table |
| Reappearance — CMC@1 | **65.73%** | ResNet101-IBN | paper, reappearance scenario table |

Long-term re-ID (matching a person's identity across a 7-month recording
window, real clothing changes included) is the **hardest** of CHIRLA's
scenarios by a wide margin — even the best of ~30 tested backbones tops
out under 19% CMC@1. Reappearance (re-identifying a person after they
leave every camera's view and later re-enter, without necessarily
crossing the full 7-month window) is **comparatively easy** — 65.73%
CMC@1, more than 3.5x the long-term figure. Multi-camera long-term (the
combination of simultaneous multi-view re-ID and the long-term window) is
reported as the **second-hardest** scenario, between these two; this
document does not carry that scenario's specific CMC/mAP figures because
they were not supplied for citation here, and inventing a number to fill
the gap would be exactly the kind of unvalidated figure this project's
eval discipline exists to prevent.

CHIRLA's own benchmark split names four scenarios in total (`long_term`,
`multi_camera`, `multi_camera_long_term`, `reappearance` — see
`configs/datasets.yaml`'s CHIRLA `validity_matrix_cell`); this table
carries confirmed figures for two of the four (`long_term`,
`reappearance`) plus a qualitative ranking for a third
(`multi_camera_long_term`), and no figure for `multi_camera` alone.

## The sanity check this becomes, once real

`chirla-license-verification-confirm` (the ledger's fast item) unblocks
CHIRLA for internal benchmarking; `chirla-deployment-scope-legal-review`
(the ledger's slow item) is a separate, independent gate on anything
beyond that. If and when this project's own bake-off harness eventually
runs against CHIRLA for real, its long-term-scenario mAP is the number to
compare against this table's **23.24%** — a measured result landing
**wildly above** 18.81% CMC@1 / 23.24% mAP on the long-term scenario
should trigger suspicion of a methodology error (wrong split, train/test
leakage, a metric computed differently than the paper's) **before**
celebration, not after. This is the same discipline
`src/data/scorecard.py`'s `SUPERSEDED_GATE_MEASUREMENTS` applies to this
project's own numbers, aimed here at a number this project has not
produced yet.

**No timing, throughput, CPU, or latency claim is made anywhere in this
document** — same hard scope rule as every other page in this project;
CMC/mAP are accuracy metrics, not performance ones, and nothing here
changes that boundary.
