# ADR 0008 — The production V-JEPA2 artifact's provenance is permanently lost

- **Status:** Accepted
- **Date:** 2026-08-07
- **Decides for:** how to record and close a forensic objective that
  cannot be answered, and what process change stops the next one from
  reaching this state
- **Supersedes:** nothing — no prior ADR addressed artifact provenance
- **Related:** [[iron-blocked-on-humans]] item 2 (production artifact,
  first surfaced Day 6, ~12 days old at close); [[iron-data-model-day13]];
  ADR 0007 (evidence commitments — the claims-layer analogue of "a record
  that outlives what it was built from")

## Context

Day 3 (`FOUNDATION_REPORT.md`, "Day 3 (resumed, post-amendment)") opened
an objective — internally "Objective 1" — to produce a forensic verdict
on `models/int8/vjepa2_vitl_int8.xml`/`.bin`: the artifact production
actually ran inference against. The question was concrete and answerable
in principle — does the production artifact match the reference model's
token geometry (tubelet = 2, confirmed three independent ways at the
reference level in D3.5) and temporal alignment, the same way the Day-3
scripted export at `models/export/2026-07-31/` was verified to — but it
requires the production `.xml`/`.bin` files themselves, from whoever ran
production. They were never supplied. The objective was marked **PARKED**
at Day 3 and re-surfaced in every subsequent day's blocked-list
(`[[iron-blocked-on-humans]]`) without resolution: 8 days old at Day 6,
11 days old at Day 14, and by today, Day 15, still zero movement.

A fresh export was deliberately never substituted for the missing
artifact — `src/config.py`'s `PathsConfig.vjepa_xml` docstring states
the reserved-path rule directly: "Only the artifact that production
actually ran may ever occupy it... A freshly exported model placed here
would silently become 'the production artifact' in every later forensic
comparison, and the real one can never be reconstructed." That
discipline was correct and is not what this ADR revisits. What this ADR
addresses is the status of the *question itself*, which has been carried
as "pending" for twelve days on the tacit assumption that the artifact
would eventually arrive and the verdict would eventually be computed.

That assumption has no basis. Nothing about engineering effort resolves
this — it was blocked on a person on Day 3 and remains blocked on the
same person today. Carrying it as "pending" indefinitely misrepresents
its status: a pending item implies forward motion is possible from
inside this repository, and none is. The 22 artifacts under `outputs/`
that this objective's verdict would have retroactively validated or
condemned (`scripts/rebuild_index.py --audit`: "0 artifact(s) carry
provenance, 22 are void or unverifiable", first measured Day 3, D3.8)
inherit the same problem: they are recorded as `void or unverifiable`,
which — like "pending" — still reads as a state that resolves once the
missing input shows up. It does not. Those 22 artifacts were produced by
a pipeline stage (the pre-normalization-fix V-JEPA2 path, D3.4: p1 cosine
0.331903 against the official reference) that itself no longer exists in
this form; even if the production `.xml`/`.bin` arrived tomorrow and
turned out to match the reference exactly, that would say nothing about
what process actually produced those 22 specific files, because they
were never regenerated under a manifested export and their generating
run left no record to forensically re-examine.

## Decision

1. **The parked forensic objective is closed, unanswered, not
   abandoned.** What it was: a side-by-side verdict — production artifact
   vs. the 2026-07-31 reference export — on token count and temporal
   alignment. What is actually known: the tubelet=2 / 392-token geometry
   is confirmed at the *reference and Day-3-export* level, three
   independent ways (D3.5) — checkpoint config, a live reference forward
   pass, and the export's own token-count abort check. None of those three
   confirmations touch the production artifact. **The tubelet and
   temporal-alignment questions are answered only for the Day-3 scripted
   export; the production artifact's geometry, and whether it carries the
   same normalization fix D3.4 measured (p1 cosine 0.331903 → 0.999987),
   remain and will remain unknown.** This is recorded here as the
   objective's terminal state, not as a status that further code can move.

2. **The 22 void artifacts are unattributable, not pending.** Every
   record and registry entry that currently reads `void`,
   `unverifiable`, or `pending_verification` for these 22 files
   (`outputs/*.npy` depth maps, `tracks.npy`, `visibility.npy`) is
   corrected to **`unattributable`**. The distinction: `pending` says a
   future action resolves the state; `unattributable` says no action
   can, because the run that produced them kept no manifest and the code
   path that produced them has since changed. They remain permanently
   refused by the artifact guard (`src/artifacts.py::require_compatible`)
   — this ADR changes their recorded status, not their refusal.

3. **Process change: every shipped artifact carries an export manifest;
   an unmanifested artifact is refused at load.** This is what actually
   prevents a second occurrence of this ADR. `scripts/export_vjepa_ov.py`
   already writes `export_manifest.json` beside every export it produces,
   naming `source_checkpoint` (path + per-file sha256), the artifact's own
   sha256, `preprocess_sha`, and a golden-vector verification block — see
   `models/export/2026-07-31/export_manifest.json` for the shape. What
   was missing was enforcement on the *read* side: nothing stopped an
   artifact with no manifest — exactly the state the production
   `.xml`/`.bin` would be in if it appeared tomorrow with no accompanying
   record of what checkpoint produced it — from being loaded and treated
   as trustworthy anyway. That gap is closed in the same change as this
   ADR: `src/provenance.py::require_export_manifest` now runs inside
   `SemanticExtractor.__init__`, before `ov.Core().compile_model()`, and
   raises `ManifestError` if `export_manifest.json` is absent from beside
   the model file or does not name that file among its recorded
   artifacts. A production artifact that shows up without export
   provenance is refused at load, not silently trusted because it landed
   on the reserved path.

## Consequences

- Nobody needs to re-ask "is Objective 1 still pending" in a future
  day's report — it is answered here: no, it is closed, and the answer
  is that the question cannot be answered with what exists. The human
  blocker (production `.xml`/`.bin` from whoever ran it) stays open in
  `[[iron-blocked-on-humans]]`, because the artifact could still arrive —
  but its arrival now answers a *new* forensic question about that file,
  not a resumption of this one, since this one's context (the 22 void
  artifacts it would have validated) is already permanently
  unattributable regardless.
- If the production artifact does eventually arrive, it must go through
  the same `require_export_manifest` gate as everything else — meaning
  it will be refused unless whoever supplies it also supplies its export
  manifest. This is intentional: a bare `.xml`/`.bin` with no
  accompanying record of what checkpoint and preprocessing produced it
  is exactly the situation this ADR exists to stop recurring, regardless
  of whether the file sits on the reserved production path.
- `require_export_manifest` only guards the V-JEPA2 IR load path
  (`SemanticExtractor.__init__`). Other model artifacts loaded elsewhere
  in the pipeline (CoTracker3's `.pth`, Depth-Anything-V2's weights) are
  not yet covered by this specific gate and are tracked as a Day-16 item
  rather than claimed done here.

## Open questions

- Whether a partial forensic verdict is possible without the full
  artifact — e.g., if whoever ran production can supply just the
  artifact's sha256 and export log without the multi-gigabyte file
  itself — has not been asked of them. Worth trying before the next
  status carry-forward, since it is a smaller ask than the full binary.
