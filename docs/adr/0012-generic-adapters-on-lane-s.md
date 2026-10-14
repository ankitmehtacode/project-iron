# ADR 0012 — Generic re-ID adapters may train on lane S

- **Status:** Proposed
- **Date:** 2026-10-14 (Day 41)
- **Decides for:** whether a re-identification adapter that enrols no one
  may be trained on lane-S (synthetic, commercially cleared) data, and how
  that adapter relates to the person-specific adaptation that stays lane C
- **Amends:** ADR 0001 (identity capability lives in small per-deployment
  adapters over frozen backbones; checkpoint manifests accept lane C only)
- **Related:** `src/identity/checkpoint.py` (`write_checkpoint_manifest`,
  lane-C only; append-only history since Day 41); `src/identity/lane_gate.py`;
  `src/data/registry.py` (lanes, `DeploymentScopeRestriction`, the SMPL
  trap); `configs/datasets.yaml` entry `PhysicalAI-SmartSpaces`;
  `docs/blocker_ledger.yaml` entries `smartspaces-nvidia-terms-of-service`,
  `smartspaces-smpl-provenance`, `adr-0012-acceptance`

> **Claude Code must not change the status line.** Acceptance is a human
> decision, recorded by the human editing `Status:` to `Accepted` (or
> `Rejected`) — see `docs/blocker_ledger.yaml`'s `adr-0012-acceptance`.

## Context

ADR 0001 keeps identity capability out of shared weights: backbones are
frozen and trained only on lane S/C non-identity data, and everything that
can recognise a person lives in a small per-deployment adapter whose
checkpoint manifest accepts lane-C (consented, self-collected) data only.
That rule was written when no lane-S re-ID data existed, so "re-ID adapter"
and "person-specific adapter" were the same thing in practice.

PhysicalAI-SmartSpaces (registered and fetched 2026-10-14) changes that.
It is lane S — synthetic, CC-BY-4.0 per its card — with cross-camera-
consistent object ids over 23 fetched scenes and 481 person entities
(integrity report, Day 41). That is enough to learn *how to re-identify*
— which appearance cues survive a viewpoint change — without learning
*who anyone is*: the characters are rendered assets, and none of them is a
person this system will ever be deployed to recognise.

The question is whether that generic skill may be learned on lane S, with
only the deployment-specific part (the people a site enrols) left to
lane C.

## Decision (proposed)

Generic re-ID adapters with **no enrolled people** may train on lane-S
data. Person-specific adaptation stays lane C, and every person-specific
adapter records its generic parent, so withdrawal of a person's consent
retrains only the top (site) layer, never the generic one.

### Conditions

1. **Generic adapters only.** An adapter trained on lane S enrols no one
   and is never the artifact that recognises a deployed person. It is a
   separate artifact type from the site adapter, written by its own
   manifest writer; `write_checkpoint_manifest` keeps refusing anything
   outside lane C.
2. **Selection on SmartSpaces validation data only.** Hyperparameters and
   checkpoint selection for a generic adapter use SmartSpaces' own
   validation scenes (Warehouse_020–022) and nothing else — **never
   CHIRLA**. Tuning against CHIRLA turns its evaluation into training: the
   number would then measure fit to the benchmark, not generalisation to
   real people. CHIRLA stays a held-out, internal benchmark (lane R,
   `deployment_scope_restriction` unresolved).
3. **Promotion stays blocked** while either of these is unresolved: the
   NVIDIA terms-of-service clause on the SmartSpaces card (recorded as an
   unresolved `deployment_scope_restriction`; the terms are not linked from
   the card), and the SMPL-provenance question (whether any SmartSpaces
   character asset derives from SMPL, whose licence is not commercial).
   Both carry into any checkpoint manifest through
   `unresolved_restrictions`, and a manifest with any open restriction
   refuses `promoted=True`.

### What a site adapter records

The person-specific adapter's manifest names its generic parent (artifact
sha and manifest path). On a withdrawal, only the site layer retrains;
the generic parent is untouched because it never saw the person. The
erasure drill is re-run against the two-layer arrangement before any site
adapter built this way is promoted.

## Consequences

- The re-ID bake-off can include a generic adapter trained on SmartSpaces,
  evaluated on CHIRLA against the Day-40 floors (chance, colour).
- New code is needed before any of this trains: a generic-adapter artifact
  type and writer, the parent link in the site manifest, and a re-run of
  the erasure drill. None is written by this ADR.
- Known data caveats that a generic adapter would inherit: five of the 23
  fetched scenes are Cosmos Transfer re-renders with byte-identical ground
  truth (train/val must split by source scene, not by scene name); the
  card's ground-truth key spelling differs from the files (the strict
  reader handles this); and loaded Transporters' 2D boxes include their
  payload (Day 41 reprojection), which matters for any crop-based training.

## Not decided here

- No training run. Day 41 runs no SmartSpaces → CHIRLA experiment.
- Whether the NVIDIA terms or SMPL provenance permit product use — those
  are counsel and verification questions on the ledger.
