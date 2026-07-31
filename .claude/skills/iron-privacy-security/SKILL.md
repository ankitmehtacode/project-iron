---
name: iron-privacy-security
description: Privacy architecture, identity-tier boundaries, biometric-data law, audit logging, and platform security for project-iron's monitoring product. MUST be consulted when writing anything that touches identity (session IDs, enrollment, re-ID, gait), person data storage/retention, query/export of footage or events, network-facing code, RTSP/camera ingest, update mechanisms, or Tier-2/Tier-3 features. Triggers on: "identity", "enrollment", "re-ID", "gait", "employee", "retention", "export", "RTSP", "camera stream", "update", "auth", "who queried", "delete data". A surveillance platform that gets this wrong ends the company; these rules are not negotiable per-feature.
---

# Iron Privacy & Security — The System That Watches Must Be Watchable

## Identity Tier Boundaries (architectural, not configurable)

- **Tier 1 is strictly anonymous.** Session-scoped IDs only; no enrollment code paths compiled
  into T1 builds; no cross-session identity persistence. This is what makes T1 zero-friction
  legally — do not "helpfully" add identity features behind a flag.
- **Enrolled identity (T2/T3)** requires: consented enrollment flow, per-employee opt-out
  handling, retention limits enforced BY THE PLATFORM (a customer promise is not enforcement),
  and the customer as data controller. Movement/gait-based identification of enrolled people
  is biometric processing (DPDP 2023; GDPR Art. 9 if EU) — any new identity signal gets a
  legal-review checkbox in its PR before merge.
- **Gait never identifies alone.** It enters the fusion as one weak weighted signal. Code that
  fires an identification from a single-signal gait match is a defect regardless of its
  measured accuracy.
- Identity output is always `(EntityRef, confidence)` with full signal provenance. Revocation
  exists and propagates (see iron-events corrections).

## Data Handling Law

- Video stays on-prem by default; cloud gets metadata + event clips per tier policy. Continuous
  video streaming off-site is a design decision requiring explicit sign-off, never a
  convenience.
- Retention is per-class (video / events / identity separately), config-enforced, with deletion
  actually verified by test. T1 default: faces blurred in stored clips.
- **The audit log audits itself**: every identity resolution, every human query, every export,
  every admin action → immutable audit records. Building a query path that skips audit logging
  is a P0. For T3 customers this self-surveillance is a selling point; treat it as product,
  not overhead.
- No person data in URLs, filenames, or log lines. Logs carry EntityRef ids, never names.

## Platform Security Law

- **Every RTSP stream is hostile input.** Cameras are the most-compromised device class in
  existence. Decoders run sandboxed/hardened; camera VLAN is segmented from inference network;
  fuzz the ingest path.
- Edge boxes: signed updates only, secure boot where hardware allows, per-site key isolation
  (one compromised site yields nothing about another), stores encrypted at rest.
- T3 (sovereign): fully air-gapped, no external runtime dependency (local LLM, in-rack
  training), reproducible builds with SBOM, hardware attestation. A T3 feature that phones
  home for anything is rejected at design time.
- Secrets never in code, config files in git, or manifests. Pen test before T2 GA; red team
  before any T3 trial — schedule them, they are release gates.

## When Uncertain

If a feature's privacy posture is unclear, the default is the more restrictive interpretation
plus a flagged question in the PR — never the permissive interpretation plus silence.
