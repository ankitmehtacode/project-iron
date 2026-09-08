# Registry deployment-scope audit — Day 38, Objective 2

## Why this exists

Objective 1 found that CHIRLA's HuggingFace dataset card carries an
"Out-of-Scope Use" clause that appears on **neither** its GitHub README nor
its published paper. That clause was invisible to this repository until a
human went and read `huggingface.co` directly — Claude Code's network access
does not reach `huggingface.co`, `sciencedb.cn`, or `arxiv.org` (confirmed
directly: `docs/chirla_verification_checklist.md`, Day 36).

Every dataset in `configs/datasets.yaml` was registered from whatever this
environment could reach — in practice, almost always GitHub only. If CHIRLA
had a deployment-scope clause sitting on a host this project could not read,
any other registered dataset that is *also* hosted (or documented) somewhere
outside this environment's reach has the same structural blind spot, and
**none of them have ever been checked for it.**

## Method, and its limit

This is a **triage pass, not a re-verification pass** — no external host was
fetched or read to produce this document, per the Day-38 prompt's explicit
instruction. The ranking below comes from two things already recorded in
`configs/datasets.yaml` itself:

1. **`consent_posture`.** An entry marked `public_cctv_no_consent` is, by
   definition, real footage of real, non-consenting people captured by
   surveillance-style cameras — exactly the category CHIRLA's clause names
   ("surveillance, identification, or monitoring of real people"). These are
   the entries most likely to carry a similar clause on whatever page
   actually hosts them.
2. **`subsystem`.** `reid`, `tracking`, `gait`, and `anomaly` entries are
   product-relevant to re-ID/surveillance claims in a way `depth`, `twin`,
   and pure `detection`/`hoi` entries are not — matching the Day-38 prompt's
   own example ranking (MEVA, MMPTRACK, ChokePoint, PRW, CUHK-SYSU, WILDTRACK
   above DA-2K/ETH3D).

**What this method cannot do:** with the single exception of CHIRLA, no entry
in this registry records an actual hosting URL — `hypothesis_class` and
`notes` describe licensing beliefs and provenance in prose, not where the
bytes live. So this triage is priority ranking from what a person would
already know about these named, publicly-known datasets, filtered through
what the registry itself records about consent posture and subsystem — it is
**not** a claim that any specific entry below is confirmed to be hosted
outside this environment's reach, only that it is plausible enough to be
checked before something with a weaker prior. Closing that gap for real is
exactly what `registry_deployment_scope_reaudit` (Objective 3) is for.

## Scope: 63 registered, 50 triaged

- **63** total entries in `configs/datasets.yaml`.
- **9** excluded — this project's own first-party captures/synthetic sets,
  which have no third-party terms to re-read: `synthetic-indoor-v1`,
  `synthetic-indoor-v3`, `synthetic-indoor-v4-gate`,
  `synthetic-indoor-v4.1-gate`, `synthetic-indoor-v5-cessation`,
  `synthetic-indoor-v6-motion`, `site-zero`, `office-capture-v1`,
  `thinkwill-cctv-archive`.
- **3** excluded — permanently blocked (`DukeMTMC`, `DukeMTMC-reID`,
  `MS-Celeb`): a deployment-scope clause is moot on a dataset that is never
  fetched, retracted or not.
- **1** excluded — `CHIRLA` itself, the confirmed case Objective 1 already
  handled; it is the reference point for this audit, not an item on it.
- **50** remain, triaged into three tiers below.

## Tier 1 — highest priority for a human re-read (21 entries)

Re-ID/tracking/gait/anomaly subsystem, and/or `consent_posture:
public_cctv_no_consent` — the same shape of dataset CHIRLA is, real people
captured on camera without a documented consent process, exactly the
category a surveillance-specific out-of-scope clause would target.

| dataset | subsystem | consent_posture | why it ranks here |
|---|---|---|---|
| MEVA | activity | staged_actors | Explicitly named in the Day-38 prompt; single highest product-impact pending item on the whole registry (`docs/blocker_ledger.yaml`); multi-camera surveillance-style footage. |
| MMPTRACK | tracking | staged_actors | Explicitly named; multi-camera indoor tracking, closest subsystem match to this product. |
| WILDTRACK | tracking | public_cctv_no_consent | Explicitly named; real, uncontrolled pedestrian traffic. |
| ChokePoint | reid | public_cctv_no_consent | Explicitly named; re-ID benchmark, real subjects. |
| PRW | reid | public_cctv_no_consent | Explicitly named; person-search, real subjects. |
| CUHK-SYSU | reid | public_cctv_no_consent | Explicitly named; person-search, real subjects. |
| MEVID | reid | staged_actors | Clothing-change re-ID, inherits its posture from MEVA — same blind spot as MEVA. |
| Market-1501 | reid | public_cctv_no_consent | Canonical re-ID benchmark, scraped pedestrians. |
| MSMT17 | reid | public_cctv_no_consent | Canonical re-ID benchmark, scraped pedestrians. |
| i-LIDS | activity | staged_actors | UK Home Office AVSS surveillance benchmark. |
| PETS2009 | tracking | staged_actors | Multi-camera surveillance-workshop benchmark. |
| CAVIAR | activity | staged_actors | EU surveillance-camera project dataset. |
| UBnormal | anomaly | unknown | Anomaly-detection-in-video corpus; "surveillance/monitoring" is the use case by construction. |
| ShanghaiTech-Campus | anomaly | public_cctv_no_consent | Real campus surveillance footage of real, uncontrolled pedestrians (Day 35 finding). |
| CUHK-Avenue | anomaly | public_cctv_no_consent | Real fixed-camera footage of a real campus avenue (Day 35 finding). |
| UCF-Crime | anomaly | public_cctv_no_consent | Real surveillance footage, crime-related — highest-sensitivity content in the registry. |
| UCSD-Anomaly-Detection | anomaly | public_cctv_no_consent | Real pedestrian-walkway surveillance footage. |
| CASIA-B | gait | staged_actors | Gait recognition — biometric-identification-adjacent by construction. |
| OU-MVLP | gait | staged_actors | Gait recognition, application-gated distribution (own flag already in the registry). |
| GREW | gait | unknown | Gait recognition "in the wild" — real, uncontrolled subjects; application-gated. |
| Gait3D | gait | unknown | Gait recognition, real subjects. |

## Tier 2 — medium priority (4 entries)

Re-ID subsystem but synthetic subjects — the deployment-USE clause CHIRLA
carries is about real people, so these are less likely to carry the identical
clause, but they are still re-ID/surveillance-product-relevant enough, and
distributed through the same application-gated / non-GitHub channels
(per the registry's own "application-gated" hypothesis_class flag on
adjacent entries), to be worth a cheap re-read before the Tier-3 sets.

| dataset | subsystem | why medium, not low |
|---|---|---|
| RandPerson | reid | Synthetic re-ID; lane S conditional on asset clearance (SMPL trap) — a use-scope clause is a different question from the asset-license one already tracked. |
| UnrealPerson | reid | Synthetic re-ID, same shape as RandPerson. |
| ClonedPerson | reid | Synthetic re-ID, same shape. |
| PersonX | reid | Synthetic re-ID, same shape. |

## Tier 3 — lower priority (25 entries)

Activity/HOI/detection/depth/geometry sets — per the Day-38 prompt's own
example (DA-2K, ETH3D), a pure geometry or object-detection benchmark is far
less likely to carry a surveillance-specific out-of-scope clause than a
re-ID/tracking/gait/anomaly set of real people. Listed for completeness, not
because the risk is zero — `registry_deployment_scope_reaudit`'s effort
should still reach these eventually, just after Tiers 1–2.

OA18, NTU-RGBD-120, Toyota-Smarthome, InHARD, MECCANO, Charades, VIRAT,
Something-Something-v2, EPIC-KITCHENS, COCO, CrowdHuman, PeopleSansPeople,
DA-2K, ETH3D, iBims-1, DIODE-indoor, NYUv2, SUN-RGBD, Hypersim, ScanNet,
Matterport3D, HM3D, Kubric, Infinigen-Indoors, Infinigen-Indoors-depth-candidate.

## Count, for the Blocker Ledger

**50 datasets triaged; 21 Tier 1, 4 Tier 2, 25 Tier 3 — 0 re-read yet.** This
is Objective 3's input, not a solved problem: see
`docs/blocker_ledger.yaml`'s `registry_deployment_scope_reaudit` entry.
