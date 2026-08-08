# Office capture runbook

Day 16. The single remaining obstacle to a real Tier-1 economic claim is 24
hours of real office footage, including nights and weekends — v3-indoor and
v4-gate are both synthetic and both say so on every scorecard they produce
(see `docs/day16/motion_gate_v3_rescored.json` /
`motion_gate_v4_gate.json`). This is the exact, ordered sequence for the
capture that produces that footage, so that day is a checklist, not an
improvisation.

Every command below has been dry-run against local fixtures — no camera —
in this repository. `scripts/discover_cameras.py` was run against the real
local subnet and reported honestly (0 devices, exit 0). The rest of the
chain (RTSP connect → decode → content-addressed store → registry entry,
lane C → Gap records on a dropped frame → condition tagging → consent
refusal) was exercised end-to-end against
`tests/fixtures/minimal_rtsp_server.py` by `scripts/capture_dry_run.py`, all
six stages passing. **What remains manual is everything that genuinely
needs a camera in a room**: placement, the walkthrough, and the overnight
window itself.

---

## 0. Before anyone shows up with a camera

1. **Consent, per participant, before any recording.** Use
   `docs/site_zero_consent_TEMPLATE.md` — **marked DRAFT, not legally
   reviewed**; counsel review of §7 (the erasure limitation) is still
   [[iron-blocked-on-humans]] item 4. Do not record anyone without a signed
   record; `scripts/ingest_capture.py` refuses ingestion without one
   (`ConsentRecord.load`), and that refusal is the feature, not a bug to
   route around.
2. **Recruit beyond the immediate team.** Per Day 5's finding, consent from
   people who all already know each other and know they're being measured
   is weak evidence of freely-given consent and a poor sample of a real
   deployment.
3. Confirm hardware is on hand per the Day-5 shopping list
   (`FOUNDATION_REPORT.md` §D5.4): PoE IP cameras (both 1080p and 720p —
   the product claims both), a PoE switch, adjustable mounts, an NVR or
   recording host with several TB free.

## 1. Camera placement — a measurement decision, not a convenience one

Three placement requirements are non-negotiable, because each one is the
only source of a specific kind of ground truth and none can be retrofitted
by annotation after the fact:

| # | Placement | Why | Condition tag(s) to record |
|---|---|---|---|
| 1 | **One overlapping pair** — two cameras with a shared field of view | The only source of cross-camera association GT | `single_person` / `crowded` as applicable |
| 2 | **One deliberate coverage gap** between two zones — a corridor stretch or doorway no camera covers | The source of the handoff/`observed=false` inferred-event cases the event schema is built for (Day 13's `Coverage.Gap`) | tag the adjacent covered segments; the gap itself is inferred from absence, not tagged |
| 3 | **One long-corridor view at the envelope's far edge** | Populates `Condition.FAR_FIELD` — "subject at the far edge of the capability envelope" (`src/data/golden.py`) — the regime the measured envelope curve (`src/cascade/envelope.py`) is thinnest on real data | `far_field` |

Mount at realistic CCTV height and angle (per Day 5: "not desk height").
Record each camera's resolution, mount height, and approximate coverage
polygon in the capture log before recording starts — this is what lets a
clip be labelled with `covers` the same way the synthetic sets are.

## 2. The overnight / empty segment — do not skip this

**This is the segment the Tier-1 economic claim depends on, and the one
most likely to be skipped as boring.** Day 15 found that v3-indoor cannot
evaluate the motion gate because its `moving_frame_fraction` is 0.9667 —
every synthetic set built so far, including v4-gate, is authored, and
authored quiet is not the same as real quiet (v4-gate's own scorecard says
so — see `docs/day16/motion_gate_v4_gate.json`'s dataset note). A real
empty corridor has sensor noise, HVAC-driven shadow drift, and compression
artifacts that no synthetic scene in this repository has ever produced.
Those are exactly what a background-subtraction gate reacts to, which
means they are exactly what this capture has to contain to mean anything.

- **Minimum window: one full overnight period** (e.g. building close to
  building open) on all cameras, continuously, with nobody in frame for
  the great majority of it.
- **Also cover at least one full weekend day**, if the schedule allows —
  the product's claim is about 24 hours including nights *and* weekends,
  not one representative overnight.
- Record whatever genuinely happens: a cleaner passing through, HVAC
  cycling the lights, a delivery at a side door. Do not stage anything —
  the entire point is the footage nobody scripted.
- Condition-tag the segment `empty` for the stretches with nobody present,
  and add whatever else actually occurred (`lights_transient`,
  `evening_artificial`) rather than leaving it as one undifferentiated
  block.

## 3. During the scripted sessions

For the daytime, occupied portion (mirrors Day 5's volume targets: 8–12
participants, 30–40 scripted sessions):

- Script the same walkthrough patterns the synthetic sets encode where
  possible — entries, exits, occlusion by furniture, an agent crossing
  between the overlapping pair's two fields of view — so the real footage
  is comparable to, not just adjacent to, v2/v3/v4-gate.
- Tag conditions per segment as they're recorded: `daylight` /
  `evening_artificial`, `single_person` / `crowded`,
  `occlusion_crossing` / `occlusion_furniture` where staged, `glare` if a
  window catches direct sun during a session.
- Note any camera bump, lens smudge, or other physical incident
  (`camera_bump`, `lens_smudge`) — these are conditions the taxonomy
  already has tags for and a real capture is the first chance to populate
  them.

## 4. Commands, in order

```bash
# 1. Discover — expect 0 devices on most networks; that is a clean,
#    honest result, not a failure (exit code 0 either way). If a camera
#    answers ONVIF, its RTSP URI and stream profiles print with
#    credentials masked.
python scripts/discover_cameras.py --out outputs/capture/discovered_cameras.json

# 1b. If discovery finds nothing (the common case) or a device's ONVIF
#     Media service doesn't answer, enter the RTSP URL by hand — read it
#     off the camera's own admin page, never guessed from a vendor
#     template:
python scripts/discover_cameras.py --manual \
    --host <camera-ip> --rtsp-url rtsp://<camera-ip>/<stream-path>

# 2. Verify the dry-run chain still passes on the machine that will do the
#    real capture, before driving to the office. This exercises RTSP
#    connect, decode, content-addressed store, Gap detection, and consent
#    refusal against the local fixture -- no camera needed for this step.
python scripts/capture_dry_run.py

# 3. Record. However the NVR/recording host captures RTSP to disk as
#    files (this repository's live-RTSP-to-store path is a dry run today,
#    not yet a single production script -- see the caveat below).

# 4. Ingest each recorded file, per session/segment, with its consent
#    record and condition tags. Ingestion REFUSES without --consent; that
#    refusal is deliberate.
python scripts/ingest_capture.py \
    --source data/raw/office_capture_v1/<segment> \
    --consent docs/consent/<participant-or-segment>.signed.json \
    --conditions daylight single_person \
    --store data/captures/office-capture-v1
```

## 5. What remains manual

Discovery, RTSP transport, decode, content-addressing, registry shape (lane
C), Gap accounting, and condition tagging are all built and dry-run tested
against fixtures — zero hardware needed to trust that this machinery works.
What is not built, and does not need to be for a one-afternoon capture, is
a single script that goes directly from a live RTSP stream to the
content-addressed store (`scripts/capture_dry_run.py` demonstrates every
piece of that chain works; `scripts/ingest_capture.py` is the file-based
path that actually writes the store today). For this capture: record to
files with whatever NVR/recording host is on hand, then run
`ingest_capture.py` per file. Bridging the two into one live script is real
work a real camera would motivate, not a blocker to running the capture
itself.

Everything else is physically manual by nature: camera placement, running
the cabling, walking the scripted sessions, and — the part this runbook
exists to make sure does not get skipped — leaving the cameras running,
untouched, overnight.
