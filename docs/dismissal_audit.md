# The dismissal audit (Day 32, Objective 1)

**A dismissal is a hypothesis and carries the same evidence bar as any
other claim. A diagnosis that lets you skip work is the one to test
hardest.**

Eleven findings before Day 31 were instruments producing WRONG output;
the fix each time was a better instrument. The two found on Day 31 were
instruments producing CORRECT output that a human explained away. This
document is the census of that second class: every place in
`FOUNDATION_REPORT.md`, the ADRs, and the code where an instrument,
test, check or gate fired and the response was a dismissal rather than a
measurement.

Method: regex sweep over `FOUNDATION_REPORT.md`, `docs/adr/*.md`,
`src/`, `scripts/` and `tests/` for `flake`, `flaky`, `contention`,
`transient`, `known issue`, `pre-existing`, `expected`, `unrelated`,
`harmless`, `out of scope`, `not a regression`, `intermittent`,
`environment`, `spurious`. 115 candidate lines, of which most are
scoping declarations ("X is out of scope for this objective", stated in
advance) or descriptive uses ("two unrelated histories") rather than
dismissals of a firing instrument. **12 are genuine dismissals** and are
classified below.

Classification:

- **measured** — backed by an actual measurement at the time. Cited.
- **asserted** — plausible mechanism, no measurement. This is the class
  that produced both Day-31 defects.
- **still-open** — dismissed and never revisited.

## The table

| # | Dismissal | Where | Class | Re-tested today | Outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | "Confirmed CPU-contention flake, not a regression" — a suite failure at 17-18s | Day 29, report §7919-7925 | **asserted** | yes (Day 31) | **REFUTED.** Root cause: `test_stationary_v5_cessation_agent_silhouette_is_bit_identical` (Day 23) and `test_stationary_v6_motion_agent_silhouette_is_bit_identical` (Day 30) rendered every held frame (~46-48 of them at 1280x720) instead of sampling. Both were correctly unmarked when Day 24 measured them (≤5.39s then, per that commit's own measurement) and grew slow later without being re-flagged — not the same defect as #3, which was already closed by Day 24. 20.7s and 18.4s reproduced in isolation with no contention. Fixed Day 31 by sampling first/middle/last held frame. |
| 2 | Background run reported "completed, exit code 0" while carrying a failure | Day 31, harness | **asserted** | yes (Day 31) | **REFUTED** on reading the actual output. Objective 2 makes exit codes structurally insufficient. |
| 3 | "`tests/test_synthetic_indoor.py`'s rendering tests are not marked `@pytest.mark.slow`… **Not fixed today (out of scope)** — flagged so it does not go unnoticed" | Day 23, report §5893-5899 | still-open (**correction, see below**) | yes | **This entry is not a Day-32 finding — the draft's first pass misdated it.** `git blame` shows the 13 tests were marked `@pytest.mark.slow` and the duration-threshold gate added to `conftest.py` on **Day 24** (commit `0abd8c7`), one day after being flagged, not "open 8 days" and not closed "today." Re-verified today that the fix still holds: `pytest -m "not slow"` on this file runs 37 passed / 13 deselected. It is **not** the root of #1 and #2 — see the corrected #1 below; that is a separate, later defect in two specific tests that were genuinely fast (≤5.39s, correctly unmarked) when Day 24 measured them and grew slow afterward without being re-flagged. Keeping this row for completeness of the grep sweep, but it counts as **measured-and-resolved-on-Day-24**, not as a live Day-32 result. |
| 4 | "Confidence in '0 failures repo-wide' rests on… not on watching the last 2% finish" | Day 23, report §5934-5940 | **asserted** | n/a | **CONFIRMED as a gap.** A suite result recorded without a completed run. Objective 2 makes this unrecordable. |
| 5 | `black`/`flake8` "pre-existing drift in ~23-25 files, unrelated to today" | Days 21, 22, 23 | measured *at the time* (touched-file list cross-checked) | **yes** | **STILL OPEN AND GROWING: 28 files** fail `black --check` today, 26 fail `flake8`, against 23-25 then. Each day's dismissal was individually true and the aggregate was never anyone's job. |
| 6 | `--seconds` below 3 "divides to zero static-scenario frames (a pre-existing edge case…) and is avoided rather than fixed here" | `tests/test_cascade_bench_artifact_gate.py:14` | **still-open** | **yes** | **CONFIRMED, but the draft's first pass named the wrong exception.** `static_scenario(args.seconds // 3, …)` with `args.seconds < 3` produces zero frames (no crash at construction — `range(0)` is legal); the crash is downstream, in `run_gate` at `cascade_bench.py:267`, where `np.percentile` on the empty `durations_ms` array raises **`IndexError`**, not `ZeroDivisionError`. Reproduced directly before touching anything. **Fixed today**: `scripts/cascade_bench.py:337` now reads `static_scenario(max(1, args.seconds // 3), …)`; regression test `test_seconds_below_three_does_not_crash` added and passing (`tests/test_cascade_bench_artifact_gate.py`, 5 passed). A free finding, as the objective predicted — just not quite the finding first drafted. |
| 7 | mypy "35 pre-existing `--strict` errors, unchanged" | Days 15, 17, 18 | **measured** — "verified both ways via `git stash`", "reproduced EXACTLY" | partially | **SUPERSEDED, not refuted.** That invocation no longer exists: `mypy.ini`'s scope has changed and CI runs plain `mypy`, which is clean at 0 errors / 64 files. A wider `mypy --strict src/` today reports 179 errors in 26 files, but that is a *different* command and is not evidence against the original claim. |
| 8 | Transient mypy "6 errors" (never committed) | Day 16 | **measured** — traced to an unpinned `.venv`, since deleted | n/a | Holds. The provenance table records the mechanism. |
| 9 | Cascade bench 5.10% / 9.28%: "a wrong number for a machine-state reason unrelated to the interpreter" | Day 26, report §4126-4131 | **measured** — `CPU_Speed_Limit = 46` read directly, and the second figure taken *after* the concurrent load cleared | n/a | Holds, **and it is the counter-example that makes #1 diagnosable.** Same contention hypothesis, same project; here it was measured, and the number was correctly not quoted. |
| 10 | v3-indoor gate GT defect: "a rendering artifact unrelated to real motion… **Not fixed today**" | Day 18, report §3455-3470 | **measured** — cited the `occupied` bucket's wake_fraction of exactly 0.0000 as the discriminating evidence | n/a | Holds, and is now **superseded**: Day 17's gait fix, and Day 31 retired v3's wake numbers outright. |
| 11 | "A missing checkpoint is an environment problem, not a regression" | `src/endurance/runner.py:53` | **enforced** | n/a | Not a dismissal — a designed refusal with a test (`tests/test_endurance_gates.py:334`). |
| 12 | Environment-gate contention/battery notes | `src/bench/environment.py:5` | **measured** — figures recorded inline | n/a | Holds. |

## Counts

*Corrected from the first pass: #3 moved out of `still-open` into
`measured` — it was resolved on Day 24, not left open until today (see
its row above). That leaves `still-open` with one member, not two.*

| class | n | re-tested today | held / superseded | failed re-test |
| --- | ---: | ---: | ---: | ---: |
| asserted | 3 (#1, #2, #4) | 3 | 0 | **3** |
| still-open | 1 (#6) | 1 | 0 | **1** |
| measured | 7 (#3, #5, #7, #8, #9, #10, #12) | 3 (#3, #5, #7) | 6 (2 superseded: #7, #10) | 0 |
| enforced | 1 (#11) | — | 1 | 0 |

**Every asserted dismissal (3/3) and the one re-testable still-open
dismissal (1/1) failed re-test — 4 for 4.** No measured dismissal failed
re-test, though one of the seven (#5) is a survival with an asterisk: the
original per-day claims were individually true, and the *aggregate* they
never added up to is an unresolved, currently-growing gap, not a clean
pass. That is the honest split on a small sample, and it is the sample
this project actually has.

## What the failures have in common

All four confirmed-wrong entries (#1, #4, #6, and arguably #2) are about
**long-running or long-avoided operations**, and in every case the
dismissal avoided the cost of waiting or of touching a slow path.

- **#4 (Day 23):** the repo-wide run had 2% left. *Didn't wait.*
- **#1 (Day 29):** an unmarked slow test tripped the duration gate.
  *Called it contention.* Cost of the isolation re-run that would have
  refuted it: about thirty seconds.
- **#2 (Day 31):** a two-hour background run reported exit 0. *Trusted
  the notification.* Cost of reading the output file: one command.
- **#6 (Day 23, closed Day 32):** `--seconds` below 3 crashes. *Avoided
  rather than fixed.* Cost of the fix: one `max(1, ...)`, plus a
  regression test — about five minutes, for a bug that was open across
  nine days of report edits.

The common factor is not carelessness and it is not a bad instrument. In
all four the instrument was right. **The common factor is that the
dismissal saved time on a slow operation, and the re-test that would have
refuted it was cheap in absolute terms but felt expensive relative to the
thing being dismissed.**

That was diagnosable at three and is now diagnosable at four, which is
why it gets a process rule rather than four corrections:

> **When a dismissal would save you from waiting, price the re-test
> before accepting it.** If the re-test costs minutes, run it. Day 29's
> cost thirty seconds and would have saved two days.

The corollary, from #9: this project already knows how to do this. The
same contention hypothesis, measured properly six days earlier, produced
a correct call and a number correctly withheld. The gap is not capability.

**Does this audit turn up a third instance of Day-31's specific
species — a real signal misdiagnosed as noise?** Checked deliberately,
because the prompt asks to lead with it if so: no, not cleanly. #6 is a
known bug deprioritized as out-of-scope, not a signal explained away as
fake — different species (still-open, not asserted-and-wrong-about-
mechanism). #4 is closer (an unearned "0 failures" claim, same shape as
#2's untrusted exit code) but predates both Day-29 and Day-31 rather
than following them, so it would be the *first* instance, not a third,
and calling it that after the fact would be exactly the kind of
retrofitted pattern-matching this document exists to catch. The new live
instance found below turned out to be a **correctly measured** contention
call, not a wrong one — it demonstrates the fix working, not a third
failure of it. Two confirmed instances stands.

## A live instance, found while re-testing #3 (not from the grep sweep)

Re-verifying #3's "still holds" claim (`pytest -m "not slow"` on
`test_synthetic_indoor.py`) surfaced a genuine, undismissed instrument
firing in real time: `test_superseded_gate_numbers_are_stamped_on_the_scorecard`
**failed** on the first run, at 19.25s against the 15s duration gate —
the exact shape of #1. In isolation it passed at 6.03s.

Applying today's own rule instead of writing "flake" and moving on:

1. Read `uname`/`uptime` at the time: 1-/5-minute load average
   21.27 / 23.07 on a 12-logical-core machine (~1.9/core) — a real,
   independently-observable high-load condition, not an assumption.
2. Re-ran the same command immediately after: **37 passed**, this test at
   4.75s, load average down to 2.05.
3. The test itself is not a pure-CPU microbenchmark like #9's cascade
   run — it calls `src.data.scorecard.compute` over a real golden-set
   clip, genuine motion-gate work whose wall-clock cost scales with
   contention the same way #9's did.

This one is **measured**, not asserted: there is a load reading
bracketing the failure and an immediate clean reproduction, which is
what #1 lacked. It is not in the counts table above because it is not a
grep hit — it is a new instrument firing during this audit's own
re-testing, which is itself the mechanism this document argues for. Not
fixed today: the test is correct and the gate is correct; what would
still help is recording load average alongside any duration-gate
failure automatically, so the next occurrence doesn't require a human to
think to run `uptime` by hand. That is a Day-33 candidate, not a
Day-32 fix, since Objective 2 today is about suite-count truthfulness,
not per-test environment capture.

## Not re-testable today

- **#7's original invocation.** The Day-15 `mypy --strict` file set no
  longer exists as a scope. What would settle it: nothing — the claim is
  superseded by a clean gate, and reconstructing a retired scope to
  re-litigate a number nobody quotes is not worth it. Recorded as
  superseded rather than verified.
- **#5's aggregate.** 28 files failing `black --check` is measured; what
  is *not* measured is whether any of that drift changes behaviour. It
  does not, by construction (formatting only), but nothing enforces that
  claim. What would settle it: a `black --check` gate in CI, which is
  Day-33 item.
