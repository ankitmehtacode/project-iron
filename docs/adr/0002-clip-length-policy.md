# ADR 0002 — Clip length policy is blocked on the temporal-slot fix

- **Status:** Accepted
- **Date:** 2026-08-01
- **Decides for:** V-JEPA2 clip length (`T`) for every semantic call —
  export, wrapper, cascade stage 3, and the calibration set
- **Supersedes:** nothing
- **Depends on:** the frame→slot fix in `src/semantics/patch_mapping.py`
  (commit `0277c84`)

## Context

The plan proposed a 16-frame clip policy. The checkpoint is natively 64.
Choosing between them looked like a straightforward
latency-versus-context trade, and it is not, because until commit
`0277c84` the frame→temporal-slot rule was a clamp:

```python
t_out = min(t, T_out - 1)      # what the code did
t_out = t // tubelet           # what the encoder's layout requires
```

The two agree only at the bottom of the range. How badly they disagree
is a function of clip length, and it grows almost linearly with it:

| `T` (frames) | `T_out` slots | misassigned frames | share |
| ---: | ---: | ---: | ---: |
| 4 | 2 | 1 | 25% |
| 8 | 4 | 5 | 62% |
| **16** | 8 | 13 | **81%** |
| 32 | 16 | 29 | 91% |
| **64** | 32 | 61 | **95%** |

The audit that found the defect ran at `T=4`, where it misassigns a
single frame — frame 1 reads slot 1 instead of slot 0. That is the
mildest configuration the bug has, and it is the one that hid it: a
single off-by-one temporal slot on a 4-frame clip produces semantic
vectors that are wrong but not obviously wrong.

**The clip-length decision and the clamp fix are therefore coupled, and
the coupling runs in the dangerous direction.** The clamp's severity is
lowest exactly where a quick sanity check would be run, and highest
exactly where the product wants to operate. Had the 16-frame policy
shipped on the pre-fix mapper, 81% of frames would have read the wrong
temporal slot, and at the checkpoint's native 64 it would have been 95% —
a system in which nearly every semantic vector describes the wrong
moment, while every shape check, every token count, and every smoke test
still passes.

## Decision

1. **No clip-length policy is adopted on a mapper that clamps.** The fix
   in `0277c84` is a precondition of this ADR, not a companion to it.
   `TemporalSpan` now validates `n_temporal == frames_covered // tubelet`
   and refuses a mismatch rather than clamping it, so the failure mode is
   now loud.

2. **Any change to `T` is re-measured, not reasoned about.** Clip length
   changes the number of temporal slots, which changes what every
   downstream semantic consumer reads. It is a measured change under the
   eval discipline — scorecard delta on the frozen golden set — and never
   a config edit justified by latency arithmetic alone.

3. **The 16-versus-64 choice stays open** until there is a semantic
   metric to decide it with. The motion-gate scorecard cannot: it does
   not reach stage 3. Deciding clip length against a metric that does not
   exercise the encoder would be picking a number and calling it
   measured.

## Consequences

The clip-length decision is deferred, and deliberately. The cost is that
the export and calibration paths keep carrying an unfixed `T` longer than
planned. The benefit is that when the decision is made it will be made
against a mapper whose slot arithmetic is correct and validated, so the
measurement will describe clip length rather than the interaction between
clip length and a latent indexing bug.

The general form of this, worth keeping: **a bug whose severity scales
with a parameter must be fixed before that parameter is tuned.** Tuning
first produces a measurement of the bug's gradient and reads it as a
property of the system.

## Open questions

- What semantic metric decides 16 versus 64? Same-object retrieval mAP
  and temporal embedding stability are the candidates from the eval
  discipline, and neither is implemented.
- Does the tubelet stay at 2 across every export? The table above assumes
  it does; a different tubelet rescales the whole column.
