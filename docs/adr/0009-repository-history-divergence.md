# ADR 0009 — `main` and `origin/main` are unrelated histories

- **Status:** Proposed
- **Date:** 2026-08-10
- **Decides for:** whether and how to reconcile the local `main`/
  `foundation/*` line with the pre-existing history already on
  `origin/main`, and what to do about `git push --all origin` failing on
  `main` every day since Day 16
- **Supersedes:** nothing — first ADR to address repository history
- **Related:** [[iron-day-workflow]] (the push-at-start/push-at-end
  standing rule); Day-16 and Day-17 `FOUNDATION_REPORT.md` sections, which
  first noted the divergence and left it out of scope; Day-18 Objective 0
  (push verification)

## Context

`git push --all origin` has failed on `main` specifically since Day 16,
with `[rejected] main -> main (non-fast-forward)`. Every day's report has
noted this and moved on, out of scope for that day's objectives. Objective
4 today is to actually investigate it, because the cost of leaving it
alone grows with every branch added, and eighteen `foundation/day-N`
branches now exist.

**What is actually on `origin/main`:** 34 commits, `2026-02-17` through
`2026-07-13`, five distinct author identities (`Dalbirsingh Matharu`,
`priyanshu sharma`, `Radhetare`, `Rishi Joshi` / `Rishiijoshi` — a
duplicate identity for the same person under two capitalizations), merged
through five pull requests referencing branches that still exist on the
remote today (`optimize-cotracker-input`, `temporal-stitching`, both
merged into `origin/main`). This is a real, multi-person, PR-based
development history for a project also called "project-iron" — same file
layout, same module names (`src/geometry`, `src/graph`,
`src/interface/ui`, `src/rag_agent.py`, `src/vector_database.py`, all
named in Day 1's report as present in the imported working tree), 82 of
its 82 files matching local `main`'s 90 by name. It is the same project,
built by other people, on a timeline that ends 18 days before local
`main` begins.

**What local `main` is:** a single commit, `41e924c` —
`"chore: import project-iron working tree as baseline"`, dated
`2026-07-31`, authored by the git identity this repository has been
committed under throughout (`Ankit Mehta`). Day 1's own report says so
directly (§7): *"The repo was not a git repository. No `.git`, no
remote, no history. A baseline commit was created so the ten objective
commits had something to sit on."* Every `foundation/day-N` branch
descends from that one commit. `git merge-base foundation/day-1
origin/main` and the same check against `foundation/day-17` both return
nothing — there is no common ancestor. These are, in git's own terms,
unrelated histories that happen to share a remote URL.

**Why this matters now rather than later:** a normal `git clone` of
`ankitmehtacode/project-iron` gets `origin/main`'s real history as its
`main` and every `foundation/day-N` branch alongside it — nothing is
broken for a fresh clone. The cost is entirely local and entirely
future: anyone who runs `git log --all --graph` in a clone that already
has both refs sees two disconnected root commits with no explanation;
`git push --all origin` will keep failing on `main` indefinitely, which
is exactly the kind of check that trains people to stop looking at
push failures (the same failure mode this project has spent two days —
16 and 17 — fixing elsewhere, applied to git hygiene instead of a test
gate); and the eventual reconciliation, whenever it happens, gets more
expensive the more `foundation/day-N` branches exist to carry through
it or leave stranded.

## Options considered

**A — Graft the foundation line onto `origin/main`'s history.**
Rebase `foundation/day-1` (and, transitively, every `foundation/day-N`
branch built on it) so its root commit becomes a child of `origin/main`'s
tip, unifying the timeline into one connected graph. *Consequence:*
rewrites every commit hash across all 18 branches — roughly 100+ commits
— which invalidates every SHA already pushed and cited (this ADR's own
`FOUNDATION_REPORT.md` cross-references commit hashes by name in several
places; all of them would go stale). Requires force-pushing every
`foundation/day-N` branch. The single most disruptive option, and the
only one that actually produces one connected history.

**B — Keep the two lines permanently separate; stop implying they are
the same branch.** `origin/main` stays exactly as it is — untouched,
still the history the other four collaborators' clones expect. Locally,
rename the current `main` (the single-baseline-commit trunk that
`foundation/day-N` branches from) to a name that does not collide with
`main`'s conventional meaning — e.g. `foundation-baseline` — so a future
`git log --all --graph` reads as "two deliberately separate projects
sharing a remote," not as an unexplained fork in `main` itself. `git
push --all origin` continues to skip `main` (non-fast-forward, as today)
because nothing about local `main` is pushed as `origin/main` going
forward; only `foundation/day-N` branches push, exactly as they already
do. *Consequence:* the confusion this ADR exists to resolve is answered
by documentation and a rename, not by unifying the graph — a future
contributor who asks "why are there two roots" gets an immediate,
correct answer instead of an investigation, which was the actual ask.
Zero risk: no shared ref is touched, nothing is force-pushed, every
existing SHA (local and on every collaborator's clone) stays exactly
what it is today.

**C — Archive `origin/main`'s current tip under a tag and replace it
with the foundation line.** Tag `origin/main`'s current commit (e.g.
`archive/pre-foundation-main`) for permanent, discoverable reference,
then force-update `main` — locally and on `origin` — to the foundation
line's history. *Consequence:* this is the option that would actually
make `foundation/day-N` the project's new mainline, but it requires
force-pushing over history that four other people's clones depend on.
Their next `git pull` on `main` would either fail outright or (with
`git pull --rebase` misused, or a hard reset) silently discard local
work built on the old `main`. This is not this repository's call to
make unilaterally — it changes what `origin/main` means for people who
are not part of this conversation and have not been asked.

## Recommendation

**Option B.** It is reversible, touches nothing outside this local
checkout, and directly answers the actual complaint (confusion on a
future clone or a future contributor) without gambling with four other
people's history. Options A and C both assume this "Foundation Day"
work is meant to *become* the project's mainline, which is a real
possibility but not this repository's decision to infer — it depends on
whether `ankitmehtacode/project-iron` is meant to diverge permanently
from `Dalbirsm03/project-iron` (visible in `origin/main`'s own merge
commit messages, e.g. `"Merge branch 'main' of
https://github.com/Dalbirsm03/project-iron"`) as an independent solo
exercise, or eventually reconcile with it. That is a five-minute
question for whoever owns this relationship to answer; it is not one
this ADR can answer from repository state alone.

**This ADR takes no action.** Per the objective, nothing here is
executed — no branch is renamed, no history is rewritten, no tag is
created. Status stays `Proposed` until the human decision above is
made; see Consequences for what accepting Option B would actually
change on disk.

## Consequences (if Option B is accepted)

- Local `main` is renamed (e.g. to `foundation-baseline`); every
  `foundation/day-N` branch's tracking/base reference updates to match,
  but no commit on any of them changes hash.
- `git push --all origin` stops attempting to push a `main` that was
  never going to fast-forward `origin/main` in the first place — the
  rejection disappears not because the conflict was resolved, but
  because local `main` no longer exists under that name to be pushed.
  `origin/main` is untouched and untouchable by this repository's push
  routine from that point on.
- A future contributor cloning `ankitmehtacode/project-iron` still sees
  `origin/main`'s real history as `main`, plus every `foundation/day-N`
  branch — unchanged from today, since Option B does not touch `origin`
  at all. The improvement is entirely for people already working in a
  local checkout that has both refs, this machine included.
- If Option A or C is chosen later instead, this ADR's status moves to
  `Superseded` by whichever ADR records that decision and its execution
  — this document is not the place that graft or force-push gets
  authorized.

## Open questions

- What `ankitmehtacode/project-iron` and `Dalbirsm03/project-iron`
  actually are to each other (fork, shared team repo, something else)
  was not asked of anyone and cannot be determined from repository
  state alone — the merge commit messages establish that `origin/main`
  pulled from `Dalbirsm03/project-iron` at some point, not which
  direction the relationship runs today.
- Whether the four other identities on `origin/main` are aware this
  divergence exists, and whether any of them still actively push to
  that remote, is unknown and worth confirming before any option
  involving `origin/main` (B's documentation-only change excepted) is
  acted on.
