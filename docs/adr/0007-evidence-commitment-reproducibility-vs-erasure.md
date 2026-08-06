# ADR 0007 — Evidence commitments outlive their inputs

- **Status:** Accepted
- **Date:** 2026-08-06
- **Decides for:** how a claim stays provable after its underlying
  observations are deleted, by retention policy or by an erasure request
- **Supersedes:** nothing — no prior commitment scheme existed
- **Related:** [[iron-blocked-on-humans]] item 4, counsel review of
  `docs/site_zero_consent_TEMPLATE.md` §7 (does deleting footage while
  retaining a model trained on it satisfy DPDP Act erasure obligations);
  this ADR is the architectural half of the same question applied to
  claims rather than models

## Context

This system's output may end up in an internal investigation or a
security audit. A claim in that setting has to survive scrutiny that
arrives after the fact — sometimes long after — and this product also has
a retention policy that deletes raw footage and observations on a
schedule, plus a consent framework (ADR 0001) that must honour withdrawal
requests. Those two facts collide: the observations a claim was built
from will, predictably, stop existing before every question about that
claim has been asked.

Retrofitting a commitment scheme after the first erasure request lands is
too late — every claim made before that point would have no way to prove
what it was built from, because the observations needed to compute a
commitment over would already be gone. Objective 5's data model
(`src/model/evidence.py`) is written now so that every future claim
carries a commitment from the moment it is made, whether or not it is
ever the subject of a retention or erasure event.

## Decision

`EvidenceCommitment.compute()` builds a Merkle root over the SHA-256
hashes of every contributing observation, computed **at claim time** —
not retroactively, not on demand when someone asks. The stored record
(`EvidenceCommitment`) keeps only `merkle_root` and `leaf_count`; it
never retains the leaf hashes themselves. This is deliberate: a record
whose entire purpose is to survive after its inputs are erased must not
itself become a place where erased data's fingerprints linger — storing
the leaf hashes would mean "erasing" the observations still left every
hash of them sitting in every commitment that ever referenced them.

`status` (`reproducible | inputs_expired | inputs_erased`) records
whether the original observations are still available to re-verify
against, independent of whether the commitment itself is still valid.
`EvidenceCommitment.verify(observation_hashes)` recomputes the root from
a supplied hash list and compares — usable only while the inputs are
still around, which is exactly the property `status` exists to make
explicit rather than something a caller has to infer from a 404 further
down the stack.

This is distinct from — and does not replace — `Evidence.reproducible`
and `Evidence.reproduce_command`, which answer "can the derivation be
re-run" (a pipeline-determinism question). `EvidenceCommitment` answers
"can what was committed to be verified" (a data-integrity question) even
after the pipeline that could re-run it has nothing left to run against.

## Consequences

Every claim that wants a commitment now needs its contributing
observation hashes at claim time, which means the observation layer
(`src/model/observation.py`) needs a stable, hashable representation —
satisfied today by `Observation` being a frozen dataclass, though no
canonical hashing function is implemented yet (see Open questions). The
commitment adds one Merkle computation per claim; at this data volume
that cost is not measured to matter, and no measurement has been made —
consistent with the eval-discipline rule that a performance claim
without a number is not a claim.

The unresolved legal question this ADR is the architectural half of —
whether a Merkle commitment over erased data satisfies a DPDP Act
erasure obligation, or whether even a root+count is retained information
that must itself be deleted — is explicitly not decided here. That is
counsel's question, tracked in [[iron-blocked-on-humans]], not
engineering's to resolve by shipping code.

## Open questions

- No canonical `Observation -> hash` function exists yet. `compute()`
  takes pre-hashed strings; whoever calls it must decide the hashing
  scheme (which fields, what encoding) themselves for now. A Day-14+ item
  is a single `observation_content_hash()` so two callers cannot hash the
  same observation two different ways and get commitments that silently
  fail to compare.
- Whether `inputs_erased` commitments must themselves eventually expire —
  and what "erasing a commitment" even means when it holds only a root
  and a count — is unresolved pending the counsel review above.
