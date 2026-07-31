# ADR 0001 — Person-specific capability lives in deletable adapters

- **Status:** Accepted
- **Date:** 2026-08-01
- **Decides for:** all identity-bearing work (re-ID, gait, face, any
  person-specific recognition), including Phase 3
- **Supersedes:** nothing
- **Open question for counsel:** §Open questions

## Context

The product must eventually answer "is this the same person as before".
That capability is the one part of the system whose weights are derived
from identifiable individuals, and it is therefore the part where a
withdrawal-of-consent request lands.

The consent template (`docs/site_zero_consent_TEMPLATE.md` §7) currently
tells participants:

> If a model has already been trained using your data, deleting your
> footage does not remove whatever that model learned from it. We cannot
> reverse training.

That paragraph is honest for a monolithic fine-tune, and it is a bad
position to be in. It asks a person to accept that their withdrawal is
partially unenforceable, and it invites counsel's sharpest question:
whether erasure obligations under the DPDP Act are satisfied by deleting
footage while retaining a model trained on it.

The architectural choice made *before* any identity work starts
determines whether that paragraph has to stay.

There is a second forcing constraint. All four public gait datasets are
lane R (eval only), and the re-ID sets that are not lane R carry consent
debt or an unverified SMPL dependency. The trainable identity corpus is
Site Zero — consented, self-collected, and revocable **by individuals we
know by name and who can walk up to us and withdraw**. Identity training
data is therefore small, personal, and subject to withdrawal by
construction. An architecture that assumes a large anonymous corpus is
mismatched with the only data we may lawfully use.

## Decision

**Person-specific capability lives in a small adapter over a frozen
backbone. It never lives in fine-tuned backbone weights.**

Concretely:

1. The backbone (V-JEPA2, DA-V2, CoTracker3) stays **frozen** and is
   trained only on lane S and lane C non-identity data. No gradient from
   an identity objective ever reaches backbone parameters.
2. Identity capability is an **adapter**: a small head, or a gallery of
   embeddings, or both — parameterised by orders of magnitude fewer
   weights than the backbone and trainable in hours on CPU.
3. Adapters are **per-deployment**, not global. A site's adapter is
   trained on that site's consented population and does not ship to
   another site.
4. Any artifact containing identity-derived parameters is **labelled as
   such in its `.meta.json` sidecar**, with the consent-record set it
   was derived from. An identity artifact whose contributing consent set
   is unknown must be refused, not shipped — the same posture the
   registry already takes toward unlicensed data.

## Consequences

### Erasure becomes an operation instead of an apology

With a frozen backbone, the full set of parameters derived from any
individual is exactly the adapter. Honouring a withdrawal means: delete
the footage, delete the derived embeddings, retrain the adapter without
them, ship it. That is hours of CPU, not weeks, and it is a routine
operation rather than an exceptional one.

Under a monolithic fine-tune the same request requires re-running the
entire training pipeline to produce a model provably free of one
person's data. At that cost, the honest institutional answer is "we
won't", and the consent form has to say so.

This is the whole argument for the decision. Everything else is
secondary.

### The three-tier deletion design

Withdrawal is served by three tiers with different costs, and the
architecture is what keeps the expensive tier from ever being needed:

| Tier | What is deleted | Cost | When |
| --- | --- | --- | --- |
| 1. Gallery eviction | The person's enrolled embeddings and their entry in the identity gallery | Seconds. No retraining. | Immediately on request; stops all recognition of that person at once |
| 2. Adapter retrain | The adapter is retrained from the remaining consented set | Hours, CPU | Next scheduled adapter build, or immediately if requested |
| 3. Backbone retrain | Not applicable by design | Weeks | Never, because no identity gradient reaches the backbone |

Tier 1 is what makes the response *immediate*: recognition stops the
moment the gallery entry is removed, before any retraining runs. Tier 2
removes the residual influence on the learned metric. Tier 3 is the
column that the decision exists to keep empty.

The confirmation we give a participant is therefore specific and
checkable — gallery evicted on date X, adapter rebuilt without your data
as build Y — rather than a general assurance.

### The consent form can make a stronger promise

Once identity work exists under this architecture, the §7 limitation
narrows from "we cannot reverse training" to a statement that the
identity model is rebuilt without the withdrawn data and that the
frozen backbone never encoded the person in the first place. That is a
materially better answer for the participant and a materially better
position for counsel.

**This ADR does not by itself license changing that paragraph.** The
paragraph is accurate today and must stay until an adapter architecture
is actually built and the rebuild has been demonstrated end to end. The
cross-reference added to the template points here so that the two are
revised together.

### Compliance advantage over monolithic fine-tuning

- **Purpose limitation** is enforceable per artifact: an adapter has one
  purpose and one population, so "used only for what was consented to"
  is a property of the file rather than a policy claim.
- **Data-protection impact assessments** scope to a small, inspectable
  component instead of the whole model.
- **Customer isolation** falls out of per-deployment adapters. One
  site's identity data cannot leak into another site's model, because
  the shipped backbone contains none of it.
- **Provenance is auditable**: the sidecar names the consent set, so the
  question "whose data is in this artifact" has a recorded answer.

### Costs accepted

- An adapter over a frozen backbone will underperform a full fine-tune
  on any benchmark that rewards domain adaptation. We accept a lower
  ceiling in exchange for erasure being real. **This must be measured,
  not assumed** — the gap belongs on a scorecard before Phase 3 commits.
- Per-deployment adapters multiply the artifacts to version, evaluate
  and ship. The `.meta.json` coupling and golden-set machinery already
  built are prerequisites rather than nice-to-haves.
- A gallery of embeddings is itself personal data at rest and needs the
  same protection as footage. Deletable is not the same as unprotected.

## Constraint on Phase 3

Phase 3 identity work inherits this as a hard constraint:

1. No training run may unfreeze backbone weights under an identity
   objective. A run that does is rejected at review regardless of the
   metric it produces.
2. Identity artifacts ship with a sidecar naming the consent set;
   missing provenance is a refusal, matching the existing artifact
   coupling rule.
3. The adapter-versus-fine-tune gap is measured and published on the
   internal scorecard before any identity capability is demonstrated to
   a customer.
4. A withdrawal drill — evict, retrain, confirm — is exercised on
   synthetic identities before Site Zero enrolment begins. An erasure
   path that has never been run is a claim, not a capability.
5. Gait remains auxiliary and never a sole identifier, unchanged by this
   ADR and independently forced by the data reality.

## Alternatives considered

**Monolithic fine-tuning of the backbone.** Best expected accuracy.
Rejected because it makes erasure practically impossible, forces the
consent form to disclaim a right the participant is being asked to rely
on, and couples every customer's data into one shipped artifact.

**Machine unlearning on a fine-tuned backbone.** Would allow fine-tuning
while approximating deletion. Rejected for now: the guarantees are
approximate and contested, verifying them is research rather than
engineering, and telling a participant their data was "approximately
removed" is worse than telling them it was not removed. Worth revisiting
if the field produces certifiable methods; it does not change the
default.

**Gallery-only, no learned adapter** (nearest-neighbour on frozen
backbone embeddings). Maximally deletable — tier 1 is the whole system.
Rejected as the sole approach because frozen general-purpose embeddings
are weak at cross-camera re-ID under clothing change, which is a
condition the golden set explicitly tracks. It remains the correct
*starting* point, and the adapter is added only when measurement shows
the gallery alone is insufficient.

## Open questions

- **For counsel:** does tier 1 plus tier 2 satisfy the erasure
  obligation under the DPDP Act, or must a participant be offered
  something stronger? This is the same question already flagged in the
  consent template §7, and this ADR is the architectural answer we
  propose to it — but whether it is legally sufficient is not ours to
  decide.
- Does an embedding gallery constitute biometric data under the Act
  independently of the footage it came from? It is treated as such here
  regardless, but the answer affects retention periods.
- Retention: how long may an adapter trained on a departed employee's
  data persist before it must be rebuilt even absent a request?
