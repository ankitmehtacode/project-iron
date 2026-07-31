# Site Zero — Participant Consent Form

> **DRAFT — NOT LEGALLY REVIEWED. DO NOT USE FOR ACTUAL CAPTURE.**
>
> This is an engineering scaffold: it records what the capture protocol and the
> product's DPDP posture require a consent form to cover, so that counsel is
> reviewing a concrete document rather than starting from a blank page. It is
> not legal advice and it has not been checked against the Digital Personal
> Data Protection Act, 2023, its rules, or any employment-law constraint on
> obtaining consent from employees.
>
> **Required before any recording:** review by qualified counsel, particularly
> on (a) whether employer-to-employee consent is freely given for DPDP
> purposes, (b) retention periods, (c) the withdrawal mechanism's practical
> limits once data is in a trained model, and (d) whether a Data Protection
> Officer must be named.
>
> Placeholders in `⟨angle brackets⟩` must be filled before review.

---

## 1. Who is collecting this data

**Data Fiduciary:** ⟨legal entity name⟩, ⟨registered address⟩
**Contact for data questions:** ⟨name, role, email, phone⟩
**Grievance officer:** ⟨name, email⟩ — responds within ⟨N⟩ working days.

## 2. What is being recorded

Video from ⟨N⟩ fixed cameras in ⟨locations⟩ at ⟨resolutions⟩, during
⟨scheduled windows⟩. Audio is **not** recorded.

Recordings capture your appearance, clothing, movement, posture, and
interactions with objects and other participants. From these, the system
derives:

- **Tracks**: your position over time within a camera's view.
- **Activity labels**: a closed vocabulary of 19 verbs (entering, exiting,
  sitting, picking up an object, and similar).
- **Appearance embeddings**: numerical descriptions used to recognise that two
  recordings show the same person.
- **Gait characteristics**: how you walk.

Appearance embeddings and gait characteristics are **biometric data** under the
DPDP Act. They are treated as such throughout — see §5.

## 3. Why

⟨Company⟩ is developing indoor video-analytics software. This recording is used
to:

1. **Build evaluation sets** measuring whether the software works.
2. **Train and calibrate models** that run inside the product.
3. **Demonstrate the product** to prospective customers — *only* if you tick
   the separate box in §8. You may refuse this and still take part.

This footage will be used for these purposes and no others. It will not be sold,
shared with third parties, published, or used to make any decision about you —
including any assessment of your work, attendance, or conduct.

## 4. This is voluntary, and refusing costs you nothing

Participation is entirely optional. Declining, or withdrawing later, will not
affect your employment, engagement, standing, or any assessment of you, and
will not be recorded anywhere that could influence a decision about you.

Scheduled capture windows are posted ⟨N⟩ days ahead. If you do not wish to be
recorded, you may avoid the area, or ask ⟨contact⟩ to pause capture — no reason
required.

## 5. How it is stored and protected

- Encrypted at rest and in transit; access limited to named engineers on
  ⟨company⟩'s ML team.
- Never copied to personal devices or third-party cloud services outside
  ⟨named infrastructure⟩.
- Biometric derivatives (§2) are stored separately from footage, under stricter
  access control, and are never exported from ⟨named systems⟩.
- Consent records are stored with the data they authorise, so any use can be
  traced to the permission that allowed it.

## 6. How long it is kept

| Item | Retention |
|---|---|
| Raw footage | ⟨period⟩ from capture |
| Annotated evaluation clips | ⟨period⟩ |
| Appearance / gait derivatives | ⟨period⟩ |
| Consent records | ⟨period, typically longer than the data⟩ |

At the end of each period the item is deleted, not archived. ⟨Counsel to
confirm each period is defensible as "no longer necessary for the purpose".⟩

## 7. Your rights

You may, at any time and without giving a reason:

- **Access** what has been recorded of you.
- **Correct** anything inaccurate.
- **Withdraw consent** and have your footage and derivatives deleted.
- **Complain** to the grievance officer (§1), and to the Data Protection Board
  of India.

To exercise any of these, contact ⟨contact⟩. Deletion is completed within
⟨N⟩ days and confirmed to you in writing.

> **An honest limitation, stated plainly.** If a model has already been trained
> using your data, deleting your footage does not remove whatever that model
> learned from it. We cannot reverse training. What we commit to is: deleting
> your footage and derivatives, excluding them from every future training run,
> and — if you ask — telling you which released models were trained on data
> including yours.
>
> ⟨Counsel: does this satisfy the erasure obligation, or must retraining be
> offered? This is the sharpest open question in this document.⟩

## 8. What you are agreeing to

Tick each that applies. **You may tick some and not others.**

- ☐ I agree to be recorded as described in §2.
- ☐ I agree my recordings may be used to **evaluate** the software (§3.1).
- ☐ I agree my recordings may be used to **train and calibrate** models (§3.2).
- ☐ I agree my recordings may be shown in **demonstrations to prospective
  customers** (§3.3). *Optional — you may decline this and still take part.*
- ☐ I agree that **biometric derivatives** (appearance embeddings, gait) may be
  computed from my recordings and stored as described in §5.

I confirm that I have read and understood this form, that I have had the chance
to ask questions, and that I am giving consent freely.

**Name:** ⟨_______________⟩  **Signature:** ⟨_______________⟩

**Date:** ⟨__________⟩

**Witness / person who explained this form:** ⟨_______________⟩

---

## Notes for whoever runs the session

Not part of the consent form; keep on file with the protocol.

1. **Non-participants must be excluded, not blurred.** Blurring is a
   post-processing step that can fail, be forgotten, or be undone from the
   original. Capture windows are scheduled so that only consenting participants
   are present.
2. **Recruit beyond the team.** Consent from a small group of colleagues who
   all know each other is weak evidence of freely-given consent, and a dataset
   of eight engineers is a poor sample of the people a deployment will see.
3. **Re-consent when purposes change.** A new purpose needs new consent; it is
   not covered by a form that did not mention it.
4. **This is a rehearsal.** The Tier-2 enrollment flow will need the same
   mechanics — purpose limitation, withdrawal, biometric handling — at customer
   scale. Problems found here are cheap; the same problems found at a customer
   site are not.
