# CHIRLA verification checklist — five minutes, human only

**Do not run `scripts/fetch_dataset.py --verify-license CHIRLA --i-have-read-it`
until every box below is checked by a person who actually read the pages.**
That flag is a signature, not a formality — see `scripts/fetch_dataset.py`'s
own docstring. Claude Code must never check these boxes on its own judgment
(and did not): its network access reaches `github.com` /
`raw.githubusercontent.com`, not `huggingface.co`, `sciencedb.cn`, or
`arxiv.org`, so none of the four items below were verified when this
checklist was written (2026-09-07, Day 36) — only GitHub's `README.md` at
commit `fcb6f53359d5888b6e8fb745b65a411697dcc22c` was read, and that page
alone does not clear a single one of these boxes.

**Note on scope:** the Day-36 task that produced this checklist asked for
"the exact three URLs" but named a check (the ethics/consent statement) that
does not live on any of the three — it lives in the arXiv paper body. Rather
than silently drop that check to make the count match, this list has four
items. Flagging the mismatch here instead of hiding it is the same discipline
this project applies to its own numbers elsewhere.

---

## 1. HuggingFace dataset card

**Open:** https://huggingface.co/datasets/bdager/CHIRLA

**Look for:** the license badge/field HuggingFace itself records for the
dataset (its own structured `license:` tag, not just prose on the page —
Hugging Face dataset cards carry a license field separate from whatever the
README says). Does it say `cc-by-4.0`, a different CC-BY version, or
something else? GitHub's README only says "CC-BY" with no version number.

- [ ] License tag recorded: ______________________
- [ ] Matches GitHub README's unversioned "CC-BY" claim: yes / no / GitHub
      says less than this page does

## 2. ScienceDB DOI page

**Open:** https://doi.org/10.57760/sciencedb.20543

**Look for:** the license terms ScienceDB itself states for the deposited
dataset (ScienceDB depositors set their own terms at upload time, which can
differ from what a linked GitHub repo claims), and whether the page states
any access restriction (registration wall, application-gated, embargo) that
the GitHub README does not mention.

- [ ] License/terms stated on this page: ______________________
- [ ] Any access restriction not mentioned on GitHub: yes / no — if yes, what:
      ______________________

## 3. GitHub README's License section (no separate LICENSE file exists)

**Open:** https://github.com/bdager/CHIRLA/blob/main/README.md#license
(the repository root has no `LICENSE` file — `.gitignore`, `README.md`,
`assets/`, `benchmark/`, `data/`, `downloader/`, `requirements.txt` is the
complete top-level listing as of the commit above; the only license text in
the repo is this one line in the README)

**Look for:** confirm the line reads exactly "The dataset is publicly
available under the **CC-BY** license." with no version number and no
separate terms file, and note whether anything in `benchmark/README.md` or
`downloader/README.md` (not yet read by anything in this repository) states
terms that differ from the top-level README's one-line claim.

- [ ] Confirmed exact text, no version number: yes / no
- [ ] `benchmark/README.md` / `downloader/README.md` checked for
      conflicting terms: yes / no

## 4. arXiv paper body (not the abstract)

**Open:** https://arxiv.org/pdf/2502.06681 (the PDF — the abstract page at
https://arxiv.org/abs/2502.06681 will not contain this; it must be in the
body, likely an Ethics/Data Collection/Institutional Review subsection)

**Look for:** an ethics or data-collection consent statement naming who the
22 individuals were (e.g. lab members, students, staff of the hosting
research group) and how they consented to being recorded for seven months
and released in a public dataset. `configs/datasets.yaml`'s CHIRLA entry
currently records `consent_posture: unknown` specifically because this
question is unanswered by anything reachable from this environment — the
subjects read as a known, small, lab-affiliated population (unlike scraped
CCTV of the public), but the README documents no consent process the way
MEVA's entry documents "hired, consented actors."

- [ ] Ethics/consent statement found: yes / no
- [ ] If yes, who are the subjects and what consent process is described:
      ______________________
- [ ] Resulting `consent_posture` for `configs/datasets.yaml`: staged_actors /
      public_cctv_no_consent / leave as unknown (state why)

---

## After checking all four

Only then does `scripts/fetch_dataset.py --verify-license CHIRLA
--license-url <the page with the authoritative terms> --i-have-read-it`
become appropriate — and even then it only records a `LicenseSnapshot`
(`DatasetRegistry.require_fetchable`'s gate); it does not change
`consent_posture`, `lane`, or anything else in `configs/datasets.yaml`,
which must be hand-edited separately with whatever items 1-4 above actually
found.
