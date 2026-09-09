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

**Day-39 update, replacing this section's original instruction:** the
original text here said to point `--license-url` at "the page with the
authoritative terms" — i.e. the rendered HuggingFace dataset-card page
itself. Do not do that. That page embeds per-request-volatile state
(`lastModified`, live download/like counts, discussion stats) around the
license text; hashing it whole produces a different hash on every fetch
even when the license itself is unchanged — this was tried, on this exact
dataset, and confirmed live: two fetches minutes apart, two different
hashes, identical license text underneath. `scripts/fetch_dataset.py` now
refuses that URL shape outright for `hosting: huggingface` entries
(`configs/datasets.yaml`'s CHIRLA entry records `hosting: huggingface`, and
the fixed tool reads it before touching any URL).

Run this instead — the raw, non-rendered file at a commit the tool pins
for you automatically (you do not need to look up a commit sha by hand):

```
python scripts/fetch_dataset.py --verify-license CHIRLA \
  --license-url https://huggingface.co/datasets/bdager/CHIRLA/raw/main/README.md \
  --i-have-read-it \
  --verified-class "CC-BY-4.0" \
  --verified-by "<your name>"
```

This still only records a `LicenseSnapshot`
(`DatasetRegistry.require_fetchable`'s gate); it does not change
`consent_posture`, `lane`, or anything else in `configs/datasets.yaml`,
which must be hand-edited separately with whatever items 1-4 above actually
found. It also does not, by itself, satisfy `chirla-license-verification-
confirm` on `docs/blocker_ledger.yaml` if a prior snapshot already exists
against the volatile page shape — CHIRLA's own `notes` field carries a Day-39
correction record explaining that state; re-read it before assuming this
step is done.

## Fetching the six benchmark scenarios (once verified)

Requires the cached `huggingface-cli login` token (or `--hf-token`) and
network access to huggingface.co, neither available inside Claude Code's
sandbox — run this from a human terminal, same as the verification command
above. The dataset's own declared configs are fetched by name, never a raw
directory glob; `videos` is deliberately excluded (it accounts for most of
the repository's ~10.9GB and is not needed for any of the six benchmarks
below, which total ~932.5MB combined — see CHIRLA's `notes` field for where
that figure comes from):

```
python scripts/fetch_dataset.py CHIRLA \
  --hf-config reid_long_term \
  --hf-config reid_multi_cam \
  --hf-config reid_multi_cam_long_term \
  --hf-config reid_reappearance \
  --hf-config tracking_brief \
  --hf-config tracking_multi
```

Each config is idempotent and independently verified (HuggingFace's own
per-file hash, re-checked after download; a mismatch deletes that config's
staged files and refuses rather than installing something that doesn't
match) — a partial failure on one config does not need the whole command
re-run, and re-running the whole command after a partial success re-fetches
only what is missing. Each writes a `_manifest.json` under
`data/raw/CHIRLA/<commit_sha>/<config_name>/` recording the commit, the
files stored, and the per-split example counts/byte sizes HuggingFace's own
card reports.

**Cross-check once this runs for real:** compare the manifest's split
counts against the dataset's own published reference figures — for
`reid_long_term`, gallery/query/train/val should read 368/4903/65/1177. A
mismatch there means something is wrong with the fetch (a stale commit, a
config-pattern change upstream), not with the reference numbers, since those
came directly from the dataset's own published metadata, read the same
night this checklist was updated.
