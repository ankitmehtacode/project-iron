#!/usr/bin/env bash
# Fresh-clone check (Day 42, Objective A1).
#
# Clones a commit of THIS repository into an empty directory, builds a venv
# from locking-requirements.txt alone, installs the checkout with -e, checks
# that pip freeze matches the lockfile, and runs the whole suite. The verdict
# is read from the junit artifact's counts, never from pytest's exit status
# alone: a collection error or a crashed worker can exit in ways a status
# check misreads, and the counts are what the report quotes.
#
#   scripts/fresh_clone_check.sh [REF] [WORKDIR]
#
# REF defaults to HEAD. WORKDIR defaults to a new mktemp directory. The junit
# XML and log are copied to artifacts/pytest/fresh_clone_<sha12>.{xml,log}.
# Exit 0 only when install, lockfile check and suite all pass with
# 0 failures and 0 errors.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REF="${1:-HEAD}"
WORK="${2:-$(mktemp -d)}"
PYTHON="${PYTHON:-python3.10}"
SHA="$(git -C "$REPO" rev-parse "$REF")"
OUT="$REPO/artifacts/pytest/fresh_clone_${SHA:0:12}"

echo "fresh clone of $SHA into $WORK"
rm -rf "$WORK/clone" "$WORK/venv"
git clone -q "$REPO" "$WORK/clone"
git -C "$WORK/clone" checkout -q "$SHA"
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip==24.0
"$WORK/venv/bin/pip" install -q -r "$WORK/clone/locking-requirements.txt"
"$WORK/venv/bin/pip" install -q -e "$WORK/clone"
"$WORK/venv/bin/python" "$WORK/clone/scripts/check_lockfile.py"

set +e
(cd "$WORK/clone" && "$WORK/venv/bin/python" -m pytest -p no:cacheprovider \
    --junitxml="$WORK/junit.xml" -ra -q) >"$WORK/pytest.log" 2>&1
STATUS=$?
set -e
mkdir -p "$(dirname "$OUT")"
cp "$WORK/junit.xml" "$OUT.xml"
cp "$WORK/pytest.log" "$OUT.log"

"$WORK/venv/bin/python" - "$OUT.xml" "$STATUS" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
total = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
passed = total["tests"] - total["failures"] - total["errors"] - total["skipped"]
print(
    f"fresh clone: {passed} passed, {total['failures']} failed, "
    f"{total['errors']} errors, {total['skipped']} skipped "
    f"(pytest exit status {sys.argv[2]}; counts from {sys.argv[1]})"
)
for case in root.iter("testcase"):
    if case.find("failure") is not None or case.find("error") is not None:
        print(f"  FAILED {case.get('classname')}::{case.get('name')}")
sys.exit(1 if total["failures"] or total["errors"] or total["tests"] == 0 else 0)
PY
