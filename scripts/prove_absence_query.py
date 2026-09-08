"""Day 37, Objective 4 -- real `prove_absence` results, persisted for the
Inspector's new Coverage/Absence view.

Day 35's own punch list named this the most valuable stale item: "the 'was
anyone in the vault between 02:00 and 04:00' falsification test has never
been visually inspectable." This primitive (src/model/coverage.py) is what
makes this project an evidence engine rather than a search product -- a
plain event log cannot distinguish "we looked and saw nothing" from "we
never looked", and `prove_absence` is the only sanctioned way to turn a
Coverage record into a negative claim. It has real, passing tests
(tests/test_model_coverage.py, tests/test_falsification.py's falsification
test #1) but nothing had ever run it and put the result somewhere a human
could look at it.

This project's real planned capture uses "zone"/"corridor" language
(docs/capture_runbook.md), not "vault" -- the prompt's "vault" is
illustrative, not a zone this project has actually planned, so the query
below uses a real zone id in that same style instead of inventing one that
implies a plan that does not exist.

Every query result below is run through the REAL `prove_absence` function
against REAL `Coverage`/`Gap`/`Interval` dataclass instances -- never a
hand-written JSON payload standing in for what the function would return.
Each result carries its own `scenario_realism` label, self-labelled the
same way Day 36 required of `PromotionResult`:

  * "actual_project_state" -- the input is not merely realistic, it is
    literally true right now. The one query below with an EMPTY coverage
    log is this kind: zero Coverage records exist anywhere in this
    repository for any Site Zero zone, because zero minutes of Site Zero
    footage have ever been captured (see docs/blocker_ledger.yaml's
    site-zero-capture entry) -- checked directly (`git grep -l
    "site-zero" outputs/ data/` finds nothing) rather than assumed.
  * "constructed_from_real_types" -- real dataclasses, run through the
    real function, but built to exercise a code path (a degraded-coverage
    half, an internal Gap, a fully-covered proof) rather than reporting
    something that occurred. Three of the four queries below are this
    kind, and two of them reuse the EXACT Coverage records already proven
    correct in tests/test_falsification.py's falsification test #1 and
    tests/test_model_coverage.py -- not new numbers invented for this
    script.

Output: one outputs/coverage_queries/<query_id>.json per query (same
"generated, gitignored" convention as outputs/associations/), served by
the Inspector's new Coverage/Absence view.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.cascade.envelope import MeasuredEnvelope
from src.model.coverage import (
    Absence,
    CannotEstablish,
    Coverage,
    Gap,
    Interval,
    prove_absence,
)

HOUR_NS = 3_600_000_000_000
BASE_TS_NS = 1_800_000_000 * 1_000_000_000
"""An arbitrary, clearly-not-real epoch far from any actual recording --
the interval's absolute placement is illustrative; only its SHAPE (a
2-hour overnight window) and the coverage arithmetic around it are the
point of any of these queries."""

REAL_ENVELOPE_PATH = Path("configs/envelope/gate_320x180.envelope.json")


def _interval_dict(interval: Interval) -> dict[str, int]:
    return {"start_ns": interval.start_ns, "end_ns": interval.end_ns}


def _gap_dict(gap: Gap) -> dict[str, Any]:
    return {
        "camera_id": gap.camera_id,
        "interval": _interval_dict(gap.interval),
        "reason": gap.reason,
    }


def _coverage_dict(cov: Coverage) -> dict[str, Any]:
    return {
        "subject_id": cov.subject_id,
        "subject_kind": cov.subject_kind,
        "interval": _interval_dict(cov.interval),
        "status": cov.status,
        "envelope_ref": cov.envelope_ref,
        "gaps": [_gap_dict(g) for g in cov.gaps],
        "manifest_sha": cov.manifest_sha,
    }


def _result_dict(result: Absence | CannotEstablish) -> dict[str, Any]:
    if isinstance(result, Absence):
        return {
            "kind": "absence",
            "predicate": result.predicate,
            "coverage_basis": [_coverage_dict(c) for c in result.coverage_basis],
        }
    return {
        "kind": "cannot_establish",
        "reason": result.reason,
        "uncovered_subintervals": [
            _interval_dict(i) for i in result.uncovered_subintervals
        ],
        "envelope_violations": list(result.envelope_violations),
        "gaps": [_gap_dict(g) for g in result.gaps],
    }


def _query(
    *,
    query_id: str,
    scenario_realism: str,
    scenario_note: str,
    subject_id: str,
    subject_kind: str,
    predicate_description: str,
    predicate: Any,
    coverage_log: list[Coverage],
    interval: Interval,
) -> dict[str, Any]:
    result = prove_absence(subject_id, interval, predicate, coverage_log)
    return {
        "query_id": query_id,
        "scenario_realism": scenario_realism,
        "scenario_note": scenario_note,
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "query_interval": _interval_dict(interval),
        "predicate_description": predicate_description,
        **_result_dict(result),
    }


def _never_present(_subject_id: str, _interval: Interval) -> bool:
    return False


def main() -> int:
    envelope_sha = MeasuredEnvelope.load(REAL_ENVELOPE_PATH).sha
    query_interval = Interval(BASE_TS_NS, BASE_TS_NS + 2 * HOUR_NS)

    queries: list[dict[str, Any]] = []

    # 1. ACTUAL project state: zero Coverage records exist for any Site
    # Zero zone, because zero minutes of Site Zero footage have ever been
    # captured (docs/blocker_ledger.yaml: site-zero-capture, 39 days old
    # as of Day 37). This is not a constructed absence-of-data scenario --
    # it is what querying this project's real state returns today.
    queries.append(
        _query(
            query_id="site-zero-corridor-a__overnight-window",
            scenario_realism="actual_project_state",
            scenario_note=(
                "Zero Coverage records exist anywhere in this repository for "
                "any Site Zero zone -- checked directly, not assumed. This is "
                "the real, current answer to a real query against this "
                "project's actual state, not a constructed example."
            ),
            subject_id="site-zero-corridor-a",
            subject_kind="zone",
            predicate_description="was any person present in this corridor zone",
            predicate=_never_present,
            coverage_log=[],
            interval=query_interval,
        )
    )

    # 2. Constructed from real types -- reproduces
    # tests/test_falsification.py::test_falsification_absence_under_
    # degraded_coverage exactly (falsification test #1): a camera that
    # goes degraded partway through the window must block a proven
    # absence, even though it saw nothing the whole time.
    live_half = Coverage(
        subject_id="cam-dock-02",
        subject_kind="camera",
        interval=Interval(query_interval.start_ns, query_interval.start_ns + HOUR_NS),
        status="live",
        envelope_ref=f"motion_gate@cam-dock-02@envelope_sha={envelope_sha[:12]}",
        gaps=(),
        manifest_sha="sha-cov-1",
    )
    degraded_half = Coverage(
        subject_id="cam-dock-02",
        subject_kind="camera",
        interval=Interval(query_interval.start_ns + HOUR_NS, query_interval.end_ns),
        status="degraded",
        envelope_ref=f"motion_gate@cam-dock-02@envelope_sha={envelope_sha[:12]}",
        gaps=(),
        manifest_sha="sha-cov-2",
    )
    queries.append(
        _query(
            query_id="cam-dock-02__degraded-half-blocks-absence",
            scenario_realism="constructed_from_real_types",
            scenario_note=(
                "Reproduces tests/test_falsification.py's falsification test "
                "#1 exactly: real Coverage records, run through the real "
                "prove_absence, not a scenario invented for this view. "
                "Demonstrates envelope_violations rendering."
            ),
            subject_id="cam-dock-02",
            subject_kind="camera",
            predicate_description="did anyone enter the loading dock",
            predicate=_never_present,
            coverage_log=[live_half, degraded_half],
            interval=query_interval,
        )
    )

    # 3. Constructed from real types -- reproduces
    # tests/test_model_coverage.py::test_gap_inside_live_coverage_blocks_
    # absence_and_is_reported: live status throughout, but one dropped-frame
    # Gap inside the interval. Demonstrates `gaps` rendering, distinct from
    # an envelope (status) violation.
    gap = Gap(
        camera_id="cam-lobby-01",
        interval=Interval(query_interval.start_ns + 100, query_interval.start_ns + 200),
        reason="dropped_frame",
    )
    covered_with_gap = Coverage(
        subject_id="cam-lobby-01",
        subject_kind="camera",
        interval=query_interval,
        status="live",
        envelope_ref=f"motion_gate@cam-lobby-01@envelope_sha={envelope_sha[:12]}",
        gaps=(gap,),
        manifest_sha="sha-cov-3",
    )
    queries.append(
        _query(
            query_id="cam-lobby-01__dropped-frame-gap-blocks-absence",
            scenario_realism="constructed_from_real_types",
            scenario_note=(
                "Reproduces tests/test_model_coverage.py::test_gap_inside_"
                "live_coverage_blocks_absence_and_is_reported: live status "
                "throughout, one dropped-frame Gap inside the window. "
                "Demonstrates `gaps` rendering, distinct from an "
                "envelope_violation (a Gap can occur during otherwise-live "
                "coverage; a status violation cannot)."
            ),
            subject_id="cam-lobby-01",
            subject_kind="camera",
            predicate_description="did anyone enter the lobby",
            predicate=_never_present,
            coverage_log=[covered_with_gap],
            interval=query_interval,
        )
    )

    # 4. Constructed from real types -- reproduces
    # tests/test_model_coverage.py::test_unbroken_live_coverage_with_no_
    # occurrence_proves_absence: unbroken live coverage, no gaps, predicate
    # never held. The one PROVEN Absence in this set.
    unbroken_live = Coverage(
        subject_id="cam-lobby-01",
        subject_kind="camera",
        interval=query_interval,
        status="live",
        envelope_ref=f"motion_gate@cam-lobby-01@envelope_sha={envelope_sha[:12]}",
        gaps=(),
        manifest_sha="sha-cov-4",
    )
    queries.append(
        _query(
            query_id="cam-lobby-01__unbroken-coverage-proves-absence",
            scenario_realism="constructed_from_real_types",
            scenario_note=(
                "Reproduces tests/test_model_coverage.py::test_unbroken_live_"
                "coverage_with_no_occurrence_proves_absence: unbroken live "
                "coverage, no gaps, predicate never held. The one query in "
                "this set that resolves to a proven Absence."
            ),
            subject_id="cam-lobby-01",
            subject_kind="camera",
            predicate_description="did anyone enter the lobby",
            predicate=_never_present,
            coverage_log=[unbroken_live],
            interval=query_interval,
        )
    )

    out_dir = Path("outputs/coverage_queries")
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = {"absence": 0, "cannot_establish": 0}
    for q in queries:
        path = out_dir / f"{q['query_id']}.json"
        path.write_text(json.dumps(q, indent=2, sort_keys=True) + "\n")
        kinds[q["kind"]] += 1
        print(f"wrote {path} ({q['kind']})")

    print(
        f"{len(queries)} quer(y/ies): {kinds['absence']} absence, "
        f"{kinds['cannot_establish']} cannot_establish"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
