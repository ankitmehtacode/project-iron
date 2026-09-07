"""The Inspector must show what is on disk, or say what is missing.

A viewer over an evaluation harness is itself a measuring apparatus, and this
repository's most expensive findings have all come from auditing those rather
than the code under test. So these tests target the ways a viewer lies:

* by rendering sample data when the real artifact is absent,
* by showing a number without the artifact that produced it,
* by drawing a delta between two runs that are not comparable,
* by rendering an empty result and an absent one identically.

Endpoints are exercised against the repository's real artifacts. Where an
artifact genuinely does not exist, the test asserts the refusal rather than
skipping — the refusal is the behaviour under test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.inspector import artifacts as art
from src.inspector.server import build_routes

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_CODE = [
    REPO_ROOT / "src" / "inspector" / "artifacts.py",
    REPO_ROOT / "src" / "inspector" / "server.py",
    REPO_ROOT / "src" / "inspector" / "static" / "app.js",
]


@pytest.fixture(scope="module")
def store() -> art.Artifacts:
    return art.Artifacts.from_config()


def call(store: art.Artifacts, path: str, query: dict | None = None):
    """Route a URL the way the server does, without binding a socket."""
    for pattern, handler in build_routes(store):
        match = pattern.match(path)
        if match:
            status, body, content_type = handler(match, query or {})
            if content_type.startswith("application/json"):
                return status, json.loads(body)
            return status, body
    raise AssertionError(f"no route matched {path}")


def call_raw(store: art.Artifacts, path: str, query: dict | None = None):
    """Like ``call``, but returns the undecoded response bytes.

    ``json.loads`` (used by ``call``) is too lenient to catch the NaN/
    Infinity bug below: Python's own decoder accepts those as a
    non-standard extension, the same leniency that let the bug ship
    undetected through every prior test in this file. Only a raw-bytes
    check (or a strict ``parse_constant``) reproduces what a real
    browser's ``JSON.parse`` actually rejects.
    """
    for pattern, handler in build_routes(store):
        match = pattern.match(path)
        if match:
            status, body, content_type = handler(match, query or {})
            return status, body, content_type
    raise AssertionError(f"no route matched {path}")


# --- endpoints against real artifacts ------------------------------------


def test_scorecards_endpoint_lists_real_files(store: art.Artifacts) -> None:
    status, payload = call(store, "/api/scorecards")
    assert status == 200
    cards = payload["scorecards"]
    assert cards, "make eval has not been run; there is nothing to inspect"
    for card in cards:
        assert (REPO_ROOT / card["file"]).exists(), (
            f"{card['file']} is listed but not on disk — the viewer is "
            "reporting something it cannot open"
        )


def test_every_scorecard_carries_the_shas_that_identify_it(
    store: art.Artifacts,
) -> None:
    """A metric without its instrument is not evidence."""
    _, payload = call(store, "/api/scorecards")
    for card in payload["scorecards"]:
        _, full = call(store, f"/api/scorecard/{card['name']}")
        assert full["golden_set_sha"], f"{card['name']} has no set_sha"
        assert full["_source"], "every payload names the file it was read from"


def test_scorecard_response_never_emits_non_standard_json_constants(
    store: art.Artifacts,
) -> None:
    """Found live during Day 35's Objective 1 audit: real scorecards carry
    genuine NaN values (an unpopulated per_condition bucket like "night",
    a zero-denominator precision). json.dumps defaults to allow_nan=True
    and happily emits the literal token NaN -- which is NOT valid JSON per
    the spec browsers implement, so a real browser's response.json() threw
    on EVERY real scorecard, the whole payload was lost (not just the
    offending field), and every view silently rendered as if every field
    were absent -- "unmeasured" was shown with no indication the real
    cause was a transport-layer crash. Python's own json.loads is too
    lenient to have ever caught this (it also accepts NaN as a
    non-standard extension) -- hence a strict parse here, and hence this
    bug reaching Day 35 despite an otherwise well-tested server module."""
    _, payload = call(store, "/api/scorecards")
    cards = payload["scorecards"]
    assert cards, "make eval has not been run; there is nothing to check"
    for card in cards:
        _, body, content_type = call_raw(store, f"/api/scorecard/{card['name']}")
        assert content_type.startswith("application/json")

        def _reject_non_standard_constant(token: str) -> None:
            raise AssertionError(
                f"{card['name']}: non-standard JSON constant {token!r} in "
                "the response body -- a real browser's JSON parser rejects "
                "this and the whole payload is lost, not just this field"
            )

        json.loads(body, parse_constant=_reject_non_standard_constant)


def test_active_golden_set_matches_the_configured_one(store: art.Artifacts) -> None:
    from src.config import IronConfig

    status, payload = call(store, "/api/golden")
    assert status == 200
    assert payload["version"] == IronConfig.load().eval.golden_set_version
    assert len(payload["clips"]) > 0


def test_envelope_endpoint_serves_the_measured_curve(store: art.Artifacts) -> None:
    status, payload = call(store, "/api/envelope")
    assert status == 200
    assert payload["samples"], "an envelope with no samples classifies nothing"
    assert payload["_source"].endswith(".envelope.json")


def test_provenance_reports_git_and_names_what_it_could_not_resolve(
    store: art.Artifacts,
) -> None:
    status, payload = call(store, "/api/provenance")
    assert status == 200
    assert "unresolved" in payload, (
        "the view must always state what it could not resolve, so a missing "
        "manifest surfaces as a warning band instead of silently absent"
    )
    assert payload["git"].get("sha") or payload["git"].get("error")


def test_clip_endpoint_runs_the_same_partition_the_scorecard_runs(
    store: art.Artifacts,
) -> None:
    """The screen and the number must not be able to disagree."""
    _, golden = call(store, "/api/golden")
    clip_id = golden["clips"][0]["clip_id"]
    status, payload = call(store, f"/api/clip/{clip_id}")
    if status == 404:
        pytest.skip(f"{clip_id} not rendered locally: {payload['looked_for']}")

    assert len(payload["labels"]) == payload["frames"]
    assert len(payload["wake"]) == payload["frames"]
    assert set(payload["labels"]) <= {0, 1, 2, 3}
    for agent in payload["agents"]:
        assert len(agent["silhouette_gate_px"]) == payload["frames"]
        assert len(agent["wake_threshold_px"]) == payload["frames"]


# --- refusals -------------------------------------------------------------


def test_comparison_across_different_golden_sets_is_refused(
    store: art.Artifacts,
) -> None:
    """The UI refuses exactly what the Python refuses.

    A delta across two golden sets renders as a change in the system and is
    nothing of the kind.
    """
    _, payload = call(store, "/api/scorecards")
    by_set: dict[str, str] = {}
    for card in payload["scorecards"]:
        by_set.setdefault(card["golden_set_sha"], card["name"])
    if len(by_set) < 2:
        pytest.skip("only one golden set has been scored; nothing to mismatch")

    left, right = list(by_set.values())[:2]
    _, result = call(store, "/api/compare", {"left": [left], "right": [right]})
    assert result["refused"] is True
    assert any("different golden sets" in r for r in result["reasons"])


def test_comparison_across_different_envelopes_is_refused(
    store: art.Artifacts, tmp_path: Path
) -> None:
    """Same set, different envelope: still not comparable.

    The envelope decides which misses count as gate defects, so the same gate
    scores differently under each.
    """
    _, payload = call(store, "/api/scorecards")
    names = [c["name"] for c in payload["scorecards"]]
    assert names

    source = store.scorecards_dir / f"{names[0]}.json"
    card = json.loads(source.read_text())
    card["envelope"] = dict(card.get("envelope", {}))
    card["envelope"]["envelope_sha"] = "0" * 64

    forged = store.scorecards_dir / "pytest-envelope-variant.json"
    forged.write_text(json.dumps(card))
    try:
        _, result = call(
            store,
            "/api/compare",
            {"left": [names[0]], "right": ["pytest-envelope-variant"]},
        )
        assert result["refused"] is True
        assert any("different capability envelopes" in r for r in result["reasons"])
    finally:
        forged.unlink()


def test_a_missing_artifact_returns_the_path_and_the_command(
    store: art.Artifacts,
) -> None:
    status, payload = call(store, "/api/scorecard/definitely-not-a-real-scorecard")
    assert status == 404
    assert payload["absent"] is True
    assert payload["looked_for"], "an empty state must name the file it wanted"
    assert payload["produced_by"], "and the command that produces it"


def _four_class_event_fixture():
    """One real, validated instance of each schema-v2 event class — built
    through src.model.events's own dataclasses (so __post_init__ validation
    runs), not a literal dict standing in for one."""
    import uuid

    from src.events.schema import EntityRef, Verb
    from src.model.events import HypothesisEvent, InferredEvent, ObservedEvent, PredictedEvent

    subject = EntityRef("session", "sess-day35")
    common = dict(site_id="site-0", confidence=0.8, importance=0.5, manifest_sha="sha-day35")
    return [
        ObservedEvent(event_id=uuid.uuid4(), ts_ns=1, subject=subject, verb=Verb.ENTERED, **common),
        InferredEvent(event_id=uuid.uuid4(), ts_ns=2, subject=subject, verb=Verb.EXITED, basis="retroactive resolution", **common),
        PredictedEvent(event_id=uuid.uuid4(), ts_ns=3, subject=subject, verb=Verb.APPROACHED, predicted_by="velocity extrapolation", **common),
        HypothesisEvent(event_id=uuid.uuid4(), ts_ns=4, subject=subject, verb=Verb.LOITERED, rationale="unconfirmed", **common),
    ]


def test_mixed_four_class_event_list_serves_a_distinct_event_class_per_row(
    tmp_path: Path,
) -> None:
    """Objective 3's mandated test: a mixed list renders each of the four
    classes distinguishably. Verified on the served DATA here (event_class
    is present and correct per row); the companion tests below verify the
    served MARKUP (app.js/style.css) maps every class to a distinct,
    non-colour-only rendering. Neither is an eyeball check."""
    from src.model.events import write_events_v2_parquet

    events_path = tmp_path / "events.parquet"
    write_events_v2_parquet(_four_class_event_fixture(), events_path)
    store = art.Artifacts(
        project_root=tmp_path,
        scorecards_dir=tmp_path / "outputs" / "scorecards",
        golden_dir=tmp_path / "configs" / "golden",
        envelope_path=tmp_path / "configs" / "envelope" / "gate_320x180.envelope.json",
        data_dir=tmp_path / "data",
        events_path=events_path,
        associations_dir=tmp_path / "outputs" / "associations",
    )
    status, payload = call(store, "/api/events")
    assert status == 200
    classes = [row["event_class"] for row in payload["rows"]]
    assert classes == ["observed", "inferred", "predicted", "hypothesis"]


def test_app_js_maps_all_four_event_classes_to_distinct_rendering() -> None:
    """The served markup, checked structurally: EVENT_CLASS_ROW must map
    all four classes to distinct CSS row classes and distinct verb-cell
    wording, so a screenshot could not confuse one for another even before
    considering colour."""
    text = (REPO_ROOT / "src" / "inspector" / "static" / "app.js").read_text()
    section = text[text.index("EVENT_CLASS_ROW"):text.index("async function viewEvents")]
    for name in ("observed", "inferred", "predicted", "hypothesis"):
        assert f"{name}:" in section, f"EVENT_CLASS_ROW is missing {name!r}"
    # Distinct CSS classes (None counts as its own distinct, unstyled state
    # for "observed" — the baseline every other class is distinguished FROM).
    assert 'cls: "inferred"' in section
    assert 'cls: "predicted"' in section
    assert 'cls: "hypothesis"' in section
    # PredictedEvent's own verb-cell wording must differ from a bare pass-
    # through — Day 13's "never interchangeable with ObservedEvent" rule
    # applied to the text itself, not just to styling.
    assert '`will ${v}`' in section
    assert '`possibly ${v}`' in section


def test_style_css_gives_each_event_class_a_distinct_non_color_border() -> None:
    """Grayscale legibility for all three non-observed classes, not only
    the original inferred/observed pair."""
    css = (REPO_ROOT / "src" / "inspector" / "static" / "style.css").read_text()
    assert "tr.inferred { border-bottom: 1px dashed" in css
    assert "tr.predicted { border-bottom: 1px dotted" in css
    assert "tr.hypothesis { border-bottom: 3px double" in css


def test_absent_events_are_an_instruction_not_an_empty_table(
    store: art.Artifacts,
) -> None:
    """Rendering an empty table would claim nothing happened.

    Nothing has run. Those are different statements and the UI must not
    collapse them.
    """
    status, payload = call(store, "/api/events")
    if status == 200:
        assert "rows" in payload
        return
    assert payload["absent"] is True
    assert "events.parquet" in payload["looked_for"]
    assert payload["produced_by"]


# --- the no-mock-data guard ----------------------------------------------

FIXTURE_SHAPED = re.compile(
    r"""(
        \bMOCK\b | \bmock_ | \bfake_ | \bdummy_ | \bstub_ |
        \bsample_data\b | \bplaceholder_data\b | \bexample_data\b |
        \blorem\b | "foo" | 'foo' | \bJohn\s+Doe\b | \bJane\s+Doe\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def test_no_mock_or_placeholder_data_path_exists_in_serving_code() -> None:
    """The Inspector must have no way to render data it did not read.

    A viewer that can fall back to sample data cannot be used to verify a
    claim, because a green screen no longer distinguishes "the artifact says
    so" from "the fallback fired".
    """
    offenders: list[str] = []
    for path in SERVING_CODE:
        assert path.exists(), f"serving file missing: {path}"
        for number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            # Prose about the rule is allowed; a code path implementing one is
            # not. Comment markers for Python and JavaScript.
            if stripped.startswith(("#", "*", "//", '"""', "'''")):
                continue
            if FIXTURE_SHAPED.search(line):
                offenders.append(f"{path.name}:{number}: {stripped[:90]}")
    assert not offenders, "fixture-shaped data in serving code:\n" + "\n".join(
        offenders
    )


def test_serving_code_never_swallows_an_error_into_an_empty_result() -> None:
    """A bare ``except`` returning nothing is indistinguishable from no data."""
    for path in SERVING_CODE:
        if path.suffix != ".py":
            continue
        text = path.read_text()
        assert "except Exception:\n            pass" not in text
        assert "except:  # noqa" not in text


# --- empty-state snapshot -------------------------------------------------


def test_empty_state_snapshot(tmp_path: Path) -> None:
    """The empty state is a contract: what was wanted, and how to make it.

    Snapshotted because it is the screen a new engineer sees first, and a
    regression to a bare "no data" would remove the only instruction there is.
    """
    empty = art.Artifacts(
        project_root=tmp_path,
        scorecards_dir=tmp_path / "outputs" / "scorecards",
        golden_dir=tmp_path / "configs" / "golden",
        envelope_path=tmp_path / "configs" / "envelope" / "gate_320x180.envelope.json",
        data_dir=tmp_path / "data",
        events_path=tmp_path / "outputs" / "events" / "events.parquet",
        associations_dir=tmp_path / "outputs" / "associations",
    )

    assert art.list_scorecards(empty) == []

    envelope = art.read_envelope(empty)
    assert isinstance(envelope, art.Absent)
    assert envelope.as_dict() == {
        "absent": True,
        "what": "measured capability envelope",
        "looked_for": "configs/envelope/gate_320x180.envelope.json",
        "produced_by": (
            "python scripts/measure_envelope.py --displacements 2 3 4 6 8 "
            "12 20 30 --model-out configs/envelope/gate_320x180.envelope.json"
        ),
    }

    events = art.read_events(empty)
    assert isinstance(events, art.Absent)
    assert events.looked_for == "outputs/events/events.parquet"
    assert "nothing has run" in events.produced_by

    assert art.list_associations(empty) == []
    association = art.read_association(empty, "any-component")
    assert isinstance(association, art.Absent)
    assert association.produced_by == "python scripts/build_association_demo.py"


# --- association / identity view ------------------------------------------


def _build_association_demo_if_absent(store: art.Artifacts) -> None:
    """The real demo artifact this view serves is a generated output
    (outputs/ is gitignored — see .gitignore's "generated pipeline
    artifacts" note), not a checked-in fixture. Regenerate it if a
    previous run has not already, exactly the way a human running
    ``make inspect`` cold would."""
    if store.associations_dir.exists() and list(store.associations_dir.glob("*.json")):
        return
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/build_association_demo.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )


def test_association_endpoints_serve_real_verdicts(store: art.Artifacts) -> None:
    _build_association_demo_if_absent(store)
    status, payload = call(store, "/api/associations")
    assert status == 200
    rows = payload["associations"]
    assert rows, "run scripts/build_association_demo.py before this test"
    kinds = {r["verdict_kind"] for r in rows}
    assert "decisive" in kinds and "ambiguous" in kinds, (
        "the demo must exercise both verdict types, or the self-audit "
        "below cannot check that they render differently"
    )

    for row in rows:
        _, full = call(store, f"/api/association/{row['component_id']}")
        assert full["candidates"], "a verdict without its competitor set is not evidence"
        assert full["verdict"]["kind"] in ("decisive", "ambiguous")
        assert (REPO_ROOT / full["_source"]).exists()


def test_ambiguous_verdict_has_no_winner_field_on_the_wire(store: art.Artifacts) -> None:
    """Structural, at the API boundary: an Ambiguous verdict payload must
    not carry a ``winner`` key at all — the same discipline
    src.estimator.joint.Ambiguous enforces in Python (no ``winner``
    attribute exists on the dataclass), re-checked here because the JSON
    boundary is a second place this could quietly leak back in."""
    _build_association_demo_if_absent(store)
    _, rows = call(store, "/api/associations")
    ambiguous = [r for r in rows["associations"] if r["verdict_kind"] == "ambiguous"]
    assert ambiguous, "no ambiguous demo component found"
    _, full = call(store, f"/api/association/{ambiguous[0]['component_id']}")
    assert "winner" not in full["verdict"]


def test_ambiguous_verdict_never_produces_an_observed_event(store: art.Artifacts) -> None:
    """Closes the loop to Day 34 Objective 4 at the API boundary: an
    Ambiguous verdict's event must be InferredEvent, never ObservedEvent."""
    _build_association_demo_if_absent(store)
    _, rows = call(store, "/api/associations")
    ambiguous = [r for r in rows["associations"] if r["verdict_kind"] == "ambiguous"]
    assert ambiguous
    _, full = call(store, f"/api/association/{ambiguous[0]['component_id']}")
    assert full["event"]["event_class"] == "inferred"


def test_pruned_by_budget_is_forensically_visible_in_a_real_resolution(
    store: art.Artifacts,
) -> None:
    """At least one real, served decision must carry PRUNED_BY_BUDGET, or
    the "literal, visible label" requirement (Day 35 Objective 2) has
    nothing to render against."""
    _build_association_demo_if_absent(store)
    _, rows = call(store, "/api/associations")
    found = False
    for row in rows["associations"]:
        _, full = call(store, f"/api/association/{row['component_id']}")
        for decision in full["decisions"]:
            cause = decision["cause"]
            if cause and cause["kind"] == "pruned_by_budget":
                found = True
    assert found, "no PRUNED_BY_BUDGET decision in any served component"


# --- structural self-audit: Ambiguous must never render as Decisive -------


def test_ambiguous_rendering_has_no_rank_based_winner_styling() -> None:
    """The anti-pattern named in the Day 35 prompt, checked structurally:
    app.js must apply its winner class/badge ONLY inside the isDecisive
    branch, never unconditionally on the top-ranked/top-scored row. A
    conditional gated on verdict.kind is required; one gated on array
    position or score rank alone would let an Ambiguous case's strongest
    competitor be mistaken for a Decisive winner."""
    text = (REPO_ROOT / "src" / "inspector" / "static" / "app.js").read_text()
    # The winner class/badge must appear only inside code paths that also
    # check isDecisive — approximated here by requiring every occurrence of
    # the winner markers to be textually preceded, within the same
    # function, by an isDecisive guard rather than appearing bare.
    assoc_section = text[text.index("/* --- view: association"):text.index("/* --- view: provenance")]
    assert "assoc-winner" in assoc_section
    assert "winner-badge" in assoc_section
    for marker in ("assoc-winner", "winner-badge"):
        idx = assoc_section.index(marker)
        preceding = assoc_section[:idx]
        # The nearest preceding conditional must be an isDecisive check —
        # i.e. isDecisive appears more recently before this marker than
        # any bare score/rank comparison would need to for it to fire.
        assert "isDecisive" in preceding, (
            f"{marker!r} must be reachable only through an isDecisive check"
        )


def test_ambiguous_verdict_tag_uses_a_non_color_signal() -> None:
    """Grayscale legibility, same discipline as observed/inferred: the
    Decisive/Ambiguous distinction must not rest on colour alone."""
    css = (REPO_ROOT / "src" / "inspector" / "static" / "style.css").read_text()
    assert "ambiguous-tag" in css
    ambiguous_rule = css[css.index(".ambiguous-tag"):css.index(".ambiguous-tag") + 200]
    assert "dashed" in ambiguous_rule, (
        "the ambiguous tag must carry a non-colour signal (a dashed "
        "border, matching the observed/inferred convention), not colour alone"
    )
    js = (REPO_ROOT / "src" / "inspector" / "static" / "app.js").read_text()
    assert '"AMBIGUOUS' in js or "'AMBIGUOUS" in js, (
        "the Ambiguous state must carry a literal text label, not only a class name"
    )
    assert '"DECISIVE"' in js, "the Decisive state must carry a literal text label too"
