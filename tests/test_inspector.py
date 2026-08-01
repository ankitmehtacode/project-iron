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
