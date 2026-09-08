"""Day 37, Objective 3 -- a REAL association sweep, replacing two hand-picked
components with a systematic one across an entire golden set.

``scripts/build_association_demo.py`` (Day 35, retired by this script) built
real geometry-grounded ``AssociationCandidate``s, but for exactly two
components a human had already picked BY searching the clip for a near-tie
and a landslide. It proved the Inspector could render both verdict shapes.
It never asked what ``resolve_data_association`` actually returns across a
real golden set, because nothing in this project had ever run it that way.
This script does.

Why not v6-motion
------------------
The Day 37 prompt that asked for this named v6-motion as the default target.
Checked directly against ``configs/golden/v6-motion.golden.json`` and every
one of its 19 clips' own ``agent_xyz`` arrays (never the manifest's free-text
notes): EVERY v6-motion clip has exactly 1 agent.
``resolve_data_association``'s job is choosing among COMPETING hypotheses
for what a detection continues; with one agent in the scene there is never
more than one candidate, so every "resolution" would be a vacuous Decisive
verdict with no real competitor -- testing nothing about the association
mechanism itself. v6-motion was built for the motion-gate/cessation work
(Days 23-31), not multi-agent identity resolution, and v5-cessation (also
checked directly) is the same story.

Only v2-indoor (9 clips) and v3-indoor (30 clips) have multi-agent clips.
v3-indoor is chosen here: larger, and the same clip family
``build_association_demo.py`` already drew its two hand-picked components
from -- generalized to every multi-agent clip it has, not two.

Note on v3-indoor's own DO-NOT-QUOTE caveat: Day 31 found v3-indoor's ground
truth is unrealistically constant-velocity, which invalidates it as a source
of PRODUCT motion-gate numbers (wake_fraction, recall). That finding is
about the GATE's own threshold behaviour against unrealistic acceleration,
not about whether v3-indoor's multi-agent positions are valid input to a
DATA-ASSOCIATION test -- association only needs relative agent geometry at
an instant, which is exactly as real in v3-indoor as in any other synthetic
set. No product timing/accuracy claim is made from this script's output.

What "systematic" means, stated before any result below was inspected --
so nothing here can be read as tuned to produce a chosen answer:
  * every clip in the golden set with >= 2 agents, checked directly against
    each clip's own ``agent_xyz.shape[1]`` (never the manifest's notes)
  * ONE frame per clip: the middle frame (``num_frames // 2``), fixed by
    this rule for every clip alike, not chosen per-clip
  * EVERY agent in that clip, in turn, as the "detected" agent -- an
    N-agent clip contributes N components, one per agent, not one
  * ``budget=1`` for every component, uniformly -- the realistic
    real-world constraint (one accepted continuation per detection),
    never tuned per-component to force a particular death cause

Output: one ``outputs/associations/<component_id>.json`` per component
(same schema ``build_association_demo.py`` used, so the Inspector's existing
rendering needs no change), plus a printed summary answering this
objective's own named question: how many real verdicts are Decisive vs
Ambiguous, and which component_ids are Ambiguous, by name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import IronConfig
from src.data.golden import GoldenSetError, load_golden_set
from src.estimator.association_from_geometry import resolve_geometric_component
from src.events.schema import Verb

ASSUMED_FPS = 12.0
"""This repo's synthetic clips carry no explicit frame-rate field, so 12 fps
is a stated assumption, not a measured one -- unchanged from Day 35."""

MIN_AGENTS_FOR_COMPETITION = 2
BUDGET = 1

BOUNDARY_FIXTURE_CLIP = "crowded_6agents__cam_a"
BOUNDARY_FIXTURE_DATASET = "synthetic-indoor-v3"
BOUNDARY_FIXTURE_FRAME = 13
BOUNDARY_FIXTURE_DETECTED_AGENT = 0
"""The one hand-picked case Day 35's now-retired build_association_demo.py
found by searching v3-indoor's real distances for a near-tie: agents 0 and 3
are 1.8 cm apart at frame 15, already close at frame 13. Kept as a SEPARATE,
explicitly self-labelled fixture (selection="boundary_fixture_hand_selected"
in the payload) so the Association view's Ambiguous rendering path has a
real example to be tested against even though this script's own systematic
sweep (see main()) found zero -- never counted toward, or confused with,
the systematic sweep's own answer."""


def _multi_agent_clips(
    golden_dir: Path, data_dir: Path, version: str
) -> list[tuple[str, str, Path, np.ndarray]]:
    """Every clip in ``version`` with >= 2 agents, checked directly against
    the clip's own array shape. Returns (clip_id, source_dataset, path, xyz)."""
    golden = load_golden_set(golden_dir, version)
    result = []
    for clip in golden.clips:
        dataset = clip.source_dataset or version
        path = data_dir / "synthetic" / dataset / f"{clip.clip_id}.npz"
        if not path.exists():
            print(f"skip {clip.clip_id}: no clip file at {path}", file=sys.stderr)
            continue
        with np.load(path) as sample:
            xyz = np.asarray(sample["agent_xyz"])
        if xyz.shape[1] < MIN_AGENTS_FOR_COMPETITION:
            continue
        result.append((clip.clip_id, dataset, path, xyz))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-version", default="v3-indoor")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/associations"))
    parser.add_argument(
        "--clear",
        action="store_true",
        help="remove every existing *.json in --out-dir first (e.g. the "
        "retired build_association_demo.py's ambiguous-demo.json / "
        "decisive-demo.json) so the Inspector serves only this sweep",
    )
    parser.add_argument(
        "--no-boundary-fixture",
        action="store_false",
        dest="boundary_fixture",
        help="skip writing the hand-selected near-tie fixture (see "
        "BOUNDARY_FIXTURE_* above) -- off by default only for a run that "
        "wants the systematic sweep's own answer with nothing else mixed in",
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    golden_dir = config.paths.resolve(config.eval.golden_sets_dir)
    data_dir = config.paths.resolved_data_dir

    try:
        clips = _multi_agent_clips(golden_dir, data_dir, args.golden_version)
    except GoldenSetError as exc:
        print(f"associate_golden_set: {exc}", file=sys.stderr)
        return 1

    if not clips:
        print(
            f"associate_golden_set: no multi-agent clip found in "
            f"{args.golden_version!r} -- nothing to resolve",
            file=sys.stderr,
        )
        return 1

    if args.clear and args.out_dir.exists():
        for old in args.out_dir.glob("*.json"):
            old.unlink()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = config.config_sha()

    decisive_ids: list[str] = []
    ambiguous_ids: list[str] = []
    skipped_short_clips: list[str] = []

    for clip_id, dataset, path, xyz in clips:
        n_frames, n_agents = xyz.shape[0], xyz.shape[1]
        frame = n_frames // 2
        if frame < 2:
            skipped_short_clips.append(clip_id)
            continue
        for detected_agent in range(n_agents):
            component_id = (
                f"{args.golden_version}__{clip_id}__f{frame}__agent-{detected_agent}"
            )
            payload = resolve_geometric_component(
                xyz,
                component_id=component_id,
                clip_id=clip_id,
                source_path=str(path),
                site_id=f"golden-{args.golden_version}",
                frame=frame,
                detected_agent=detected_agent,
                budget=BUDGET,
                assumed_fps=ASSUMED_FPS,
                verb=Verb.APPROACHED,
                importance=0.5,
                caller_confidence=0.9,
                manifest_sha=manifest_sha,
                config_sha=manifest_sha,
                event_namespace="iron://day37-associate-golden-set",
                selection="systematic_sweep",
            )
            out_path = args.out_dir / f"{component_id}.json"
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            kind = payload["verdict"]["kind"]
            (decisive_ids if kind == "decisive" else ambiguous_ids).append(component_id)

    total = len(decisive_ids) + len(ambiguous_ids)
    print(
        f"resolved {total} real component(s) from {args.golden_version} "
        f"({len(clips)} multi-agent clip(s), 1 frame each, every agent as "
        "detected, budget=1)"
    )
    print(f"  decisive : {len(decisive_ids)}")
    print(f"  ambiguous: {len(ambiguous_ids)}")
    if ambiguous_ids:
        print("  ambiguous component_ids:")
        for cid in ambiguous_ids:
            print(f"    - {cid}")
    else:
        print(
            "  no Ambiguous verdict occurred in this systematic sweep -- "
            "reported, not omitted"
        )
    if skipped_short_clips:
        print(
            f"  skipped (fewer than 3 frames, cannot form a velocity "
            f"prediction): {skipped_short_clips}"
        )
    print(f"wrote {total} file(s) under {args.out_dir}")

    if args.boundary_fixture:
        fixture_path = (
            data_dir
            / "synthetic"
            / BOUNDARY_FIXTURE_DATASET
            / f"{BOUNDARY_FIXTURE_CLIP}.npz"
        )
        with np.load(fixture_path) as sample:
            fixture_xyz = np.asarray(sample["agent_xyz"])
        fixture_id = (
            f"boundary-fixture__{BOUNDARY_FIXTURE_CLIP}__f{BOUNDARY_FIXTURE_FRAME}"
            f"__agent-{BOUNDARY_FIXTURE_DETECTED_AGENT}"
        )
        fixture_payload = resolve_geometric_component(
            fixture_xyz,
            component_id=fixture_id,
            clip_id=BOUNDARY_FIXTURE_CLIP,
            source_path=str(fixture_path),
            site_id="boundary-fixture",
            frame=BOUNDARY_FIXTURE_FRAME,
            detected_agent=BOUNDARY_FIXTURE_DETECTED_AGENT,
            budget=BUDGET,
            assumed_fps=ASSUMED_FPS,
            verb=Verb.APPROACHED,
            importance=0.5,
            caller_confidence=0.9,
            manifest_sha=manifest_sha,
            config_sha=manifest_sha,
            event_namespace="iron://day37-associate-golden-set",
            selection="boundary_fixture_hand_selected",
        )
        (args.out_dir / f"{fixture_id}.json").write_text(
            json.dumps(fixture_payload, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"wrote 1 additional hand-selected boundary fixture "
            f"({fixture_id}, verdict={fixture_payload['verdict']['kind']}) -- "
            "NOT part of the systematic sweep above; kept only so the "
            "Association view's Ambiguous rendering path has a real "
            "example, since the sweep itself found none"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
