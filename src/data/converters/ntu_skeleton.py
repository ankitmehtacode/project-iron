"""NTU-RGB+D skeleton files → POSE-verb ground truth in the event schema.

The first concrete converter, chosen because NTU is the primary POSE-verb
benchmark and its format is simple enough to prove the canonical interface
end to end.

What the format gives us
------------------------
An NTU sample is a text file named like ``S001C002P003R002A008.skeleton``: the
``A008`` field is the action class, and the file body carries per-frame
skeleton joints. For verb GT the filename's action class and the frame count
are what matter; the joints themselves feed a future keypoint pipeline, not
this converter.

The mapping is deliberately conservative
----------------------------------------
Only NTU classes whose meaning maps *exactly* onto a schema verb are
converted; everything else lands in ``skipped`` with the reason. A wrong GT
label is strictly worse than a missing one: missing GT shrinks the eval,
wrong GT corrupts it while inflating the score. Growing this mapping is
welcome — one class at a time, each with its NTU definition quoted, never by
"close enough".

NTU is lane R
-------------
This GT is for **evaluation only**. Converting it does not change its lane,
and the converter does not need to re-check that here — the loaders that feed
training enforce lanes at the data boundary (:mod:`src.data.registry`).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.data.converters.base import CanonicalClip, CanonicalClips
from src.events import EntityRef, Event, Verb, deterministic_event_id

# NTU records at 30 fps; timestamps are synthesized from frame indices.
NTU_FPS = 30.0
NANOSECONDS_PER_SECOND = 1_000_000_000

# S=setup, C=camera, P=performer, R=replication, A=action class.
FILENAME_PATTERN = re.compile(
    r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<performer>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})$"
)

# NTU action class -> schema verb, for classes whose NTU definition matches
# the verb exactly. Each entry quotes the NTU class name it was checked
# against. POSE verbs carry no object; FELL is GEOMETRIC and also object-free,
# so nothing here can violate verb/object coherence.
ACTION_TO_VERB: dict[int, Verb] = {
    8: Verb.SAT,  # A008 "sitting down"
    9: Verb.STOOD,  # A009 "standing up (from sitting position)"
    43: Verb.FELL,  # A043 "falling"
    80: Verb.BENT_DOWN,  # A080 "squat down" — torso lowers; matches bent_down
}

SITE_ID = "dataset:ntu-rgbd-120"


class NtuParseError(ValueError):
    """Raised when a .skeleton file does not parse.

    An unparseable GT file is refused, not skipped: ``skipped`` is for samples
    the mapping does not cover, which is a policy outcome. A corrupt file is a
    data problem someone must look at, and folding it into ``skipped`` would
    hide it among hundreds of legitimately unmapped classes.
    """


def parse_frame_count(path: Path) -> int:
    """Read the frame count: the first line of a .skeleton file.

    The rest of the body (per-frame joint blocks) is validated no further
    here; this converter consumes the temporal extent only.
    """
    with open(path) as handle:
        first = handle.readline().strip()
    try:
        frames = int(first)
    except ValueError as exc:
        raise NtuParseError(
            f"{path.name}: first line must be the frame count, got {first!r}"
        ) from exc
    if frames <= 0:
        raise NtuParseError(f"{path.name}: nonsensical frame count {frames}")
    return frames


class NtuSkeletonConverter:
    """Converter for NTU-RGB+D ``.skeleton`` directories."""

    def __init__(self, manifest_sha: str) -> None:
        """
        Args:
            manifest_sha: Manifest of the conversion run. GT events carry it
                like production events carry theirs — conversions have bugs
                too, and GT that cannot be traced to the code that wrote it
                cannot be re-examined when a metric looks wrong.
        """
        if not manifest_sha:
            raise ValueError("a conversion run needs a manifest_sha")
        self._manifest_sha = manifest_sha

    @property
    def dataset(self) -> str:
        return "NTU-RGBD-120"

    def convert(self, raw_dir: Path) -> CanonicalClips:
        """Translate every ``.skeleton`` file under ``raw_dir``.

        Raises:
            NtuParseError: on a malformed file. See :class:`NtuParseError` for
                why corruption is not folded into ``skipped``.
        """
        result = CanonicalClips(dataset=self.dataset)

        for path in sorted(raw_dir.rglob("*.skeleton")):
            clip_id = path.stem
            match = FILENAME_PATTERN.match(clip_id)
            if match is None:
                raise NtuParseError(
                    f"{path.name}: filename does not follow the NTU "
                    "SxxxCxxxPxxxRxxxAxxx convention; refusing to guess the "
                    "action class"
                )

            frames = parse_frame_count(path)
            action = int(match.group("action"))
            performer = int(match.group("performer"))

            result.clips.append(
                CanonicalClip(
                    clip_id=clip_id,
                    source_dataset=self.dataset,
                    frames=frames,
                    fps=NTU_FPS,
                )
            )

            verb = ACTION_TO_VERB.get(action)
            if verb is None:
                result.skipped[clip_id] = (
                    f"action A{action:03d} has no exact verb mapping; "
                    "deliberately unconverted rather than approximated"
                )
                continue

            # The action is the whole clip, so the event timestamp is the clip
            # midpoint — synthesized from frame count at NTU's fixed 30 fps.
            # +1 so a 1-frame clip still gets a positive, nonzero timestamp.
            midpoint_ns = int(
                (frames / 2.0) / NTU_FPS * NANOSECONDS_PER_SECOND
            ) + 1

            result.events.append(
                Event(
                    event_id=deterministic_event_id(
                        SITE_ID, clip_id, verb.value
                    ),
                    site_id=SITE_ID,
                    ts_ns=midpoint_ns,
                    # Performers are numbered, not identified. A session ref
                    # keeps NTU subject numbers in the anonymous keyspace,
                    # exactly as an unenrolled person would be in production.
                    subject=EntityRef("session", f"ntu-performer-{performer:03d}"),
                    verb=verb,
                    confidence=1.0,  # ground truth: annotated, not detected
                    observed=True,
                    # Importance is a production triage score, not a property
                    # of GT. Zero states "unscored" honestly; evaluating the
                    # importance model against GT it wrote itself would be
                    # circular.
                    importance=0.0,
                    manifest_sha=self._manifest_sha,
                )
            )

        return result
