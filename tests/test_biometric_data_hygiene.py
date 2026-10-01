"""Day 40: the first real biometric data in this repository (CHIRLA, lane R)
lands under data/raw/. These checks keep it — and any image crop taken from
it — out of everything git tracks.

Rules they back (FOUNDATION_REPORT.md §Day-40): evaluation reads parquet in
memory; no CHIRLA image appears in docs, artifacts, screenshots, test
fixtures or the Inspector; test fixtures stay synthetic. The first two are
checked here mechanically. In-memory-only reading is a property of the eval
code, not of git, so it is stated there rather than pretended here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Leading bytes of the formats a person crop or a fetched shard would arrive in.
_MAGIC = {
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF8": "gif",
    b"RIFF": "riff/webp",
    b"PAR1": "parquet",
}
_MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".parquet",
    ".npy",
    ".npz",
    ".h5",
    ".mp4",
    ".avi",
}

# Media tracked before Day 40, none of it from CHIRLA. Adding to this set is a
# reviewed decision, never a way to make the check pass.
PRE_DAY40_MEDIA = {
    "src/interface/ui/data/raw/test_video.mp4": "UI demo clip, pre-Day-40",
    "tests/golden/clips/real_test_video.npz": "golden-vector clip, pre-Day-40",
    "tests/golden/clips/synthetic_seeded_noise.npz": "synthetic golden clip",
    "tests/golden/clips/synthetic_spatial_gradient.npz": "synthetic golden clip",
    "tests/golden/clips/synthetic_tubelet_probe.npz": "synthetic golden clip",
    "tests/golden/reference/real_test_video.npz": "golden reference output",
    "tests/golden/reference/synthetic_seeded_noise.npz": "golden reference output",
    "tests/golden/reference/synthetic_spatial_gradient.npz": "golden reference output",
    "tests/golden/reference/synthetic_tubelet_probe.npz": "golden reference output",
}


def _tracked(*pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *pathspec],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def test_nothing_under_data_is_tracked() -> None:
    assert _tracked("data/") == []


def test_no_tracked_file_carries_image_or_dataset_bytes() -> None:
    offenders = []
    for rel in _tracked():
        if rel in PRE_DAY40_MEDIA:
            continue
        path = REPO_ROOT / rel
        if path.suffix.lower() in _MEDIA_SUFFIXES:
            offenders.append(f"{rel} (suffix)")
            continue
        try:
            with open(path, "rb") as handle:
                head = handle.read(4)
        except OSError:  # deleted in the working tree; git still lists it
            continue
        kind = next((k for magic, k in _MAGIC.items() if head.startswith(magic)), None)
        if kind:
            offenders.append(f"{rel} ({kind} bytes)")
    assert not offenders, offenders


def test_the_pre_day40_media_list_is_not_stale() -> None:
    """An entry that is no longer tracked is removed, not left as a
    standing exemption for a future file at the same path."""
    tracked = set(_tracked())
    assert set(PRE_DAY40_MEDIA) <= tracked
