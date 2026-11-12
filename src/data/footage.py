"""Which local footage belongs to which registered dataset (Day 45).

``iron run`` takes video paths, not dataset names, so the consent gate is
applied by path: a video under a registered footage root is opened only if
its dataset passes :meth:`src.data.registry.DatasetRegistry.open_for_eval`
(for lane C, a signed consent record whose file still hashes to the recorded
sha). Video outside every root is not footage of our subjects and is not
gated here (SmartSpaces, synthetic sets).
"""

from __future__ import annotations

from pathlib import Path

from src.data.registry import DatasetRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "datasets.yaml"

FOOTAGE_ROOTS: dict[str, Path] = {
    "site-zero-solo": REPO_ROOT / "data" / "raw" / "site_zero_solo",
}


def dataset_for(video: Path) -> str | None:
    resolved = video.resolve()
    for name, root in FOOTAGE_ROOTS.items():
        if resolved.is_relative_to(root.resolve()):
            return name
    return None


def require_consented(
    videos: list[Path], registry: DatasetRegistry | None = None
) -> list[str]:
    """Raise :class:`src.data.registry.LaneViolation` for any video whose
    dataset is not cleared; return the datasets the videos belong to."""
    names = sorted({n for v in videos if (n := dataset_for(v)) is not None})
    if names:
        registry = registry or DatasetRegistry.load(REGISTRY_PATH)
        for name in names:
            registry.open_for_eval(name)
    return names
