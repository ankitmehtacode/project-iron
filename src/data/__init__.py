"""Data infrastructure: registry, converters, golden sets, annotation round-trip.

Everything here exists to keep three facts queryable forever: where every byte
of training/eval data came from, what its license actually permits, and which
artifacts were built from it. See :mod:`src.data.registry` for the lane rules.
"""

from src.data.registry import (
    LANE_DESCRIPTIONS,
    PERMANENT_BLOCKLIST,
    TRAINING_PATH_MARKER,
    BlockedDataset,
    ConsentPosture,
    ConsentRecord,
    DatasetEntry,
    DatasetRegistry,
    Lane,
    LaneViolation,
    LicenseNotVerified,
    LicenseSnapshot,
    RegistryError,
    UnknownDataset,
    normalise,
)

__all__ = [
    "LANE_DESCRIPTIONS",
    "PERMANENT_BLOCKLIST",
    "TRAINING_PATH_MARKER",
    "BlockedDataset",
    "ConsentPosture",
    "ConsentRecord",
    "DatasetEntry",
    "DatasetRegistry",
    "Lane",
    "LaneViolation",
    "LicenseNotVerified",
    "LicenseSnapshot",
    "RegistryError",
    "UnknownDataset",
    "normalise",
]
