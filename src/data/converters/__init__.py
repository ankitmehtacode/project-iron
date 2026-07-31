"""Dataset converters into the canonical clips + GT-event form."""

from src.data.converters.base import CanonicalClip, CanonicalClips, Converter
from src.data.converters.ntu_skeleton import (
    ACTION_TO_VERB,
    NtuParseError,
    NtuSkeletonConverter,
)

__all__ = [
    "ACTION_TO_VERB",
    "CanonicalClip",
    "CanonicalClips",
    "Converter",
    "NtuParseError",
    "NtuSkeletonConverter",
]
