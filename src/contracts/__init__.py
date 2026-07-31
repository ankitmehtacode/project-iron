"""Typed boundary contracts: arrays that carry their own meaning.

Every production defect this repository's audit turned up was a *boundary* bug:
wrong units crossing a function call, stale focal lengths crossing a resize, a
wrong temporal token count crossing a docstring. None of them raised an
exception. All of them produced correctly-shaped, entirely wrong numbers.

The fix is structural rather than a matter of care. A bare ``np.ndarray``
carries no statement about what its values mean, so nothing can check it. The
envelopes here carry that statement, and consumers verify it at the boundary:

===================  ====================================  ==========================
Envelope             Carries                               Bug it makes impossible
===================  ====================================  ==========================
``DepthField``       units, geometry, valid mask           disparity labelled metres
``PatchTokens``      span (with tubelet), grid, encoder    temporal off-by-2
``Intrinsics``       fx, fy, cx, cy, distortion, valid_for stale focal after resize
``AffineTransform``  stage space to canonical space        letterbox / stretch drift
===================  ====================================  ==========================

Consumers raise, they never coerce. :func:`unproject` rejects non-metric depth
instead of guessing a conversion, because a silent auto-conversion recreates
the original bug with more confidence attached.

The canonical frame is the original video's pixel space, with pixel centres at
half-integer coordinates; see :mod:`src.contracts.frames` for why that
convention was chosen and what it buys.
"""

from src.contracts.errors import ContractError, GeometryMismatch, UnitsError
from src.contracts.fields import ALL_UNITS, METRIC_UNITS, DepthField, Units
from src.contracts.frames import AffineTransform, FrameGeometry, Intrinsics
from src.contracts.geometry import unproject
from src.contracts.tokens import PatchTokens, TemporalSpan

__all__ = [
    "ALL_UNITS",
    "METRIC_UNITS",
    "AffineTransform",
    "ContractError",
    "DepthField",
    "FrameGeometry",
    "GeometryMismatch",
    "Intrinsics",
    "PatchTokens",
    "TemporalSpan",
    "Units",
    "UnitsError",
    "unproject",
]
