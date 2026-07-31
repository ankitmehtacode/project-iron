"""Exceptions raised when a boundary contract is violated.

These are deliberately distinct from ``ValueError`` so that callers can catch
the specific class of mistake — and so that a ``except ValueError`` somewhere
up the stack, written for an unrelated reason, cannot quietly swallow a units
or frame-geometry violation.

Consumers raise; they never coerce. A convenience auto-conversion from
disparity to metres would recreate the exact bug these types exist to kill,
with extra steps and more confidence.
"""

from __future__ import annotations


class ContractError(Exception):
    """Base class for all boundary-contract violations."""


class UnitsError(ContractError):
    """Raised when an array's units are not what the consumer requires.

    The motivating defect: Depth-Anything-V2 emits *relative inverse depth*
    (disparity), and downstream code treated it as metres. Nothing crashed. The
    resulting 3D points were plausibly-shaped and entirely wrong, which is the
    worst possible failure mode because it survives eyeballing.
    """


class GeometryMismatch(ContractError):
    """Raised when two operands describe different pixel frames.

    The motivating defect: intrinsics measured on a full-resolution frame,
    applied to a downscaled one. ``fx`` is in pixels, so it is only meaningful
    together with the frame size it was measured at.
    """
