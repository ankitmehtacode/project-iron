"""Uncertainty as a required, typed field — never an implicit point estimate.

An observation with no stated uncertainty is not "precise", it is
unaudited: nothing downstream can tell a laser-accurate badge reader from
a monocular depth guess. :class:`Uncertainty` forces every observation to
say which shape its error takes and gives its parameters, so a consumer
that needs a specific kind (e.g. a Gaussian sigma to feed a filter) can
check ``kind`` and fail loudly rather than reinterpret a number that was
never that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

UncertaintyKind = Literal[
    "gaussian_px",
    "gaussian_3d",
    "categorical_error_rate",
    "interval",
    "exact",
]
"""The shape an observation's error takes.

- ``gaussian_px``: isotropic or diagonal pixel-space sigma (detector boxes).
- ``gaussian_3d``: metres, after unprojection.
- ``categorical_error_rate``: a discrete sensor's known false-accept /
  false-reject rate (badge readers, door contacts).
- ``interval``: a bounded range with no distributional assumption.
- ``exact``: the sensor's own spec guarantees zero error (a monotonic
  counter, a wall-clock read). Rare — most sensors are not this.
"""

ALL_UNCERTAINTY_KINDS: tuple[UncertaintyKind, ...] = get_args(UncertaintyKind)

_REQUIRED_PARAMS: dict[UncertaintyKind, tuple[str, ...]] = {
    "gaussian_px": ("sigma_x", "sigma_y"),
    "gaussian_3d": ("sigma_m",),
    "categorical_error_rate": ("false_accept_rate", "false_reject_rate"),
    "interval": ("low", "high"),
    "exact": (),
}


class UncertaintyError(ValueError):
    """Raised when an :class:`Uncertainty` is malformed for its kind."""


@dataclass(frozen=True)
class Uncertainty:
    """A required statement of how much an observation's value might be wrong.

    Attributes:
        kind: Which error shape :attr:`params` should be read as.
        params: The parameters for that shape. Validated against
            :data:`_REQUIRED_PARAMS` so a ``gaussian_px`` uncertainty
            cannot be built without both sigmas, and an ``interval`` cannot
            be built with ``low > high``.
    """

    kind: UncertaintyKind
    params: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.kind not in ALL_UNCERTAINTY_KINDS:
            raise UncertaintyError(
                f"unknown uncertainty kind {self.kind!r}; expected one of "
                f"{ALL_UNCERTAINTY_KINDS}"
            )
        keys = [k for k, _ in self.params]
        if len(set(keys)) != len(keys):
            raise UncertaintyError(f"duplicate parameter keys in {self.params!r}")
        present = set(keys)
        required = set(_REQUIRED_PARAMS[self.kind])
        missing = required - present
        if missing:
            raise UncertaintyError(
                f"Uncertainty(kind={self.kind!r}) is missing required "
                f"parameters {sorted(missing)}"
            )
        as_dict = dict(self.params)
        if self.kind == "interval" and as_dict["low"] > as_dict["high"]:
            raise UncertaintyError(
                f"interval uncertainty has low={as_dict['low']} > "
                f"high={as_dict['high']}"
            )
        for name in ("sigma_x", "sigma_y", "sigma_m"):
            if name in as_dict and as_dict[name] < 0:
                raise UncertaintyError(
                    f"{name} must be non-negative, got {as_dict[name]}"
                )

    def get(self, key: str) -> float:
        for k, v in self.params:
            if k == key:
                return v
        raise KeyError(key)

    @classmethod
    def exact(cls) -> "Uncertainty":
        """The uncertainty of a sensor whose spec guarantees zero error."""
        return cls(kind="exact", params=())

    @classmethod
    def gaussian_px(cls, sigma_x: float, sigma_y: float) -> "Uncertainty":
        return cls(
            kind="gaussian_px", params=(("sigma_x", sigma_x), ("sigma_y", sigma_y))
        )

    @classmethod
    def categorical(
        cls, false_accept_rate: float, false_reject_rate: float
    ) -> "Uncertainty":
        return cls(
            kind="categorical_error_rate",
            params=(
                ("false_accept_rate", false_accept_rate),
                ("false_reject_rate", false_reject_rate),
            ),
        )
