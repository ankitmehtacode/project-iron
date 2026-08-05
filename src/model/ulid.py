"""ULID identifiers for records that must sort by creation time.

Why not uuid4
-------------
``Observation.observation_id`` needs two properties uuid4 does not give:
lexicographic sortability (so a store can range-scan "observations since
ts") and a decodable creation timestamp (so a corrupt or mis-timestamped
record is detectable from its own id). ULID (Crockford Base32, 48-bit
millisecond timestamp + 80 bits of randomness) has both, is a stable spec,
and needs no new dependency — the encoding is thirty lines.

Monotonicity within a millisecond
----------------------------------
Two observations produced in the same process within the same millisecond
must still sort in generation order. :func:`generate_ulid` increments the
random component when called twice in the same millisecond by the same
:class:`_MonotonicState`, per the ULID spec's monotonic variant.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ALPHABET_INDEX = {c: i for i, c in enumerate(_CROCKFORD_ALPHABET)}
_TIME_LEN = 10
_RANDOM_LEN = 16
_ULID_LEN = _TIME_LEN + _RANDOM_LEN
_MAX_TIME_MS = (1 << 48) - 1
_MAX_RANDOM = (1 << 80) - 1


class InvalidULID(ValueError):
    """Raised when a string is not a well-formed ULID."""


def _encode(value: int, length: int) -> str:
    chars = ["0"] * length
    for i in range(length - 1, -1, -1):
        chars[i] = _CROCKFORD_ALPHABET[value & 0x1F]
        value >>= 5
    if value != 0:
        raise ValueError(f"value does not fit in {length} base32 characters")
    return "".join(chars)


def _decode(text: str) -> int:
    value = 0
    for char in text:
        try:
            value = (value << 5) | _ALPHABET_INDEX[char]
        except KeyError as exc:
            raise InvalidULID(f"{char!r} is not a Crockford Base32 character") from exc
    return value


@dataclass(frozen=True)
class ULID:
    """A validated 26-character Crockford Base32 ULID.

    Construction validates format and range; there is no way to hold a
    malformed id in this type. Compare and sort by :attr:`value` directly —
    ULID's encoding is designed so lexicographic string order equals
    creation-time order.
    """

    value: str

    def __post_init__(self) -> None:
        text = self.value.upper()
        if len(text) != _ULID_LEN:
            raise InvalidULID(
                f"ULID must be {_ULID_LEN} characters, got {len(self.value)}: "
                f"{self.value!r}"
            )
        time_part = _decode(text[:_TIME_LEN])
        if time_part > _MAX_TIME_MS:
            raise InvalidULID(f"ULID timestamp overflows 48 bits: {self.value!r}")
        _decode(text[_TIME_LEN:])
        object.__setattr__(self, "value", text)

    @property
    def timestamp_ms(self) -> int:
        """The embedded creation timestamp, milliseconds since the epoch."""
        return _decode(self.value[:_TIME_LEN])

    def __str__(self) -> str:
        return self.value


class _MonotonicState:
    """Per-process state guaranteeing strictly increasing ids within a ms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_random = 0

    def next(self, now_ms: int, random_bits: int) -> tuple[int, int]:
        with self._lock:
            if now_ms == self._last_ms:
                random_bits = self._last_random + 1
                if random_bits > _MAX_RANDOM:
                    # Random space exhausted within one millisecond: force the
                    # clock forward rather than wrap and silently reuse an id.
                    now_ms += 1
                    random_bits = 0
            self._last_ms = now_ms
            self._last_random = random_bits
            return now_ms, random_bits


_STATE = _MonotonicState()


def generate_ulid(now_ns: int | None = None) -> ULID:
    """Generate a new ULID, monotonic within this process for a given ms.

    Args:
        now_ns: Wall-clock time in nanoseconds since the epoch. Defaults to
            the current time; passing it explicitly makes id generation
            reproducible in tests.
    """
    if now_ns is None:
        now_ns = time.time_ns()
    now_ms = now_ns // 1_000_000
    if not 0 <= now_ms <= _MAX_TIME_MS:
        raise ValueError(f"timestamp out of ULID range: {now_ns} ns")
    random_bits = int.from_bytes(os.urandom(10), "big") & _MAX_RANDOM
    now_ms, random_bits = _STATE.next(now_ms, random_bits)
    return ULID(_encode(now_ms, _TIME_LEN) + _encode(random_bits, _RANDOM_LEN))
