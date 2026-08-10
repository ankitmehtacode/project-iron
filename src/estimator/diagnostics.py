"""Sequence-level filter diagnostics — innovation whiteness (Day 21).

A single NIS value says whether one update was self-consistent with its
own predicted uncertainty. It says nothing about whether *nearby* updates
are related to each other, which is exactly what distinguishes a
tuning problem from a structural one:

- A filter with the right model but a poorly-tuned Q still produces
  **white** innovations — each one is still an independent surprise,
  just scaled wrong on average (that shows up as calibration bias, not
  autocorrelation).
- A filter using the **wrong dynamics** (e.g. one constant-velocity mode
  asked to cover both steady walking and a motion-onset transient)
  produces innovations that are **correlated with each other**, because
  the model is failing to predict something structured — the same
  mis-fit recurring frame after frame — not a fresh independent surprise
  each time.

This is the measurement Day 21 uses to decide "wrong Q" (a tuning fix
would be enough) vs "wrong model" (a tuning fix cannot be enough, however
it is chosen — Q could not have been tuned to fit residual structure this
does not admit).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]


def lag1_autocorrelation(sequences: list[list[float]]) -> tuple[float, int]:
    """Pooled lag-1 Pearson autocorrelation across independent sequences.

    Each element of ``sequences`` is one track's ordered scalar innovation
    series. Pairs are formed only *within* a sequence — never across the
    boundary between two tracks, which would manufacture spurious
    correlation out of nothing but concatenation.

    Returns:
        ``(correlation, n_pairs)``. ``(nan, 0)`` when fewer than 2 pairs
        exist to correlate, or every value is constant (zero variance).
    """
    lag0: list[float] = []
    lag1: list[float] = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        lag0.extend(seq[:-1])
        lag1.extend(seq[1:])
    if len(lag0) < 2:
        return float("nan"), 0
    x = np.array(lag0)
    y = np.array(lag1)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), len(lag0)
    correlation = float(np.corrcoef(x, y)[0, 1])
    return correlation, len(lag0)


def white_noise_bound(n_pairs: int, confidence: float = 0.95) -> float:
    """Two-sided bound on sample lag-1 autocorrelation for a white-noise
    sequence of this many pairs.

    Bartlett's formula: for large ``n``, a white sequence's sample
    autocorrelation is approximately Normal(0, 1/n) (Bar-Shalom et al.,
    the same textbook the NIS/NEES chi-square bounds in
    :mod:`src.estimator.consistency` come from). A correlation outside
    this bound is evidence of a non-white sequence, not sampling noise.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = float(norm.ppf(0.5 + confidence / 2))
    return float(z / np.sqrt(n_pairs))


def is_white(
    sequences: list[list[float]], confidence: float = 0.95
) -> tuple[bool, float, float, int]:
    """Whether the pooled lag-1 autocorrelation falls inside its white-noise bound.

    Returns:
        ``(white, correlation, bound, n_pairs)``. ``white`` is ``False``
        whenever ``n_pairs`` is too small to say anything (fewer than 10
        pairs) — a diagnosis needs enough evidence to be a diagnosis, not
        just an absence of a signal.
    """
    correlation, n_pairs = lag1_autocorrelation(sequences)
    if n_pairs < 10 or np.isnan(correlation):
        return False, correlation, float("nan"), n_pairs
    bound = white_noise_bound(n_pairs, confidence)
    return bool(abs(correlation) <= bound), correlation, bound, n_pairs
