"""Retrieval metrics shared across eval domains.

Extracted from ``scripts/eval_semantics.py`` (Day 36) so
``src.identity.promotion``'s promotion gate could import it directly instead
of reaching into a ``scripts/`` entrypoint from library code — a layering
inversion mypy's strict scope would otherwise have to special-case.
``scripts/eval_semantics.py`` now imports :func:`average_precision` from
here rather than defining it a second time.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def average_precision(
    query_idx: int,
    sim: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    query_label: int,
) -> float | None:
    """AP for one query against a ranked candidate pool.

    ``sim`` is the query's similarity to every pool member (higher = closer
    rank). Returns ``None`` when the query has no true positive in the pool.
    """
    order = np.argsort(-sim)
    ranked_labels = labels[order]
    hits = ranked_labels == query_label
    n_pos = int(hits.sum())
    if n_pos == 0:
        return None
    precisions = np.cumsum(hits, dtype=np.float64) / (np.arange(len(hits)) + 1)
    return float((precisions * hits).sum() / n_pos)
