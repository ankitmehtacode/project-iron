"""Trivial baselines: the score a strategy gets by ignoring the capability.

Why this is a required, not-optional layer
------------------------------------------
Three times now, a metric on this project has produced a confident number
for a degenerate reason. AbsRel 0.1538 on depth was the near-constant
prediction landing on the dominant distance band. The one-cm motion
threshold generated 48 false positives and hid 42 real misses while the
recall number sat in the mid-nineties. mAP 18x chance on semantics turned
out to be 79% of tracks never leaving their 16-px patch. Each was caught
by an ad-hoc diagnostic AFTER the number had been reported.

The pattern is a metric that fails to distinguish the system under test
from a trivial solution which knows nothing about the capability. The
structural fix is not to add another diagnostic per finding; it is to
require every metric to declare, at emit time, the score of that trivial
solution — and to render the metric's value against that score wherever
the number appears.

What "trivial baseline" means in practice
-----------------------------------------
The cheapest strategy that ignores the capability being measured. For
motion the always-wake and never-wake decisions. For depth a constant
prediction. For retrieval / semantics a position-only lookup and the
1 / n_labels chance floor. For tracking copy-previous-frame and
static-point. Not curated to be beatable — the point is to catch metrics
that measure something other than what they claim, and a beatable baseline
is the case that never surfaces the failure.

Enforcement
-----------
:func:`require_baseline` raises ``BaselineMissing`` for any metric name
that has not been registered. Consumers call ``require_baseline`` at emit
time — inside :class:`src.data.scorecard.Scorecard.add_metric`, inside
``scripts/eval_tracking.py``, inside ``scripts/eval_semantics.py`` — and
the raise happens before the metric can be shown to a human. The same
discipline the validity gates take: a capability without evidence gets a
refusal, never a blank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

BaselineComputer = Callable[..., "list[Baseline]"]


@dataclass(frozen=True)
class Baseline:
    """One trivial strategy's score against a specific metric.

    ``description`` is the strategy in plain language ("always wake",
    "constant prediction at the dataset median", "position-only lookup on
    the query frame"), and it is expected to be short enough to render on
    the scorecard line next to the number.

    ``flag_worthy`` distinguishes an achievable adversary from a
    theoretical bound. always-wake's recall of 1.0 and always-wake's
    false-negative count of 0 are both boundaries no real gate can beat,
    so flagging against them fires unconditionally and drowns out the
    signal the flag exists to carry. Boundary baselines are still reported
    — their value is informational context on how much room there is — but
    the flag computation ignores them.
    """

    name: str
    """Short slug for the baseline, unique within a metric."""

    value: float
    """The baseline strategy's score under the same metric."""

    description: str
    """Plain-language statement of what the strategy actually does."""

    flag_worthy: bool = True
    """Set False for theoretical bounds that no real system can beat.

    When False the baseline still renders on the scorecard line and still
    round-trips through JSON — the reader sees it, the flag calculation
    ignores it. Use for always_wake's 1.0 recall, always_wake's 0
    false-negatives, never_wake's 0 false-positives, and any other "trivial
    optimum on this metric alone" that a real strategy would only match by
    ignoring the capability entirely."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "description": self.description,
            "flag_worthy": bool(self.flag_worthy),
        }


class BaselineMissing(RuntimeError):
    """Raised when a metric is emitted without any registered baseline.

    Deliberately a runtime error at emit time, not a lint at import time:
    a lint would let a new metric slip in as long as the code compiled.
    The raise here fires the moment someone tries to put an
    unbaseline-checked number into a scorecard or a JSON payload — which
    is exactly the moment the number would otherwise start travelling.
    """


class _Registry:
    """Maps a metric name to its list of baseline computer callables.

    Kept as a class rather than a module dict so tests can construct a
    fresh instance in isolation — global mutable state that tests share is
    how a "clear the registry between tests" pattern silently becomes
    "clear it in a fixture nobody remembers to add".
    """

    def __init__(self) -> None:
        self._computers: dict[str, list[BaselineComputer]] = {}

    def register(self, metric_name: str, computer: BaselineComputer) -> None:
        self._computers.setdefault(metric_name, []).append(computer)

    def has(self, metric_name: str) -> bool:
        return metric_name in self._computers

    def compute(self, metric_name: str, /, **context: Any) -> list[Baseline]:
        computers = self._computers.get(metric_name)
        if not computers:
            raise BaselineMissing(
                f"no baseline is registered for metric {metric_name!r}. "
                "Every metric must declare the score of a trivial strategy "
                "that ignores the capability being measured — see "
                "src/eval/baselines.py for the pattern. Registering the "
                "metric with an empty baseline list is not allowed; if no "
                "trivial strategy meaningfully applies, that itself is a "
                "finding, and the metric should be reconsidered."
            )
        out: list[Baseline] = []
        for fn in computers:
            out.extend(fn(**context))
        return out

    def require(self, metric_name: str) -> None:
        if not self.has(metric_name):
            raise BaselineMissing(
                f"metric {metric_name!r} has no registered baseline. Add "
                "one in src/eval/baselines.py before emitting the metric."
            )

    def clear(self) -> None:
        self._computers.clear()


baseline_registry = _Registry()


def register_baseline(metric_name: str, computer: BaselineComputer) -> None:
    """Register a baseline computer for one metric name.

    A computer takes keyword arguments (the metric's own context) and
    returns a list of :class:`Baseline`; several may be registered against
    the same metric name and their results concatenate.
    """
    baseline_registry.register(metric_name, computer)


def compute_baselines(metric_name: str, /, **context: Any) -> list[Baseline]:
    return baseline_registry.compute(metric_name, **context)


def require_baseline(metric_name: str) -> None:
    baseline_registry.require(metric_name)


def margin(metric_value: float, baselines: list[Baseline], higher_is_better: bool) -> float:
    """Metric's advantage over the STRONGEST FLAG-WORTHY trivial baseline.

    A margin at or below zero means the metric is not distinguishing the
    system under test from an achievable trivial strategy — either the
    system is not doing better than chance, or the metric is measuring
    something the trivial strategy also achieves. In both cases the report
    calls it out.

    Boundary baselines (``flag_worthy=False`` — always-wake's recall 1.0,
    always-wake's false-negative count of 0, and so on) are ignored here:
    no real gate can beat a theoretical optimum, and flagging every gate
    against it turns the flag into noise. Those baselines still surface
    on the scorecard line as informational context; only the flag math
    skips them.

    ``NaN`` is returned when no flag-worthy baselines were supplied — the
    caller sees the metric render as "n/a" for its margin and no flag,
    which is honest about the absence of an achievable comparison.
    """
    if not baselines:
        return float("nan")
    scoring = [b for b in baselines if b.flag_worthy and np.isfinite(b.value)]
    if not scoring:
        return float("nan")
    values = [b.value for b in scoring]
    if higher_is_better:
        return metric_value - max(values)
    return min(values) - metric_value


# ---------------------------------------------------------------------------
# Registered baselines for every metric this project currently emits.
#
# The imports are deferred so this module has no runtime dependency on the
# metric-computing code — the registry loads baselines at import of this
# module, but the computers themselves only need numpy and the context
# their caller passes in.
# ---------------------------------------------------------------------------


def _motion_gate_recall_baselines(**ctx: Any) -> list[Baseline]:
    """Always-wake gives recall 1.0 by definition; never-wake gives 0.0.

    Both are boundary baselines — no real gate can exceed 1.0 recall, and
    beating 0.0 is a bar every functioning gate clears — so both are
    reported as informational and neither is flag-worthy. The flag on
    recall would fire on every real gate, drowning out the signal it
    exists to carry. Precision and F1 do the flagging for the gate.
    """
    return [
        Baseline(
            "always_wake",
            1.0,
            "wake on every frame; recall is 1.0 by construction",
            flag_worthy=False,
        ),
        Baseline(
            "never_wake",
            0.0,
            "sleep on every frame; recall is 0.0",
            flag_worthy=False,
        ),
    ]


def _motion_gate_precision_baselines(*, moving_fraction: float, **_: Any) -> list[Baseline]:
    """Always-wake's precision equals the fraction of scored frames that move.

    ``moving_fraction`` is the strongest fair baseline: an oracle-free
    strategy that wakes on every frame is right exactly as often as motion
    is present, and any gate that scores below this fraction has produced
    a lower-precision output than the do-nothing strategy.
    """
    return [
        Baseline(
            "always_wake",
            float(moving_fraction),
            "wake on every scored frame; precision equals the moving-frame "
            "fraction of the set",
        )
    ]


def _motion_gate_f1_baselines(*, moving_fraction: float, **_: Any) -> list[Baseline]:
    """Harmonic mean of always-wake's precision and recall."""
    p = float(moving_fraction)
    r = 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return [
        Baseline(
            "always_wake",
            f1,
            "F1 of the always-wake strategy: harmonic mean of the moving "
            "fraction and 1.0",
        )
    ]


def _motion_gate_false_negatives_baselines(**_: Any) -> list[Baseline]:
    """Always-wake produces zero false negatives — a theoretical floor.

    No real gate can go below 0 FN, so this baseline is reported as
    informational context (how much room there is between the gate and
    the trivial optimum) but does not drive the flag. Recall carries the
    achievable comparison already.
    """
    return [
        Baseline(
            "always_wake",
            0.0,
            "always wake; no missed frame is possible",
            flag_worthy=False,
        ),
    ]


def _motion_gate_false_positives_baselines(*, non_moving_frames: int, **_: Any) -> list[Baseline]:
    """Always-wake FP is flag-worthy; never-wake FP is a boundary at 0.

    ``non_moving_frames`` is a beatable target: a gate that discriminates
    at all produces fewer FPs than a strategy that wakes on every frame.
    Never-wake produces 0 FP but at the cost of every recall; the flag on
    FP against that boundary would fire on every functioning gate.
    """
    return [
        Baseline(
            "always_wake",
            float(non_moving_frames),
            "always wake; every non-moving scored frame becomes a false "
            "positive",
        ),
        Baseline(
            "never_wake",
            0.0,
            "never wake; no false positive is possible",
            flag_worthy=False,
        ),
    ]


def _envelope_metric_baselines(**_: Any) -> list[Baseline]:
    """Envelope descriptors are counts about the SET, not model results.

    Reported alongside model metrics because a scorecard whose set
    difficulty is unstated is unreadable, but the metric is not being
    compared to a strategy — the baseline is "the number is what it is".
    Recorded as such so the registry check passes and the render reads
    ``value (baseline: set property, margin: n/a)``.
    """
    return [
        Baseline(
            "set_property",
            float("nan"),
            "descriptor of the set, not of any model — no strategy applies",
        )
    ]


def _coverage_metric_baselines(**_: Any) -> list[Baseline]:
    """Coverage descriptors are set properties, same as envelope.*."""
    return [
        Baseline(
            "set_property",
            float("nan"),
            "descriptor of the set, not of any model — no strategy applies",
        )
    ]


def _gt_metric_baselines(**_: Any) -> list[Baseline]:
    """GT descriptors (occluded_track_fraction, ...) are set properties."""
    return [
        Baseline(
            "set_property",
            float("nan"),
            "descriptor of the set, not of any model — no strategy applies",
        )
    ]


def _semantics_map_baselines(
    *, n_tracks: int, position_only_map: float | None = None, **_: Any
) -> list[Baseline]:
    """Chance mAP and position-only mAP for same-object retrieval.

    Position-only requires the caller to compute it (it is a proxy for the
    patch-index-identity retrieval Day 11's diagnostic surfaced); when the
    caller has not, only chance is returned. That is deliberately not enough
    to hide a positional collapse — see ``scripts/eval_semantics.py``.
    """
    b = [
        Baseline(
            "chance",
            1.0 / n_tracks if n_tracks > 0 else float("nan"),
            "1 / n_tracks — the retrieval mAP a uniform-random ranker gets",
        )
    ]
    if position_only_map is not None:
        b.append(
            Baseline(
                "position_only",
                float(position_only_map),
                "retrieve by patch-index identity on the query frame — a "
                "strategy that ignores the encoder entirely",
            )
        )
    return b


def _semantics_temporal_cosine_baselines(**_: Any) -> list[Baseline]:
    """Same-embedding constant baseline is cosine 1.0 by construction."""
    return [
        Baseline(
            "constant_embedding",
            1.0,
            "return the query frame's embedding for every frame — cosine is "
            "1.0 by construction, so a real encoder's stability shows only in "
            "the gap between 1.0 and its own number",
        )
    ]


def _semantics_patch_boundary_baselines(**_: Any) -> list[Baseline]:
    """No trivial ceiling; report as a set descriptor for now."""
    return [
        Baseline(
            "set_property",
            float("nan"),
            "descriptor of the encoder's output geometry on this set — no "
            "adversary strategy defined yet",
        )
    ]


def _tracking_pts_within_baselines(
    *, static_point_pts_within: float | None = None, copy_prev_pts_within: float | None = None,
    **_: Any,
) -> list[Baseline]:
    b: list[Baseline] = []
    if static_point_pts_within is not None:
        b.append(
            Baseline(
                "static_point",
                float(static_point_pts_within),
                "keep the query point at its frame-0 location for every "
                "frame; ignores motion entirely",
            )
        )
    if copy_prev_pts_within is not None:
        b.append(
            Baseline(
                "copy_previous",
                float(copy_prev_pts_within),
                "copy the previous frame's GT position as the prediction; the "
                "trivial motion model an actual tracker must beat",
            )
        )
    if not b:
        # Registered so require_baseline() succeeds; without any context the
        # caller gets a NaN and the margin surfaces as such — flagged.
        b.append(
            Baseline(
                "trivial_placeholder",
                float("nan"),
                "no baseline context was supplied by the caller; the "
                "measurement is unaudited and its margin will render as NaN",
            )
        )
    return b


def _tracking_occlusion_accuracy_baselines(
    *, always_visible_accuracy: float | None = None, **_: Any
) -> list[Baseline]:
    b: list[Baseline] = []
    if always_visible_accuracy is not None:
        b.append(
            Baseline(
                "always_visible",
                float(always_visible_accuracy),
                "predict every frame as visible; accuracy equals the visible "
                "fraction of frames",
            )
        )
    if not b:
        b.append(
            Baseline(
                "trivial_placeholder",
                float("nan"),
                "no baseline context was supplied; margin renders as NaN",
            )
        )
    return b


def _tracking_average_jaccard_baselines(**_: Any) -> list[Baseline]:
    """No closed-form trivial AJ; report as chance-like placeholder.

    ``compute_tapvid_metrics`` averages Jaccard across pixel thresholds and
    is not straightforward to model in closed form for a "static" or
    "copy-previous" strategy. Left as a placeholder so the require check
    passes; a Day-13+ item is to compute AJ for the static/copy-previous
    baselines from the same query batch and report them here.
    """
    return [
        Baseline(
            "unimplemented",
            float("nan"),
            "no trivial-strategy AJ computed yet; margin renders as NaN",
        )
    ]


def _depth_rank_correlation_baselines(**_: Any) -> list[Baseline]:
    """Constant prediction has rank correlation ~0 by construction."""
    return [
        Baseline(
            "constant_prediction",
            0.0,
            "predict the same disparity at every pixel; rank correlation is "
            "0 by construction (Spearman on a constant is undefined; treated "
            "as chance)",
        )
    ]


def _depth_absrel_baselines(*, per_band_absrel: dict[str, float] | None = None, **_: Any) -> list[Baseline]:
    """Constant prediction (dataset median) baseline; per-band already reported.

    A constant depth prediction produces low AbsRel on whichever band
    dominates the pixel count — Day-9 showed AbsRel 0.1538 while the
    3-8 m band containing every agent scored 0.0000. The per-band
    breakdown is the actual baseline check; a scalar here is a
    placeholder.
    """
    return [
        Baseline(
            "constant_median",
            float("nan"),
            "predict the dataset median depth everywhere; AbsRel score is "
            "dominated by whichever band holds the most pixels — see "
            "per_band breakdown for the honest evidence",
        )
    ]


def _register_defaults() -> None:
    """Register every metric name the project currently emits.

    Import side-effect: this runs once at ``src.eval.baselines`` import so
    that ``require_baseline`` sees the full set. Adding a new metric name
    requires adding it here — that is the point of the requirement.
    """
    # -- motion gate ------------------------------------------------------
    register_baseline("motion_gate.recall", _motion_gate_recall_baselines)
    register_baseline("motion_gate.precision", _motion_gate_precision_baselines)
    register_baseline("motion_gate.f1", _motion_gate_f1_baselines)
    register_baseline(
        "motion_gate.false_negatives", _motion_gate_false_negatives_baselines
    )
    register_baseline(
        "motion_gate.false_positives", _motion_gate_false_positives_baselines
    )
    # -- envelope / coverage / gt descriptors -----------------------------
    for name in (
        "envelope.limited_misses",
        "envelope.unobservable_frames",
        "envelope.wakes_outside_envelope",
    ):
        register_baseline(name, _envelope_metric_baselines)
    for name in ("coverage.frames_scored", "coverage.observable_fraction"):
        register_baseline(name, _coverage_metric_baselines)
    register_baseline("gt.occluded_track_fraction", _gt_metric_baselines)
    # -- semantics --------------------------------------------------------
    register_baseline("semantics.mAP", _semantics_map_baselines)
    register_baseline("semantics.temporal_cosine", _semantics_temporal_cosine_baselines)
    register_baseline(
        "semantics.patch_boundary_l2", _semantics_patch_boundary_baselines
    )
    # -- tracking (TAP-Vid family) ---------------------------------------
    register_baseline(
        "tracking.pts_within_avg", _tracking_pts_within_baselines
    )
    register_baseline(
        "tracking.occlusion_accuracy", _tracking_occlusion_accuracy_baselines
    )
    register_baseline(
        "tracking.average_jaccard", _tracking_average_jaccard_baselines
    )
    # -- depth ------------------------------------------------------------
    register_baseline("depth.rank_correlation", _depth_rank_correlation_baselines)
    register_baseline("depth.absrel", _depth_absrel_baselines)


_register_defaults()
