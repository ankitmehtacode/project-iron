"""Evidence, EvidenceCommitment, and calibrated Confidence.

Evidence is what a claim can point back to
--------------------------------------------
:class:`Evidence` is the ordered chain from raw inputs to a claim: which
clips, which state-graph references, which observations, which stages ran
in what order with which model shas, and whether the whole chain can be
re-run — with the exact command, not just a "yes". A claim whose evidence
cannot be re-derived is a claim nobody can check.

EvidenceCommitment outlives its inputs
-----------------------------------------
Retention policy will eventually delete the observations behind an old
claim; erasure requests can force it sooner. :class:`EvidenceCommitment`
is computed once, at claim time, as a Merkle root over the contributing
observation hashes, and stores only the root and a leaf count — never the
hashes themselves. After the observations are gone, the commitment still
proves that *something specific* was committed to at claim time, and
:attr:`EvidenceCommitment.status` records why it can or cannot be
re-verified now (``reproducible``, ``inputs_expired``, ``inputs_erased``).
Implemented now, ahead of retention expiry itself, because a commitment
scheme retrofitted after the first erasure request cannot cover any claim
made before it.

Confidence cannot claim to be a probability by accident
-----------------------------------------------------------
:class:`UncalibratedScore` and :class:`CalibratedProbability` are
different types, not one type with an optional field. An
:class:`UncalibratedScore` has no ``probability`` attribute at all —
reading one raises ``AttributeError``, not "returns an uncalibrated
number under a probability-shaped name". :class:`CalibratedProbability`
requires a :class:`CalibrationRecord` as a non-optional constructor
argument, so a probability claim with no calibration behind it cannot be
built. A report or UI layer that wants a probability has to hold a
:class:`CalibratedProbability` value to get one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Sequence, get_args

CommitmentStatus = Literal["reproducible", "inputs_expired", "inputs_erased"]
COMMITMENT_STATUSES: tuple[CommitmentStatus, ...] = get_args(CommitmentStatus)


class EvidenceError(ValueError):
    """Raised when an Evidence, Commitment, or Confidence record is malformed."""


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivationStep:
    """One stage in the ordered chain from raw inputs to a claim."""

    stage: str
    producer_sha: str

    def __post_init__(self) -> None:
        if not self.stage:
            raise EvidenceError("DerivationStep.stage must not be empty")
        if not self.producer_sha:
            raise EvidenceError("DerivationStep.producer_sha must not be empty")


@dataclass(frozen=True)
class Evidence:
    """What backs a claim, and whether it can be checked.

    Attributes:
        evidence_id: Unique id.
        clip_refs: Footage this evidence points to.
        state_refs: References into the state graph (see
            :mod:`src.model.episode`) this evidence was derived against.
        observation_refs: The underlying :class:`~src.model.observation.Observation`
            ids. Required and non-empty — evidence with no observations
            behind it is not evidence.
        derivation_chain: Ordered stages from raw input to claim. Required
            and non-empty: an unrecorded derivation cannot be defended.
        producer_shas: Every model/component sha involved.
        reproducible: Whether re-running ``reproduce_command`` on the same
            inputs reproduces this evidence.
        reproduce_command: The exact command, always recorded — even when
            ``reproducible`` is False, in which case it documents what was
            tried and is expected to explain why it does not reproduce
            (e.g. a live, unseeded sensor read).
    """

    evidence_id: str
    clip_refs: tuple[str, ...]
    state_refs: tuple[str, ...]
    observation_refs: tuple[str, ...]
    derivation_chain: tuple[DerivationStep, ...]
    producer_shas: tuple[str, ...]
    reproducible: bool
    reproduce_command: str

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise EvidenceError("Evidence.evidence_id must not be empty")
        if not self.observation_refs:
            raise EvidenceError(
                "Evidence.observation_refs must not be empty: evidence with "
                "no underlying observations is not evidence"
            )
        if not self.derivation_chain:
            raise EvidenceError(
                "Evidence.derivation_chain must not be empty: a claim whose "
                "derivation was not recorded cannot be defended in review"
            )
        if not self.reproduce_command:
            raise EvidenceError(
                "Evidence.reproduce_command must not be empty, even when "
                "reproducible=False — it should document what was tried"
            )


# ---------------------------------------------------------------------------
# EvidenceCommitment — Merkle root, survives expiry and erasure.
# ---------------------------------------------------------------------------


def compute_merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Merkle root over hex-encoded leaf hashes, combined in the given order.

    Pairwise SHA-256; an odd node at any level is paired with itself (the
    standard duplicate-last-node convention). Deterministic only in the
    order the leaves are supplied — callers must agree on an ordering
    (e.g. observation_id ascending) or two honest recomputations of the
    same evidence set will disagree on the root.

    Raises:
        EvidenceError: if ``leaf_hashes`` is empty or contains a non-hex
            string.
    """
    if not leaf_hashes:
        raise EvidenceError("compute_merkle_root requires at least one leaf hash")
    try:
        level = [bytes.fromhex(h) for h in leaf_hashes]
    except ValueError as exc:
        raise EvidenceError(f"leaf hash is not valid hex: {exc}") from exc
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
    return level[0].hex()


@dataclass(frozen=True)
class EvidenceCommitment:
    """A Merkle commitment over contributing observations, made at claim time.

    Deliberately does not store the leaf hashes themselves — only the
    root and how many there were. That is what lets this record survive
    retention expiry or an erasure request: after the observations are
    gone, ``merkle_root`` and ``leaf_count`` still prove that something
    specific was committed to, without retaining the erased bytes' hashes
    inside the very record that is supposed to outlive them.

    Attributes:
        commitment_id: Unique id.
        merkle_root: Hex SHA-256 root, validated against ``leaf_count`` at
            construction only insofar as ``leaf_count >= 1`` — the root
            itself cannot be re-verified without the original hashes; use
            :meth:`verify` when they are still available.
        leaf_count: How many observation hashes contributed.
        computed_at_ns: When the commitment was made — at claim time, not
            retroactively.
        status: Whether the inputs are (still) available to re-verify
            against.
    """

    commitment_id: str
    merkle_root: str
    leaf_count: int
    computed_at_ns: int
    status: CommitmentStatus

    def __post_init__(self) -> None:
        if not self.commitment_id:
            raise EvidenceError("EvidenceCommitment.commitment_id must not be empty")
        if not self.merkle_root:
            raise EvidenceError("EvidenceCommitment.merkle_root must not be empty")
        if self.leaf_count < 1:
            raise EvidenceError(
                "EvidenceCommitment.leaf_count must be >= 1: a commitment "
                "over zero observations proves nothing"
            )
        if self.status not in COMMITMENT_STATUSES:
            raise EvidenceError(
                f"unknown commitment status {self.status!r}; expected one of "
                f"{COMMITMENT_STATUSES}"
            )

    @classmethod
    def compute(
        cls,
        commitment_id: str,
        observation_hashes: Sequence[str],
        computed_at_ns: int,
        status: CommitmentStatus = "reproducible",
    ) -> "EvidenceCommitment":
        """Build a commitment by hashing ``observation_hashes`` now, at claim time."""
        root = compute_merkle_root(observation_hashes)
        return cls(
            commitment_id=commitment_id,
            merkle_root=root,
            leaf_count=len(observation_hashes),
            computed_at_ns=computed_at_ns,
            status=status,
        )

    def verify(self, observation_hashes: Sequence[str]) -> bool:
        """Whether ``observation_hashes`` recomputes this commitment's root.

        Only meaningful while the inputs are still available — call it
        before retention expiry or an erasure request makes that
        impossible. Returns False (never raises) on a malformed input list,
        since "does not verify" is the correct answer for those too.
        """
        if len(observation_hashes) != self.leaf_count:
            return False
        try:
            return compute_merkle_root(observation_hashes) == self.merkle_root
        except EvidenceError:
            return False


# ---------------------------------------------------------------------------
# Confidence and Calibration — separate types by construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    """How a score was mapped to a probability, and on what evidence.

    Attributes:
        calibration_id: Unique id.
        method: e.g. ``"isotonic_regression"``, ``"platt_scaling"``,
            ``"temperature_scaling"``.
        fit_on: Reference to the dataset/golden set the calibration was
            fit on.
        valid_for: The capability/model version this calibration applies
            to (e.g. ``"motion_gate@v3-indoor"``). A calibration fit for
            one capability does not transfer to another.
        manifest_sha: The run that produced this calibration.
    """

    calibration_id: str
    method: str
    fit_on: str
    valid_for: str
    manifest_sha: str

    def __post_init__(self) -> None:
        for name in ("calibration_id", "method", "fit_on", "valid_for", "manifest_sha"):
            if not getattr(self, name):
                raise EvidenceError(f"CalibrationRecord.{name} must not be empty")


@dataclass(frozen=True)
class UncalibratedScore:
    """A raw model or detector score with no calibration behind it.

    STRUCTURAL: has no ``probability`` attribute. There is no way to read
    a probability-shaped value off this type by accident — only
    :attr:`score`, and :meth:`as_dict` serializes it exclusively under the
    key ``uncalibrated_score``.
    """

    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise EvidenceError(
                f"UncalibratedScore.score must lie in [0, 1], got {self.score}"
            )

    def as_dict(self) -> dict[str, float]:
        return {"uncalibrated_score": self.score}


@dataclass(frozen=True)
class CalibratedProbability:
    """A probability, backed by the :class:`CalibrationRecord` that produced it.

    ``calibration`` is a required constructor argument with no default:
    there is no way to build a value carrying a ``probability`` attribute
    without also supplying the calibration that justifies calling it one.
    """

    probability: float
    calibration: CalibrationRecord

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise EvidenceError(
                f"CalibratedProbability.probability must lie in [0, 1], got "
                f"{self.probability}"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "probability": self.probability,
            "calibration_id": self.calibration.calibration_id,
            "calibration_method": self.calibration.method,
        }


Confidence = UncalibratedScore | CalibratedProbability
"""Either an :class:`UncalibratedScore` or a :class:`CalibratedProbability`.

Never a third, ambiguous shape — a consumer that needs to know whether a
number is a real probability checks ``isinstance(c, CalibratedProbability)``
and gets a type-level answer, not a boolean flag that could be set wrong.
"""
