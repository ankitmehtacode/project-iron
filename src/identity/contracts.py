"""Torch-native envelopes for the identity-adapter training path.

Why these are not :class:`src.contracts.tokens.PatchTokens`
-------------------------------------------------------------
``src.contracts`` (see the ``iron-contracts`` skill) enforces one rule
project-wide: no raw array crosses a public module boundary undocumented.
:class:`~src.contracts.tokens.PatchTokens` already does this for the
*production* cascade — encoder output as a frozen ``numpy`` array with an
``encoder_sha``, consumed after the OpenVINO export boundary.

Training an adapter needs the same discipline applied to a different kind of
boundary: a **gradient-carrying** ``torch.Tensor`` moving from a frozen
backbone into a trainable adapter head, inside one training process, before
anything is exported. Reusing ``PatchTokens`` here would mean either
detaching to ``numpy`` (which silently discards the graph a training loop
needs) or smuggling a ``torch.Tensor`` into a dataclass whose field is typed
``npt.NDArray`` (which mypy would rightly reject). So this module is a
sibling of ``src.contracts.tokens``, not a duplicate of it: same rule
(labelled envelope, not a bare tensor; ``__post_init__`` validation; raise,
never coerce), applied on the training side of the export boundary that
``PatchTokens`` governs on the production side.

ADR 0001 (``docs/adr/0001-identity-adapter-architecture.md``) is what these
exist to serve: it requires every identity-derived artifact to be traceable
to the backbone and adapter that produced it. ``backbone_sha``/
``adapter_sha`` on these envelopes are that traceability, carried through
every tensor operation rather than attached only to the final checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class ContractError(ValueError):
    """Raised when a training-path envelope's own invariants do not hold."""


@dataclass(frozen=True, eq=False)
class FeatureTensor:
    """A frozen backbone's output for one batch, labelled with its source.

    Attributes:
        data: ``[batch, dim]`` (pooled) or ``[batch, tokens, dim]``
            (patch/token) features. Whichever shape a given
            :class:`~src.identity.backbone.FrozenBackbone` implementation
            produces — the adapter that consumes this is written against
            the shape it declares, not against a fixed rank.
        backbone_sha: Content hash of the backbone that produced ``data``.
            Required: a feature tensor that cannot name its source backbone
            cannot be checked against a checkpoint's recorded
            ``backbone_sha`` later, which is the entire point of recording
            one (ADR 0001 §"Any artifact containing identity-derived
            parameters is labelled as such").

    Equality is disabled, matching :class:`~src.contracts.tokens.PatchTokens`
    and :class:`~src.contracts.fields.DepthField`: ``==`` on a wrapped tensor
    returns a tensor, not a ``bool``.
    """

    data: torch.Tensor
    backbone_sha: str

    def __post_init__(self) -> None:
        if self.data.ndim not in (2, 3):
            raise ContractError(
                "FeatureTensor.data must be [batch, dim] or "
                f"[batch, tokens, dim], got shape {tuple(self.data.shape)}"
            )
        if self.data.shape[0] == 0:
            raise ContractError("FeatureTensor.data has an empty batch dimension")
        if not self.backbone_sha:
            raise ContractError(
                "FeatureTensor.backbone_sha is required: features from "
                "different backbones are not comparable, and a checkpoint "
                "manifest cannot verify a feature block that cannot name "
                "its source"
            )

    @property
    def dim(self) -> int:
        return int(self.data.shape[-1])

    @property
    def batch(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True, eq=False)
class IdentityEmbedding:
    """An adapter's output embedding, labelled with both of its producers.

    Every adapter forward pass returns this, never a bare tensor — the
    downstream consumer (a gallery, a loss, a checkpoint) always has both
    ``backbone_sha`` and ``adapter_sha`` in hand without having to thread
    them through separately, which is exactly the kind of separate-forwarding
    ``src.data.registry``'s stack-walk module docstring already calls out as
    the failure mode ("trusting every helper... to forward a flag").
    """

    data: torch.Tensor
    backbone_sha: str
    adapter_sha: str

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ContractError(
                f"IdentityEmbedding.data must be [batch, dim], got shape "
                f"{tuple(self.data.shape)}"
            )
        if not self.backbone_sha:
            raise ContractError("IdentityEmbedding.backbone_sha is required")
        if not self.adapter_sha:
            raise ContractError("IdentityEmbedding.adapter_sha is required")

    @property
    def dim(self) -> int:
        return int(self.data.shape[-1])
