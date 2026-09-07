"""``Adapter``: the only trainable surface identity capability may live in.

Per ADR 0001 (``docs/adr/0001-identity-adapter-architecture.md``): person-
specific capability lives in a small adapter over a frozen backbone, never
in fine-tuned backbone weights. "Small" is not a style preference here — it
is the entire erasure argument (Tier 2 of the ADR's three-tier deletion
design costs "hours, CPU" specifically because the adapter is orders of
magnitude smaller than the backbone). :data:`MAX_ADAPTER_PARAMETERS` makes
that a constructor-time refusal instead of a code-review norm someone can
grow past one layer at a time.
"""

from __future__ import annotations

import hashlib
import json

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from src.identity.contracts import FeatureTensor, IdentityEmbedding

# "Orders of magnitude fewer weights than the backbone" (ADR 0001). A
# V-JEPA2-ViT-L backbone is ~300M parameters; 2M is comfortably three orders
# of magnitude below that and still ample for a metric-learning head.
MAX_ADAPTER_PARAMETERS = 2_000_000


class AdapterConfigError(ValueError):
    """Raised when an adapter configuration cannot be built honestly."""


class AdapterConfig(BaseModel):
    """Everything that determines an adapter's architecture.

    Frozen and hashable (:meth:`config_sha`) so a checkpoint's manifest can
    name the exact configuration that produced it — the same
    ``config_sha()`` discipline :class:`src.config.IronConfig` already
    applies to the pipeline's own configuration.
    """

    model_config = ConfigDict(frozen=True)

    input_dim: int = Field(gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    output_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=2, ge=1, le=4)
    """Hard-capped at 4: an adapter that needs a deeper stack than that to
    do person re-identification is not a small head any more, and the
    erasure-cost argument (retrain in hours, on CPU) starts to erode."""
    version: str = "v1"

    def config_sha(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Adapter(nn.Module):
    """A small MLP head over frozen backbone features, producing an
    L2-normalized identity embedding.

    Never call this module directly on a raw tensor — it takes a
    :class:`~src.identity.contracts.FeatureTensor` and returns an
    :class:`~src.identity.contracts.IdentityEmbedding`, both labelled, so a
    consumer never has to separately track which backbone or which adapter
    version produced a given vector.
    """

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params > MAX_ADAPTER_PARAMETERS:
            raise AdapterConfigError(
                f"adapter config {config!r} produces {n_params:,} parameters, "
                f"exceeding the {MAX_ADAPTER_PARAMETERS:,} cap. ADR 0001's "
                "erasure argument depends on the adapter being orders of "
                "magnitude smaller than the backbone — an adapter this size "
                "is not a small head any more; shrink hidden_dim/num_layers "
                "or raise the cap deliberately, in review, not by accretion."
            )
        self._param_count = n_params

    @property
    def param_count(self) -> int:
        return self._param_count

    @property
    def adapter_sha(self) -> str:
        """Content hash of this adapter's CURRENT weights plus its config.

        Recomputed from live parameters rather than cached: an adapter's
        weights change every optimizer step during training, and a cached
        sha would silently describe a state the adapter no longer holds —
        the same "artifact whose producer changed underneath it" defect
        :mod:`src.artifacts` exists to catch, applied here at the source
        instead of at a downstream index.
        """
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(self.config.config_sha().encode("utf-8"))
        return digest.hexdigest()

    def forward(self, features: FeatureTensor) -> IdentityEmbedding:
        x = features.data
        if x.ndim == 3:
            # [batch, tokens, dim] -> [batch, dim]: mean-pool tokens. A
            # backbone that wants a different pooling strategy (e.g. a CLS
            # token) pools before handing features to the adapter; this is
            # the adapter's own trivial default, not a claim about what any
            # particular backbone's tokens mean.
            x = x.mean(dim=1)
        if x.shape[-1] != self.config.input_dim:
            raise AdapterConfigError(
                f"FeatureTensor has dim {x.shape[-1]}, but this adapter was "
                f"configured for input_dim={self.config.input_dim}"
            )
        raw = self.net(x)
        normalized = torch.nn.functional.normalize(raw, p=2, dim=-1)
        return IdentityEmbedding(
            data=normalized,
            backbone_sha=features.backbone_sha,
            adapter_sha=self.adapter_sha,
        )
