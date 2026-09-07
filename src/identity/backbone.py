"""``FrozenBackbone``: the typed interface any encoder must satisfy to sit
under an identity adapter.

Backbone-agnostic, deliberately unbound today
----------------------------------------------
ADR 0001 requires identity capability to live in a small adapter over a
*frozen* backbone — V-JEPA2, Depth-Anything-V2, CoTracker3, or whatever a
future backbone bake-off (Objective 4's harness) eventually selects. That
bake-off is itself blocked on data that does not exist yet (no lane-C
footage), so this module deliberately binds to no concrete backbone. Adding
one here today would be a decision dressed as infrastructure — the exact
failure this project retired with the slip-noise ratio and the velocity
floor (see FOUNDATION_REPORT.md). A ``FrozenBackbone`` implementation is
supplied by whatever code needs one — production wraps a real encoder, tests
and self-test scripts use the explicit stand-in in
:mod:`src.identity.selftest`.

Why a ``Protocol`` and not an ABC
----------------------------------
An ABC would force every backbone wrapper (present or future) to inherit
from a class defined here, coupling unrelated model-wrapper hierarchies
(``src.models.dav2_wrapper.DAv2Wrapper``, a future V-JEPA2 wrapper, ...) to
this package for no reason beyond satisfying an interface they already
satisfy structurally. ``typing.Protocol`` checks shape, not lineage — the
same reasoning that keeps ``src.models.model_wrapper.ModelWrapper`` an ABC
for its own family without every encoder in the codebase inheriting from it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from src.identity.contracts import FeatureTensor


@runtime_checkable
class FrozenBackbone(Protocol):
    """Structural interface: anything with these two members qualifies.

    Attributes:
        backbone_sha: Content hash identifying this backbone's weights.
            Declared, not computed on every call — a real implementation
            hashes its weight file(s) once at load time (see
            :func:`src.provenance.hash_model_files` for the existing
            content-hashing helper; a backbone wrapper should call that,
            not reimplement hashing).

    Frozen is enforced by the caller, not by this protocol
    --------------------------------------------------------
    ``FrozenBackbone`` does not itself set ``requires_grad = False`` on
    anything — a Protocol has no ``__init__`` to do that in, and a backbone
    may be a non-torch object entirely (e.g. an OpenVINO-compiled model with
    no notion of gradients at all). The frozen-ness this ADR requires is
    enforced at the training entrypoint (:func:`extract_features_no_grad`
    below, which wraps every call in ``torch.no_grad()`` and detaches the
    result) and verified by
    ``tests/test_identity_adapter.py::test_frozen_backbone_has_no_gradient_path``,
    which deliberately uses a backbone that did NOT freeze its own
    parameters, constructs a real backward pass through an adapter, and
    asserts no backbone parameter ever receives a gradient regardless. The
    type system states the shape of the interface; the test proves the
    property the ADR actually cares about, and proves it holds even when a
    backbone implementation is careless about its own ``requires_grad``.
    """

    backbone_sha: str

    def extract_features(self, clip: torch.Tensor) -> FeatureTensor:
        """Encode a clip/batch into a :class:`FeatureTensor`.

        Args:
            clip: Backbone-specific input shape (e.g.
                ``[batch, frames, channels, height, width]`` for a video
                encoder) — deliberately unconstrained here; the shape
                contract belongs to each concrete backbone, not to this
                protocol.

        Returns:
            Features labelled with this backbone's ``backbone_sha``.
        """
        ...


def extract_features_no_grad(
    backbone: FrozenBackbone, clip: torch.Tensor
) -> FeatureTensor:
    """The single enforcement point for ADR 0001's frozen-backbone
    constraint on the training path.

    Every training/calibration call site must reach a backbone through this
    function, never through ``backbone.extract_features`` directly. Whatever
    a specific implementation does or does not do about ``requires_grad`` on
    its own parameters, no gradient can propagate out of a ``torch.no_grad()``
    region — so a backbone wrapper that forgot to freeze itself is still safe
    when reached this way. The output is defensively ``.detach()``-ed on top
    of that, in case an implementation's ``extract_features`` folds in a
    tensor constructed outside the ``no_grad`` block (unusual, but not
    something this function can rule out for an arbitrary implementation).
    """
    with torch.no_grad():
        features = backbone.extract_features(clip)
    return FeatureTensor(
        data=features.data.detach(), backbone_sha=features.backbone_sha
    )
