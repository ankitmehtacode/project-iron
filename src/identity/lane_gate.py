"""The ONLY way adapter training may resolve a dataset name.

Reuses ``src.data.registry.DatasetRegistry.open_for_training`` directly —
does not reimplement lane logic. Two things make that reuse structural
rather than a convention:

1. :data:`IRON_TRAINING_PATH` at module level, which is
   ``src.data.registry``'s stack-walk marker
   (``src.data.registry.TRAINING_PATH_MARKER``). Any lane-R loader reached
   through THIS module — or through anything this module calls, at any call
   depth — is refused by ``DatasetRegistry.open_for_eval``'s stack walk, the
   same protection ``scripts/build_calibration_set.py`` gets by living on a
   training-marked path. A helper added later that forgets to forward a
   "this is training" flag cannot silently defeat this, because there is no
   flag to forward — the walk inspects the real call stack.
2. :func:`require_training_dataset` calls ``open_for_training`` and nothing
   else. It does not re-check ``entry.lane``, does not special-case
   ``C_pending_consent``, and does not duplicate any refusal message —
   every refusal a caller sees originates in ``src.data.registry``, so a
   change to the lane rules there changes adapter training's behaviour
   automatically instead of requiring a second edit here.
"""

from __future__ import annotations

from src.data.registry import DatasetEntry, DatasetRegistry

IRON_TRAINING_PATH = True


def require_training_dataset(registry: DatasetRegistry, name: str) -> DatasetEntry:
    """Resolve ``name`` for adapter training, or raise.

    Raises:
        Whatever ``DatasetRegistry.open_for_training`` raises: ``LaneViolation``
        for lane R or for ``C_pending_consent`` with no attached
        ``ConsentRecord``; ``LicenseNotVerified`` for an unverified entry;
        ``UnknownDataset``/``BlockedDataset`` for a bad name. See that
        method's docstring — this function adds no cases of its own.
    """
    return registry.open_for_training(name)
