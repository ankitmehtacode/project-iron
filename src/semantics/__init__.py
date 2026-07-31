"""Semantics: embeddings and semantic reasoning over the geometry pipeline.

Components:
    patch_mapping      Map tracked pixel coordinates onto encoder patch tokens
    semantic_extractor CoTracker3 tracks -> V-JEPA2 embeddings (heavy imports)
    pca_reducer        1024-d embeddings -> compact vectors via PCA

Import policy
-------------
This package deliberately imports nothing at module scope.
``semantic_extractor`` pulls in CoTracker and OpenVINO, and re-exporting it
here meant ``import src.semantics.anything`` required a tracking checkpoint —
so the token-indexing arithmetic in :mod:`patch_mapping` could not be executed,
let alone tested, without the whole pipeline installed. That coupling is how a
one-line index bug survived. Same convention as :mod:`src.graph`: import the
submodule you need.

    from src.semantics.patch_mapping import map_tracks_to_embeddings
    from src.semantics.semantic_extractor import SemanticExtractor  # heavy

Author: Radhe Tare
Team  : Semantics (language, embeddings, reasoning)
"""

__version__ = "0.2.0"
__author__ = "Radhe Tare"

__all__: list[str] = []
