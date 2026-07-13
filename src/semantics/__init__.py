"""
src/semantics/__init__.py
=========================
Semantics module for Project Iron.

Responsible for language understanding, feature embeddings, and semantic
reasoning over the outputs of the geometry pipeline (depth maps, tracked
points, spatial features from V-JEPA2).

Components (Tasks 7.1 & 7.2):
    SemanticExtractor : Maps CoTracker3 pixel tracks → V-JEPA2 1024-d embeddings
    PCAReducer        : Reduces 1024-d embeddings → compact 64-d vectors via PCA

Author: Radhe Tare
Team  : Semantics (language, embeddings, reasoning)
"""

from .semantic_extractor import SemanticExtractor, pixel_to_patch_index
from .pca_reducer import PCAReducer

__version__ = "0.2.0"
__author__ = "Radhe Tare"

__all__ = [
    "SemanticExtractor",
    "pixel_to_patch_index",
    "PCAReducer",
]
