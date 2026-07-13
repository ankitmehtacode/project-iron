"""
pca_reducer.py
==============
Task 7.2 — Dimensionality Reduction (1024-d → 64-d via PCA)

Trains an incremental PCA model on V-JEPA2 semantic embeddings to
reduce the 1024-dimensional patch vectors to a compact 64-dimensional
representation, preserving maximum variance.

─────────────────────────────────────────────────────────────────────
WHY PCA?
─────────────────────────────────────────────────────────────────────
V-JEPA2 outputs 1024-d embeddings per patch. Storing or comparing
these directly across many frames and tracks is expensive:
    100 tracks × 4 frames × 1024d = 409,600 floats per video clip

After PCA reduction to 64-d:
    100 tracks × 4 frames × 64d  =  25,600 floats  (16× smaller)

PCA preserves the directions of maximum variance in the embedding
space, keeping the most semantically discriminative information.

─────────────────────────────────────────────────────────────────────
DESIGN: IncrementalPCA
─────────────────────────────────────────────────────────────────────
Standard PCA loads all data into RAM at once. V-JEPA embeddings from
many video clips can exceed available RAM (7.4 GB system). We use
sklearn's IncrementalPCA which processes data in mini-batches:
    - Trains one batch at a time → constant ~O(n_components) RAM
    - Produces identical results to full-batch PCA

─────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────

    from pca_reducer import PCAReducer

    reducer = PCAReducer(n_components=64)

    # Train on extracted embeddings
    # embeddings: list of [B, T, N, 1024] arrays from SemanticExtractor
    reducer.fit(embeddings_list)
    reducer.save("models/pca/vjepa_pca_64.pkl")

    # Reduce new embeddings
    reduced = reducer.transform(new_embeddings)  # [..., 64]

    # Or load a saved model
    reducer2 = PCAReducer.load("models/pca/vjepa_pca_64.pkl")
    reduced = reducer2.transform(embeddings)

Author: Radhe Tare
Task  : 7.2 — Dimensionality Reduction (PCA 1024→64)
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
EMBED_DIM_IN = 1024  # V-JEPA2 ViT-L embedding dimension (input)
EMBED_DIM_OUT = 64  # target reduced dimension (output)
BATCH_SIZE = 512  # IncrementalPCA mini-batch size (number of vectors)


class PCAReducer:
    """
    Trains and applies PCA to reduce V-JEPA2 embeddings from 1024-d to 64-d.

    Internally uses sklearn IncrementalPCA so the full embedding matrix
    never needs to be in memory simultaneously.

    Attributes:
        n_components: Target dimensionality after reduction.
        _pca        : The underlying IncrementalPCA instance.
        _is_fitted  : Whether the model has been trained.
    """

    def __init__(self, n_components: int = EMBED_DIM_OUT):
        """
        Args:
            n_components: Number of PCA dimensions to keep. Default 64.
        """
        self.n_components = n_components
        self._pca = IncrementalPCA(
            n_components=n_components,
            batch_size=BATCH_SIZE,
        )
        self._is_fitted = False

    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────

    def fit(self, embeddings_list: list[np.ndarray]) -> "PCAReducer":
        """
        Train PCA incrementally on a list of semantic embedding arrays.

        Each array can be any shape as long as the last dimension is
        1024 (the V-JEPA embedding dimension). Typical shape:
        [B, T, N, 1024] from SemanticExtractor.extract().

        Args:
            embeddings_list: List of float32 ndarrays, each with last
                             dim == 1024. Can be a single-element list.

        Returns:
            self (for method chaining)

        Example:
            reducer.fit([clip1_embeddings, clip2_embeddings, ...])
        """
        print(f"Training PCA: {EMBED_DIM_IN}-d → {self.n_components}-d")
        total_vectors = 0

        for i, emb in enumerate(embeddings_list):
            # Flatten all leading dimensions → [N_vectors, 1024]
            flat = emb.reshape(-1, EMBED_DIM_IN).astype(np.float32)
            total_vectors += len(flat)

            # Feed to IncrementalPCA in mini-batches
            for start in range(0, len(flat), BATCH_SIZE):
                batch = flat[start : start + BATCH_SIZE]
                # partial_fit requires at least n_components samples per batch
                if len(batch) >= self.n_components:
                    self._pca.partial_fit(batch)

            print(
                f"  Processed clip {i+1}/{len(embeddings_list)} "
                f"({len(flat):,} vectors, running total: {total_vectors:,})"
            )

        self._is_fitted = True

        # Report explained variance
        explained = self._pca.explained_variance_ratio_.sum() * 100
        print("\nPCA training complete.")
        print(f"  Components       : {self.n_components}")
        print(f"  Total vectors    : {total_vectors:,}")
        print(f"  Variance retained: {explained:.1f}%")

        return self

    # ─────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings from 1024-d to n_components-d.

        Args:
            embeddings: float32 ndarray with last dim == 1024.
                        Any leading shape is preserved.
                        Typical: [B, T, N, 1024]

        Returns:
            reduced: float32 ndarray with last dim == n_components.
                     Same leading shape as input.
                     Typical: [B, T, N, 64]

        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError("PCAReducer is not fitted. Call fit() or load() first.")

        original_shape = embeddings.shape
        flat = embeddings.reshape(-1, EMBED_DIM_IN).astype(np.float32)

        # Apply PCA projection in batches to avoid RAM spikes
        reduced_parts = []
        for start in range(0, len(flat), BATCH_SIZE):
            batch = flat[start : start + BATCH_SIZE]
            reduced_parts.append(self._pca.transform(batch))

        reduced_flat = np.concatenate(reduced_parts, axis=0)

        # Restore original leading shape with new last dim
        new_shape = original_shape[:-1] + (self.n_components,)
        return reduced_flat.reshape(new_shape).astype(np.float32)

    def fit_transform(self, embeddings_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        Train PCA and immediately reduce the training data.

        Convenience method that calls fit() then transform() on each array.

        Args:
            embeddings_list: List of float32 ndarrays (last dim = 1024).

        Returns:
            List of reduced arrays (last dim = n_components).
        """
        self.fit(embeddings_list)
        return [self.transform(e) for e in embeddings_list]

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the trained PCA model to disk as a pickle file.

        Args:
            path: Destination file path (e.g. "models/pca/vjepa_pca_64.pkl")
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted PCAReducer.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pca": self._pca,
                    "n_components": self.n_components,
                    "is_fitted": self._is_fitted,
                },
                f,
            )
        size_kb = os.path.getsize(path) / 1024
        print(f"PCA model saved: {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: str) -> "PCAReducer":
        """
        Load a previously saved PCAReducer from disk.

        Args:
            path: Path to the .pkl file saved by save().

        Returns:
            A fitted PCAReducer instance ready for transform().
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        reducer = cls(n_components=data["n_components"])
        reducer._pca = data["pca"]
        reducer._is_fitted = data["is_fitted"]
        print(
            f"PCA model loaded: {path}  "
            f"({data['n_components']}-d, fitted={data['is_fitted']})"
        )
        return reducer

    # ─────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────

    def explained_variance_report(self) -> None:
        """Print a cumulative explained variance report per component."""
        if not self._is_fitted:
            raise RuntimeError("PCAReducer is not fitted.")

        ratios = self._pca.explained_variance_ratio_
        cumulative = np.cumsum(ratios) * 100

        print(f"\nExplained variance report ({self.n_components} components):")
        milestones = [10, 20, 32, 48, 64]
        for k in milestones:
            if k <= len(cumulative):
                print(
                    f"  First {k:2d} components: "
                    f"{cumulative[k-1]:.1f}% variance retained"
                )
        print(
            f"  All {self.n_components:2d} components: "
            f"{cumulative[-1]:.1f}% variance retained"
        )


# ─────────────────────────────────────────────────────────────────────
# Quick sanity test (no real embeddings needed)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== PCAReducer sanity test ===\n")

    # Simulate V-JEPA embeddings for 3 video clips
    # Shape: [B=1, T=4, N=100, 1024]
    np.random.seed(42)
    clips = [np.random.randn(1, 4, 100, 1024).astype(np.float32) for _ in range(3)]

    # Train PCA
    reducer = PCAReducer(n_components=64)
    reducer.fit(clips)

    # Transform one clip
    reduced = reducer.transform(clips[0])
    print(f"\nInput  shape: {clips[0].shape}")  # (1, 4, 100, 1024)
    print(f"Output shape: {reduced.shape}")  # (1, 4, 100, 64)
    assert reduced.shape == (1, 4, 100, 64), "Shape mismatch!"
    print("Shape test passed ✅")

    # Variance report
    reducer.explained_variance_report()

    # Save and reload
    reducer.save("/tmp/test_pca.pkl")
    reducer2 = PCAReducer.load("/tmp/test_pca.pkl")
    reduced2 = reducer2.transform(clips[0])
    assert np.allclose(reduced, reduced2, atol=1e-5), "Reload mismatch!"
    print("\nSave/load test passed ✅")
