"""
FAISS Vector Database for Identity Embeddings
Stores and retrieves V-JEPA embeddings for tracked objects.

Task 8.1: Setup local vector store with FAISS
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


@dataclass
class IdentityEmbedding:
    """Represents a tracked object's identity."""

    track_id: int
    embedding: np.ndarray
    metadata: Dict[str, Any]  # {class_name, first_frame, last_frame, etc.}


class VectorDatabase:
    """
    FAISS-based vector database for semantic search over tracked objects.
    """

    def __init__(
        self,
        db_path: str = ".vectordb/faiss_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            db_path: Directory to save/load FAISS index
            embedding_model: HuggingFace model for text embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Text embedding model (for queries like "red car")
        self.text_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = None
        self.track_id_mapping = {}  # doc_id -> track_id

    def index_embeddings(self, identities: List[IdentityEmbedding]) -> None:
        """
        Index identity embeddings into FAISS.

        Args:
            identities: List of IdentityEmbedding objects
        """
        if len(identities) == 0:
            raise ValueError("No embeddings to index")

        documents = []

        for identity in identities:
            # Create searchable text description
            text = (
                f"{identity.metadata.get('class_name', 'object')} "
                f"track_{identity.track_id}"
            )

            # Create LangChain Document
            doc = Document(
                page_content=text,
                metadata={
                    "track_id": identity.track_id,
                    "embedding_vector": identity.embedding.tolist(),
                    **identity.metadata,
                },
            )
            documents.append(doc)

            # Map document index to track_id
            self.track_id_mapping[len(documents) - 1] = identity.track_id

        # Create FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=documents, embedding=self.text_embeddings
        )

        print(f"[VectorDB] Indexed {len(identities)} identity embeddings")

    def add_embedding(self, identity: IdentityEmbedding) -> None:
        """Add a single embedding to existing index."""
        if self.vectorstore is None:
            self.index_embeddings([identity])
            return

        text = (
            f"{identity.metadata.get('class_name', 'object')} track_{identity.track_id}"
        )

        doc = Document(
            page_content=text,
            metadata={
                "track_id": identity.track_id,
                "embedding_vector": identity.embedding.tolist(),
                **identity.metadata,
            },
        )

        self.vectorstore.add_documents([doc])
        doc_id = len(self.track_id_mapping)
        self.track_id_mapping[doc_id] = identity.track_id

        print(f"[VectorDB] Added track_id={identity.track_id}")

    def search(
        self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for tracked objects.

        Args:
            query: Natural language query (e.g., "red car")
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {"class_name": "car"})

        Returns:
            List of {track_id, score, metadata}
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not initialized. Call index_embeddings() first."
            )

        # Search FAISS
        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(
                query=query, k=k, filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)

        # Format results
        output = []
        for doc, score in results:
            output.append(
                {
                    "track_id": doc.metadata["track_id"],
                    "score": float(score),
                    "class_name": doc.metadata.get("class_name", "unknown"),
                    "metadata": doc.metadata,
                }
            )

        return output

    def save(self) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore is None:
            raise RuntimeError("No vector store to save")

        # Save FAISS index
        self.vectorstore.save_local(str(self.db_path))

        # Save mapping
        with open(self.db_path / "mapping.pkl", "wb") as f:
            pickle.dump(self.track_id_mapping, f)

        print(f"[VectorDB] Saved to {self.db_path}")

    def load(self) -> None:
        """Load FAISS index from disk."""
        if not (self.db_path / "index.faiss").exists():
            raise FileNotFoundError(f"No index found at {self.db_path}")

        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            str(self.db_path),
            embeddings=self.text_embeddings,
            allow_dangerous_deserialization=True,
        )

        # Load mapping
        with open(self.db_path / "mapping.pkl", "rb") as f:
            self.track_id_mapping = pickle.load(f)

        print(f"[VectorDB] Loaded from {self.db_path}")


def example_usage():
    """Example: Index and search identity embeddings."""

    # === CREATE SAMPLE DATA ===
    identities = [
        IdentityEmbedding(
            track_id=0,
            embedding=np.random.rand(512).astype(np.float32),
            metadata={"class_name": "car", "color": "red", "first_frame": 0},
        ),
        IdentityEmbedding(
            track_id=1,
            embedding=np.random.rand(512).astype(np.float32),
            metadata={"class_name": "car", "color": "blue", "first_frame": 10},
        ),
        IdentityEmbedding(
            track_id=2,
            embedding=np.random.rand(512).astype(np.float32),
            metadata={"class_name": "person", "color": "unknown", "first_frame": 5},
        ),
    ]

    # === INDEX ===
    db = VectorDatabase(db_path=".vectordb/faiss_index")
    db.index_embeddings(identities)

    # === SEARCH ===
    print("\n=== Query: 'red car' ===")
    results = db.search("red car", k=3)
    for r in results:
        print(
            f"Track ID: {r['track_id']}, "
            f"Score: {r['score']:.3f}, "
            f"Class: {r['class_name']}"
        )

    print("\n=== Query: 'person' ===")
    results = db.search("person", k=3)
    for r in results:
        print(
            f"Track ID: {r['track_id']}, "
            f"Score: {r['score']:.3f}, "
            f"Class: {r['class_name']}"
        )

    # === FILTER SEARCH ===
    print("\n=== Query: 'vehicle' with filter class_name='car' ===")
    results = db.search(
        "vehicle",
        k=3,
        filter_dict={"class_name": "car"},
    )
    for r in results:
        print(
            f"Track ID: {r['track_id']}, "
            f"Score: {r['score']:.3f}, "
            f"Color: {r['metadata']['color']}"
        )

    # === SAVE/LOAD ===
    db.save()

    db2 = VectorDatabase(db_path=".vectordb/faiss_index")
    db2.load()
    print("\n=== After reload ===")
    results = db2.search("red car", k=1)
    print(f"Top result: Track ID {results[0]['track_id']}")


if __name__ == "__main__":
    example_usage()
