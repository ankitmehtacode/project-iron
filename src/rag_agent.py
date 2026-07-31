"""
RAG Agent for Semantic Object Search
Converts natural language queries to vectors and retrieves tracked objects.

Task 8.2: Implement RAG with FAISS search
Task 8.3: Define agent tools (get_track_location, filter_by_class)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from vector_database import VectorDatabase


class AgentTools:
    """
    Tool functions for the RAG agent.
    """

    def __init__(self, parquet_path: str):
        """
        Args:
            parquet_path: Path to tracking results Parquet file
        """
        self.parquet_path = Path(parquet_path)
        self.df = None

        if self.parquet_path.exists():
            self.df = pd.read_parquet(parquet_path)
            print(f"[Tools] Loaded {len(self.df)} tracking records")

    def get_track_location(
        self, track_id: int, frame_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get location(s) of a tracked object.

        Args:
            track_id: Object ID to retrieve
            frame_idx: Specific frame (None = all frames)

        Returns:
            {track_id, locations: [{frame_idx, x, y, z, confidence}]}
        """
        if self.df is None:
            raise RuntimeError("No tracking data loaded")

        # Filter by track_id
        track_data = self.df[self.df["track_id"] == track_id]

        if len(track_data) == 0:
            return {"track_id": track_id, "found": False, "locations": []}

        # Filter by frame if specified
        if frame_idx is not None:
            track_data = track_data[track_data["frame_idx"] == frame_idx]

        # Format locations
        locations = []
        for _, row in track_data.iterrows():
            locations.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "confidence": float(row["confidence"]),
                }
            )

        return {
            "track_id": track_id,
            "found": True,
            "num_frames": len(locations),
            "locations": locations,
        }

    def filter_by_class(
        self, class_name: str, frame_idx: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all objects of a specific class.

        Args:
            class_name: Object class (e.g., "car", "person")
            frame_idx: Specific frame (None = all frames)

        Returns:
            List of {track_id, class_name, location}
        """
        # This requires class metadata in Parquet
        # For now, return placeholder
        # In production, add class_name column to Parquet schema

        return [
            {
                "note": "Class filtering requires class metadata in Parquet file",
                "requested_class": class_name,
                "frame_idx": frame_idx,
            }
        ]

    def get_track_summary(self, track_id: int) -> Dict[str, Any]:
        """
        Get summary statistics for a track.

        Args:
            track_id: Object ID

        Returns:
            {track_id, first_frame, last_frame, avg_depth, path_length}
        """
        if self.df is None:
            raise RuntimeError("No tracking data loaded")

        track_data = self.df[self.df["track_id"] == track_id]

        if len(track_data) == 0:
            return {"track_id": track_id, "found": False}

        # Sort by frame
        track_data = track_data.sort_values("frame_idx")

        # Compute path length
        coords = track_data[["x", "y"]].values
        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            path_length = 0.0

        return {
            "track_id": track_id,
            "found": True,
            "first_frame": int(track_data["frame_idx"].min()),
            "last_frame": int(track_data["frame_idx"].max()),
            "num_frames": len(track_data),
            "avg_depth": float(track_data["z"].mean()),
            "path_length_pixels": path_length,
            "avg_confidence": float(track_data["confidence"].mean()),
        }


class RAGAgent:
    """
    Retrieval-Augmented Generation agent for object queries.
    Combines semantic search (FAISS) with structured data (Parquet).
    """

    def __init__(
        self,
        vector_db_path: str = ".vectordb/faiss_index",
        parquet_path: str = "output/tracking_results.parquet",
    ):
        """
        Args:
            vector_db_path: Path to FAISS index
            parquet_path: Path to tracking Parquet file
        """
        self.vector_db = VectorDatabase(db_path=vector_db_path)

        # Try to load existing index
        try:
            self.vector_db.load()
        except FileNotFoundError:
            print("[RAG] No existing vector DB found. Call index_embeddings() first.")

        self.tools = AgentTools(parquet_path=parquet_path)

    def query(
        self, query: str, k: int = 5, include_locations: bool = True
    ) -> Dict[str, Any]:
        """
        Query for objects using natural language.

        Args:
            query: Natural language (e.g., "Show me the red car")
            k: Number of results
            include_locations: Include full location data

        Returns:
            {query, results: [{track_id, score, summary, locations}]}
        """
        # Step 1: Semantic search in FAISS
        search_results = self.vector_db.search(query, k=k)

        # Step 2: Retrieve tracking data for each result
        enriched_results = []

        for result in search_results:
            track_id = result["track_id"]

            # Get summary
            summary = self.tools.get_track_summary(track_id)

            # Optionally get locations
            if include_locations:
                locations = self.tools.get_track_location(track_id)
            else:
                locations = None

            enriched_results.append(
                {
                    "track_id": track_id,
                    "similarity_score": result["score"],
                    "class_name": result.get("class_name", "unknown"),
                    "summary": summary,
                    "locations": locations,
                }
            )

        return {
            "query": query,
            "num_results": len(enriched_results),
            "results": enriched_results,
        }

    def get_track(self, track_id: int) -> Dict[str, Any]:
        """Direct lookup by track ID."""
        return self.tools.get_track_location(track_id)

    def filter_by_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Filter objects by class."""
        return self.tools.filter_by_class(class_name)


def example_usage():
    """Example: RAG agent queries."""

    # Assume vector DB and Parquet are already created
    # (See vector_database.py and parquet_writer.py examples)

    agent = RAGAgent(
        vector_db_path=".vectordb/faiss_index",
        parquet_path="output/tracking_results.parquet",
    )

    # === SEMANTIC QUERY ===
    print("=== Query: 'Show me the red car' ===")
    result = agent.query("red car", k=3, include_locations=False)

    print(f"Found {result['num_results']} results:")
    for r in result["results"]:
        print(f"\nTrack ID: {r['track_id']}")
        print(f"  Similarity: {r['similarity_score']:.3f}")
        print(f"  Class: {r['class_name']}")
        print(f"  Frames: {r['summary']['first_frame']} - {r['summary']['last_frame']}")
        print(f"  Avg depth: {r['summary']['avg_depth']:.2f}m")

    # === DIRECT LOOKUP ===
    print("\n=== Direct lookup: Track ID 0 ===")
    locations = agent.get_track(track_id=0)
    print(f"Found: {locations['found']}")
    if locations["found"]:
        print(f"Locations across {locations['num_frames']} frames")
        print(f"First location: {locations['locations'][0]}")


if __name__ == "__main__":
    example_usage()
