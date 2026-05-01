"""Vector database storage using FAISS.

This module provides the VectorDB class for storing and searching
embedding vectors using Facebook's FAISS library.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorDB:
    """Vector database for semantic search using FAISS.

    Provides efficient storage and retrieval of embedding vectors with
    L2 distance-based similarity search. Stores metadata alongside vectors
    for result reconstruction.

    Attributes:
        db_path: Path to the FAISS index file.
        dimension: Dimension of embedding vectors.
        index: FAISS index instance.
        metadata_path: Path to metadata JSON file.
        metadata: List of metadata dictionaries for each vector.

    Example:
        >>> vdb = VectorDB(Path("~/.open_notes/vectors"), dimension=384)
        >>> vdb.add_vectors([[0.1] * 384], [{"chunk_id": "1", "note_id": "note1"}])
        >>> results = vdb.search([0.1] * 384, top_k=5)
    """

    def __init__(self, db_path: Path, dimension: int = 384):
        """Initialize VectorDB with a storage path.

        Args:
            db_path: Path where the FAISS index will be stored.
            dimension: Dimension of vectors to be stored (default: 384).
        """
        self.db_path = db_path
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.metadata_path = self.db_path.with_suffix(".meta.json")
        self.metadata: list[dict[str, Any]] = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load existing index and metadata from disk."""
        if self.db_path.exists():
            self.index = faiss.read_index(str(self.db_path))
            if self.metadata_path.exists():
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
        else:
            self.index = None
            self.metadata = []

    def _create_index(self) -> None:
        """Create a new FAISS index with L2 distance."""
        index = faiss.IndexFlatL2(self.dimension)
        self.index = index

    def add_vectors(
        self, vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        """Add vectors to the database.

        Args:
            vectors: List of embedding vectors (must match dimension).
            metadata: List of metadata dictionaries for each vector.

        Example:
            >>> vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            >>> metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
            >>> vdb.add_vectors(vectors, metadata)
        """
        if self.index is None:
            self._create_index()

        vectors_array = np.array(vectors, dtype=np.float32)
        self.index.add(vectors_array)
        self.metadata.extend(metadata)

        self._save()

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[int, float, dict]]:
        """Search for similar vectors.

        Performs L2 distance search and converts distances to similarity scores.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of tuples containing (index, score, metadata).
            Score is computed as 1/(1+distance), so higher is better.

        Example:
            >>> results = vdb.search(query_embedding, top_k=5)
            >>> for idx, score, meta in results:
            ...     print(f"Score: {score:.3f}, Chunk: {meta['chunk_id']}")
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_array = np.array([query_vector], dtype=np.float32)
        # FIX: Fix parameters
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                score = 1.0 / (1.0 + dist)
                results.append((idx, score, self.metadata[idx]))

        return results

    # HACK: Refactor
    def delete_by_note_id(self, note_id: str) -> None:
        """Delete all vectors associated with a note.

        Args:
            note_id: ID of the note whose vectors should be deleted.

        Example:
            >>> vdb.delete_by_note_id("note123")
        """
        if not self.metadata:
            return

        indices_to_remove = [
            i for i, m in enumerate(self.metadata) if m.get("note_id") == note_id
        ]

        if not indices_to_remove:
            return

        if self.index is None:
            return

        new_metadata = []
        for i, m in enumerate(self.metadata):
            if i not in indices_to_remove:
                new_metadata.append(m)

        if not new_metadata:
            self.index = None
            self.metadata = []
            self._clear_files()
            return

        all_vectors = self._get_all_vectors()
        new_vectors = [v for i, v in enumerate(all_vectors) if i not in indices_to_remove]

        self.index = None
        self._clear_files()

        if new_vectors:
            self._create_index()
            self.add_vectors(new_vectors, new_metadata)

    def _get_all_vectors(self) -> list[list[float]]:
        """Retrieve all vectors from the index.

        Returns:
            List of all stored vectors.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        vectors = []
        for i in range(self.index.ntotal):
            vector = self.index.reconstruct(i)
            vectors.append(vector.tolist())
        return vectors

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.db_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def _clear_files(self) -> None:
        """Remove index files from disk."""
        if self.db_path.exists():
            os.remove(self.db_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector database.

        Returns:
            Dictionary containing:
                - total_vectors: Number of vectors stored
                - dimension: Vector dimension
        """
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.dimension}

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
        }
