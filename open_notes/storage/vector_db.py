from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorDB:
    def __init__(self, db_path: Path, dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.metadata_path = self.db_path.with_suffix(".meta.json")
        self.metadata: list[dict[str, Any]] = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            self.index = faiss.read_index(str(self.db_path))
            if self.metadata_path.exists():
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
        else:
            self.index = None
            self.metadata = []

    def _create_index(self) -> None:
        index = faiss.IndexFlatL2(self.dimension)
        self.index = index

    def add_vectors(
        self, vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        if self.index is None:
            self._create_index()

        vectors_array = np.array(vectors, dtype=np.float32)
        self.index.add(vectors_array)
        self.metadata.extend(metadata)

        self._save()

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[int, float, dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                score = 1.0 / (1.0 + dist)
                results.append((idx, score, self.metadata[idx]))

        return results

    def delete_by_note_id(self, note_id: str) -> None:
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
        if self.index is None:
            return []
        vectors = faiss.extract_indexed(self.index)
        return vectors.tolist() if hasattr(vectors, "tolist") else []

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.db_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def _clear_files(self) -> None:
        if self.db_path.exists():
            os.remove(self.db_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path)

    def get_stats(self) -> dict[str, Any]:
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.dimension}

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
        }
