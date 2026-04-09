from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_notes.models import SearchResult


def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []

    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        for r in results:
            r["normalized_score"] = 1.0
        return results

    for r in results:
        r["normalized_score"] = (r["score"] - min_s) / (max_s - min_s)

    return results


def reciprocal_rank_fusion(
    results_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    doc_scores: dict[str, float] = {}

    for results in results_lists:
        for rank, result in enumerate(results):
            doc_id = result.get("chunk_id") or result.get("id")
            if not doc_id:
                continue

            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0.0

            doc_scores[doc_id] += 1.0 / (k + rank + 1)

    fused_results = []
    for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
        for results in results_lists:
            for result in results:
                if (result.get("chunk_id") or result.get("id")) == doc_id:
                    fused_results.append(
                        {
                            **result,
                            "score": score,
                            "id": doc_id,
                        }
                    )
                    break

    return fused_results[:len(max(results_lists, key=len))]


class QueryEngine:
    def __init__(
        self,
        vector_db: Any,
        keyword_index: Any,
        embedding: Any,
    ):
        self.vector_db = vector_db
        self.keyword_index = keyword_index
        self.embedding = embedding

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if mode == "vector":
            return self._vector_search(query, top_k)
        elif mode == "keyword":
            return self._keyword_search(query, top_k)
        elif mode == "hybrid":
            return self._hybrid_search(query, top_k, vector_weight, keyword_weight)

        return []

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self.embedding.embed([query])[0]
        results = self.vector_db.search(query_embedding, top_k)

        search_results = []
        for idx, score, meta in results:
            search_results.append(
                SearchResult(
                    chunk_id=meta.get("chunk_id", ""),
                    note_id=meta.get("note_id", ""),
                    note_path=Path(meta.get("note_path", "")),
                    content=meta.get("content", ""),
                    heading_path=meta.get("heading_path", ""),
                    score=score,
                    source="vector",
                )
            )

        return search_results

    def _keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        results = self.keyword_index.search(query, top_k)

        search_results = []
        for r in results:
            search_results.append(
                SearchResult(
                    chunk_id=r.get("chunk_id", ""),
                    note_id=r.get("note_id", ""),
                    note_path=Path(""),
                    content=r.get("content", ""),
                    heading_path=r.get("heading_path", ""),
                    score=r.get("score", 0),
                    source="keyword",
                )
            )

        return search_results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> list[SearchResult]:
        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)

        if not vector_results and not keyword_results:
            return []

        vector_dicts = [
            {
                "chunk_id": r.chunk_id,
                "note_id": r.note_id,
                "note_path": r.note_path,
                "content": r.content,
                "heading_path": r.heading_path,
                "score": r.score,
            }
            for r in vector_results
        ]
        keyword_dicts = [
            {
                "chunk_id": r.chunk_id,
                "note_id": r.note_id,
                "note_path": r.note_path,
                "content": r.content,
                "heading_path": r.heading_path,
                "score": r.score,
            }
            for r in keyword_results
        ]

        vector_dicts = normalize_scores(vector_dicts)
        keyword_dicts = normalize_scores(keyword_dicts)

        fused = reciprocal_rank_fusion([vector_dicts, keyword_dicts])

        search_results = []
        for r in fused:
            v_score = next((x["normalized_score"] for x in vector_dicts if x["chunk_id"] == r["chunk_id"]), 0)
            k_score = next((x["normalized_score"] for x in keyword_dicts if x["chunk_id"] == r["chunk_id"]), 0)

            final_score = v_score * vector_weight + k_score * keyword_weight
            source = "vector" if v_score > k_score else "keyword" if k_score > v_score else "hybrid"

            search_results.append(
                SearchResult(
                    chunk_id=r["chunk_id"],
                    note_id=r["note_id"],
                    note_path=r["note_path"],
                    content=r["content"],
                    heading_path=r["heading_path"],
                    score=final_score,
                    source=source,
                )
            )

        return search_results[:top_k]
