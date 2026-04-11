"""Query engine for hybrid search across the knowledge base.

This module provides the QueryEngine class for searching notes using
vector search, keyword search, or a hybrid combination of both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_notes.models import SearchResult


def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize scores to [0, 1] range using min-max normalization.

    Args:
        results: List of result dictionaries with 'score' key.

    Returns:
        Results with added 'normalized_score' key.
    """
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
    """Fuse ranked lists using Reciprocal Rank Fusion (RRF).

    RRF combines multiple ranked lists into a single unified ranking.
    Documents appearing in multiple lists benefit from their combined ranks.

    Args:
        results_lists: List of ranked result lists to fuse.
        k: Constant for RRF formula (default: 60). Higher values
            give more weight to lower ranks.

    Returns:
        Fused and reranked results.

    Example:
        >>> vector_results = [{"chunk_id": "1", "score": 0.9}, {"chunk_id": "2", "score": 0.8}]
        >>> keyword_results = [{"chunk_id": "2", "score": 0.7}, {"chunk_id": "3", "score": 0.6}]
        >>> fused = reciprocal_rank_fusion([vector_results, keyword_results])
    """
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
    """Hybrid search engine combining vector and keyword search.

    Provides unified search across the knowledge base using either vector
    similarity, keyword matching, or both with reciprocal rank fusion.

    Attributes:
        vector_db: Vector database for semantic search.
        keyword_index: Keyword index for full-text search.
        embedding: Embedding model for text vectorization.

    Example:
        >>> engine = QueryEngine(vdb, kwdx, emb)
        >>> results = engine.search("machine learning", mode="hybrid", top_k=5)
    """

    def __init__(
        self,
        vector_db: Any,
        keyword_index: Any,
        embedding: Any,
    ):
        """Initialize QueryEngine with search components.

        Args:
            vector_db: VectorDB instance for semantic search.
            keyword_index: KeywordIndex instance for full-text search.
            embedding: BaseEmbedding instance for text vectorization.
        """
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
        """Search the knowledge base.

        Performs search using the specified mode and returns ranked results.

        Args:
            query: Search query string.
            mode: Search mode - "vector", "keyword", or "hybrid" (default: "hybrid").
            top_k: Number of results to return (default: 5).
            vector_weight: Weight for vector search in hybrid mode (default: 0.7).
            keyword_weight: Weight for keyword search in hybrid mode (default: 0.3).

        Returns:
            List of SearchResult objects ranked by relevance.

        Example:
            >>> results = engine.search("neural networks", mode="hybrid", top_k=10)
            >>> for r in results:
            ...     print(f"{r.heading_path}: {r.score:.3f}")
        """
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
        """Perform vector-based semantic search.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of SearchResult from vector search.
        """
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
        """Perform keyword-based full-text search.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of SearchResult from keyword search.
        """
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
        """Perform hybrid search combining vector and keyword search.

        Uses normalized scores and reciprocal rank fusion to combine results.

        Args:
            query: Search query.
            top_k: Number of results.
            vector_weight: Weight for vector search.
            keyword_weight: Weight for keyword search.

        Returns:
            List of SearchResult from hybrid search.
        """
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
