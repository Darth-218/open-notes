"""Tests for query engine module."""

from __future__ import annotations

import pytest

from open_notes.models import SearchResult
from open_notes.query.engine import QueryEngine, normalize_scores, reciprocal_rank_fusion


class TestNormalizeScores:
    """Tests for normalize_scores function."""

    def test_normalize_empty(self):
        """Test normalizing empty list."""
        result = normalize_scores([])
        assert result == []

    def test_normalize_single(self):
        """Test normalizing single result."""
        result = normalize_scores([{"score": 5.0}])
        assert result[0]["normalized_score"] == 1.0

    def test_normalize_multiple(self):
        """Test normalizing multiple results."""
        results = [{"score": 1.0}, {"score": 5.0}, {"score": 3.0}]
        result = normalize_scores(results)
        
        assert result[0]["normalized_score"] == 0.0
        assert result[1]["normalized_score"] == 1.0
        assert result[2]["normalized_score"] == 0.5

    def test_normalize_same_scores(self):
        """Test normalizing when all scores are the same."""
        results = [{"score": 5.0}, {"score": 5.0}]
        result = normalize_scores(results)
        
        assert result[0]["normalized_score"] == 1.0
        assert result[1]["normalized_score"] == 1.0


class TestReciprocalRankFusion:
    """Tests for reciprocal_rank_fusion function."""

    def test_rrf_empty(self):
        """Test RRF with empty lists."""
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_single_list(self):
        """Test RRF with single list."""
        results = [{"chunk_id": "1", "score": 1.0}, {"chunk_id": "2", "score": 0.8}]
        fused = reciprocal_rank_fusion([results])
        
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "1"

    def test_rrf_two_lists(self):
        """Test RRF with two different result lists."""
        list1 = [{"chunk_id": "1", "score": 1.0}, {"chunk_id": "2", "score": 0.8}]
        list2 = [{"chunk_id": "2", "score": 0.9}, {"chunk_id": "3", "score": 0.7}]
        
        fused = reciprocal_rank_fusion([list1, list2])
        
        assert len(fused) == 2
        chunk_ids = [r["chunk_id"] for r in fused]
        assert "2" in chunk_ids

    def test_rrf_k_parameter(self):
        """Test RRF with different k values."""
        results = [{"chunk_id": "1"}, {"chunk_id": "2"}]
        
        fused_k1 = reciprocal_rank_fusion([results], k=1)
        fused_k60 = reciprocal_rank_fusion([results], k=60)
        
        assert fused_k1[0]["chunk_id"] == "1"
        assert fused_k60[0]["chunk_id"] == "1"


class TestQueryEngine:
    """Tests for QueryEngine class."""

    def test_create_query_engine(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test creating a QueryEngine instance."""
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        assert engine.vector_db is not None
        assert engine.keyword_index is not None

    def test_search_empty_query(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test searching with empty query."""
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        results = engine.search("", top_k=5)
        
        assert results == []

    def test_search_vector_only(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test vector-only search mode."""
        vectors = [[0.0] * 384, [1.0] * 384]
        metadata = [
            {"chunk_id": "c1", "note_id": "n1", "note_path": "/test/1.md", "content": "python", "heading_path": "Intro"},
            {"chunk_id": "c2", "note_id": "n2", "note_path": "/test/2.md", "content": "java", "heading_path": "Intro"},
        ]
        temp_vector_db.add_vectors(vectors, metadata)
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        results = engine.search("python", mode="vector", top_k=2)
        
        assert len(results) <= 2
        if results:
            assert results[0].source == "vector"

    def test_search_keyword_only(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test keyword-only search mode."""
        chunks = [
            {"id": "c1", "content": "Python programming is great", "heading_path": "Intro"},
        ]
        temp_keyword_index.index_chunks(chunks, "n1")
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        results = engine.search("Python", mode="keyword", top_k=5)
        
        if results:
            assert results[0].source == "keyword"

    def test_search_hybrid(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test hybrid search mode."""
        vectors = [[0.0] * 384]
        metadata = [{"chunk_id": "c1", "note_id": "n1", "note_path": "/test.md", "content": "python code", "heading_path": ""}]
        temp_vector_db.add_vectors(vectors, metadata)
        
        chunks = [{"id": "c1", "content": "python code", "heading_path": ""}]
        temp_keyword_index.index_chunks(chunks, "n1")
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        results = engine.search("python", mode="hybrid", top_k=5)
        
        assert isinstance(results, list)

    def test_search_top_k(self, temp_vector_db, temp_keyword_index, embedding_model):
        """Test that top_k is respected."""
        for i in range(10):
            vectors = [[float(i) * 0.1] * 384]
            metadata = [{"chunk_id": f"c{i}", "note_id": f"n{i}", "note_path": f"/test{i}.md", "content": f"content {i}", "heading_path": ""}]
            temp_vector_db.add_vectors(vectors, metadata)
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        results = engine.search("content", mode="vector", top_k=3)
        
        assert len(results) <= 3