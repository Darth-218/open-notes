"""Tests for RAG pipeline module."""

from __future__ import annotations

import pytest

from open_notes.rag.pipeline import RAGPipeline, RAGResponse
from open_notes.query.engine import QueryEngine


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_create_rag_response(self):
        """Test creating a RAGResponse instance."""
        result = RAGResponse(
            answer="Test answer",
            sources=[],
        )
        assert result.answer == "Test answer"
        assert result.sources == []

    def test_rag_response_sources(self):
        """Test RAGResponse with sources."""
        from open_notes.models import SearchResult
        from pathlib import Path
        
        sources = [
            SearchResult(
                chunk_id="c1",
                note_id="n1",
                note_path=Path("/test.md"),
                content="content",
                heading_path="",
                score=0.9,
                source="vector",
            )
        ]
        
        result = RAGResponse(answer="Answer", sources=sources)
        
        assert len(result.sources) == 1
        assert result.sources[0].chunk_id == "c1"


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    def test_create_rag_pipeline(self, temp_vector_db, temp_keyword_index, embedding_model, mock_llm):
        """Test creating a RAGPipeline instance."""
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        pipeline = RAGPipeline(
            query_engine=engine,
            llm=mock_llm,
            prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        
        assert pipeline.query_engine is not None

    def test_query_no_results(self, temp_vector_db, temp_keyword_index, embedding_model, mock_llm):
        """Test querying when no results found."""
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        pipeline = RAGPipeline(
            query_engine=engine,
            llm=mock_llm,
            prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        
        result = pipeline.query("test query", top_k=5)
        
        assert result.answer != ""
        assert isinstance(result.sources, list)

    def test_query_with_results(self, temp_vector_db, temp_keyword_index, embedding_model, mock_llm):
        """Test querying with search results."""
        vectors = [[0.0] * 384]
        metadata = [{"chunk_id": "c1", "note_id": "n1", "note_path": "/test.md", "content": "Python is great", "heading_path": ""}]
        temp_vector_db.add_vectors(vectors, metadata)
        
        chunks = [{"id": "c1", "content": "Python is great", "heading_path": ""}]
        temp_keyword_index.index_chunks(chunks, "n1")
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        pipeline = RAGPipeline(
            query_engine=engine,
            llm=mock_llm,
            prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        
        result = pipeline.query("Python", top_k=5)
        
        assert isinstance(result, RAGResponse)
        assert result.answer == "Mock response"

    def test_query_top_k_parameter(self, temp_vector_db, temp_keyword_index, embedding_model, mock_llm):
        """Test that top_k is passed to query engine."""
        for i in range(10):
            vectors = [[float(i) * 0.1] * 384]
            metadata = [{"chunk_id": f"c{i}", "note_id": f"n{i}", "note_path": f"/test{i}.md", "content": f"content {i}", "heading_path": ""}]
            temp_vector_db.add_vectors(vectors, metadata)
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        pipeline = RAGPipeline(
            query_engine=engine,
            llm=mock_llm,
            prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        )
        
        result = pipeline.query("content", top_k=3)
        
        assert len(result.sources) <= 3

    def test_prompt_template_variables(self, temp_vector_db, temp_keyword_index, embedding_model, mock_llm):
        """Test that prompt template variables are replaced."""
        vectors = [[0.0] * 384]
        metadata = [{"chunk_id": "c1", "note_id": "n1", "note_path": "/test.md", "content": "Test content", "heading_path": ""}]
        temp_vector_db.add_vectors(vectors, metadata)
        
        chunks = [{"id": "c1", "content": "Test content", "heading_path": ""}]
        temp_keyword_index.index_chunks(chunks, "n1")
        
        engine = QueryEngine(temp_vector_db, temp_keyword_index, embedding_model)
        pipeline = RAGPipeline(
            query_engine=engine,
            llm=mock_llm,
            prompt_template="CONTEXT: {context} QUESTION: {question} ANSWER:",
        )
        
        result = pipeline.query("test", top_k=1)
        
        assert "Mock response" in result.answer