"""Tests for embedding module."""

from __future__ import annotations

import pytest

from open_notes.embedding.transformers import SentenceTransformerEmbedding


class TestSentenceTransformerEmbedding:
    """Tests for SentenceTransformerEmbedding class."""

    def test_create_embedding(self):
        """Test creating a SentenceTransformerEmbedding instance."""
        emb = SentenceTransformerEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            batch_size=8,
        )
        assert emb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert emb.device == "cpu"

    def test_dimension(self, embedding_model):
        """Test embedding dimension."""
        assert embedding_model.dimension == 384

    def test_embed_single(self, embedding_model):
        """Test embedding a single text."""
        result = embedding_model.embed(["hello world"])
        
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_embed_multiple(self, embedding_model):
        """Test embedding multiple texts."""
        texts = ["hello", "world", "test"]
        result = embedding_model.embed(texts)
        
        assert len(result) == 3
        assert all(len(e) == 384 for e in result)

    def test_embed_empty_list(self, embedding_model):
        """Test embedding empty list."""
        result = embedding_model.embed([])
        
        assert result == []

    def test_embed_batch(self, embedding_model):
        """Test embedding respects batch size."""
        texts = ["text"] * 10
        result = embedding_model.embed(texts)
        
        assert len(result) == 10

    def test_embeddings_deterministic(self, embedding_model):
        """Test that same text produces same embedding."""
        text = "test sentence"
        result1 = embedding_model.embed([text])
        result2 = embedding_model.embed([text])
        
        assert result1 == result2

    def test_different_texts_different_embeddings(self, embedding_model):
        """Test that different texts produce different embeddings."""
        result1 = embedding_model.embed(["hello"])
        result2 = embedding_model.embed(["goodbye"])
        
        assert result1 != result2