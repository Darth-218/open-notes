"""Tests for vector database storage."""

from __future__ import annotations

import numpy as np

import pytest

from open_notes.storage.vector_db import VectorDB


class TestVectorDB:
    """Tests for VectorDB class."""

    def test_create_vector_db(self, temp_dir):
        """Test creating a VectorDB instance."""
        db = VectorDB(temp_dir / "vectors", dimension=384)
        assert db.dimension == 384

    def test_add_vectors(self, temp_vector_db):
        """Test adding vectors to the database."""
        vectors = [[0.1] * 384, [0.2] * 384]
        metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
        
        temp_vector_db.add_vectors(vectors, metadata)
        
        stats = temp_vector_db.get_stats()
        assert stats["total_vectors"] == 2

    def test_search_empty(self, temp_vector_db):
        """Test searching an empty database."""
        results = temp_vector_db.search([0.1] * 384, top_k=5)
        assert results == []

    def test_search_with_data(self, temp_vector_db):
        """Test searching with data in the database."""
        vectors = [[0.0] * 384, [1.0] * 384]
        metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]
        temp_vector_db.add_vectors(vectors, metadata)
        
        results = temp_vector_db.search([0.0] * 384, top_k=1)
        
        assert len(results) == 1
        assert results[0][2]["chunk_id"] == "1"

    def test_search_top_k(self, temp_vector_db):
        """Test searching with top_k parameter."""
        vectors = [[0.0] * 384, [0.5] * 384, [1.0] * 384]
        metadata = [{"chunk_id": str(i)} for i in range(3)]
        temp_vector_db.add_vectors(vectors, metadata)
        
        results = temp_vector_db.search([0.1] * 384, top_k=2)
        
        assert len(results) == 2

    def test_delete_by_note_id(self, temp_vector_db):
        """Test deleting vectors by note ID."""
        vectors = [[0.1] * 384] * 3
        metadata = [
            {"chunk_id": "c1", "note_id": "n1"},
            {"chunk_id": "c2", "note_id": "n1"},
            {"chunk_id": "c3", "note_id": "n2"},
        ]
        temp_vector_db.add_vectors(vectors, metadata)
        
        temp_vector_db.delete_by_note_id("n1")
        
        stats = temp_vector_db.get_stats()
        assert stats["total_vectors"] == 1

    def test_get_stats_empty(self, temp_vector_db):
        """Test getting stats from empty database."""
        stats = temp_vector_db.get_stats()
        assert stats["total_vectors"] == 0
        assert stats["dimension"] == 384

    def test_get_stats_with_data(self, temp_vector_db):
        """Test getting stats with data."""
        vectors = [[0.1] * 384] * 5
        metadata = [{"chunk_id": str(i)} for i in range(5)]
        temp_vector_db.add_vectors(vectors, metadata)
        
        stats = temp_vector_db.get_stats()
        assert stats["total_vectors"] == 5

    def test_persistence(self, temp_dir):
        """Test that vectors are persisted to disk."""
        db_path = temp_dir / "vectors"
        db1 = VectorDB(db_path, dimension=384)
        db1.add_vectors([[0.1] * 384], [{"chunk_id": "1"}])
        
        db2 = VectorDB(db_path, dimension=384)
        stats = db2.get_stats()
        assert stats["total_vectors"] == 1