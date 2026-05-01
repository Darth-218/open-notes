"""Tests for keyword index storage."""

from __future__ import annotations

import pytest

from open_notes.storage.keyword_index import KeywordIndex


class TestKeywordIndex:
    """Tests for KeywordIndex class."""

    def test_create_keyword_index(self, temp_dir):
        """Test creating a KeywordIndex instance."""
        idx = KeywordIndex(temp_dir / "keywords.db")
        assert idx.db_path == temp_dir / "keywords.db"

    def test_index_chunks(self, temp_keyword_index):
        """Test indexing chunks."""
        chunks = [
            {"id": "c1", "content": "Python programming language", "heading_path": "Intro"},
            {"id": "c2", "content": "Machine learning models", "heading_path": "ML"},
        ]
        
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        stats = temp_keyword_index.get_stats()
        assert stats["total_chunks"] == 2

    def test_search_empty(self, temp_keyword_index):
        """Test searching an empty index."""
        results = temp_keyword_index.search("python", top_k=5)
        assert results == []

    def test_search_with_data(self, temp_keyword_index):
        """Test searching with indexed data."""
        chunks = [
            {"id": "c1", "content": "Python is a programming language", "heading_path": "Intro"},
            {"id": "c2", "content": "JavaScript is also popular", "heading_path": "Web"},
        ]
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        results = temp_keyword_index.search("Python", top_k=5)
        
        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_search_top_k(self, temp_keyword_index):
        """Test searching with top_k parameter."""
        chunks = [
            {"id": f"c{i}", "content": f"content {i}", "heading_path": f"Section {i}"}
            for i in range(10)
        ]
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        results = temp_keyword_index.search("content", top_k=3)
        
        assert len(results) == 3

    def test_delete_note(self, temp_keyword_index):
        """Test deleting chunks by note ID."""
        chunks = [
            {"id": "c1", "content": "content 1", "heading_path": "S1"},
            {"id": "c2", "content": "content 2", "heading_path": "S2"},
        ]
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        temp_keyword_index.delete_by_note_id("note-1")
        
        stats = temp_keyword_index.get_stats()
        assert stats["total_chunks"] == 0

    def test_get_stats_empty(self, temp_keyword_index):
        """Test getting stats from empty index."""
        stats = temp_keyword_index.get_stats()
        assert stats["total_chunks"] == 0

    def test_get_stats_with_data(self, temp_keyword_index):
        """Test getting stats with data."""
        chunks = [{"id": f"c{i}", "content": f"test {i}", "heading_path": ""} for i in range(5)]
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        stats = temp_keyword_index.get_stats()
        assert stats["total_chunks"] == 5

    def test_persistence(self, temp_dir):
        """Test that index is persisted to disk."""
        db_path = temp_dir / "keywords.db"
        idx1 = KeywordIndex(db_path)
        idx1.index_chunks([{"id": "c1", "content": "test", "heading_path": ""}], "note-1")
        
        idx2 = KeywordIndex(db_path)
        stats = idx2.get_stats()
        assert stats["total_chunks"] == 1

    def test_search_partial_match(self, temp_keyword_index):
        """Test partial matching in search."""
        chunks = [
            {"id": "c1", "content": "artificial intelligence neural networks", "heading_path": "AI"},
        ]
        temp_keyword_index.index_chunks(chunks, "note-1")
        
        results = temp_keyword_index.search("neural", top_k=5)
        
        assert len(results) > 0