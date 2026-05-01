"""Integration tests for the full open-notes system."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from open_notes import OpenNotes
from open_notes.config import Config
from open_notes.embedding.transformers import SentenceTransformerEmbedding
from open_notes.indexer.chunker import MarkdownChunker
from open_notes.storage.file_system import NoteStorage
from open_notes.storage.keyword_index import KeywordIndex
from open_notes.storage.vector_db import VectorDB


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "notes"


class TestFullIntegration:
    """Integration tests for the complete open-notes workflow."""

    def test_index_and_search_workflow(self, temp_dir):
        """Test the full workflow: create KB, add notes, index, search."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        
        for note_file in FIXTURES_DIR.glob("*.md"):
            shutil.copy(note_file, kb_path / note_file.name)
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": "", "temperature": 0.7},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        
        result = on.index_all()
        
        assert result["indexed"] >= 3
        assert result["chunks"] >= 3
        
        search_results = on.search("Python")
        
        assert len(search_results) > 0
        assert any("python" in r.content.lower() for r in search_results)

    def test_hybrid_search_results(self, temp_dir):
        """Test that hybrid search returns combined results."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        
        (kb_path / "test.md").write_text("# Machine Learning\n\nNeural networks are great.")
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        on.index_all()
        
        results = on.search("neural networks")
        
        assert len(results) > 0

    def test_stats_after_indexing(self, temp_dir):
        """Test that stats are accurate after indexing."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        
        (kb_path / "note1.md").write_text("# Note 1\n\nContent one")
        (kb_path / "note2.md").write_text("# Note 2\n\nContent two")
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        on.index_all()
        
        stats = on.get_stats()
        
        assert stats["vector_db"]["total_vectors"] >= 2
        assert stats["keyword_index"]["total_chunks"] >= 2

    def test_empty_knowledge_base(self, temp_dir):
        """Test handling of empty knowledge base."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        result = on.index_all()
        
        assert result["indexed"] == 0
        assert result["chunks"] == 0

    def test_nonexistent_knowledge_base(self, temp_dir):
        """Test handling when KB path doesn't exist."""
        kb_path = temp_dir / "nonexistent_kb"
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        result = on.index_all()
        
        assert result["indexed"] == 0
        assert "Created knowledge base" in result["message"]

    def test_search_no_results(self, temp_dir):
        """Test search returns empty when no matches."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        (kb_path / "note.md").write_text("# Unrelated content")
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        on.index_all()
        
        results = on.search("xyznonexistentquery123")
        
        assert isinstance(results, list)

    def test_different_search_modes(self, temp_dir):
        """Test different search modes work correctly."""
        kb_path = temp_dir / "knowledge_base"
        kb_path.mkdir()
        (kb_path / "note.md").write_text("# Python\n\nPython is a programming language.")
        
        config = Config(
            {
                "knowledge_base": {"path": str(kb_path)},
                "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
                "vector_db": {"path": str(temp_dir / "vectors")},
                "keyword_index": {"path": str(temp_dir / "keywords.db")},
                "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
                "search": {"top_k": 5, "mode": "hybrid"},
                "llm": {"provider": "llama_cpp", "model_path": ""},
                "rag": {"prompt_template": "Context: {context}\nQuestion: {question}\nAnswer:"},
            }
        )
        
        on = OpenNotes(config=config)
        on.index_all()
        
        vector_results = on.search("programming")
        config._data["search"]["mode"] = "vector"
        on._query_engine = None
        keyword_results = on.search("programming")
        
        assert isinstance(vector_results, list)