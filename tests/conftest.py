"""Pytest configuration and shared fixtures for open-notes tests."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from open_notes.config import Config
from open_notes.embedding.transformers import SentenceTransformerEmbedding
from open_notes.indexer.chunker import MarkdownChunker
from open_notes.llm.base import BaseLLM
from open_notes.storage.file_system import NoteStorage
from open_notes.storage.keyword_index import KeywordIndex
from open_notes.storage.vector_db import VectorDB


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_kb(temp_dir):
    """Create a temporary knowledge base directory."""
    kb_path = temp_dir / "knowledge_base"
    kb_path.mkdir()
    return kb_path


@pytest.fixture
def temp_vector_db(temp_dir):
    """Create a temporary vector database."""
    db_path = temp_dir / "vectors"
    return VectorDB(db_path, dimension=384)


@pytest.fixture
def temp_keyword_index(temp_dir):
    """Create a temporary keyword index."""
    db_path = temp_dir / "keywords.db"
    return KeywordIndex(db_path)


@pytest.fixture
def sample_note_paths():
    """Return paths to sample note fixtures."""
    return list(FIXTURES_DIR.glob("*.md"))


@pytest.fixture
def copy_notes_to_kb(sample_note_paths, temp_kb):
    """Copy sample notes to temporary knowledge base."""
    for note_path in sample_note_paths:
        shutil.copy(note_path, temp_kb / note_path.name)
    return temp_kb


@pytest.fixture
def note_storage(temp_kb):
    """Create a NoteStorage instance with temporary KB."""
    return NoteStorage(base_path=temp_kb)


@pytest.fixture
def chunker():
    """Create a MarkdownChunker with default settings."""
    return MarkdownChunker(max_chars=500, overlap_chars=50)


@pytest.fixture
def embedding_model():
    """Create a real embedding model (sentence-transformers)."""
    model = SentenceTransformerEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=8,
    )
    return model


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""

    class MockLLM(BaseLLM):
        def __init__(self):
            self._name = "mock"

        def generate(self, prompt: str, **kwargs) -> str:
            return "Mock response"

        def chat(self, messages: list, **kwargs) -> str:
            return "Mock chat response"

        @property
        def name(self) -> str:
            return self._name

    return MockLLM()


@pytest.fixture
def config(temp_dir, temp_kb):
    """Create a test configuration."""
    config_dict = {
        "knowledge_base": {"path": str(temp_kb)},
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 8,
        },
        "vector_db": {"path": str(temp_dir / "vectors")},
        "keyword_index": {"path": str(temp_dir / "keywords.db")},
        "chunker": {"max_chars_per_chunk": 500, "overlap_chars": 50},
        "search": {"top_k": 5, "mode": "hybrid", "vector_weight": 0.7, "keyword_weight": 0.3},
        "llm": {"provider": "llama_cpp", "model_path": "", "temperature": 0.7, "max_tokens": 2048},
        "rag": {"prompt_template": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"},
    }
    return Config(config_dict)


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)