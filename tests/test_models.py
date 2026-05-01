"""Tests for data models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from open_notes.models import Chunk, Note, SearchResult, ChatMessage


class TestNote:
    """Tests for Note dataclass."""

    def test_create_note(self):
        """Test creating a Note instance."""
        note = Note(
            id="test-123",
            path=Path("/notes/test.md"),
            title="Test Note",
            content="# Test\n\nContent here",
        )
        assert note.id == "test-123"
        assert note.title == "Test Note"
        assert note.content == "# Test\n\nContent here"

    def test_note_to_dict(self):
        """Test converting Note to dictionary."""
        note = Note(
            id="test-123",
            path=Path("/notes/test.md"),
            title="Test Note",
            content="Content",
        )
        d = note.to_dict()
        assert d["id"] == "test-123"
        assert d["title"] == "Test Note"
        assert d["path"] == "/notes/test.md"

    def test_note_default_values(self):
        """Test Note with default values."""
        note = Note(id="test", path=Path("test.md"), title="Test")
        assert note.frontmatter == {}
        assert note.content == ""
        assert note.created_at is not None
        assert note.updated_at is not None


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_chunk(self):
        """Test creating a Chunk instance."""
        chunk = Chunk(
            id="chunk-1",
            note_id="note-1",
            content="This is chunk content",
            heading_path="Introduction",
            position=0,
            char_count=20,
        )
        assert chunk.id == "chunk-1"
        assert chunk.note_id == "note-1"
        assert chunk.content == "This is chunk content"

    def test_chunk_default_values(self):
        """Test Chunk with default values."""
        chunk = Chunk(id="c1", note_id="n1", content="test")
        assert chunk.heading_path == ""
        assert chunk.position == 0
        assert chunk.char_count == 0


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            chunk_id="chunk-1",
            note_id="note-1",
            note_path=Path("/notes/test.md"),
            content="Relevant content",
            heading_path="Section 1",
            score=0.85,
            source="hybrid",
        )
        assert result.chunk_id == "chunk-1"
        assert result.score == 0.85
        assert result.source == "hybrid"

    def test_search_result_to_dict(self):
        """Test converting SearchResult to dictionary."""
        result = SearchResult(
            chunk_id="chunk-1",
            note_id="note-1",
            note_path=Path("/notes/test.md"),
            content="Content",
            heading_path="Section",
            score=0.9,
            source="vector",
        )
        d = result.to_dict()
        assert d["chunk_id"] == "chunk-1"
        assert d["score"] == 0.9
        assert d["source"] == "vector"

    def test_search_result_sources(self):
        """Test different search result sources."""
        for source in ["vector", "keyword", "hybrid"]:
            result = SearchResult(
                chunk_id="c1",
                note_id="n1",
                note_path=Path("test.md"),
                content="test",
                heading_path="",
                score=0.5,
                source=source,
            )
            assert result.source == source


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_chat_message(self):
        """Test creating a ChatMessage instance."""
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_chat_message_roles(self):
        """Test different chat message roles."""
        for role in ["system", "user", "assistant"]:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role

    def test_chat_message_to_dict(self):
        """Test converting ChatMessage to dictionary."""
        msg = ChatMessage(role="user", content="What's the weather?")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "What's the weather?"