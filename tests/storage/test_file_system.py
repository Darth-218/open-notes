"""Tests for file system storage module."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from open_notes.storage.file_system import NoteStorage


class TestNoteStorage:
    """Tests for NoteStorage class."""

    def test_create_storage(self, temp_kb):
        """Test creating a NoteStorage instance."""
        storage = NoteStorage(base_path=temp_kb)
        assert storage.base_path == temp_kb

    def test_scan_notes_empty(self, temp_kb):
        """Test scanning empty directory."""
        storage = NoteStorage(base_path=temp_kb)
        notes = storage.scan_notes()
        assert notes == []

    def test_scan_notes_with_files(self, copy_notes_to_kb):
        """Test scanning directory with markdown files."""
        storage = NoteStorage(base_path=copy_notes_to_kb)
        notes = storage.scan_notes()
        assert len(notes) >= 3
        assert all(p.suffix == ".md" for p in notes)

    def test_scan_notes_excludes_non_markdown(self, temp_kb):
        """Test that non-markdown files are excluded."""
        (temp_kb / "test.txt").write_text("not a note")
        (temp_kb / "test.md").write_text("# Test")
        
        storage = NoteStorage(base_path=temp_kb)
        notes = storage.scan_notes()
        assert len(notes) == 1
        assert notes[0].name == "test.md"

    def test_scan_notes_nested_directories(self, temp_kb):
        """Test scanning nested directories."""
        subdir = temp_kb / "subdir"
        subdir.mkdir()
        (temp_kb / "root.md").write_text("# Root")
        (subdir / "nested.md").write_text("# Nested")
        
        storage = NoteStorage(base_path=temp_kb)
        notes = storage.scan_notes()
        assert len(notes) == 2

    def test_read_note_simple(self, temp_kb):
        """Test reading a simple markdown note."""
        note_content = "# My Note\n\nSome content here."
        (temp_kb / "test.md").write_text(note_content)
        
        storage = NoteStorage(base_path=temp_kb)
        result = storage.read_note(temp_kb / "test.md")
        
        assert "id" in result
        assert result["content"] == note_content

    def test_read_note_with_frontmatter(self, temp_kb):
        """Test reading note with YAML frontmatter."""
        content = """---
title: My Note
tags: [test, example]
---

# Content here
"""
        (temp_kb / "test.md").write_text(content)
        
        storage = NoteStorage(base_path=temp_kb)
        result = storage.read_note(temp_kb / "test.md")
        
        assert result["title"] == "My Note"
        assert "test" in result["frontmatter"]["tags"]

    def test_read_note_no_frontmatter(self, temp_kb):
        """Test reading note without frontmatter uses first heading as title."""
        content = "# First Heading\n\nContent"
        (temp_kb / "test.md").write_text(content)
        
        storage = NoteStorage(base_path=temp_kb)
        result = storage.read_note(temp_kb / "test.md")
        
        assert result["title"] == "First Heading"

    def test_write_note(self, temp_kb):
        """Test writing a note to disk."""
        storage = NoteStorage(base_path=temp_kb)
        
        content = "# New Note\n\nThis is new content."
        frontmatter = {"title": "New Note", "tags": ["new"]}
        
        result_path = temp_kb / "new.md"
        storage.write_note(result_path, content, frontmatter)
        
        assert result_path.exists()
        text = result_path.read_text()
        assert "title: New Note" in text
        assert "This is new content" in text

    def test_read_note_nonexistent(self, temp_kb):
        """Test reading nonexistent note raises error."""
        storage = NoteStorage(base_path=temp_kb)
        
        with pytest.raises(FileNotFoundError):
            storage.read_note(temp_kb / "nonexistent.md")