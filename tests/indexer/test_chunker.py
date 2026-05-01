"""Tests for indexer parser module."""

from __future__ import annotations

import pytest

from open_notes.indexer.parser import MarkdownParser, ParsedHeading


class TestMarkdownParser:
    """Tests for MarkdownParser class."""

    def test_parse_no_headings(self):
        """Test parsing content without headings."""
        parser = MarkdownParser()
        content = "This is just plain text without any headings."
        
        headings, cleaned = parser.parse(content)
        
        assert headings == []
        assert cleaned == content

    def test_parse_single_heading(self):
        """Test parsing content with a single heading."""
        parser = MarkdownParser()
        content = "# Title\n\nSome content here."
        
        headings, cleaned = parser.parse(content)
        
        assert len(headings) == 1
        assert headings[0].text == "Title"
        assert headings[0].level == 1

    def test_parse_multiple_headings(self):
        """Test parsing content with multiple headings."""
        parser = MarkdownParser()
        content = """# Main Title

## Section One

Content one

## Section Two

Content two
"""
        
        headings, cleaned = parser.parse(content)
        
        assert len(headings) == 2
        assert headings[0].text == "Main Title"
        assert headings[1].text == "Section One"
        assert headings[2].text == "Section Two"

    def test_parse_heading_levels(self):
        """Test parsing different heading levels."""
        parser = MarkdownParser()
        content = """# H1
## H2
### H3
#### H4
##### H5
###### H6
"""
        
        headings, cleaned = parser.parse(content)
        
        assert len(headings) == 6
        assert headings[0].level == 1
        assert headings[1].level == 2
        assert headings[5].level == 6

    def test_heading_position(self):
        """Test that heading positions are correct."""
        parser = MarkdownParser()
        content = "# Title\n\nContent starts here"
        
        headings, cleaned = parser.parse(content)
        
        assert headings[0].position == 0


class TestMarkdownChunker:
    """Tests for MarkdownChunker class."""

    def test_create_chunker(self):
        """Test creating a MarkdownChunker."""
        from open_notes.indexer.chunker import MarkdownChunker
        
        chunker = MarkdownChunker(max_chars=500, overlap_chars=50)
        assert chunker.max_chars == 500
        assert chunker.overlap_chars == 50

    def test_chunk_simple_content(self, chunker):
        """Test chunking simple content without headings."""
        result = chunker.chunk(note_id="test-1", content="Simple content without headings.")
        
        assert len(result) == 1
        assert result[0]["note_id"] == "test-1"
        assert result[0]["content"] == "Simple content without headings."

    def test_chunk_with_headings(self, chunker):
        """Test chunking content with headings."""
        content = """# Section 1

Content for section 1.

# Section 2

Content for section 2.
"""
        
        result = chunker.chunk(note_id="test-1", content=content)
        
        assert len(result) >= 2

    def test_chunk_respects_max_chars(self, chunker):
        """Test that chunks respect max_chars limit."""
        long_content = "A" * 600
        
        result = chunker.chunk(note_id="test-1", content=long_content)
        
        for chunk in result:
            assert len(chunk["content"]) <= 600

    def test_chunk_empty_content(self, chunker):
        """Test chunking empty content."""
        result = chunker.chunk(note_id="test-1", content="")
        
        assert result == []

    def test_chunk_heading_path(self, chunker):
        """Test that heading path is preserved in chunks."""
        content = """# Main
## Sub

Content here
"""
        
        result = chunker.chunk(note_id="test-1", content=content)
        
        paths = [r.get("heading_path", "") for r in result]
        assert any("Main" in p for p in paths)

    def test_chunk_overlap(self, chunker):
        """Test that overlap is applied to large content."""
        long_content = "\n\n".join([f"Paragraph {i} with some content" for i in range(20)])
        
        result = chunker.chunk(note_id="test-1", content=long_content)
        
        assert len(result) > 1