"""Data models for open-notes.

This module defines the core data structures used throughout the application,
including Note, Chunk, SearchResult, and ChatMessage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Note:
    """Represents a note in the knowledge base.

    Attributes:
        id: Unique identifier for the note.
        path: File system path to the note file.
        title: Title extracted from frontmatter or first heading.
        frontmatter: YAML frontmatter metadata.
        content: Raw markdown content of the note.
        created_at: Timestamp when note was created.
        updated_at: Timestamp when note was last modified.

    Example:
        >>> note = Note(
        ...     id="abc123",
        ...     path=Path("/home/user/notes/test.md"),
        ...     title="My Note",
        ...     content="# My Note\\n\\nThis is content.",
        ... )
        >>> print(note.title)
        My Note
    """
    id: str
    path: Path
    title: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert Note to dictionary representation.

        Returns:
            Dictionary containing all Note fields.
        """
        return {
            "id": self.id,
            "path": str(self.path),
            "title": self.title,
            "frontmatter": self.frontmatter,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Chunk:
    """Represents a searchable segment of a note.

    Chunks are created by splitting notes into smaller pieces for indexing
    and retrieval. Each chunk maintains a reference to its parent note
    and position in the document.

    Attributes:
        id: Unique identifier for the chunk.
        note_id: ID of the parent note this chunk belongs to.
        content: Text content of the chunk.
        heading_path: Path of headings leading to this chunk (e.g., "Introduction/Background").
        position: Position in the original document.
        char_count: Number of characters in the chunk.

    Example:
        >>> chunk = Chunk(
        ...     id="chunk123",
        ...     note_id="note456",
        ...     content="This is a chunk of text.",
        ...     heading_path="Introduction",
        ...     position=100,
        ...     char_count=25,
        ... )
    """
    id: str
    note_id: str
    content: str
    heading_path: str = ""
    position: int = 0
    char_count: int = 0


@dataclass
class SearchResult:
    """Represents a search result from the query engine.

    Attributes:
        chunk_id: ID of the matched chunk.
        note_id: ID of the parent note.
        note_path: File system path to the note.
        content: Content of the matched chunk.
        heading_path: Heading path within the note.
        score: Relevance score (higher is better).
        source: Search source ("vector", "keyword", or "hybrid").

    Example:
        >>> result = SearchResult(
        ...     chunk_id="chunk123",
        ...     note_id="note456",
        ...     note_path=Path("/notes/test.md"),
        ...     content="Relevant content here",
        ...     heading_path="Introduction",
        ...     score=0.85,
        ...     source="hybrid",
        ... )
    """
    chunk_id: str
    note_id: str
    note_path: Path
    content: str
    heading_path: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Convert SearchResult to dictionary representation.

        Returns:
            Dictionary containing all SearchResult fields.
        """
        return {
            "chunk_id": self.chunk_id,
            "note_id": self.note_id,
            "note_path": str(self.note_path),
            "content": self.content,
            "heading_path": self.heading_path,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class ChatMessage:
    """Represents a message in a chat conversation.

    Attributes:
        role: Role of the message sender ("system", "user", or "assistant").
        content: Content of the message.

    Example:
        >>> msg = ChatMessage(role="user", content="Hello!")
        >>> msg.to_dict()
        {'role': 'user', 'content': 'Hello!'}
    """
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        """Convert ChatMessage to dictionary representation.

        Returns:
            Dictionary containing role and content.
        """
        return {"role": self.role, "content": self.content}
