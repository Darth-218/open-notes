"""Storage layer for reading and writing note files.

This module provides the NoteStorage class for file system operations
on markdown notes, including reading, writing, and scanning.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

import frontmatter


class NoteStorage:
    """Handles reading and writing markdown notes from the file system.

    This class provides methods to scan, read, and write notes in the knowledge base.
    Notes are expected to be markdown files with optional YAML frontmatter.

    Attributes:
        base_path: Root directory containing notes.

    Example:
        >>> storage = NoteStorage(Path("~/knowledge-base"))
        >>> notes = storage.scan_notes()
        >>> note = storage.read_note(notes[0])
    """

    def __init__(self, base_path: Path):
        """Initialize NoteStorage with a base path.

        Args:
            base_path: Root directory containing notes.
        """
        self.base_path = base_path

    def scan_notes(self, extensions: list[str] | None = None) -> list[Path]:
        """Scan for notes in the knowledge base.

        Recursively searches the base path for files with the given extensions.

        Args:
            extensions: List of file extensions to include. Defaults to [".md"].

        Returns:
            Sorted list of Path objects for found notes.

        Example:
            >>> storage = NoteStorage(Path("~/notes"))
            >>> all_notes = storage.scan_notes()
            >>> md_notes = storage.scan_notes([".md", ".mdx"])
        """
        if extensions is None:
            extensions = [".md"]

        notes = []
        for ext in extensions:
            notes.extend(self.base_path.rglob(f"*{ext}"))

        return sorted(notes)

    def read_note(self, path: Path) -> dict:
        """Read a note from disk.

        Parses the markdown file and extracts frontmatter, title, and content.
        Title is extracted from frontmatter metadata, first heading, or filename.

        Args:
            path: Path to the note file.

        Returns:
            Dictionary containing:
                - id: Unique note identifier
                - path: Path object for the note
                - title: Note title
                - frontmatter: YAML metadata dictionary
                - content: Raw markdown content
                - updated_at: Last modified timestamp

        Raises:
            FileNotFoundError: If the note file doesn't exist.

        Example:
            >>> storage = NoteStorage(Path("~/notes"))
            >>> note = storage.read_note(Path("~/notes/test.md"))
            >>> print(note["title"])
            My Note
        """
        if not path.exists():
            raise FileNotFoundError(f"Note not found: {path}")

        post = frontmatter.load(path)
        title = post.metadata.get("title", "")

        if not title:
            title = self._extract_title_from_content(post.content) or path.stem

        note_id = self._generate_note_id(path, post.content)

        return {
            "id": note_id,
            "path": path,
            "title": title,
            "frontmatter": dict(post.metadata),
            "content": post.content,
            "updated_at": datetime.fromtimestamp(os.path.getmtime(path)),
        }

    def write_note(
        self, path: Path, content: str, metadata: dict | None = None
    ) -> None:
        """Write a note to disk.

        Creates the note file with optional YAML frontmatter.

        Args:
            path: Path where to save the note.
            content: Markdown content for the note.
            metadata: Optional YAML metadata dictionary.

        Raises:
            IOError: If the note cannot be written.

        Example:
            >>> storage = NoteStorage(Path("~/notes"))
            >>> storage.write_note(
            ...     Path("~/notes/new.md"),
            ...     "# New Note\\n\\nContent here",
            ...     {"title": "New Note", "tags": ["test"]}
            ... )
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        post = frontmatter.Post(content)
        if metadata:
            for key, value in metadata.items():
                post.metadata[key] = value

        with open(path, "w") as f:
            f.write(frontmatter.dumps(post))

    def _extract_title_from_content(self, content: str) -> str | None:
        """Extract title from the first heading in the content.

        Args:
            content: Markdown content to search.

        Returns:
            Title text without the heading marker, or None if no heading found.
        """
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _generate_note_id(self, path: Path, content: str) -> str:
        """Generate a unique identifier for a note.

        The ID is a SHA256 hash of the path and first 1000 characters of content.

        Args:
            path: Path to the note file.
            content: Note content.

        Returns:
            16-character hex string as the note ID.
        """
        combined = f"{path}:{content[:1000]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
