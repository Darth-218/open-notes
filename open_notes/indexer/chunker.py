from __future__ import annotations

import uuid
from typing import Any

from open_notes.indexer.parser import MarkdownParser, ParsedHeading


class MarkdownChunker:
    """Chunks Markdown content into smaller sections for indexing.

    Splits content either by headings (if present and under max_chars) or by
    paragraph boundaries with a maximum character limit. Supports overlapping
    chunks to maintain context across chunk boundaries.

    Attributes:
        max_chars: Maximum number of characters per chunk.
        overlap_chars: Number of characters to overlap between chunks when
            splitting large sections.
    """

    def __init__(
        self,
        max_chars: int = 1000,
        overlap_chars: int = 100,
    ):
        """Initialize the MarkdownChunker.

        Args:
            max_chars: Maximum number of characters per chunk. Defaults to 1000.
            overlap_chars: Number of characters to overlap between chunks
                when splitting large sections. Defaults to 100.
        """
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.parser = MarkdownParser()

    def chunk(
        self, note_id: str, content: str
    ) -> list[dict[str, Any]]:
        """Chunk Markdown content into smaller sections.

        If the content contains headings, splits by headings first. If a
        heading section exceeds max_chars, further splits by paragraph or
        word boundaries. If no headings exist, chunks by size directly.

        Args:
            note_id: The unique identifier for the note being chunked.
            content: The Markdown content to chunk.

        Returns:
            A list of chunk dictionaries, each containing:
                - id: Unique chunk identifier (UUID).
                - note_id: The note identifier.
                - content: The chunk text content.
                - heading_path: The heading path in the document
                    (e.g., "Introduction/Background").
                - position: Character position of the chunk in the
                    original document.
                - char_count: Number of characters in the chunk.
        """
        headings, content = self.parser.parse(content)

        chunks = []
        if not headings:
            return self._chunk_by_size(note_id, content, "")

        sections = self._split_by_headings(content, headings)

        for section in sections:
            heading_path = section["heading_path"]
            section_content = section["content"]

            if len(section_content) <= self.max_chars:
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "note_id": note_id,
                        "content": section_content.strip(),
                        "heading_path": heading_path,
                        "position": section["position"],
                        "char_count": len(section_content),
                    }
                )
            else:
                sub_chunks = self._chunk_by_size(
                    note_id, section_content, heading_path, section["position"]
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_headings(
        self, content: str, headings: list[ParsedHeading]
    ) -> list[dict[str, Any]]:
        """Split content into sections based on heading positions.

        Args:
            content: The Markdown content to split.
            headings: List of parsed headings.

        Returns:
            A list of section dictionaries with heading_path, content,
            and position keys.
        """
        sections = []
        content_len = len(content)

        for i, heading in enumerate(headings):
            next_position = (
                headings[i + 1].position if i + 1 < len(headings) else content_len
            )

            section_content = content[heading.position:next_position].strip()
            heading_text = section_content.split("\n", 1)[0] if "\n" in section_content else section_content
            body_content = section_content[len(heading_text):].strip()

            heading_path = self._build_heading_path(headings[: i + 1])

            sections.append(
                {
                    "heading_path": heading_path,
                    "content": body_content or section_content,
                    "position": heading.position,
                }
            )

        return sections

    def _chunk_by_size(
        self,
        note_id: str,
        content: str,
        heading_path: str,
        base_position: int = 0,
    ) -> list[dict[str, Any]]:
        """Split content into chunks by maximum character size.

        Splits by paragraph boundaries first, then by word boundaries
        if paragraphs exceed max_chars. Includes overlap between
        consecutive chunks.

        Args:
            note_id: The note identifier.
            content: The content to chunk.
            heading_path: The heading path for these chunks.
            base_position: The base character position in the original
                document.

        Returns:
            A list of chunk dictionaries.
        """
        if len(content) <= self.max_chars:
            return [
                {
                    "id": str(uuid.uuid4()),
                    "note_id": note_id,
                    "content": content.strip(),
                    "heading_path": heading_path,
                    "position": base_position,
                    "char_count": len(content),
                }
            ]

        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        current_position = base_position

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "note_id": note_id,
                            "content": current_chunk.strip(),
                            "heading_path": heading_path,
                            "position": current_position,
                            "char_count": len(current_chunk),
                        }
                    )

                    overlap_start = max(0, len(current_chunk) - self.overlap_chars)
                    current_chunk = current_chunk[overlap_start:]
                    current_position = base_position + len(current_chunk)

                if len(para) > self.max_chars:
                    words = para.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= self.max_chars:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk:
                                chunks.append(
                                    {
                                        "id": str(uuid.uuid4()),
                                        "note_id": note_id,
                                        "content": temp_chunk.strip(),
                                        "heading_path": heading_path,
                                        "position": current_position,
                                        "char_count": len(temp_chunk),
                                    }
                                )
                            temp_chunk = word + " "
                            current_position = base_position + len(temp_chunk)
                    current_chunk = temp_chunk
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "note_id": note_id,
                    "content": current_chunk.strip(),
                    "heading_path": heading_path,
                    "position": current_position,
                    "char_count": len(current_chunk),
                }
            )

        return chunks

    def _build_heading_path(self, headings: list[ParsedHeading]) -> str:
        """Build a heading path string from a list of headings.

        Args:
            headings: List of headings to join.

        Returns:
            A path string with headings separated by "/".
        """
        return "/".join(h.text for h in headings)