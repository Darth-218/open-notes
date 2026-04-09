from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Note:
    id: str
    path: Path
    title: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
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
    id: str
    note_id: str
    content: str
    heading_path: str = ""
    position: int = 0
    char_count: int = 0


@dataclass
class SearchResult:
    chunk_id: str
    note_id: str
    note_path: Path
    content: str
    heading_path: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
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
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}
