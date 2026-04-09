from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

import frontmatter


class NoteStorage:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def scan_notes(self, extensions: list[str] | None = None) -> list[Path]:
        if extensions is None:
            extensions = [".md"]

        notes = []
        for ext in extensions:
            notes.extend(self.base_path.rglob(f"*{ext}"))

        return sorted(notes)

    def read_note(self, path: Path) -> dict:
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
        self, path: Path, content: str, frontmatter: dict | None = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        post = frontmatter.Post(content, **((frontmatter or {})))

        with open(path, "w") as f:
            f.write(frontmatter.dumps(post))

    def _extract_title_from_content(self, content: str) -> str | None:
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _generate_note_id(self, path: Path, content: str) -> str:
        combined = f"{path}:{content[:1000]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
