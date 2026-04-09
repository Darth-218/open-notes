from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class KeywordIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                content TEXT NOT NULL,
                heading_path TEXT,
                position INTEGER
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='rowid'
            )
        """)

        conn.commit()
        conn.close()

    def index_chunks(
        self, chunks: list[dict[str, Any]], note_id: str
    ) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        cursor.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE note_id = ?)",
            (note_id,),
        )

        for chunk in chunks:
            cursor.execute(
                """
                INSERT INTO chunks (id, note_id, content, heading_path, position)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    note_id,
                    chunk["content"],
                    chunk.get("heading_path", ""),
                    chunk.get("position", 0),
                ),
            )

            cursor.execute(
                "INSERT INTO chunks_fts (content) VALUES (?)",
                (chunk["content"],),
            )

        conn.commit()
        conn.close()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT c.id, c.note_id, c.content, c.heading_path, c.position,
                   bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "chunk_id": row[0],
                    "note_id": row[1],
                    "content": row[2],
                    "heading_path": row[3] or "",
                    "position": row[4],
                    "score": abs(row[5]) if row[5] else 0,
                }
            )

        conn.close()
        return results

    def delete_by_note_id(self, note_id: str) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        cursor.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE note_id = ?)",
            (note_id,),
        )

        conn.commit()
        conn.close()

    def rebuild(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute(
            "INSERT INTO chunks_fts (content) SELECT content FROM chunks"
        )

        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0] if cursor.fetchone() else 0

        conn.close()
        return {"total_chunks": count}
