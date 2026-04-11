"""Keyword search index using SQLite FTS5.

This module provides the KeywordIndex class for full-text search
using SQLite's FTS5 virtual table with BM25 ranking.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class KeywordIndex:
    """Full-text search index using SQLite FTS5.

    Provides keyword-based search using SQLite's FTS5 virtual table with
    BM25 ranking for relevance scoring. Works completely offline.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> idx = KeywordIndex(Path("~/.open_notes/keywords.db"))
        >>> idx.index_chunks(chunks, "note123")
        >>> results = idx.search("machine learning", top_k=5)
    """

    def __init__(self, db_path: Path):
        """Initialize KeywordIndex with a database path.

        Args:
            db_path: Path where SQLite database will be stored.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with FTS5 virtual table."""
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
        """Index chunks for keyword search.

        Replaces any existing chunks for the given note_id with new ones.

        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'heading_path', 'position'.
            note_id: ID of the note these chunks belong to.

        Example:
            >>> chunks = [
            ...     {"id": "c1", "content": "Machine learning is...", "heading_path": "Intro", "position": 0},
            ...     {"id": "c2", "content": "Deep learning uses...", "heading_path": "DL", "position": 100}
            ... ]
            >>> idx.index_chunks(chunks, "note123")
        """
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
        """Search for chunks matching the query.

        Uses FTS5 MATCH with BM25 ranking for relevance scoring.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dictionaries containing:
                - chunk_id: ID of the matching chunk
                - note_id: ID of the parent note
                - content: Chunk content
                - heading_path: Heading path in document
                - position: Position in document
                - score: BM25 score (absolute value, higher is better)

        Example:
            >>> results = idx.search("neural networks", top_k=10)
            >>> for r in results:
            ...     print(f"{r['chunk_id']}: {r['score']:.2f}")
        """
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
        """Delete all indexed chunks for a note.

        Args:
            note_id: ID of the note to remove from the index.

        Example:
            >>> idx.delete_by_note_id("note123")
        """
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
        """Rebuild the FTS5 index from the chunks table.

        Use this if the index becomes corrupted or out of sync.

        Example:
            >>> idx.rebuild()
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute(
            "INSERT INTO chunks_fts (content) SELECT content FROM chunks"
        )

        conn.commit()
        conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the keyword index.

        Returns:
            Dictionary containing:
                - total_chunks: Number of chunks indexed

        Example:
            >>> stats = idx.get_stats()
            >>> print(f"Indexed chunks: {stats['total_chunks']}")
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0] if cursor.fetchone() else 0

        conn.close()
        return {"total_chunks": count}
