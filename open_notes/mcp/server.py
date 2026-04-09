from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from open_notes.config import Config
from open_notes.models import SearchResult
from open_notes.query.engine import QueryEngine
from open_notes.rag.pipeline import RAGPipeline
from open_notes.storage.file_system import NoteStorage

mcp = FastMCP("open-notes")
_config: Config | None = None
_query_engine: QueryEngine | None = None
_rag_pipeline: RAGPipeline | None = None
_note_storage: NoteStorage | None = None


def init_mcp(
    config: Config,
    query_engine: QueryEngine,
    rag_pipeline: RAGPipeline,
    note_storage: NoteStorage,
) -> None:
    global _config, _query_engine, _rag_pipeline, _note_storage
    _config = config
    _query_engine = query_engine
    _rag_pipeline = rag_pipeline
    _note_storage = note_storage


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> str:
    if _query_engine is None:
        return "Error: Query engine not initialized"

    results = _query_engine.search(
        query=query,
        mode="hybrid",
        top_k=top_k,
    )

    if not results:
        return "No results found"

    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"[{i}] {r.heading_path or 'Note'}\n"
            f"    {r.content[:200]}...\n"
            f"    Score: {r.score:.3f} ({r.source})"
        )

    return "\n\n".join(output)


@mcp.tool()
def get_note(note_id: str) -> str:
    if _note_storage is None or _config is None:
        return "Error: Not initialized"

    notes = _note_storage.scan_notes()
    for path in notes:
        try:
            note = _note_storage.read_note(path)
            if note["id"] == note_id:
                return f"# {note['title']}\n\n{note['content']}"
        except Exception:
            continue

    return f"Note not found: {note_id}"


@mcp.tool()
def list_notes(limit: int = 10, offset: int = 0) -> str:
    if _note_storage is None:
        return "Error: Not initialized"

    notes = _note_storage.scan_notes()

    notes = notes[offset : offset + limit]

    if not notes:
        return "No notes found"

    output = []
    for path in notes:
        try:
            note = _note_storage.read_note(path)
            output.append(f"- {note['title']} ({note['id']})")
        except Exception:
            output.append(f"- {path.name}")

    return "\n".join(output)


@mcp.tool()
def rag_query(query: str, top_k: int = 5) -> str:
    if _rag_pipeline is None:
        return "Error: RAG pipeline not initialized"

    result = _rag_pipeline.query(query=query, top_k=top_k)

    output = result.answer

    if result.sources:
        output += "\n\n--- Sources ---\n"
        for i, s in enumerate(result.sources, 1):
            output += f"\n[{i}] {s.heading_path or s.note_id}\n{s.content[:100]}..."

    return output


@mcp.resource("notes://list")
def list_notes_resource() -> str:
    return list_notes()


@mcp.resource("note://{note_id}")
def get_note_resource(note_id: str) -> str:
    return get_note(note_id)
