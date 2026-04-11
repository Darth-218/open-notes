"""MCP (Model Context Protocol) server for Open Notes.

This module provides an MCP server that exposes note management functionality
through the Model Context Protocol, enabling AI assistants to interact with
the Open Notes system.

The server provides tools for searching, retrieving, listing notes, and
performing RAG (Retrieval-Augmented Generation) queries. It also exposes
notes as resources that can be accessed via URI patterns.
"""

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
    """Initialize the MCP server with required dependencies.

    Args:
        config: Application configuration instance.
        query_engine: Query engine for searching notes.
        rag_pipeline: RAG pipeline for augmented queries.
        note_storage: Storage backend for note access.

    Returns:
        None.

    Raises:
        None.
    """
    global _config, _query_engine, _rag_pipeline, _note_storage
    _config = config
    _query_engine = query_engine
    _rag_pipeline = rag_pipeline
    _note_storage = note_storage


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> str:
    """Search notes using hybrid search (BM25 + semantic).

    Performs a hybrid search across all notes combining keyword and semantic
    similarity to find the most relevant results.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return. Defaults to 5.

    Returns:
        A formatted string containing search results with titles, content
        snippets, and relevance scores. Returns "No results found" if no
        matches exist, or an error message if the query engine is not
        initialized.

    Raises:
        None.
    """
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
    """Retrieve a specific note by its unique identifier.

    Fetches the complete content of a note based on its ID. This includes
    the title and full content body.

    Args:
        note_id: The unique identifier of the note to retrieve.

    Returns:
        The note content formatted as markdown with title as heading,
        or an error message if not found or not initialized.

    Raises:
        None.
    """
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
    """List all available notes with pagination support.

    Retrieves a list of all notes in the storage system, with support for
    pagination through limit and offset parameters.

    Args:
        limit: Maximum number of notes to return. Defaults to 10.
        offset: Number of notes to skip. Defaults to 0.

    Returns:
        A formatted string listing notes with titles and IDs, or "No notes
        found" if empty, or an error message if not initialized.

    Raises:
        None.
    """
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
    """Query notes using Retrieval-Augmented Generation.

    Uses the RAG pipeline to answer questions by retrieving relevant context
    from notes and generating an answer using an LLM.

    Args:
        query: The question or query to answer.
        top_k: Number of context chunks to retrieve. Defaults to 5.

    Returns:
        The generated answer. If sources are available, includes source
        excerpts at the end. Returns error message if RAG pipeline is not
        initialized.

    Raises:
        None.
    """
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
    """List all notes as an MCP resource.

    Exposes the list_notes functionality as a resource that can be accessed
    via the URI scheme "notes://list".

    Returns:
        A formatted string containing the list of notes with titles and IDs.

    Raises:
        None.
    """
    return list_notes()


@mcp.resource("note://{note_id}")
def get_note_resource(note_id: str) -> str:
    """Get a specific note by ID as an MCP resource.

    Exposes the get_note functionality as a resource that can be accessed
    via the URI scheme "note://{note_id}".

    Args:
        note_id: The unique identifier of the note to retrieve.

    Returns:
        The note content formatted as markdown with title as heading,
        or an error message if not found.

    Raises:
        None.
    """
    return get_note(note_id)