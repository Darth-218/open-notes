from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from open_notes.config import Config
from open_notes.embedding.transformers import create_embedding
from open_notes.indexer.chunker import MarkdownChunker
from open_notes.indexer.watcher import NoteWatcher
from open_notes.llm.base import BaseLLM, DummyLLM
from open_notes.llm.llama_cpp import create_llm
from open_notes.models import SearchResult
from open_notes.query.engine import QueryEngine
from open_notes.rag.pipeline import RAGPipeline
from open_notes.storage.file_system import NoteStorage
from open_notes.storage.keyword_index import KeywordIndex
from open_notes.storage.vector_db import VectorDB


class OpenNotes:
    """Main interface for open-notes RAG system.

    This class provides a unified interface to the open-notes knowledge base
    system, providing methods for indexing notes, searching, querying with LLM,
    and watching for file changes.

    The class uses lazy initialization for all components (embedding, vector DB,
    keyword index, LLM, etc.) to minimize startup time and resource usage.
    Components are initialized on first access.

    Attributes:
        config: Configuration instance for open-notes.
        embedding: Lazy-loaded embedding model.
        vector_db: Lazy-loaded vector database.
        keyword_index: Lazy-loaded keyword search index.
        note_storage: Lazy-loaded note storage handler.
        query_engine: Lazy-loaded query engine.
        llm: Lazy-loaded LLM interface.
        rag_pipeline: Lazy-loaded RAG pipeline.

    Example:
        >>> from open_notes import OpenNotes
        >>> on = OpenNotes()
        >>> results = on.search("Python async programming")
        >>> for r in results:
        ...     print(r.content[:100])
    """

    def __init__(self, config: Config | None = None):
        """Initialize OpenNotes instance.

        Args:
            config: Optional Config instance. If not provided, loads default config.
        """
        self.config = config or Config.load()
        self._embedding = None
        self._vector_db = None
        self._keyword_index = None
        self._note_storage = None
        self._query_engine = None
        self._llm = None
        self._rag_pipeline = None

    @property
    def embedding(self):
        """Get the embedding model instance.

        Lazy-initializes the embedding model on first access using configuration
        from self.config.

        Returns:
            Embedding model instance configured from config.
        """
        if self._embedding is None:
            self._embedding = create_embedding(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedding

    @property
    def vector_db(self):
        """Get the vector database instance.

        Lazy-initializes the vector database on first access. The database
        dimension is determined by the embedding model.

        Returns:
            VectorDB instance initialized with configured path.
        """
        if self._vector_db is None:
            self._vector_db = VectorDB(
                db_path=self.config.vector_db_path,
                dimension=self.embedding.dimension,
            )
        return self._vector_db

    @property
    def keyword_index(self):
        """Get the keyword search index instance.

        Lazy-initializes the keyword index on first access using the configured
        SQLite FTS database path.

        Returns:
            KeywordIndex instance for full-text search.
        """
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex(db_path=self.config.keyword_db_path)
        return self._keyword_index

    @property
    def note_storage(self):
        """Get the note storage handler instance.

        Lazy-initializes the note storage on first access using the knowledge
        base path from configuration.

        Returns:
            NoteStorage instance for reading/writing notes.
        """
        if self._note_storage is None:
            self._note_storage = NoteStorage(base_path=self.config.knowledge_base_path)
        return self._note_storage

    @property
    def query_engine(self):
        """Get the query engine instance.

        Lazy-initializes the query engine on first access. The query engine
        combines vector search, keyword search, or both based on configuration.

        Returns:
            QueryEngine instance configured with vector and keyword indexes.
        """
        if self._query_engine is None:
            self._query_engine = QueryEngine(
                vector_db=self.vector_db,
                keyword_index=self.keyword_index,
                embedding=self.embedding,
            )
        return self._query_engine

    @property
    def llm(self) -> BaseLLM:
        """Get the LLM interface instance.

        Lazy-initializes the LLM on first access. If initialization fails,
        falls back to a dummy LLM that returns empty responses.

        Returns:
            BaseLLM implementation configured from config.
        """
        if self._llm is None:
            try:
                self._llm = create_llm(
                    provider=self.config.llm_provider,
                    model_path=self.config.llm_model_path,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
            except Exception as e:
                logging.warning(f"Failed to initialize LLM: {e}. Using dummy LLM.")
                self._llm = DummyLLM()
        return self._llm

    @property
    def rag_pipeline(self):
        """Get the RAG pipeline instance.

        Lazy-initializes the RAG pipeline on first access. The pipeline
        combines the query engine with the LLM to answer questions using
        retrieved context.

        Returns:
            RAGPipeline instance for question answering.
        """
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(
                query_engine=self.query_engine,
                llm=self.llm,
                prompt_template=self.config.rag_prompt_template,
            )
        return self._rag_pipeline

    def index_all(self) -> dict[str, Any]:
        """Index all notes in the knowledge base.

        Scans the knowledge base directory for markdown notes, chunks them
        into smaller pieces, generates embeddings, and stores them in both
        the vector database and keyword index.

        If the knowledge base directory doesn't exist, it will be created.

        Returns:
            Dictionary containing:
                - indexed: Number of notes successfully indexed
                - chunks: Total number of text chunks created
                - message: Human-readable status message
        """
        kb_path = self.config.knowledge_base_path
        if not kb_path.exists():
            kb_path.mkdir(parents=True, exist_ok=True)
            return {"indexed": 0, "chunks": 0, "message": f"Created knowledge base at {kb_path}"}

        note_paths = self.note_storage.scan_notes()

        if not note_paths:
            return {"indexed": 0, "chunks": 0, "message": "No notes found"}

        chunker = MarkdownChunker(
            max_chars=self.config.chunker_max_chars,
            overlap_chars=self.config.chunker_overlap,
        )

        total_chunks = 0
        indexed_notes = 0

        for path in tqdm(note_paths, desc="Indexing notes"):
            try:
                note_data = self.note_storage.read_note(path)
                note_id = note_data["id"]
                content = note_data["content"]

                chunks = chunker.chunk(note_id=note_id, content=content)
                total_chunks += len(chunks)

                if chunks:
                    embeddings = self.embedding.embed([c["content"] for c in chunks])

                    metadata = [
                        {
                            "chunk_id": c["id"],
                            "note_id": note_id,
                            "note_path": str(path),
                            "content": c["content"],
                            "heading_path": c.get("heading_path", ""),
                        }
                        for c in chunks
                    ]

                    self.vector_db.add_vectors(embeddings, metadata)
                    self.keyword_index.index_chunks(chunks, note_id)

                indexed_notes += 1

            except Exception as e:
                logging.warning(f"Failed to index {path}: {e}")
                continue

        return {
            "indexed": indexed_notes,
            "chunks": total_chunks,
            "message": f"Indexed {indexed_notes} notes with {total_chunks} chunks",
        }

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search the knowledge base for relevant notes.

        Performs a search using the configured search mode (vector, keyword,
        or hybrid) and returns the top matching results.

        Args:
            query: Search query string.
            top_k: Number of results to return. Defaults to config value.

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        top_k = top_k or self.config.search_top_k
        return self.query_engine.search(
            query=query,
            mode=self.config.search_mode,
            top_k=top_k,
            vector_weight=self.config.search_vector_weight,
            keyword_weight=self.config.search_keyword_weight,
        )

    def query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Query the knowledge base with an LLM.

        Uses RAG (Retrieval-Augmented Generation) to answer questions by
        retrieving relevant context and feeding it to the LLM.

        Args:
            query: Question to ask about the knowledge base.
            top_k: Number of context chunks to retrieve. Defaults to config value.

        Returns:
            Dictionary containing:
                - answer: LLM-generated answer
                - sources: List of source documents used for context
        """
        top_k = top_k or self.config.search_top_k
        result = self.rag_pipeline.query(query=query, top_k=top_k)

        return {
            "answer": result.answer,
            "sources": [s.to_dict() for s in result.sources],
        }

    def watch(self, callback: Any = None) -> None:
        """Watch the knowledge base for file changes and re-index automatically.

        Starts a file system watcher that monitors the knowledge base directory
        for changes to markdown files. When a file is created or modified,
        it is automatically re-indexed.

        Args:
            callback: Optional custom callback function. If not provided,
                uses default indexing behavior. The callback receives a Path
                object for the changed file.

        Example:
            >>> on = OpenNotes()
            >>> on.watch()  # Press Ctrl+C to stop
        """
        kb_path = self.config.knowledge_base_path
        if not kb_path.exists():
            kb_path.mkdir(parents=True, exist_ok=True)

        def on_change(path: Path) -> None:
            try:
                note_data = self.note_storage.read_note(path)
                note_id = note_data["id"]
                content = note_data["content"]

                chunker = MarkdownChunker(
                    max_chars=self.config.chunker_max_chars,
                    overlap_chars=self.config.chunker_overlap,
                )

                chunks = chunker.chunk(note_id=note_id, content=content)

                if chunks:
                    embeddings = self.embedding.embed([c["content"] for c in chunks])
                    metadata = [
                        {
                            "chunk_id": c["id"],
                            "note_id": note_id,
                            "note_path": str(path),
                            "content": c["content"],
                            "heading_path": c.get("heading_path", ""),
                        }
                        for c in chunks
                    ]
                    self.vector_db.add_vectors(embeddings, metadata)
                    self.keyword_index.index_chunks(chunks, note_id)

                print(f"Indexed: {path.name}")
            except Exception as e:
                print(f"Error indexing {path}: {e}")

        handler = callback or on_change
        watcher = NoteWatcher(kb_path, handler)

        print(f"Watching {kb_path} for changes... (Ctrl+C to stop)")
        try:
            watcher.start()
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping watcher...")
            watcher.stop()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge base and indexes.

        Returns:
            Dictionary containing:
                - vector_db: Vector database statistics
                - keyword_index: Keyword index statistics
        """
        vector_stats = self.vector_db.get_stats()
        keyword_stats = self.keyword_index.get_stats()

        return {
            "vector_db": vector_stats,
            "keyword_index": keyword_stats,
        }
