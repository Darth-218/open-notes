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
    def __init__(self, config: Config | None = None):
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
        if self._embedding is None:
            self._embedding = create_embedding(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device,
                batch_size=self.config.embedding_batch_size,
            )
        return self._embedding

    @property
    def vector_db(self):
        if self._vector_db is None:
            self._vector_db = VectorDB(
                db_path=self.config.vector_db_path,
                dimension=self.embedding.dimension,
            )
        return self._vector_db

    @property
    def keyword_index(self):
        if self._keyword_index is None:
            self._keyword_index = KeywordIndex(db_path=self.config.keyword_db_path)
        return self._keyword_index

    @property
    def note_storage(self):
        if self._note_storage is None:
            self._note_storage = NoteStorage(base_path=self.config.knowledge_base_path)
        return self._note_storage

    @property
    def query_engine(self):
        if self._query_engine is None:
            self._query_engine = QueryEngine(
                vector_db=self.vector_db,
                keyword_index=self.keyword_index,
                embedding=self.embedding,
            )
        return self._query_engine

    @property
    def llm(self) -> BaseLLM:
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
        if self._rag_pipeline is None:
            self._rag_pipeline = RAGPipeline(
                query_engine=self.query_engine,
                llm=self.llm,
                prompt_template=self.config.rag_prompt_template,
            )
        return self._rag_pipeline

    def index_all(self) -> dict[str, Any]:
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
        top_k = top_k or self.config.search_top_k
        return self.query_engine.search(
            query=query,
            mode=self.config.search_mode,
            top_k=top_k,
            vector_weight=self.config.search_vector_weight,
            keyword_weight=self.config.search_keyword_weight,
        )

    def query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or self.config.search_top_k
        result = self.rag_pipeline.query(query=query, top_k=top_k)

        return {
            "answer": result.answer,
            "sources": [s.to_dict() for s in result.sources],
        }

    def watch(self, callback: Any = None) -> None:
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
        vector_stats = self.vector_db.get_stats()
        keyword_stats = self.keyword_index.get_stats()

        return {
            "vector_db": vector_stats,
            "keyword_index": keyword_stats,
        }
