# open-notes Pipeline

Complete architecture overview of the open-notes local AI knowledge base system.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User / CLI / MCP                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenNotes (Main Interface)                          │
│                    index_all() | search() | query() | watch()               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│    Indexing Path    │   │    Query Path       │   │     Watch Path      │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

---

## Stage 1: Configuration

### Files Involved
- `config/default.yaml` - Default configuration
- `open_notes/config.py` - Config loader

### Flow
```
Config.load()
    │
    ├── 1. Load default.yaml from package
    │
    ├── 2. Load user config (~/.open_notes/config.yaml)
    │
    └── 3. Apply env overrides (OPEN_NOTES_*)
         │
         └── Returns: Config object with properties
              ├── knowledge_base_path
              ├── embedding_model, device, batch_size
              ├── vector_db_path, keyword_db_path
              ├── chunker_max_chars, chunker_overlap
              ├── search_top_k, search_mode, weights
              └── llm_provider, llm_model_path, etc.
```

---

## Stage 2: Storage Layer

### File System (`open_notes/storage/file_system.py`)

```
NoteStorage
    │
    ├── scan_notes() → List[Path]
    │       └── rglob("*.md") recursively
    │
    ├── read_note(path) → dict
    │       └── frontmatter.load() → {id, path, title, frontmatter, content, updated_at}
    │
    └── write_note(path, content, frontmatter)
            └── frontmatter.Post() → dump to file
```

### Vector DB (`open_notes/storage/vector_db.py`)

```
VectorDB (FAISS)
    │
    ├── __init__(db_path, dimension)
    │       └── faiss.IndexFlatL2 + JSON metadata
    │
    ├── add_vectors(vectors, metadata)
    │       └── np.array(vectors) → index.add()
    │
    ├── search(query_vector, top_k) → List[(idx, score, meta)]
    │       └── L2 distance → score = 1/(1+distance)
    │
    └── get_stats() → {total_vectors, dimension}
```

### Keyword Index (`open_notes/storage/keyword_index.py`)

```
KeywordIndex (SQLite FTS5)
    │
    ├── __init__(db_path)
    │       └── CREATE TABLE chunks + FTS5 virtual table
    │
    ├── index_chunks(chunks, note_id)
    │       └── INSERT INTO chunks + chunks_fts
    │
    ├── search(query, top_k) → List[dict]
    │       └── BM25(chunks_fts) ranking
    │
    ├── delete_by_note_id(note_id)
    │       └── DELETE FROM chunks + chunks_fts
    │
    └── get_stats() → {total_chunks}
```

---

## Stage 3: Embedding Layer

### File: `open_notes/embedding/transformers.py`

```
SentenceTransformerEmbedding
    │
    ├── __init__(model_name, device, batch_size)
    │       └── Lazy-load: SentenceTransformer(model_name)
    │
    ├── embed(texts) → List[List[float]]
    │       └── model.encode(texts, batch_size=batch_size)
    │
    └── dimension → int
            └── model.get_sentence_embedding_dimension()
```

**Default Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

---

## Stage 4: Indexing

### Parser (`open_notes/indexer/parser.py`)

```
MarkdownParser
    │
    └── parse(content) → (List[ParsedHeading], cleaned_content)
          │
          ├── Extract headings (# to ######)
          ├── Extract level (1-6)
          ├── Extract position in content
          └── Return: [ParsedHeading(text, level, position), ...]
```

### Chunker (`open_notes/indexer/chunker.py`)

```
MarkdownChunker
    │
    └── chunk(note_id, content) → List[dict]
          │
          ├── If no headings: _chunk_by_size()
          │
          └── If headings:
               │
               ├── _split_by_headings() → sections
               │     └── Split content by heading positions
               │
               └── For each section:
                    ├── If len(content) <= max_chars: single chunk
                    └── Else: _chunk_by_size() with overlap

    Output chunk structure:
    {
        "id": uuid,
        "note_id": str,
        "content": str,
        "heading_path": "Main/Section/Sub",
        "position": int,
        "char_count": int
    }
```

### Watcher (`open_notes/indexer/watcher.py`)

```
NoteWatcher
    │
    └── watch(callback)
          └── watchdog.Observer()
               ├── on_created
               ├── on_modified
               └── on_deleted → re-index affected files
```

---

## Stage 5: Query Engine

### File: `open_notes/query/engine.py`

```
QueryEngine
    │
    └── search(query, mode, top_k, weights) → List[SearchResult]
          │
          ├── mode="vector": _vector_search()
          │     │
          │     ├── embedding.embed([query])
          │     ├── vector_db.search(embedding, top_k)
          │     └── → List[SearchResult]
          │
          ├── mode="keyword": _keyword_search()
          │     │
          │     └── keyword_index.search(query, top_k)
          │
          └── mode="hybrid": _hybrid_search()
                │
                ├── _vector_search() → results1
                ├── _keyword_search() → results2
                ├── normalize_scores() → [0,1] range
                ├── reciprocal_rank_fusion(k=60)
                └── final_score = v_score * weight + k_score * weight

    SearchResult:
    {
        chunk_id, note_id, note_path, content,
        heading_path, score, source: "vector"|"keyword"|"hybrid"
    }
```

---

## Stage 6: RAG Pipeline

### File: `open_notes/rag/pipeline.py`

```
RAGPipeline
    │
    └── query(question, top_k) → RAGResult
          │
          ├── 1. query_engine.search(question, top_k)
          │     └── Returns: List[SearchResult]
          │
          ├── 2. Build context:
          │     context = "\n\n---\n\n".join([
          │         f"Source: {s.note_path}\n{s.content}"
          │         for s in results
          │     ])
          │
          ├── 3. Format prompt:
          │     prompt = template.format(
          │         context=context,
          │         question=question
          │     )
          │
          ├── 4. LLM.generate(prompt)
          │     └── Returns: str (answer)
          │
          └── 5. Return RAGResult(answer, sources)
```

---

## Stage 7: LLM Providers

### Base: `open_notes/llm/base.py`

```
BaseLLM (ABC)
    │
    ├── generate(prompt, **kwargs) → str
    ├── chat(messages, **kwargs) → str
    └── name → str
```

### Implementations:

| Provider | File | Model |
|----------|------|-------|
| llama_cpp | `llm/llama_cpp.py` | GGUF files |
| Ollama | `llm/llama_cpp.py` | Local Ollama server |
| OpenAI | `llm/llama_cpp.py` | OpenAI API / compatible |

**Factory**: `create_llm(provider, **kwargs)` → BaseLLM

---

## Stage 8: CLI Commands

### File: `open_notes/cli/main.py`

```
open-notes [OPTIONS] COMMAND [ARGS]...

Commands:
    │
    ├── index     → Index all notes in KB
    │
    ├── search    → Search notes (vector/keyword/hybrid)
    │
    ├── query     → RAG query with LLM
    │
    ├── serve     → Start MCP server
    │
    ├── watch     → Auto-index on file changes
    │
    ├── config    → Show current configuration
    │
    └── stats     → Show index statistics
```

---

## Stage 9: MCP Server

### File: `open_notes/mcp/server.py`

```
MCP Server (FastMCP)
    │
    ├── Tools:
    │   ├── search_notes(query, top_k)
    │   ├── get_note(note_id)
    │   ├── list_notes(limit, offset)
    │   └── rag_query(query, top_k)
    │
    └── Resources:
        ├── notes://list
        └── note://{note_id}
```

---

## Complete Data Flow

```
                    ┌──────────────────┐
                    │   User Input     │
                    │ (CLI/API/MCP)    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   OpenNotes      │
                    │   Main Class     │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│   Index Path   │  │  Query Path    │  │   Watch Path   │
└────────┬───────┘  └────────┬───────┘  └───────┬────────┘
         │                   │                  │
         ▼                   ▼                  │
┌────────────────┐  ┌────────────────┐          │
│ NoteStorage    │  │ QueryEngine    │          │
│ scan/read      │  │ search()       │          │
└────────┬───────┘  └────────┬───────┘          │
         │                   │                  │
         ▼                   ▼                  │
┌────────────────┐  ┌────────────────┐          │
│ MarkdownChunker│  │ VectorDB       │          │
│ chunk()        │  │ + KeywordIndex │          │
└────────┬───────┘  └────────┬───────┘          │
         │                   │                  │
         ▼                   ▼                  ▼
┌─────────────────┐ ┌────────────────┐  ┌────────────────┐
│ SentenceTransform││ RAGPipeline    │  │ NoteWatcher    │
│ embedding       │ │ query()        │  │ (watchdog)     │
└─────────────────┘ └────────┬───────┘  └────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ LLM Provider    │
                    │ (llama.cpp/     │
                    │  Ollama/OpenAI) │
                    └─────────────────┘
```

---

## File Structure Summary

```
open-notes/
├── config/
│   └── default.yaml              # Default configuration
│
├── open_notes/                   # Main package
│   ├── __init__.py              # OpenNotes main class
│   ├── config.py                # Config loader
│   ├── models.py                # Data models (Note, Chunk, SearchResult)
│   │
│   ├── storage/                 # Storage layer
│   │   ├── file_system.py       # Note reading/writing
│   │   ├── vector_db.py         # FAISS vector database
│   │   └── keyword_index.py     # SQLite FTS5 index
│   │
│   ├── embedding/               # Embedding layer
│   │   ├── base.py              # BaseEmbedding abstract class
│   │   └── transformers.py      # Sentence Transformers impl
│   │
│   ├── indexer/                 # Indexing
│   │   ├── parser.py            # Markdown parser
│   │   ├── chunker.py           # Content chunking
│   │   └── watcher.py           # File system watcher
│   │
│   ├── query/                   # Query engine
│   │   └── engine.py            # Hybrid search (vector + keyword)
│   │
│   ├── llm/                     # LLM providers
│   │   ├── base.py              # BaseLLM abstract class
│   │   └── llama_cpp.py         # llama.cpp, Ollama, OpenAI
│   │
│   ├── rag/                     # RAG pipeline
│   │   └── pipeline.py          # Retrieval + Generation
│   │
│   ├── mcp/                     # MCP server
│   │   └── server.py            # FastMCP server
│   │
│   └── cli/                     # CLI
│       └── main.py              # Click commands
│
├── tests/                       # Test suite
│   ├── fixtures/notes/          # Sample markdown notes
│   ├── conftest.py              # Pytest fixtures
│   ├── test_*.py                # Unit tests
│   ├── storage/                 # Storage tests
│   └── indexer/                 # Indexer tests
│
├── docs/                        # Documentation
│   ├── guides/                  # User guides
│   └── code/                    # API docs
│
├── pyproject.toml              # Python project config
├── flake.nix                   # Nix development environment
├── SPEC.md                     # Technical specification
└── README.md                   # Project readme
```

---

## Key Configuration Options

| Section | Option | Default | Description |
|---------|--------|---------|-------------|
| `knowledge_base` | path | `~/knowledge-base` | Notes directory |
| `embedding` | model | `all-MiniLM-L6-v2` | Sentence transformer model |
| `embedding` | device | `cpu` | Computation device |
| `vector_db` | path | `~/.open_notes/vectors` | FAISS index file |
| `keyword_index` | path | `~/.open_notes/keywords.db` | SQLite FTS5 DB |
| `chunker` | max_chars | 1000 | Max chunk size |
| `chunker` | overlap | 100 | Chunk overlap |
| `search` | mode | `hybrid` | vector/keyword/hybrid |
| `search` | vector_weight | 0.7 | Hybrid weight |
| `llm` | provider | `llama_cpp` | LLM backend |
| `llm` | model_path | - | GGUF model file |
| `mcp` | port | 8765 | MCP server port |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPEN_NOTES_CONFIG` | Custom config file path |
| `OPEN_NOTES_KB_PATH` | Override knowledge base path |
| `OPEN_NOTES_EMBEDDING_MODEL` | Override embedding model |
| `OPEN_NOTES_LLM_MODEL_PATH` | Override LLM model path |
