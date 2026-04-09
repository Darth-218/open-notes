# open-notes: Local AI Knowledge Base

## Project Overview

**Project Name**: open-notes  
**Type**: CLI tool / MCP Server  
**Purpose**: 100% local, offline, open-source AI knowledge base with semantic search and RAG capabilities for AI agent integration  
**Architecture**: Modular, pluggable components  

---

## Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Storage | Markdown files + YAML frontmatter | Portable, git-friendly |
| Vector DB | minDB | Best memory efficiency (3GB/100M vectors) |
| Keyword Search | SQLite FTS5 | Built-in, fast, offline |
| Embeddings | sentence-transformers | Modular, many model options |
| LLM | LangChain | Abstraction over llama.cpp/Ollama/OpenAI |
| MCP Server | MCP Python SDK (FastMCP) | Standard protocol |
| CLI | Click | Python-native, clean CLI |

---

## Project Structure

```
open_notes/
├── pyproject.toml
├── SPEC.md
├── README.md
├── config/
│   └── default.yaml
├── open_notes/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── file_system.py
│   │   ├── vector_db.py
│   │   └── keyword_index.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── transformers.py
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   └── watcher.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── ranker.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llama_cpp.py
│   │   ├── ollama.py
│   │   └── openai.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py
│   └── cli/
│       ├── __init__.py
│       └── main.py
└── tests/
    ├── __init__.py
    ├── test_indexer.py
    ├── test_query.py
    └── test_rag.py
```

---

## Configuration

### Default Configuration (`config/default.yaml`)

```yaml
knowledge_base:
  path: "~/knowledge-base"
  extensions: [".md"]
  auto_index: true
  
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  device: "cpu"
  
vector_db:
  type: "mindb"
  path: "~/.open_notes/vectors"
  preliminary_top_k: 500
  
keyword_index:
  type: "sqlite_fts"
  path: "~/.open_notes/keywords.db"
  
llm:
  provider: "llama_cpp"
  model_path: "~/models/qwen2.5-0.5b-q4_k_m.gguf"
  temperature: 0.7
  max_tokens: 2048
  
chunker:
  strategy: "by_heading"
  max_chars_per_chunk: 1000
  overlap_chars: 100
  
search:
  top_k: 5
  vector_weight: 0.7
  keyword_weight: 0.3
  mode: "hybrid"
  
mcp:
  host: "127.0.0.1"
  port: 8765
  transport: "stdio"
  
rag:
  prompt_template: |
    You are a helpful assistant. Use the following context to answer the question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
```

---

## Data Models

### Note
Represents a markdown file in the knowledge base.

```python
@dataclass
class Note:
    id: str                           # UUID or file hash
    path: Path                        # Absolute file path
    title: str                        # From frontmatter or first heading
    frontmatter: dict                 # YAML frontmatter
    content: str                      # Raw markdown content
    created_at: datetime
    updated_at: datetime
```

### Chunk
Represents a searchable segment of a note.

```python
@dataclass
class Chunk:
    id: str                           # UUID
    note_id: str                      # Reference to Note
    content: str                      # Text content
    heading_path: str                 # e.g., "Introduction/Background"
    position: int                     # Order in document
    char_count: int                   # For debugging
```

### SearchResult
Returned by query engine.

```python
@dataclass
class SearchResult:
    chunk_id: str
    note_id: str
    note_path: Path
    content: str
    heading_path: str
    score: float
    source: str                       # "vector", "keyword", or "hybrid"
```

---

## Core Components

### 1. Configuration (`config.py`)

- Load YAML config with environment variable overrides
- Support `~/.open_notes/config.yaml` and `./config.yaml`
- Validate required fields on load

### 2. Storage Layer

#### File System (`storage/file_system.py`)

- **scan_notes(path)**: Recursively find all .md files
- **read_note(path)**: Parse markdown with frontmatter
- **write_note(path, note)**: Write note to file
- **watch_notes(path, callback)**: File system watcher

#### Vector DB (`storage/vector_db.py`)

- **add_chunks(chunks, embeddings)**: Add embeddings to minDB
- **search(query_embedding, top_k)**: Search vectors
- **delete_note(note_id)**: Remove embeddings
- **get_stats()**: Return index statistics

#### Keyword Index (`storage/keyword_index.py`)

- **index_chunks(chunks)**: Add to SQLite FTS5
- **search(query, top_k)**: BM25 search
- **delete_note(note_id)**: Remove from index
- **rebuild()**: Rebuild entire index

### 3. Embedding Layer

#### Base Interface (`embedding/base.py`)

```python
class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
```

#### Implementation (`embedding/transformers.py`)

- Use `sentence-transformers` library
- Cache model in memory after first load
- Configurable device (cpu/cuda)

### 4. Indexer

#### Parser (`indexer/parser.py`)

- Extract YAML frontmatter
- Extract headings (h1-h6) with levels
- Preserve document structure

#### Chunker (`indexer/chunker.py`)

- **Strategy: by_heading**
  1. Split by headings first
  2. If chunk > max_chars, split further by paragraphs
  3. Preserve heading path metadata

- Configurable max_chars_per_chunk (default 1000)

#### Watcher (`indexer/watcher.py`)

- Use `watchdog` library
- Debounce: 500ms delay before indexing
- Events: created, modified, deleted, moved

### 5. Query Engine

#### Engine (`query/engine.py`)

- **search(query, mode, top_k)**: Main entry point
  - `mode="vector"`: Vector search only
  - `mode="keyword"`: SQLite FTS5 only  
  - `mode="hybrid"`: Both, fused

- Returns `list[SearchResult]` with scores

#### Ranker (`query/ranker.py`)

- **Reciprocal Rank Fusion (RRF)**:
  ```
  RRF(d) = Σ(1 / (k + rank(d)))
  k = 60 (constant)
  ```

- **Score normalization**: Min-max normalize scores to [0, 1]

### 6. LLM Abstraction

#### Base Interface (`llm/base.py`)

```python
@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

#### Implementations

- **LlamaCppLLM**: Uses `langchain.llms.llama_cpp`
- **OllamaLLM**: Uses `langchain.llms.ollama`
- **OpenAILLM**: Uses `langchain.llms.openai` (optional)

### 7. RAG Pipeline

```
Query → Hybrid Search → Get top-k chunks → Build context → Prompt → LLM → Response
```

- **rag_query(query, top_k, include_sources)**: Main method
- Returns response + optionally sources

### 8. MCP Server

#### Tools Exposed

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_notes` | Semantic search | `query: str`, `top_k: int` |
| `get_note` | Get full note | `note_id: str` |
| `list_notes` | List all notes | `limit: int`, `offset: int` |
| `rag_query` | RAG query | `query: str`, `top_k: int` |

#### Resources

| Resource | Description |
|----------|-------------|
| `notes://list` | List all notes |
| `note://{note_id}` | Individual note content |

#### Server Setup

- Use `FastMCP` from `mcp.server.fastmcp`
- Transport: stdio (default) or streamable-http

### 9. CLI

```
open_notes [OPTIONS] COMMAND [ARGS]...

Commands:
  index          Index all notes in knowledge base
  search         Search notes
  query          RAG query
  serve          Start MCP server
  watch          Watch mode (auto-index on changes)
  config         Show configuration
```

---

## Implementation Order

### Phase 1: Foundation
1. Config loading
2. Data models
3. File system operations

### Phase 2: Indexing
4. Embedding integration
5. Vector DB (minDB wrapper)
6. Keyword index (SQLite FTS5)
7. Chunker
8. Indexer with watcher

### Phase 3: Query
9. Query engine (hybrid)
10. RAG pipeline

### Phase 4: Integration
11. LLM abstractions
12. MCP server
13. CLI

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPEN_NOTES_CONFIG` | Config file path | `~/.open_notes/config.yaml` |
| `OPEN_NOTES_KB_PATH` | Knowledge base path | From config |
| `OPEN_NOTES_EMBEDDING_MODEL` | Override embedding model | From config |
| `OPEN_NOTES_LLM_MODEL_PATH` | Override LLM model path | From config |

---

## Error Handling

- **Indexing errors**: Log and continue, don't stop on single file failure
- **Search no results**: Return empty list, not error
- **LLM unavailable**: Raise clear error with setup instructions
- **File permissions**: Handle gracefully with user-friendly messages

---

## Future Considerations (Out of Scope)

- Note linking/suggestions
- Web UI
- Multi-user support
- Encryption at rest
- Note templates
- Tags/categories system
- Export formats
