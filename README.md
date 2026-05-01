# open-notes

100% local, offline, open-source AI knowledge base with MCP server for AI agent integration.

---

## Features

- **Semantic Search**: Vector-based similarity search using sentence-transformers embeddings
- **Hybrid Search**: Combines vector search with keyword search (SQLite FTS5)
- **RAG Pipeline**: Retrieve relevant context and generate answers with local LLMs
- **MCP Server**: Exposes tools for AI agents (Claude, GPT, etc.)
- **File Watcher**: Auto-indexes notes on file changes
- **100% Offline**: No external API calls required
- **Open Source**: MIT licensed
- **Modular LLM**: Pluggable LLM backends (llama.cpp, Ollama, OpenAI)

---

## Quick Start

### Prerequisites
- [Nix](https://nixos.org/) package manager
- (Optional) GGUF model file for LLM queries

### First Run
```bash
# Enter development shell
cd /home/darth/aaru/Projects/open_notes
nix develop

# Create your knowledge base directory
mkdir -p ~/knowledge-base

# Add some markdown notes to ~/knowledge-base

# Index your notes
open-notes index

# Search notes
open-notes search "machine learning"

# RAG query (requires LLM model)
open-notes query "what is machine learning?"

# Start MCP server
open-notes serve
```

---

## CLI Commands

### `open-notes index`
Index all notes in the knowledge base.
```bash
open-notes index
# Output: {"indexed": 5, "chunks": 23, "message": "Indexed 5 notes with 23 chunks"}
```

### `open-notes search <query>`
Search notes using semantic similarity.
```bash
# Basic search
open-notes search "python programming"

# Limit results
open-notes search "machine learning" --top-k 10

# Options:
#   -k, --top-k    Number of results to return (default: 5)
```

### `open-notes query <query>`
Query notes using RAG (Retrieval-Augmented Generation) with an LLM.
```bash
# Basic query (requires LLM model configured)
open-notes query "What is Python used for?"

# Limit context chunks
open-notes query "explain decorators" --top-k 3

# Options:
#   -k, --top-k    Number of context chunks to retrieve (default: 5)
```

### `open-notes serve`
Start the MCP server for AI agent integration.
```bash
# Start with stdio transport (default)
open-notes serve

# The server exposes these tools:
#   - search_notes(query, top_k)     - Semantic search
#   - get_note(note_id)          - Get full note content
#   - list_notes(limit, offset)  - List all notes
#   - rag_query(query, top_k)      - RAG query with context
```

### `open-notes watch`
Watch for file changes and auto-index notes.
```bash
open-notes watch
# Output: Watching /home/user/knowledge-base for changes... (Ctrl+C to stop)
# When you create/modify a .md file, it gets automatically indexed
```

### `open-notes config`
Display the current configuration.
```bash
open-notes config
# Output:
#   knowledge_base_path: /home/user/knowledge-base
#   embedding_model: sentence-transformers/all-MiniLM-L6-v2
#   embedding_dimension: 384
#   vector_db_path: /home/user/.open_notes/vectors
#   keyword_db_path: /home/user/.open_notes/keywords.db
#   llm_provider: llama_cpp
#   llm_model_path: 
#   search_mode: hybrid
#   search_top_k: 5
#   mcp_host: 127.0.0.1
#   mcp_port: 8765
```

### `open-notes stats`
Display statistics about the indexed notes.
```bash
open-notes stats
# Output:
# {
#   "vector_db": {"total_vectors": 23, "dimension": 384},
#   "keyword_index": {"total_chunks": 23}
# }
```

---

## Configuration

### Config File
Create a config file at `~/.open_notes/config.yaml`:

```yaml
# Knowledge Base
knowledge_base:
  path: "~/knowledge-base"

# Embedding Model
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"           # "cpu" or "cuda"
  batch_size: 32

# Vector Database
vector_db:
  path: "~/.open_notes/vectors"
  preliminary_top_k: 500

# Keyword Index
keyword_index:
  path: "~/.open_notes/keywords.db"

# Chunker
chunker:
  max_chars_per_chunk: 1000
  overlap_chars: 100

# Search
search:
  top_k: 5
  mode: "hybrid"          # "vector", "keyword", or "hybrid"
  vector_weight: 0.7
  keyword_weight: 0.3

# LLM Provider
llm:
  provider: "llama_cpp"     # "llama_cpp", "ollama", or "openai"
  model_path: "~/models/qwen2.5-0.5b-q4_k_m.gguf"
  temperature: 0.7
  max_tokens: 2048

# MCP Server
mcp:
  host: "127.0.0.1"
  port: 8765
  transport: "stdio"         # "stdio" or "streamable-http"

# RAG Pipeline
rag:
  prompt_template: |
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:
```

### Environment Variables
Override configuration with environment variables:

| Variable | Description |
|----------|-------------|
| `OPEN_NOTES_CONFIG` | Custom config file path |
| `OPEN_NOTES_KB_PATH` | Override knowledge base path |
| `OPEN_NOTES_EMBEDDING_MODEL` | Override embedding model |
| `OPEN_NOTES_LLM_MODEL_PATH` | Override LLM model path |

Example:
```bash
export OPEN_NOTES_KB_PATH=~/my-notes
export OPEN_NOTES_LLM_MODEL_PATH=~/models/mistral-7b.gguf
open-notes index
```

---

## Python API

Use open-notes programmatically:

```python
from open_notes import OpenNotes
import os

# Initialize with default config
on = OpenNotes()

# Or with custom config
from open_notes.config import Config
config = Config.load("/path/to/config.yaml")
on = OpenNotes(config)

# Index all notes
result = on.index_all()
print(f"Indexed {result['indexed']} notes with {result['chunks']} chunks")

# Search notes
results = on.search("python programming", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.heading_path}")
    print(f"   {r.content[:100]}...")

# RAG query (requires LLM)
result = on.query("What is Python?", top_k=3)
print("Answer:", result["answer"])
print(f"Sources: {len(result['sources'])} notes")

# Get statistics
stats = on.get_stats()
print(stats)

# Watch for changes (runs until Ctrl+C)
# on.watch()
```

---

## LLM Providers

### llama.cpp (Local GGUF Models)
```yaml
llm:
  provider: "llama_cpp"
  model_path: "~/models/qwen2.5-0.5b-q4_k_m.gguf"
  temperature: 0.7
  max_tokens: 2048
```

### Ollama (Local Server)
```yaml
llm:
  provider: "ollama"
  model_path: "llama3.2"  # Ollama model name
  temperature: 0.7
```

### OpenAI API (or Compatible)
```yaml
llm:
  provider: "openai"
  model_path: "gpt-3.5-turbo"  # Model name
  temperature: 0.7
  # Set OPENAI_API_KEY environment variable
```

---

## MCP Server for AI Agents

Start the MCP server to let AI agents query your knowledge base:

```bash
open-notes serve
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search_notes` | Semantic search across all notes |
| `get_note` | Retrieve full content of a specific note |
| `list_notes` | List all notes in the knowledge base |
| `rag_query` | RAG query with LLM-generated answer |

### MCP Clients
- Claude Desktop (via stdio transport)
- Custom MCP clients
- Any tool supporting the Model Context Protocol

---

## Search Modes

### Vector Search
Semantic similarity using embeddings:
```bash
open-notes search "python" --mode vector
```

### Keyword Search
Full-text search using SQLite FTS5:
```bash
open-notes search "python" --mode keyword
```

### Hybrid Search (Default)
Combines both with Reciprocal Rank Fusion:
```bash
open-notes search "python" --mode hybrid
```

---

## Project Structure

```
open-notes/
├── open_notes/              # Main package
│   ├── __init__.py         # OpenNotes main class
│   ├── config.py           # Configuration loader
│   ├── models.py           # Data models
│   ├── storage/            # Storage layer
│   │   ├── file_system.py  # Note reading/writing
│   │   ├── vector_db.py    # FAISS vector database
│   │   └── keyword_index.py # SQLite FTS5 index
│   ├── embedding/          # Embedding layer
│   │   └── transformers.py # Sentence transformers
│   ├── indexer/           # Indexing
│   │   ├── parser.py      # Markdown parser
│   │   ├── chunker.py     # Content chunking
│   │   └── watcher.py     # File watcher
│   ├── query/             # Query engine
│   │   └── engine.py      # Hybrid search
│   ├── llm/               # LLM providers
│   │   └── llama_cpp.py   # llama.cpp/Ollama/OpenAI
│   ├── rag/               # RAG pipeline
│   │   └── pipeline.py    # Retrieval + Generation
│   ├── mcp/               # MCP server
│   │   └── server.py      # FastMCP server
│   └── cli/               # CLI
│       └── main.py         # Click commands
├── config/                 # Configuration
│   └── default.yaml
├── tests/                  # Test suite
├── docs/                   # Documentation
├── flake.nix              # Nix development environment
├── pyproject.toml         # Python project config
└── README.md
```

---

## Running Tests

```bash
# Enter development shell
nix develop

# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_config.py -v

# Run with coverage
python -m pytest tests/ --cov=open_notes
```

---

## Requirements

- Python 3.10+
- [Nix](https://nixos.org/) (for development environment)
- ~500MB disk space (for embedding model)

---

## License

MIT

---

## Contributing

Pull requests are welcome! Please ensure tests pass and add new tests for new features.
