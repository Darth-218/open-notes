# open-notes

100% local, offline, open-source AI knowledge base with MCP server for AI agent integration.

## Features

- **Semantic Search**: Vector-based similarity search using sentence-transformers embeddings
- **Hybrid Search**: Combines vector search with keyword search (SQLite FTS5)
- **RAG Pipeline**: Retrieve relevant context and generate answers with local LLMs
- **MCP Server**: Exposes tools for AI agents (Claude, GPT, etc.)
- **File Watcher**: Auto-indexes notes on file changes
- **100% Offline**: No external API calls required
- **Open Source**: MIT licensed

## Quick Start

```bash
# Enter development shell
nix develop

# Or with direnv
direnv allow

# Index your knowledge base
open-notes index

# Search notes
open-notes search "machine learning"

# RAG query (requires LLM model)
open-notes query "what is machine learning?"

# Start MCP server
open-notes serve
```

## Configuration

Create a config file at `~/.open_notes/config.yaml`:

```yaml
knowledge_base:
  path: "~/knowledge-base"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

llm:
  provider: "llama_cpp"
  model_path: "~/models/qwen2.5-0.5b-q4_k_m.gguf"
```

Or use environment variables:
- `OPEN_NOTES_KB_PATH` - Knowledge base directory
- `OPEN_NOTES_EMBEDDING_MODEL` - Embedding model
- `OPEN_NOTES_LLM_MODEL_PATH` - Path to GGUF model file
- `OPEN_NOTES_CONFIG` - Config file path

## MCP Tools

When running `open-notes serve`, the following tools are available:

- `search_notes` - Semantic search across notes
- `get_note` - Get full note content
- `list_notes` - List all notes
- `rag_query` - RAG query with context

## Requirements

- Python 3.10+
- [Nix](https://nixos.org/) (for development environment)

## License

MIT
