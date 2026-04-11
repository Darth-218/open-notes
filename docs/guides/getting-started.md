# Getting Started with open-notes

This guide will help you set up open-notes and start using it with your markdown notes.

## Prerequisites

- Python 3.11+
- For LLM features: a GGUF model file (e.g., from HuggingFace)

## Installation

```bash
pip install open-notes
```

Or install from source:

```bash
git clone https://github.com/yourusername/open-notes.git
cd open-notes
pip install -e .
```

## Quick Start

### 1. Create a Configuration

Create a config file at `~/.open_notes/config.yaml`:

```yaml
knowledge_base:
  path: ~/notes

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

llm:
  provider: llama_cpp
  model_path: ~/models/llama-2-7b-chat.gguf
```

Or use environment variables:

```bash
export OPEN_NOTES_KB_PATH=~/notes
export OPEN_NOTES_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export OPEN_NOTES_LLM_MODEL_PATH=~/models/llama-2-7b-chat.gguf
```

### 2. Index Your Notes

```python
from open_notes import OpenNotes

# Initialize
on = OpenNotes()

# Index all notes
result = on.index_all()
print(result)
# {'indexed': 10, 'chunks': 150, 'message': 'Indexed 10 notes with 150 chunks'}
```

### 3. Search Your Notes

```python
# Simple search
results = on.search("Python async programming")

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"Content: {r.content[:200]}...")
    print("---")
```

### 4. Query with LLM

```python
# Ask a question using RAG
answer = on.query("What is async programming in Python?")

print(answer["answer"])
print("\nSources:")
for src in answer["sources"]:
    print(f"- {src['note_path']}")
```

### 5. Watch for Changes

Automatically re-index when notes change:

```python
on.watch()  # Press Ctrl+C to stop
```

## CLI Usage

open-notes includes a CLI for common operations:

```bash
# Index all notes
open-notes index

# Search
open-notes search "query"

# Query with LLM
open-notes query "your question"

# Start MCP server
open-notes serve
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         OpenNotes                            │
├─────────────────────────────────────────────────────────────┤
│  index_all()          search()          query()             │
│       │                   │                  │               │
│       ▼                   ▼                  ▼               │
│  ┌─────────┐        ┌──────────┐      ┌──────────┐         │
│  │ Markdown│        │  Query   │      │   RAG    │         │
│  │ Chunker │        │  Engine  │      │ Pipeline │         │
│  └────┬────┘        └────┬─────┘      └────┬─────┘         │
│       │                   │                  │               │
│       ▼                   ▼                  ▼               │
│  ┌─────────┐        ┌──────────┐      ┌──────────┐         │
│  │Embedding│◄───────►│ VectorDB │      │   LLM    │         │
│  │ Model   │        │          │      │          │         │
│  └─────────┘        └──────────┘      └──────────┘         │
│                           │                                   │
│                           ▼                                   │
│                    ┌────────────┐                              │
│                    │   Keyword  │                              │
│                    │   Index    │                              │
│                    └────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

- See [Configuration Reference](code/config.md) for all options
- See [Data Models](code/models.md) for API details
- Check out the [SPEC.md](../SPEC.md) for architectural details