# open-notes Wiki

Welcome to the open-notes documentation. This wiki contains everything you need to get started with and use open-notes effectively.

## Quick Links

- [Getting Started Guide](guides/getting-started.md) - Set up open-notes and index your first notes
- [Configuration Reference](code/config.md) - Detailed configuration options
- [Data Models](code/models.md) - API documentation for data structures

## What is open-notes?

open-notes is a local-first RAG (Retrieval-Augmented Generation) system designed for personal knowledge bases. It allows you to:

- **Search** your markdown notes using semantic vector search and keyword search
- **Query** your knowledge base with an LLM to get AI-powered answers
- **Watch** your note directory for changes and automatically keep your index up to date

## Project Structure

```
open-notes/
├── config/              # Default configuration files
├── open_notes/         # Main package
│   ├── config.py       # Configuration management
│   ├── models.py       # Data models
│   ├── indexer/        # Note indexing and chunking
│   ├── storage/        # Vector DB, keyword index, file storage
│   ├── embedding/      # Embedding models
│   ├── llm/            # LLM interfaces
│   ├── query/          # Query engine
│   └── rag/            # RAG pipeline
└── docs/               # Documentation
    ├── code/           # API documentation
    └── guides/         # User guides