# Configuration Module

The `Config` class manages all configuration for open-notes, loading from YAML files with environment variable overrides.

## Config Class

```python
from open_notes.config import Config
```

### Loading Configuration

```python
# Load default configuration
config = Config.load()

# Load from specific path
config = Config.load("/path/to/config.yaml")
```

Configuration is loaded from multiple sources in order of precedence:
1. Default config (`config/default.yaml` in package)
2. User config (`~/.open_notes/config.yaml`)
3. Environment variables (`OPEN_NOTES_*`)

### Getting Values

```python
# Using dot notation
model = config.get("embedding.model")

# Using properties
kb_path = config.knowledge_base_path
```

## Configuration Properties

### Knowledge Base

| Property | Type | Description |
|----------|------|-------------|
| `knowledge_base_path` | `Path` | Path to your notes directory |

### Embedding

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `embedding_model` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Model name for sentence-transformers |
| `embedding_dimension` | `int` | `384` | Embedding vector dimension |
| `embedding_device` | `str` | `"cpu"` | Device for computation (`cpu`, `cuda`) |
| `embedding_batch_size` | `int` | `32` | Batch size for embedding |

### Chunker

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `chunker_max_chars` | `int` | `1000` | Maximum characters per chunk |
| `chunker_overlap` | `int` | `100` | Character overlap between chunks |

### Search

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `search_top_k` | `int` | `5` | Default number of results |
| `search_mode` | `str` | `"hybrid"` | Search mode: `vector`, `keyword`, or `hybrid` |
| `search_vector_weight` | `float` | `0.7` | Weight for vector search in hybrid mode |
| `search_keyword_weight` | `float` | `0.3` | Weight for keyword search in hybrid mode |

### LLM

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `llm_provider` | `str` | `"llama_cpp"` | LLM provider: `llama_cpp`, `ollama`, `openai` |
| `llm_model_path` | `str` | `""` | Path to GGUF model file |
| `llm_temperature` | `float` | `0.7` | Temperature for generation |
| `llm_max_tokens` | `int` | `2048` | Maximum tokens to generate |

### MCP Server

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `mcp_host` | `str` | `"127.0.0.1"` | MCP server host |
| `mcp_port` | `int` | `8765` | MCP server port |
| `mcp_transport` | `str` | `"stdio"` | Transport: `stdio` or `streamable-http` |

### RAG

| Property | Type | Description |
|----------|------|-------------|
| `rag_prompt_template` | `str` | Prompt template with `{context}` and `{question}` variables |

## Environment Variables

| Variable | Config Key |
|----------|------------|
| `OPEN_NOTES_KB_PATH` | `knowledge_base.path` |
| `OPEN_NOTES_EMBEDDING_MODEL` | `embedding.model` |
| `OPEN_NOTES_LLM_MODEL_PATH` | `llm.model_path` |
| `OPEN_NOTES_CONFIG` | Config file path |

## Example Config File

```yaml
knowledge_base:
  path: ~/notes

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu

llm:
  provider: llama_cpp
  model_path: ~/models/llama-2-7b-chat.gguf

search:
  mode: hybrid
  top_k: 5
```