# Data Models

This document describes the core data structures used throughout open-notes.

## Note

Represents a note in the knowledge base.

```python
from open_notes.models import Note
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier |
| `path` | `Path` | File system path |
| `title` | `str` | Title from frontmatter or first heading |
| `frontmatter` | `dict` | YAML frontmatter metadata |
| `content` | `str` | Raw markdown content |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last modified timestamp |

### Methods

- `to_dict()` - Convert to dictionary representation

## Chunk

Represents a searchable segment of a note.

```python
from open_notes.models import Chunk
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique chunk identifier |
| `note_id` | `str` | Parent note ID |
| `content` | `str` | Text content |
| `heading_path` | `str` | Path of headings (e.g., "Introduction/Background") |
| `position` | `int` | Position in original document |
| `char_count` | `int` | Character count |

## SearchResult

Represents a search result from the query engine.

```python
from open_notes.models import SearchResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `chunk_id` | `str` | Matched chunk ID |
| `note_id` | `str` | Parent note ID |
| `note_path` | `Path` | File path to note |
| `content` | `str` | Matched content |
| `heading_path` | `str` | Heading path within note |
| `score` | `float` | Relevance score (higher is better) |
| `source` | `str` | Search source: `vector`, `keyword`, or `hybrid` |

### Methods

- `to_dict()` - Convert to dictionary representation

## ChatMessage

Represents a message in a chat conversation.

```python
from open_notes.models import ChatMessage
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `role` | `str` | Message role: `system`, `user`, or `assistant` |
| `content` | `str` | Message content |

### Methods

- `to_dict()` - Convert to dictionary representation