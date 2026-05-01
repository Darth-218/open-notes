"""Configuration management for open-notes.

This module provides the Config class for loading and managing
configuration from YAML files and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager for open-notes.

    Loads configuration from YAML files with the following precedence:
    1. Default config (config/default.yaml in package)
    2. User config (~/.open_notes/config.yaml)
    3. Environment variables (OPEN_NOTES_*)

    Attributes:
        _data: Internal dictionary storing configuration values.
        _knowledge_base_path: Cached knowledge base path.
        _resolved_paths: Cache for resolved path configurations.

    Example:
        >>> config = Config.load()
        >>> kb_path = config.knowledge_base_path
        >>> embedding_model = config.embedding_model
    """

    def __init__(self, config_dict: dict[str, Any]):
        """Initialize Config with a dictionary.

        Args:
            config_dict: Dictionary containing configuration values.
        """
        self._data = config_dict
        self._knowledge_base_path = None
        self._resolved_paths: dict[str, Path] = {}

    @classmethod
    def load(cls, config_path: str | None = None) -> Config:
        """Load configuration from files and environment variables.

        Configuration is loaded from multiple sources in order of precedence:
        1. Default config (config/default.yaml in package)
        2. User config (default: ~/.open_notes/config.yaml)
        3. Environment variables (OPEN_NOTES_*)

        Args:
            config_path: Optional path to user config file.
                Defaults to ~/.open_notes/config.yaml or OPEN_NOTES_CONFIG env var.

        Returns:
            Config instance with merged configuration.

        Example:
            >>> config = Config.load()
            >>> config = Config.load("/path/to/config.yaml")
        """
        config_path_str = config_path or os.environ.get("OPEN_NOTES_CONFIG", "~/.open_notes/config.yaml")
        # FIX: Fix str | None warning
        config_path = Path(config_path_str).expanduser()

        # TODO: Verify
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        config_dict = {}

        if default_path.exists():
            with open(default_path) as f:
                config_dict = yaml.safe_load(f) or {}

        if config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
                config_dict = cls._deep_merge(config_dict, user_config)

        config_dict = cls._apply_env_overrides(config_dict)
        return cls(config_dict)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override dict into base dict.

        Args:
            base: Base dictionary.
            override: Override dictionary.

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _apply_env_overrides(config: dict) -> dict:
        """Apply environment variable overrides to configuration.

        Supported environment variables:
        - OPEN_NOTES_KB_PATH: Override knowledge_base.path
        - OPEN_NOTES_EMBEDDING_MODEL: Override embedding.model
        - OPEN_NOTES_LLM_MODEL_PATH: Override llm.model_path

        Args:
            config: Configuration dictionary to modify.

        Returns:
            Configuration with environment overrides applied.
        """
        env_mappings = {
            "OPEN_NOTES_KB_PATH": ("knowledge_base", "path"),
            "OPEN_NOTES_EMBEDDING_MODEL": ("embedding", "model"),
            "OPEN_NOTES_LLM_MODEL_PATH": ("llm", "model_path"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                if section not in config:
                    config[section] = {}
                config[section][key] = value

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Dot-separated key path (e.g., "embedding.model").
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Example:
            >>> config.get("embedding.model")
            'sentence-transformers/all-MiniLM-L6-v2'
            >>> config.get("nonexistent", "default")
            'default'
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def knowledge_base_path(self) -> Path:
        """Path to the knowledge base directory.

        Returns:
            Path object for the knowledge base directory.

        Example:
            >>> config.knowledge_base_path
            PosixPath('/home/user/knowledge-base')
        """
        if self._knowledge_base_path is None:
            path = self.get("knowledge_base.path", "~/knowledge-base")
            self._knowledge_base_path = Path(path).expanduser()
        return self._knowledge_base_path

    @property
    def vector_db_path(self) -> Path:
        """Path to the vector database file.

        Returns:
            Path object for the vector database.
        """
        return self._resolve_path("vector_db.path", "~/.open_notes/vectors")

    @property
    def keyword_db_path(self) -> Path:
        """Path to the keyword search database.

        Returns:
            Path object for the SQLite FTS database.
        """
        return self._resolve_path("keyword_index.path", "~/.open_notes/keywords.db")

    def _resolve_path(self, key: str, default: str) -> Path:
        """Resolve and cache a path configuration value.

        Args:
            key: Configuration key for the path.
            default: Default path if not configured.

        Returns:
            Resolved Path object.
        """
        if key not in self._resolved_paths:
            path = self.get(key, default)
            self._resolved_paths[key] = Path(path).expanduser()
        return self._resolved_paths[key]

    @property
    def embedding_model(self) -> str:
        """Embedding model name or path.

        Returns:
            Model identifier for sentence-transformers.

        Example:
            >>> config.embedding_model
            'sentence-transformers/all-MiniLM-L6-v2'
        """
        return self.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def embedding_dimension(self) -> int:
        """Dimension of embedding vectors.

        Returns:
            Embedding vector dimension (default: 384).
        """
        return self.get("embedding.dimension", 384)

    @property
    def embedding_device(self) -> str:
        """Device for embedding computation.

        Returns:
            Device name (default: "cpu").
        """
        return self.get("embedding.device", "cpu")

    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding computation.

        Returns:
            Batch size (default: 32).
        """
        return self.get("embedding.batch_size", 32)

    @property
    def chunker_max_chars(self) -> int:
        """Maximum characters per chunk when splitting notes.

        Returns:
            Maximum chunk size in characters (default: 1000).
        """
        return self.get("chunker.max_chars_per_chunk", 1000)

    @property
    def chunker_overlap(self) -> int:
        """Character overlap between consecutive chunks.

        Returns:
            Overlap size in characters (default: 100).
        """
        return self.get("chunker.overlap_chars", 100)

    @property
    def search_top_k(self) -> int:
        """Default number of results to return from search.

        Returns:
            Number of results (default: 5).
        """
        return self.get("search.top_k", 5)

    @property
    def search_vector_weight(self) -> float:
        """Weight for vector search in hybrid search mode.

        Returns:
            Weight value between 0 and 1 (default: 0.7).
        """
        return self.get("search.vector_weight", 0.7)

    @property
    def search_keyword_weight(self) -> float:
        """Weight for keyword search in hybrid search mode.

        Returns:
            Weight value between 0 and 1 (default: 0.3).
        """
        return self.get("search.keyword_weight", 0.3)

    @property
    def search_mode(self) -> str:
        """Default search mode.

        Returns:
            Search mode: "vector", "keyword", or "hybrid" (default: "hybrid").
        """
        return self.get("search.mode", "hybrid")

    @property
    def llm_provider(self) -> str:
        """LLM provider to use.

        Returns:
            Provider name: "llama_cpp", "ollama", or "openai" (default: "llama_cpp").
        """
        return self.get("llm.provider", "llama_cpp")

    @property
    def llm_model_path(self) -> str:
        """Path to LLM model file.

        Returns:
            Path to GGUF model file for llama.cpp.
        """
        return self.get("llm.model_path", "")

    @property
    def llm_temperature(self) -> float:
        """Temperature parameter for LLM generation.

        Returns:
            Temperature value (default: 0.7).
        """
        return self.get("llm.temperature", 0.7)

    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens to generate in LLM responses.

        Returns:
            Maximum token count (default: 2048).
        """
        return self.get("llm.max_tokens", 2048)

    @property
    def mcp_host(self) -> str:
        """Host for MCP server.

        Returns:
            Hostname or IP address (default: "127.0.0.1").
        """
        return self.get("mcp.host", "127.0.0.1")

    @property
    def mcp_port(self) -> int:
        """Port for MCP server.

        Returns:
            Port number (default: 8765).
        """
        return self.get("mcp.port", 8765)

    @property
    def mcp_transport(self) -> str:
        """Transport protocol for MCP server.

        Returns:
            Transport: "stdio" or "streamable-http" (default: "stdio").
        """
        return self.get("mcp.transport", "stdio")

    @property
    def rag_prompt_template(self) -> str:
        """Prompt template for RAG queries.

        Template variables:
            - {context}: Retrieved context from knowledge base
            - {question}: User's question

        Returns:
            Prompt template string.
        """
        return self.get(
            "rag.prompt_template",
            "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

    @property
    def vector_db_preliminary_top_k(self) -> int:
        """Preliminary number of results from vector search before reranking.

        Returns:
            Number of preliminary results (default: 500).
        """
        return self.get("vector_db.preliminary_top_k", 500)
