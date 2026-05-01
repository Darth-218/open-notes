"""Tests for Config class."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from open_notes.config import Config


class TestConfigLoad:
    """Tests for Config.load() method."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = Config.load()
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.llm_provider == "llama_cpp"
        assert config.search_mode == "hybrid"

    def test_load_with_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("OPEN_NOTES_KB_PATH", "/custom/kb")
        monkeypatch.setenv("OPEN_NOTES_EMBEDDING_MODEL", "custom-model")
        
        config = Config.load()
        assert config.knowledge_base_path == Path("/custom/kb")
        assert config.embedding_model == "custom-model"

    def test_load_user_config(self, temp_dir):
        """Test loading user config file."""
        user_config = {
            "knowledge_base": {"path": str(temp_dir / "notes")},
            "embedding": {"model": "custom-embedding-model"},
            "llm": {"provider": "ollama"},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(user_config, f)
        
        config = Config.load(str(config_file))
        assert config.knowledge_base_path == temp_dir / "notes"
        assert config.embedding_model == "custom-embedding-model"
        assert config.llm_provider == "ollama"

    def test_load_user_config_overrides_default(self, temp_dir):
        """Test that user config overrides default values."""
        user_config = {
            "search": {"top_k": 10},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(user_config, f)
        
        config = Config.load(str(config_file))
        assert config.search_top_k == 10

    def test_deep_merge_config(self):
        """Test that user config deeply merges with default."""
        config = Config.load()
        
        assert config.get("embedding.model") is not None
        assert config.get("embedding.batch_size") is not None


class TestConfigProperties:
    """Tests for Config property accessors."""

    def test_knowledge_base_path_default(self):
        """Test default knowledge base path."""
        config = Config({"knowledge_base": {"path": "~/notes"}})
        assert config.knowledge_base_path == Path("~/notes").expanduser()

    def test_vector_db_path_default(self):
        """Test default vector DB path."""
        config = Config({})
        path_str = str(config.vector_db_path)
        assert ".open_notes/vectors" in path_str

    def test_keyword_db_path_default(self):
        """Test default keyword DB path."""
        config = Config({})
        path_str = str(config.keyword_db_path)
        assert ".open_notes/keywords" in path_str

    def test_embedding_model_default(self):
        """Test default embedding model."""
        config = Config({})
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dimension_default(self):
        """Test default embedding dimension."""
        config = Config({})
        assert config.embedding_dimension == 384

    def test_embedding_device_default(self):
        """Test default embedding device."""
        config = Config({})
        assert config.embedding_device == "cpu"

    def test_chunker_defaults(self):
        """Test default chunker settings."""
        config = Config({})
        assert config.chunker_max_chars == 1000
        assert config.chunker_overlap == 100

    def test_search_defaults(self):
        """Test default search settings."""
        config = Config({})
        assert config.search_top_k == 5
        assert config.search_vector_weight == 0.7
        assert config.search_keyword_weight == 0.3
        assert config.search_mode == "hybrid"

    def test_llm_defaults(self):
        """Test default LLM settings."""
        config = Config({})
        assert config.llm_provider == "llama_cpp"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 2048

    def test_mcp_defaults(self):
        """Test default MCP settings."""
        config = Config({})
        assert config.mcp_host == "127.0.0.1"
        assert config.mcp_port == 8765
        assert config.mcp_transport == "stdio"


class TestConfigGet:
    """Tests for Config.get() method."""

    def test_get_existing_key(self):
        """Test getting existing key."""
        config = Config({"embedding": {"model": "test-model"}})
        assert config.get("embedding.model") == "test-model"

    def test_get_nested_key(self):
        """Test getting deeply nested key."""
        config = Config({"a": {"b": {"c": "value"}}})
        assert config.get("a.b.c") == "value"

    def test_get_nonexistent_key(self):
        """Test getting nonexistent key returns None."""
        config = Config({})
        assert config.get("nonexistent") is None

    def test_get_with_default(self):
        """Test getting with default value."""
        config = Config({})
        assert config.get("nonexistent", "default") == "default"

    def test_get_partial_path(self):
        """Test getting partial path when full path exists."""
        config = Config({"section": {"key": "value"}})
        assert config.get("section") == {"key": "value"}


class TestConfigDeepMerge:
    """Tests for Config._deep_merge() method."""

    def test_merge_simple(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = Config._deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        """Test nested dictionary merge."""
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 3}}
        result = Config._deep_merge(base, override)
        assert result == {"a": {"b": 3, "c": 2}}

    def test_override_replaces_non_dict(self):
        """Test that non-dict values override completely."""
        base = {"a": {"b": 1}}
        override = {"a": "string"}
        result = Config._deep_merge(base, override)
        assert result == {"a": "string"}