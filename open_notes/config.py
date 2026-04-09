from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, config_dict: dict[str, Any]):
        self._data = config_dict
        self._knowledge_base_path = None
        self._resolved_paths: dict[str, Path] = {}

    @classmethod
    def load(cls, config_path: str | None = None) -> Config:
        config_path_str = config_path or os.environ.get("OPEN_NOTES_CONFIG", "~/.open_notes/config.yaml")
        config_path = Path(config_path_str).expanduser()

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
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _apply_env_overrides(config: dict) -> dict:
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
        if self._knowledge_base_path is None:
            path = self.get("knowledge_base.path", "~/knowledge-base")
            self._knowledge_base_path = Path(path).expanduser()
        return self._knowledge_base_path

    @property
    def vector_db_path(self) -> Path:
        return self._resolve_path("vector_db.path", "~/.open_notes/vectors")

    @property
    def keyword_db_path(self) -> Path:
        return self._resolve_path("keyword_index.path", "~/.open_notes/keywords.db")

    def _resolve_path(self, key: str, default: str) -> Path:
        if key not in self._resolved_paths:
            path = self.get(key, default)
            self._resolved_paths[key] = Path(path).expanduser()
        return self._resolved_paths[key]

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def embedding_dimension(self) -> int:
        return self.get("embedding.dimension", 384)

    @property
    def embedding_device(self) -> str:
        return self.get("embedding.device", "cpu")

    @property
    def embedding_batch_size(self) -> int:
        return self.get("embedding.batch_size", 32)

    @property
    def chunker_max_chars(self) -> int:
        return self.get("chunker.max_chars_per_chunk", 1000)

    @property
    def chunker_overlap(self) -> int:
        return self.get("chunker.overlap_chars", 100)

    @property
    def search_top_k(self) -> int:
        return self.get("search.top_k", 5)

    @property
    def search_vector_weight(self) -> float:
        return self.get("search.vector_weight", 0.7)

    @property
    def search_keyword_weight(self) -> float:
        return self.get("search.keyword_weight", 0.3)

    @property
    def search_mode(self) -> str:
        return self.get("search.mode", "hybrid")

    @property
    def llm_provider(self) -> str:
        return self.get("llm.provider", "llama_cpp")

    @property
    def llm_model_path(self) -> str:
        return self.get("llm.model_path", "")

    @property
    def llm_temperature(self) -> float:
        return self.get("llm.temperature", 0.7)

    @property
    def llm_max_tokens(self) -> int:
        return self.get("llm.max_tokens", 2048)

    @property
    def mcp_host(self) -> str:
        return self.get("mcp.host", "127.0.0.1")

    @property
    def mcp_port(self) -> int:
        return self.get("mcp.port", 8765)

    @property
    def mcp_transport(self) -> str:
        return self.get("mcp.transport", "stdio")

    @property
    def rag_prompt_template(self) -> str:
        return self.get(
            "rag.prompt_template",
            "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )

    @property
    def vector_db_preliminary_top_k(self) -> int:
        return self.get("vector_db.preliminary_top_k", 500)
