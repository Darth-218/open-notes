from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from open_notes.models import ChatMessage


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


class DummyLLM(BaseLLM):
    def __init__(self):
        self._name = "dummy"

    def generate(self, prompt: str, **kwargs) -> str:
        return "LLM not configured. Please configure an LLM provider."

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        return "LLM not configured. Please configure an LLM provider."

    @property
    def name(self) -> str:
        return self._name
