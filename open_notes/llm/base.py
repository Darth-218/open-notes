"""LLM base classes and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from open_notes.models import ChatMessage


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Defines the interface that all LLM implementations must follow.
    Subclasses must implement the generate, chat, and name methods.

    Attributes:
        name: The name of the LLM provider.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated text as a string.
        """
        pass

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of ChatMessage objects representing conversation history.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated response as a string.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM provider.

        Returns:
            Name string identifying the LLM.
        """
        pass


class DummyLLM(BaseLLM):
    """Dummy LLM that returns a configuration message.

    Used as a fallback when no LLM provider is configured.

    Attributes:
        _name: Internal name for the LLM.
    """

    def __init__(self):
        """Initialize DummyLLM with default name."""
        self._name = "dummy"

    def generate(self, prompt: str, **kwargs) -> str:
        """Return configuration message.

        Args:
            prompt: Ignored.
            **kwargs: Ignored.

        Returns:
            Message prompting user to configure an LLM.
        """
        return "LLM not configured. Please configure an LLM provider."

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        """Return configuration message.

        Args:
            messages: Ignored.
            **kwargs: Ignored.

        Returns:
            Message prompting user to configure an LLM.
        """
        return "LLM not configured. Please configure an LLM provider."

    @property
    def name(self) -> str:
        """Return the name of the dummy LLM.

        Returns:
            Always returns "dummy".
        """
        return self._name