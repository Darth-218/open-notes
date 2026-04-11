"""Embedding interface for text vectorization.

This module defines the BaseEmbedding abstract class and DummyEmbedding
for testing without a real embedding model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract base class for text embedding models.

    All embedding implementations must inherit from this class and implement
    the embed() method and dimension property.

    Attributes:
        name: Name of the embedding model (derived from class name).

    Example:
        >>> class MyEmbedding(BaseEmbedding):
        ...     def embed(self, texts):
        ...         # Return embeddings
        ...     @property
        ...     def dimension(self):
        ...         return 384
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Example:
            >>> embeddings = embedding_model.embed(["hello world", "goodbye"])
            >>> len(embeddings)
            2
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors.

        Returns:
            Vector dimension (e.g., 384, 768, 1024).
        """
        pass

    @property
    def name(self) -> str:
        """Name of the embedding model.

        Returns:
            Class name of the embedding model.
        """
        return self.__class__.__name__


class DummyEmbedding(BaseEmbedding):
    """Dummy embedding model for testing.

    Generates random vectors for testing without a real embedding model.
    Useful for development and testing purposes.

    Attributes:
        _dimension: Dimension of generated vectors.

    Example:
        >>> dummy = DummyEmbedding(dimension=384)
        >>> embeddings = dummy.embed(["test text"])
        >>> len(embeddings[0])
        384
    """

    def __init__(self, dimension: int = 384):
        """Initialize DummyEmbedding with a dimension.

        Args:
            dimension: Dimension of random vectors to generate (default: 384).
        """
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate random embeddings.

        Args:
            texts: List of text strings (not used, random vectors generated).

        Returns:
            List of random vectors with configured dimension.
        """
        import random

        return [
            [random.random() for _ in range(self._dimension)]
            for _ in texts
        ]

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors.

        Returns:
            Configured dimension.
        """
        return self._dimension
