"""Sentence Transformers embedding implementation.

This module provides the SentenceTransformerEmbedding class for generating
text embeddings using the sentence-transformers library.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from open_notes.embedding.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """Embedding model using sentence-transformers.

    Generates dense embeddings for text using pre-trained transformer models.
    Supports various models from Hugging Face model hub.

    Attributes:
        model_name: Name of the sentence-transformers model.
        device: Device for computation ("cpu" or "cuda").
        batch_size: Batch size for encoding.

    Example:
        >>> emb = SentenceTransformerEmbedding(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     device="cpu"
        ... )
        >>> embeddings = emb.embed(["Hello world"])
        >>> emb.dimension
        384
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """Initialize SentenceTransformerEmbedding.

        Args:
            model_name: Model name from sentence-transformers or Hugging Face.
                Default: "sentence-transformers/all-MiniLM-L6-v2".
            device: Device for computation (default: "cpu").
            batch_size: Batch size for encoding (default: 32).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model.

        Returns:
            The SentenceTransformer model instance.
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Example:
            >>> emb = SentenceTransformerEmbedding()
            >>> embeddings = emb.embed(["First text", "Second text"])
            >>> len(embeddings)
            2
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Dimension of embedding vectors produced by this model.

        Returns:
            The embedding dimension.
        """
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension


def create_embedding(
    model_name: str, device: str = "cpu", batch_size: int = 32
) -> BaseEmbedding:
    """Create an embedding model instance.

    Attempts to create a SentenceTransformerEmbedding. Falls back to
    DummyEmbedding if model loading fails.

    Args:
        model_name: Name of the embedding model.
        device: Device for computation.
        batch_size: Batch size for encoding.

    Returns:
        BaseEmbedding instance.

    Example:
        >>> emb = create_embedding("sentence-transformers/all-MiniLM-L6-v2")
        >>> isinstance(emb, SentenceTransformerEmbedding)
        True
    """
    try:
        return SentenceTransformerEmbedding(model_name, device, batch_size)
    except Exception as e:
        from open_notes.embedding.base import DummyEmbedding

        return DummyEmbedding()
