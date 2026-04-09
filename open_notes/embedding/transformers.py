from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from open_notes.embedding.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
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
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension


def create_embedding(
    model_name: str, device: str = "cpu", batch_size: int = 32
) -> BaseEmbedding:
    try:
        return SentenceTransformerEmbedding(model_name, device, batch_size)
    except Exception as e:
        from open_notes.embedding.base import DummyEmbedding

        return DummyEmbedding()
