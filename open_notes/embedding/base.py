from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class DummyEmbedding(BaseEmbedding):
    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        import random

        return [
            [random.random() for _ in range(self._dimension)]
            for _ in texts
        ]

    @property
    def dimension(self) -> int:
        return self._dimension
