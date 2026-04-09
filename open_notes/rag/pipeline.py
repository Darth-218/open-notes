from __future__ import annotations

from dataclasses import dataclass

from open_notes.llm.base import BaseLLM
from open_notes.models import ChatMessage, SearchResult
from open_notes.query.engine import QueryEngine


@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(
        self,
        query_engine: QueryEngine,
        llm: BaseLLM,
        prompt_template: str,
    ):
        self.query_engine = query_engine
        self.llm = llm
        self.prompt_template = prompt_template

    def query(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> RAGResponse:
        results = self.query_engine.search(
            query=query,
            mode="hybrid",
            top_k=top_k,
        )

        if not results:
            return RAGResponse(
                answer="No relevant context found in the knowledge base.",
                sources=[],
            )

        context = self._build_context(results)
        prompt = self.prompt_template.format(context=context, question=query)

        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            sources=results if include_sources else [],
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        context_parts = []
        for i, result in enumerate(results, 1):
            heading = result.heading_path or "Note"
            context_parts.append(
                f"[{i}] {heading}:\n{result.content[:500]}"
            )

        return "\n\n".join(context_parts)
