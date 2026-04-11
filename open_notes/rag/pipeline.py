"""RAG (Retrieval-Augmented Generation) pipeline for question answering."""

from __future__ import annotations

from dataclasses import dataclass

from open_notes.llm.base import BaseLLM
from open_notes.models import ChatMessage, SearchResult
from open_notes.query.engine import QueryEngine


@dataclass
class RAGResponse:
    """Response from a RAG pipeline query.

    Attributes:
        answer: The generated answer text from the LLM.
        sources: List of search results used as context for the answer.
    """

    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for answering user questions.

    Combines a query engine for retrieving relevant documents with an LLM
    for generating answers based on the retrieved context.

    Attributes:
        query_engine: The query engine used for semantic and hybrid search.
        llm: The language model used for generating answers.
        prompt_template: Template for formatting the prompt sent to the LLM.
    """

    def __init__(
        self,
        query_engine: QueryEngine,
        llm: BaseLLM,
        prompt_template: str,
    ):
        """Initialize the RAG pipeline.

        Args:
            query_engine: The query engine for retrieving relevant documents.
            llm: The language model for generating answers.
            prompt_template: Template string with {context} and {question} placeholders.
        """
        self.query_engine = query_engine
        self.llm = llm
        self.prompt_template = prompt_template

    def query(
        self,
        query: str,
        top_k: int = 5,
        include_sources: bool = True,
    ) -> RAGResponse:
        """Answer a question using retrieval-augmented generation.

        Args:
            query: The user's question to answer.
            top_k: Number of top search results to retrieve (default: 5).
            include_sources: Whether to include sources in the response (default: True).

        Returns:
            RAGResponse containing the generated answer and optional sources.
        """
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
        """Build a context string from search results.

        Args:
            results: List of search results to format into context.

        Returns:
            Formatted context string with numbered sources.
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            heading = result.heading_path or "Note"
            context_parts.append(
                f"[{i}] {heading}:\n{result.content[:500]}"
            )

        return "\n\n".join(context_parts)