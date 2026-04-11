"""LLM provider implementations."""

from __future__ import annotations

from pathlib import Path

from open_notes.llm.base import BaseLLM


class LlamaCppLLM(BaseLLM):
    """LLM implementation using llama.cpp library.

    Uses local GGUF models loaded via llama-cpp-python.

    Attributes:
        model_path: Path to the GGUF model file.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        n_ctx: Context window size in tokens.
        _llm: Lazily initialized llama-cpp model instance.

    Example:
        >>> llm = LlamaCppLLM("/path/to/model.gguf", temperature=0.8)
        >>> response = llm.generate("Hello, world!")
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        n_ctx: int = 2048,
    ):
        """Initialize LlamaCppLLM.

        Args:
            model_path: Path to the GGUF model file.
            temperature: Sampling temperature (default 0.7).
            max_tokens: Maximum tokens to generate (default 2048).
            n_ctx: Context window size (default 2048).
        """
        self.model_path = Path(model_path).expanduser()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self._llm = None

    @property
    def llm(self):
        """Lazily initialize and return the llama-cpp model.

        Returns:
            Llama model instance.

        Raises:
            ImportError: If llama-cpp is not installed.
        """
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self.model_path),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        return self._llm

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional llama-cpp parameters.

        Returns:
            Generated text as a string.
        """
        result = self.llm(prompt, **kwargs)
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("text", "")
        return str(result)

    def chat(self, messages: list, **kwargs) -> str:
        """Generate a response from chat messages.

        Args:
            messages: List of ChatMessage objects.
            **kwargs: Additional llama-cpp parameters.

        Returns:
            Generated response as a string.
        """
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(prompt, **kwargs)

    @property
    def name(self) -> str:
        """Return the name of the LLM.

        Returns:
            String in format "llama-cpp:model_name".
        """
        return f"llama-cpp:{self.model_path.name}"


class OllamaLLM(BaseLLM):
    """LLM implementation using Ollama API.

    Connects to a local or remote Ollama server.

    Attributes:
        model: Ollama model name to use.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        _client: Lazily initialized Ollama client.
    """

    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """Initialize OllamaLLM.

        Args:
            model: Ollama model name (default "llama2").
            temperature: Sampling temperature (default 0.7).
            max_tokens: Maximum tokens to generate (default 2048).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazily initialize and return the Ollama client.

        Returns:
            Ollama client module.

        Raises:
            ImportError: If ollama is not installed.
        """
        if self._client is None:
            import ollama

            self._client = ollama
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional Ollama parameters.

        Returns:
            Generated text as a string.
        """
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            options={"num_predict": self.max_tokens},
        )
        return response.get("response", "")

    def chat(self, messages: list, **kwargs) -> str:
        """Generate a response from chat messages.

        Args:
            messages: List of ChatMessage objects.
            **kwargs: Additional Ollama parameters.

        Returns:
            Generated response as a string.
        """
        response = self.client.chat(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return response.get("message", {}).get("response", "")

    @property
    def name(self) -> str:
        """Return the name of the LLM.

        Returns:
            String in format "ollama:model_name".
        """
        return f"ollama:{self.model}"


class OpenAILLM(BaseLLM):
    """LLM implementation using OpenAI API.

    Supports OpenAI models and compatible API endpoints.

    Attributes:
        model: Model name to use.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        api_key: OpenAI API key (reads from OPENAI_API_KEY env if None).
        base_url: Base URL for API endpoint.
        _client: Lazily initialized OpenAI client.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize OpenAILLM.

        Args:
            model: Model name (default "gpt-3.5-turbo").
            temperature: Sampling temperature (default 0.7).
            max_tokens: Maximum tokens to generate (default 2048).
            api_key: OpenAI API key (default None, reads from environment).
            base_url: Custom API base URL (default None).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        """Lazily initialize and return the OpenAI client.

        Returns:
            OpenAI client instance.

        Raises:
            ImportError: If openai is not installed.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.openai.com/v1",
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using completions API.

        Args:
            prompt: The input prompt text.
            **kwargs: Additional OpenAI parameters.

        Returns:
            Generated text as a string.
        """
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].text

    def chat(self, messages: list, **kwargs) -> str:
        """Generate a response from chat messages using chat completions API.

        Args:
            messages: List of ChatMessage objects.
            **kwargs: Additional OpenAI parameters.

        Returns:
            Generated response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        """Return the name of the LLM.

        Returns:
            String in format "openai:model_name".
        """
        return f"openai:{self.model}"


def create_llm(provider: str, **kwargs) -> BaseLLM:
    """Factory function to create LLM instances.

    Args:
        provider: Provider name ("llama_cpp", "ollama", or "openai").
            Case-insensitive.
        **kwargs: Provider-specific configuration arguments.

    Returns:
        BaseLLM instance configured for the specified provider.

    Raises:
        ValueError: If provider is not recognized.

    Example:
        >>> llm = create_llm("ollama", model="llama2", temperature=0.5)
        >>> llm = create_llm("openai", model="gpt-4", api_key="sk-...")
    """
    providers = {
        "llama_cpp": LlamaCppLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
    }

    llm_class = providers.get(provider.lower())
    if llm_class is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return llm_class(**kwargs)