from __future__ import annotations

from pathlib import Path

from open_notes.llm.base import BaseLLM


class LlamaCppLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        n_ctx: int = 2048,
    ):
        self.model_path = Path(model_path).expanduser()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self._llm = None

    @property
    def llm(self):
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
        result = self.llm(prompt, **kwargs)
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("text", "")
        return str(result)

    def chat(self, messages: list, **kwargs) -> str:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(prompt, **kwargs)

    @property
    def name(self) -> str:
        return f"llama-cpp:{self.model_path.name}"


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            options={"num_predict": self.max_tokens},
        )
        return response.get("response", "")

    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return response.get("message", {}).get("response", "")

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.openai.com/v1",
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].text

    def chat(self, messages: list, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        return f"openai:{self.model}"


def create_llm(provider: str, **kwargs) -> BaseLLM:
    providers = {
        "llama_cpp": LlamaCppLLM,
        "ollama": OllamaLLM,
        "openai": OpenAILLM,
    }

    llm_class = providers.get(provider.lower())
    if llm_class is None:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return llm_class(**kwargs)
