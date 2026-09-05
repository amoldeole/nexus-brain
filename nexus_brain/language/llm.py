"""LLM backends.

``TemplateLLM`` is the offline default: deterministic, dependency-free
rendering used by tests and demos. ``OpenAICompatibleLLM`` talks to any
OpenAI-compatible chat endpoint (OpenAI, Azure, vLLM, Ollama, LM Studio...)
via httpx. Both implement ``complete(system, user) -> str``; the language
module never needs to know which one it has.

Set ``NEXUS_LLM_BASE_URL`` / ``NEXUS_LLM_API_KEY`` / ``NEXUS_LLM_MODEL`` to
enable the remote backend from the CLI/API.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 400) -> str: ...


class TemplateLLM:
    """No model - the caller must already have produced the final text."""

    name = "template-local"

    def complete(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
        return user


class OpenAICompatibleLLM:
    name = "openai-compatible"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 30.0) -> None:
        import httpx  # local import so the core package has no hard dependency

        self.base_url = (base_url or os.environ.get("NEXUS_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("NEXUS_LLM_API_KEY", "")
        self.model = model or os.environ.get("NEXUS_LLM_MODEL", "gpt-4o-mini")
        self.name = f"openai-compatible:{self.model}"
        self._client = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def llm_from_env() -> LLM:
    if os.environ.get("NEXUS_LLM_BASE_URL") or os.environ.get("NEXUS_LLM_API_KEY"):
        try:
            return OpenAICompatibleLLM()
        except Exception:  # noqa: BLE001 - fall back to offline mode
            return TemplateLLM()
    return TemplateLLM()
