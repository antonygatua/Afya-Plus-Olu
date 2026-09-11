"""Local calls to an Ollama model for patient-facing health responses.

This is a free, offline alternative to assistant/remote.py's HealthAssistant
-- same ask(message) -> AssistantResponse interface, so callers can develop
and test against a local model without an Anthropic API key, then switch to
the real assistant by swapping which class they construct.
"""

from __future__ import annotations

from typing import Any, Optional

from assistant.remote import DEFAULT_SYSTEM_PROMPT, AssistantResponse

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 300


class LocalHealthAssistant:
    """Sends a prepared patient message to a local Ollama model.

    Same interface as HealthAssistant (assistant/remote.py) so the rest of
    the app doesn't need to know which one it's talking to. Ollama has no
    safety-classifier refusal behavior or server-side fallback to handle --
    a failed or unreachable local model just raises, the same as any other
    external-call failure elsewhere in this codebase.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the local assistant.

        Args:
            model: Ollama model name (must already be pulled locally).
            base_url: Ollama's OpenAI-compatible endpoint.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate per response.
            system_prompt: System prompt guiding the assistant's behavior.
                Shares HealthAssistant's default so both backends behave
                the same way regardless of which one answers.
            client: A pre-built OpenAI-compatible client (mainly for
                tests). A real client pointed at Ollama is created lazily
                if omitted.
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = client

    def _get_client(self):
        """Create and cache the OpenAI-compatible client for Ollama."""
        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise RuntimeError("openai is not installed. Install it with: pip install openai")

        self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        return self._client

    def ask(self, message: str) -> AssistantResponse:
        """Send a prepared patient message and return the assistant's reply.

        Raises whatever the underlying call raises -- e.g. a connection
        error if Ollama isn't running, or if the model hasn't been pulled.
        Callers are responsible for catching this, the same way they would
        for HealthAssistant.ask().
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
        )

        text = (response.choices[0].message.content or "").strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        usage = response.usage

        return AssistantResponse(
            message=text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )


def ask_local_health_assistant(message: str, **kwargs: Any) -> AssistantResponse:
    """Convenience wrapper for a one-line local assistant call."""
    return LocalHealthAssistant(**kwargs).ask(message)


__all__ = ["LocalHealthAssistant", "ask_local_health_assistant"]
