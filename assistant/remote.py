"""Anthropic-backed calls that generate the patient-facing health response.

This is the assistant that answers the patient directly, using whatever
summarizer.pipeline.prepare_message() hands it. It does not translate,
extract emergency terms, or summarize -- those are the summarizer's job.

Callers that need a fallback should catch failures from ask() themselves
(the same way translator.py falls back from Google to a local dictionary,
and summarizer.py falls back to an unsummarized message) -- this module
does not swallow its own errors. A refusal (Claude's safety classifiers
declining to answer) is raised as its own AssistantRefusalError rather than
a generic failure, since a health app may want to handle "the model
declined" differently from "the network failed."
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EFFORT = "medium"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful health assistant for AfyaPlus Health. "
    "Provide general health guidance. Never diagnose conditions "
    "or prescribe medication. Always recommend consulting a "
    "healthcare professional for serious concerns."
)


@dataclass(frozen=True)
class AssistantResponse:
    """A single question/answer exchange with the health assistant."""

    message: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class AssistantRefusalError(RuntimeError):
    """Raised when Claude's safety classifiers decline to answer.

    Distinct from a network/API failure -- a refusal is a normal HTTP 200
    with stop_reason == "refusal", not an exception from the SDK. Callers
    may want to handle this differently (e.g. point the patient to a human)
    rather than retrying it like a transient error.
    """

    def __init__(self, category: Optional[str], explanation: Optional[str]) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"The assistant declined to answer (category={category!r}).")


class HealthAssistant:
    """Sends a prepared patient message to Anthropic and returns the reply.

    Translation and summarization both happen before a message ever reaches
    here, so this class's only job is asking the question and reporting
    the response along with its token usage.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        use_server_side_fallback: bool = True,
    ) -> None:
        """Initialize the assistant.

        Args:
            model: Anthropic model to use for responses. Defaults to
                claude-opus-5, the model that generates the assistant's
                actual response; see summarizer/pipeline.py for what runs
                before a message ever reaches here.
            max_tokens: Maximum tokens to generate per response. Claude
                Opus 5 thinks by default and max_tokens caps thinking plus
                visible text combined, so this is set well above the
                expected reply length to leave room for that.
            effort: Reasoning effort ("low"/"medium"/"high"/"xhigh"/"max").
                Short conversational guidance doesn't need Opus 5's top
                effort tiers -- "medium" is the recommended starting point
                for chat-shaped workloads.
            system_prompt: System prompt guiding the assistant's behavior.
            api_key: Anthropic API key. Defaults to the ANTHROPIC_API_KEY
                environment variable.
            client: A pre-built Anthropic-compatible client (mainly for
                tests). A real client is created lazily if omitted.
            use_server_side_fallback: When Claude Opus 5's safety
                classifiers decline a request, automatically retry it on
                Anthropic's recommended fallback model instead of raising
                AssistantRefusalError. Enabled by default.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_server_side_fallback = use_server_side_fallback
        self._client = client

    def _get_client(self):
        """Create and cache the Anthropic client."""
        if self._client is not None:
            return self._client

        if anthropic is None:
            raise RuntimeError("anthropic is not installed. Install it with: pip install anthropic")

        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def ask(self, message: str) -> AssistantResponse:
        """Send a prepared patient message and return the assistant's reply.

        Raises AssistantRefusalError if Claude's safety classifiers decline
        and no fallback recovers it, or whatever the underlying API call
        raises on a transport/API failure. Callers that need a fallback of
        their own (e.g. a local model, or a canned response) are
        responsible for catching this, the same way translator.py and
        summarizer.py catch failures from their own external calls.
        """
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": message}],
            "output_config": {"effort": self.effort},
        }

        if self.use_server_side_fallback:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        response = client.beta.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            details = response.stop_details
            raise AssistantRefusalError(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if not text:
            raise RuntimeError("Anthropic API returned an empty response.")

        usage = response.usage

        return AssistantResponse(
            message=text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
        )


def ask_health_assistant(message: str, **kwargs: Any) -> AssistantResponse:
    """Convenience wrapper for a one-line assistant call."""
    return HealthAssistant(**kwargs).ask(message)


__all__ = ["AssistantResponse", "AssistantRefusalError", "HealthAssistant", "ask_health_assistant"]
