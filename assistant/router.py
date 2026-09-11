"""Unified entrypoint that picks between the real and local health assistant.

assistant.remote.HealthAssistant is the real, paid Anthropic backend;
assistant.local.LocalHealthAssistant is the free, offline Ollama backend.
Assistant.ask() auto-selects between them the same way Summarizer.use_llm
auto-selects based on API key presence, and falls back from remote to local
on a transient failure -- never the other way, since silently spending
money to recover from a local outage would be the wrong default.

A refusal (AssistantRefusalError) is not treated as a transient failure and
is never silently rerouted to the unfiltered local model: Claude's safety
classifiers declining a request is a deliberate signal, not an outage, and
routing around it would defeat the point of the refusal.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from assistant.local import LocalHealthAssistant
from assistant.remote import AssistantRefusalError, AssistantResponse, HealthAssistant

logger = logging.getLogger(__name__)


class Assistant:
    """Answers a prepared patient message using the best available backend.

    Defaults to the real assistant (assistant.remote.HealthAssistant) when
    an Anthropic API key is configured, and the local one
    (assistant.local.LocalHealthAssistant) otherwise. A remote failure that
    isn't a refusal falls back to local by default; a refusal is raised as
    it is.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        fallback_to_local: bool = True,
        remote: Optional[Any] = None,
        local: Optional[Any] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the router.

        Args:
            backend: Force "remote" or "local". Defaults to "remote" when
                an Anthropic API key is available, else "local".
            fallback_to_local: When the remote backend raises anything
                other than AssistantRefusalError, retry once on the local
                backend instead of failing outright. Never falls back in
                the other direction, since a local outage shouldn't
                silently start spending money.
            remote: A HealthAssistant-compatible instance to reuse. A
                default one is created lazily if omitted.
            local: A LocalHealthAssistant-compatible instance to reuse. A
                default one is created lazily if omitted.
            api_key: Anthropic API key used only to decide the default
                backend when `backend` is omitted. Defaults to the
                ANTHROPIC_API_KEY environment variable.
        """
        self.fallback_to_local = fallback_to_local
        self._remote = remote
        self._local = local
        self._last_model: Optional[str] = None

        has_api_key = bool(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.backend = backend if backend is not None else ("remote" if has_api_key else "local")

        if self.backend not in ("remote", "local"):
            raise ValueError(f"backend must be 'remote' or 'local', got {backend!r}")

    def _get_remote(self) -> HealthAssistant:
        if self._remote is None:
            self._remote = HealthAssistant()
        return self._remote

    def _get_local(self) -> LocalHealthAssistant:
        if self._local is None:
            self._local = LocalHealthAssistant()
        return self._local

    @property
    def last_model(self) -> Optional[str]:
        """The model that actually answered the most recent ask() call.

        Needed because a remote failure can fall back to local mid-call --
        callers that need to know which model produced a response (e.g. to
        record its cost correctly) can't assume it's always the configured
        remote model.
        """
        return self._last_model

    def ask(self, message: str) -> AssistantResponse:
        """Answer a prepared patient message.

        Raises AssistantRefusalError immediately if the remote backend
        declines -- that's never rerouted to the unfiltered local model.
        Any other remote failure falls back to the local backend when
        fallback_to_local is enabled; otherwise it's raised as it is.
        """
        if self.backend == "local":
            local = self._get_local()
            response = local.ask(message)
            self._last_model = local.model
            return response

        remote = self._get_remote()
        try:
            response = remote.ask(message)
            self._last_model = remote.model
            return response
        except AssistantRefusalError:
            raise
        except Exception as exc:
            if not self.fallback_to_local:
                raise
            logger.warning("Remote assistant failed, falling back to local. Error: %s", exc)
            local = self._get_local()
            response = local.ask(message)
            self._last_model = local.model
            return response


def ask(message: str, **kwargs: Any) -> AssistantResponse:
    """Convenience wrapper for a one-line assistant call."""
    return Assistant(**kwargs).ask(message)


__all__ = ["Assistant", "ask"]
