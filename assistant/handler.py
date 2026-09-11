"""End-to-end request handling: prepare a raw message, answer it, track cost.

This is the composition summarizer.pipeline.MessagePipeline and
assistant.router.Assistant were each built to feed into and be fed by:
given a raw patient message, it translates/detects/summarizes it, asks the
assistant, and -- when a CostTracker is supplied -- records what that
answer actually cost, tagged by whether summarization ran. That's what
turns "does summarizing before answering save money?" from an assumption
into a number: compare cost_tracker.summary("summarized") against
cost_tracker.summary("not_summarized").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from assistant.cost import CostTracker
from assistant.remote import AssistantResponse
from assistant.router import Assistant
from summarizer.emergency_terms import EmergencyExtractionResult
from summarizer.pipeline import MessagePipeline


@dataclass(frozen=True)
class HandledMessage:
    """Everything about one end-to-end request/response cycle."""

    original_text: str
    translated_text: str
    final_text: str
    emergency: EmergencyExtractionResult
    was_summarized: bool
    response: AssistantResponse


class MessageHandler:
    """Wires prepare -> answer -> (optionally) track cost into one call.

    summarizer.pipeline.MessagePipeline and assistant.router.Assistant
    remain independently usable and testable; this class just composes
    them, the same way summarizer/pipeline.py composes translator,
    emergency_terms, and summarizer.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        assistant: Optional[Any] = None,
        cost_tracker: Optional[CostTracker] = None,
        cost_model: Optional[str] = None,
    ) -> None:
        """Initialize the handler.

        Args:
            pipeline: A MessagePipeline-compatible instance. A default one
                is created if omitted.
            assistant: An Assistant-compatible instance (anything with a
                matching .ask()). A default one is created if omitted.
            cost_tracker: When given, every call records its actual cost
                here, labeled "summarized" or "not_summarized" from
                was_summarized.
            cost_model: The model to record cost against. Only relevant
                when cost_tracker is given. Defaults to the assistant's
                `last_model` (set after each call by assistant.router.
                Assistant, so it reflects a remote/local fallback
                correctly) or, failing that, its `model` attribute.
        """
        self.pipeline = pipeline or MessagePipeline()
        self.assistant = assistant or Assistant()
        self.cost_tracker = cost_tracker
        self.cost_model = cost_model

    def handle(self, raw_text: str) -> HandledMessage:
        """Prepare, answer, and (optionally) cost-track one patient message."""
        prepared = self.pipeline.prepare(raw_text)
        response = self.assistant.ask(prepared.final_text)

        if self.cost_tracker is not None:
            model = (
                self.cost_model
                or getattr(self.assistant, "last_model", None)
                or getattr(self.assistant, "model", None)
            )
            if model is None:
                raise ValueError("cost_tracker is set but no model is known -- pass cost_model explicitly.")

            label = "summarized" if prepared.was_summarized else "not_summarized"
            self.cost_tracker.record(response, model=model, label=label)

        return HandledMessage(
            original_text=prepared.original_text,
            translated_text=prepared.translated_text,
            final_text=prepared.final_text,
            emergency=prepared.emergency,
            was_summarized=prepared.was_summarized,
            response=response,
        )


def handle_message(raw_text: str, **kwargs: Any) -> HandledMessage:
    """Convenience wrapper for a one-line end-to-end call."""
    return MessageHandler(**kwargs).handle(raw_text)


__all__ = ["HandledMessage", "MessageHandler", "handle_message"]
