"""Tests for the prepare -> answer -> cost-track composition, no real calls."""

import pytest

from assistant.cost import CostTracker
from assistant.handler import MessageHandler
from assistant.remote import AssistantResponse
from summarizer.pipeline import MessagePipeline
from summarizer.summarizer import Summarizer
from summarizer.translator import MedicalTranslator


class _FakeAssistant:
    def __init__(self, response, model="claude-opus-5", last_model=None):
        self._response = response
        self.model = model
        self.last_model = last_model
        self.calls = []

    def ask(self, message):
        self.calls.append(message)
        return self._response


def _local_pipeline():
    return MessagePipeline(translator=MedicalTranslator(use_google_translate=False), summarizer=Summarizer(use_llm=False))


def test_handle_wires_prepared_text_into_the_assistant():
    assistant = _FakeAssistant(response=AssistantResponse(message="reply", prompt_tokens=5, completion_tokens=5))
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant)

    result = handler.handle("Nina homa")

    assert "fever" in result.translated_text.lower()
    assert assistant.calls == [result.final_text]
    assert result.response.message == "reply"


def test_handle_without_cost_tracker_does_not_require_a_model():
    assistant = _FakeAssistant(response=AssistantResponse(message="reply", prompt_tokens=1, completion_tokens=1), model=None)
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant)

    result = handler.handle("Nina homa")

    assert result.response.message == "reply"


def test_handle_records_cost_labeled_by_was_summarized():
    tracker = CostTracker()
    assistant = _FakeAssistant(response=AssistantResponse(message="reply", prompt_tokens=1_000_000, completion_tokens=1_000_000))
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant, cost_tracker=tracker)

    result = handler.handle("Nina homa")

    assert not result.was_summarized
    summary = tracker.summary("not_summarized")
    assert summary.call_count == 1
    assert summary.total_cost == pytest.approx(30.00)  # claude-opus-5 rates


def test_handle_prefers_last_model_over_static_model_for_cost():
    tracker = CostTracker()
    # Simulates assistant.router.Assistant having fallen back to local mid-call:
    # .model stays whatever the router itself reports, but .last_model is the
    # backend that actually answered and must be what gets priced.
    assistant = _FakeAssistant(
        response=AssistantResponse(message="reply", prompt_tokens=1_000_000, completion_tokens=1_000_000),
        model="claude-opus-5",
        last_model="claude-haiku-4-5",
    )
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant, cost_tracker=tracker)

    handler.handle("Nina homa")

    summary = tracker.summary("not_summarized")
    assert summary.total_cost == pytest.approx(6.00)  # claude-haiku-4-5 rates, not opus-5's


def test_handle_raises_when_cost_tracker_set_but_model_unknown():
    tracker = CostTracker()
    assistant = _FakeAssistant(response=AssistantResponse(message="reply", prompt_tokens=1, completion_tokens=1), model=None)
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant, cost_tracker=tracker)

    with pytest.raises(ValueError):
        handler.handle("Nina homa")


def test_emergency_message_is_labeled_correctly_for_cost_tracking():
    tracker = CostTracker()
    assistant = _FakeAssistant(response=AssistantResponse(message="reply", prompt_tokens=100, completion_tokens=100))
    handler = MessageHandler(pipeline=_local_pipeline(), assistant=assistant, cost_tracker=tracker)

    result = handler.handle("Help! I have severe chest pain")

    assert result.emergency.is_emergency
    assert not result.was_summarized
    assert tracker.summary("not_summarized").call_count == 1
