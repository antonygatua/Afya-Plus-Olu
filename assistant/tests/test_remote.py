"""Tests for the Anthropic-backed health assistant, without any real API calls."""

import pytest

from assistant.remote import AssistantRefusalError, HealthAssistant


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeStopDetails:
    def __init__(self, category, explanation):
        self.category = category
        self.explanation = explanation


class _FakeResponse:
    def __init__(self, text=None, input_tokens=10, output_tokens=5, stop_reason="end_turn", stop_details=None):
        self.content = [_FakeTextBlock(text)] if text is not None else []
        self.usage = _FakeUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class _FakeMessages:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception:
            raise self._exception
        return self._response


class _FakeBeta:
    def __init__(self, messages):
        self.messages = messages


class _FakeClient:
    def __init__(self, response=None, exception=None):
        self.messages = _FakeMessages(response=response, exception=exception)
        self.beta = _FakeBeta(self.messages)


def test_ask_returns_message_and_token_usage():
    response = _FakeResponse(text="Rest, hydrate, and see a doctor if it persists.", input_tokens=42, output_tokens=13)
    client = _FakeClient(response=response)
    assistant = HealthAssistant(client=client)

    result = assistant.ask("I have had a headache for three days.")

    assert result.message == "Rest, hydrate, and see a doctor if it persists."
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 13
    assert result.total_tokens == 55


def test_ask_sends_server_side_fallback_by_default():
    client = _FakeClient(response=_FakeResponse(text="ok"))
    assistant = HealthAssistant(client=client)

    assistant.ask("What are some tips for better sleep?")

    call = client.messages.calls[0]
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert call["model"] == "claude-opus-5"
    assert call["output_config"] == {"effort": "medium"}


def test_ask_can_disable_server_side_fallback():
    client = _FakeClient(response=_FakeResponse(text="ok"))
    assistant = HealthAssistant(client=client, use_server_side_fallback=False)

    assistant.ask("What are some tips for better sleep?")

    call = client.messages.calls[0]
    assert "betas" not in call
    assert "fallbacks" not in call


def test_ask_raises_refusal_error_on_declined_request():
    response = _FakeResponse(
        stop_reason="refusal",
        stop_details=_FakeStopDetails(category="cyber", explanation="policy decline"),
    )
    client = _FakeClient(response=response)
    assistant = HealthAssistant(client=client)

    with pytest.raises(AssistantRefusalError) as exc_info:
        assistant.ask("Some declined request")

    assert exc_info.value.category == "cyber"
    assert exc_info.value.explanation == "policy decline"


def test_ask_raises_on_empty_response():
    client = _FakeClient(response=_FakeResponse(text=""))
    assistant = HealthAssistant(client=client)

    with pytest.raises(RuntimeError):
        assistant.ask("Anything")


def test_ask_propagates_api_errors():
    client = _FakeClient(exception=ConnectionError("network down"))
    assistant = HealthAssistant(client=client)

    with pytest.raises(ConnectionError):
        assistant.ask("Anything")
