"""Tests for the Ollama-backed local health assistant, without a real Ollama server."""

import pytest

from assistant.local import LocalHealthAssistant


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content=None, prompt_tokens=8, completion_tokens=4):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletions:
    def __init__(self, response=None, exception=None):
        self._response = response
        self._exception = exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception:
            raise self._exception
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, response=None, exception=None):
        self.completions = _FakeCompletions(response=response, exception=exception)
        self.chat = _FakeChat(self.completions)


def test_ask_returns_message_and_token_usage():
    response = _FakeResponse(content="Rest and stay hydrated.", prompt_tokens=20, completion_tokens=6)
    client = _FakeClient(response=response)
    assistant = LocalHealthAssistant(client=client)

    result = assistant.ask("I have a mild cough.")

    assert result.message == "Rest and stay hydrated."
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 6
    assert result.total_tokens == 26


def test_ask_sends_system_and_user_messages():
    client = _FakeClient(response=_FakeResponse(content="ok"))
    assistant = LocalHealthAssistant(client=client)

    assistant.ask("What helps with a headache?")

    call = client.completions.calls[0]
    assert call["model"] == "llama3.2"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "What helps with a headache?"}


def test_ask_raises_on_empty_response():
    client = _FakeClient(response=_FakeResponse(content=""))
    assistant = LocalHealthAssistant(client=client)

    with pytest.raises(RuntimeError):
        assistant.ask("Anything")


def test_ask_propagates_connection_errors():
    client = _FakeClient(exception=ConnectionError("Ollama is not running"))
    assistant = LocalHealthAssistant(client=client)

    with pytest.raises(ConnectionError):
        assistant.ask("Anything")


def test_shares_system_prompt_with_remote_assistant():
    from assistant.remote import DEFAULT_SYSTEM_PROMPT

    assistant = LocalHealthAssistant(client=_FakeClient(response=_FakeResponse(content="ok")))

    assert assistant.system_prompt == DEFAULT_SYSTEM_PROMPT
