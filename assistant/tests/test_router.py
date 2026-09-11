"""Tests for the remote/local assistant router -- no real API or Ollama calls."""

import pytest

from assistant.remote import AssistantRefusalError, AssistantResponse
from assistant.router import Assistant


class _FakeBackend:
    def __init__(self, response=None, exception=None, model="fake-model"):
        self._response = response
        self._exception = exception
        self.model = model
        self.calls = []

    def ask(self, message):
        self.calls.append(message)
        if self._exception:
            raise self._exception
        return self._response


def _response(text="ok"):
    return AssistantResponse(message=text, prompt_tokens=10, completion_tokens=5)


def test_defaults_to_remote_when_api_key_present():
    remote = _FakeBackend(response=_response("from remote"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key")

    result = assistant.ask("question")

    assert result.message == "from remote"
    assert remote.calls == ["question"]
    assert local.calls == []


def test_defaults_to_local_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    remote = _FakeBackend(response=_response("from remote"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(remote=remote, local=local, api_key=None)

    result = assistant.ask("question")

    assert result.message == "from local"
    assert remote.calls == []
    assert local.calls == ["question"]


def test_explicit_backend_overrides_api_key_detection():
    remote = _FakeBackend(response=_response("from remote"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(backend="local", remote=remote, local=local, api_key="sk-fake-key")

    result = assistant.ask("question")

    assert result.message == "from local"
    assert remote.calls == []


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        Assistant(backend="carrier-pigeon")


def test_remote_failure_falls_back_to_local_by_default():
    remote = _FakeBackend(exception=ConnectionError("network down"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key")

    result = assistant.ask("question")

    assert result.message == "from local"
    assert local.calls == ["question"]


def test_remote_failure_raises_when_fallback_disabled():
    remote = _FakeBackend(exception=ConnectionError("network down"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key", fallback_to_local=False)

    with pytest.raises(ConnectionError):
        assistant.ask("question")

    assert local.calls == []


def test_refusal_never_falls_back_to_local():
    remote = _FakeBackend(exception=AssistantRefusalError(category="cyber", explanation="declined"))
    local = _FakeBackend(response=_response("from local"))
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key")

    with pytest.raises(AssistantRefusalError):
        assistant.ask("question")

    assert local.calls == []


def test_last_model_reflects_the_backend_that_actually_answered():
    remote = _FakeBackend(response=_response("from remote"), model="claude-opus-5")
    local = _FakeBackend(response=_response("from local"), model="llama3.2")
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key")

    assistant.ask("question")

    assert assistant.last_model == "claude-opus-5"


def test_last_model_reflects_local_after_a_fallback():
    remote = _FakeBackend(exception=ConnectionError("network down"), model="claude-opus-5")
    local = _FakeBackend(response=_response("from local"), model="llama3.2")
    assistant = Assistant(remote=remote, local=local, api_key="sk-fake-key")

    assistant.ask("question")

    assert assistant.last_model == "llama3.2"
