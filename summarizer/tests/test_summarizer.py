"""Tests for LLM-backed summarization, without any real API calls."""

from summarizer.summarizer import Summarizer

LONG_APPOINTMENT_MESSAGE = (
    "Good morning, I wanted to reach out about scheduling a routine check up "
    "some time in the next couple of weeks, whatever slot works best for the "
    "clinic, since I have not had a general check up in quite a while and "
    "would like to get one booked in before the end of the month if possible."
)

LONG_FEVER_MESSAGE = (
    "For the past three days I have had a mild fever that comes and goes, "
    "along with a dry cough in the evenings, and I have been drinking plenty "
    "of fluids and resting but it has not fully gone away yet even though it "
    "is not getting any worse either as far as I can tell right now."
)


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text=None, exception=None):
        self._response_text = response_text
        self._exception = exception
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception:
            raise self._exception
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text=None, exception=None):
        self.messages = _FakeMessages(response_text=response_text, exception=exception)


def test_short_message_is_not_summarized():
    client = _FakeClient(response_text="should not matter")
    summarizer = Summarizer(use_llm=True, client=client)

    result = summarizer.summarize("I have a mild headache")

    assert result.text == "I have a mild headache"
    assert not result.was_summarized
    assert client.messages.calls == []


def test_emergency_message_skips_llm_even_if_long():
    text = LONG_APPOINTMENT_MESSAGE + " I have severe chest pain and cannot breathe."
    client = _FakeClient(response_text="should not be used")
    summarizer = Summarizer(use_llm=True, client=client)

    result = summarizer.summarize(text)

    assert result.text == text.strip()
    assert not result.was_summarized
    assert result.emergency.is_emergency
    assert client.messages.calls == []


def test_long_non_emergency_message_gets_summarized():
    client = _FakeClient(response_text="Wants a routine check up booked within two weeks.")
    summarizer = Summarizer(use_llm=True, client=client)

    result = summarizer.summarize(LONG_APPOINTMENT_MESSAGE)

    assert result.was_summarized
    assert result.text == "Wants a routine check up booked within two weeks."
    assert len(client.messages.calls) == 1


def test_summary_missing_medium_severity_term_gets_patched():
    client = _FakeClient(response_text="Mild ongoing cough for a few days, improving.")
    summarizer = Summarizer(use_llm=True, client=client)

    result = summarizer.summarize(LONG_FEVER_MESSAGE)

    assert result.was_summarized
    assert "fever" in result.text.lower()
    assert result.emergency.highest_severity == "medium"


def test_llm_failure_falls_back_to_original_message():
    client = _FakeClient(exception=RuntimeError("network error"))
    summarizer = Summarizer(use_llm=True, client=client)

    result = summarizer.summarize(LONG_APPOINTMENT_MESSAGE)

    assert not result.was_summarized
    assert result.text == LONG_APPOINTMENT_MESSAGE.strip()


def test_llm_disabled_returns_original_message_unchanged():
    summarizer = Summarizer(use_llm=False)

    result = summarizer.summarize(LONG_APPOINTMENT_MESSAGE)

    assert not result.was_summarized
    assert result.text == LONG_APPOINTMENT_MESSAGE.strip()


def test_empty_message_returns_empty_result():
    summarizer = Summarizer(use_llm=False)

    result = summarizer.summarize("   ")

    assert result.text == ""
    assert not result.was_summarized
    assert result.emergency.matches == []
