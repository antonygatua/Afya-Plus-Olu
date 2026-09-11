"""Tests for the end-to-end translate -> detect -> summarize pipeline."""

from summarizer.pipeline import MessagePipeline
from summarizer.summarizer import Summarizer
from summarizer.translator import MedicalTranslator

LONG_APPOINTMENT_MESSAGE = (
    "Good morning, I wanted to reach out about scheduling a routine check up "
    "some time in the next couple of weeks, whatever slot works best for the "
    "clinic, since I have not had a general check up in quite a while and "
    "would like to get one booked in before the end of the month if possible."
)


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


def _local_only_translator():
    return MedicalTranslator(use_google_translate=False)


def test_empty_message_short_circuits():
    pipeline = MessagePipeline(translator=_local_only_translator(), summarizer=Summarizer(use_llm=False))

    result = pipeline.prepare("   ")

    assert result.original_text == ""
    assert result.final_text == ""
    assert not result.was_summarized
    assert result.emergency.matches == []


def test_short_message_is_translated_but_not_summarized():
    pipeline = MessagePipeline(translator=_local_only_translator(), summarizer=Summarizer(use_llm=False))

    result = pipeline.prepare("Nina homa")

    assert "fever" in result.translated_text.lower()
    assert result.final_text == result.translated_text
    assert not result.was_summarized


def test_raw_text_catches_emergency_that_local_translation_would_miss():
    # "kuanguka" (collapsed) is in emergency_terms.json's swahili_emergency_terms
    # but has no entry in swahili_medical_terms.json's translation dictionary,
    # so the local-dictionary fallback leaves it untranslated -- the
    # translated text alone would never trigger emergency detection.
    text = "Amekuwa na maumivu na kuanguka nyumbani"
    pipeline = MessagePipeline(translator=_local_only_translator(), summarizer=Summarizer(use_llm=False))

    result = pipeline.prepare(text)

    assert "collapsed" not in result.translated_text.lower()
    assert "kuanguka" in result.emergency.matched_terms
    assert result.emergency.is_emergency
    assert not result.was_summarized
    assert result.final_text == result.translated_text


def test_long_ordinary_message_gets_summarized():
    client = _FakeClient(response_text="Wants a routine check up booked within two weeks.")
    pipeline = MessagePipeline(
        translator=_local_only_translator(),
        summarizer=Summarizer(use_llm=True, client=client),
    )

    result = pipeline.prepare(LONG_APPOINTMENT_MESSAGE)

    assert result.was_summarized
    assert result.final_text == "Wants a routine check up booked within two weeks."
    assert not result.emergency.is_emergency
