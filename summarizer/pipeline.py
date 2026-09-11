"""End-to-end message preparation: translate, detect emergencies, summarize.

This is the entrypoint the rest of the app should call before handing a user
message to the health assistant (assistant/remote.py). It wires together the
three pieces built so far:

1. translator.py -- normalize and translate Swahili/mixed text to English.
2. emergency_terms.py -- rule-based detection of curated emergency phrases.
3. summarizer.py -- shrink the message with an LLM, safe for emergencies.

One detail worth calling out: emergency terms are checked on BOTH the raw original text and the translated text, not just the translated one. 
This guards against local dictionary translation quietly dropping an emergency signal (e.g. a Swahili phrase the dictionary doesn't have a full
translation for) -- if either version looks like an emergency, the whole message is treated as one and summarization is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from summarizer.emergency_terms import (
    EmergencyExtractionResult,
    EmergencyTermExtractor,
    merge_emergency_results,
)
from summarizer.summarizer import Summarizer
from summarizer.translator import MedicalTranslator


@dataclass(frozen=True)
class PipelineResult:
    """Everything downstream code needs to know about a prepared message."""

    original_text: str
    translated_text: str
    final_text: str  # what should actually be sent to the health assistant
    emergency: EmergencyExtractionResult
    was_summarized: bool


class MessagePipeline:
    """Translates, checks for emergencies, and summarizes a user message."""

    def __init__(
        self,
        translator: Optional[Any] = None,
        summarizer: Optional[Any] = None,
        extractor: Optional[EmergencyTermExtractor] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            translator: A MedicalTranslator-compatible instance. A default
                one is created if omitted.
            summarizer: A Summarizer-compatible instance. A default one
                (sharing this pipeline's extractor) is created if omitted.
            extractor: EmergencyTermExtractor used to check the raw,
                pre-translation text. A new one is created if omitted.
        """
        self.extractor = extractor or EmergencyTermExtractor()
        self.translator = translator or MedicalTranslator()
        self.summarizer = summarizer or Summarizer(extractor=self.extractor)

    def prepare(self, text: Optional[str]) -> PipelineResult:
        """Translate, check, and summarize a raw user message."""
        original = (text or "").strip()

        if not original:
            return PipelineResult(
                original_text="",
                translated_text="",
                final_text="",
                emergency=EmergencyExtractionResult(),
                was_summarized=False,
            )

        translated = self.translator.translate_to_english(original)
        raw_emergency = self.extractor.extract(original)

        if raw_emergency.is_emergency:
            # The raw text alone already signals an emergency -- don't wait
            # on (or trust) the translated version before treating it as one.
            translated_emergency = self.extractor.extract(translated)
            emergency = merge_emergency_results(raw_emergency, translated_emergency)
            return PipelineResult(
                original_text=original,
                translated_text=translated,
                final_text=translated,
                emergency=emergency,
                was_summarized=False,
            )

        summary_result = self.summarizer.summarize(translated)
        emergency = merge_emergency_results(raw_emergency, summary_result.emergency)

        return PipelineResult(
            original_text=original,
            translated_text=translated,
            final_text=summary_result.text,
            emergency=emergency,
            was_summarized=summary_result.was_summarized,
        )


def prepare_message(text: Optional[str], **kwargs: Any) -> PipelineResult:
    """Convenience wrapper for a one-line pipeline run."""
    return MessagePipeline(**kwargs).prepare(text)


__all__ = ["PipelineResult", "MessagePipeline", "prepare_message"]
