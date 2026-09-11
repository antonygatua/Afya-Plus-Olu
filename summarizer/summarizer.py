"""Cost-saving summarization for symptom messages, safe for emergencies.

Shrinks an already-translated user message before it's sent to the health
assistant (assistant/remote.py), while guaranteeing that curated emergency
phrases (see emergency_terms.py) always survive in the output -- even if
that means skipping summarization altogether.

Summarization strategy:

1. Extract any curated emergency terms from the message.
2. If the message is a high-severity emergency, or already short enough
   that summarizing it wouldn't save meaningful cost, return it unchanged.
   Urgent messages skip the extra API round-trip entirely rather than risk
   an LLM softening or dropping critical detail.
3. Otherwise, ask a cheap LLM (Anthropic) for a short summary.
4. Verify every matched emergency term still appears in the summary; append
   any that went missing rather than silently losing them.
5. If the LLM call is unavailable or fails, fall back to the original
   message -- never guess at a truncated summary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from summarizer.emergency_terms import EmergencyExtractionResult, EmergencyTermExtractor

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MIN_WORDS_TO_SUMMARIZE = 40
DEFAULT_MAX_SUMMARY_WORDS = 40

_SUMMARY_SYSTEM_PROMPT = (
    "You compress patient symptom messages for a medical triage system. "
    "Rewrite the message in at most {max_words} words, in English, keeping "
    "every symptom, timing detail, and severity word. Do not add "
    "information, diagnoses, or advice. Reply with only the rewritten "
    "message."
)


@dataclass(frozen=True)
class SummaryResult:
    """The outcome of summarizing a message."""

    text: str
    emergency: EmergencyExtractionResult
    was_summarized: bool


class Summarizer:
    """Shrinks symptom messages while preserving emergency signals."""

    def __init__(
        self,
        use_llm: Optional[bool] = None,
        model: str = DEFAULT_MODEL,
        min_words_to_summarize: int = DEFAULT_MIN_WORDS_TO_SUMMARIZE,
        max_summary_words: int = DEFAULT_MAX_SUMMARY_WORDS,
        extractor: Optional[EmergencyTermExtractor] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            use_llm: Explicitly enable/disable the LLM summarization call.
                Defaults to enabled when an Anthropic API key is available.
            model: Anthropic model used for summarization. Defaults to a
                fast model, since this call exists to shrink the message
                before it reaches the health assistant, not to answer it.
            min_words_to_summarize: Messages with fewer words than this are
                returned unchanged -- summarizing them wouldn't save enough
                to be worth the extra API call.
            max_summary_words: Target word budget for the LLM summary.
            extractor: EmergencyTermExtractor to reuse. A new one is created
                if omitted.
            api_key: Anthropic API key. Defaults to the ANTHROPIC_API_KEY
                environment variable.
            client: A pre-built Anthropic-compatible client (mainly for
                tests). A real client is created lazily if omitted.
        """
        self.model = model
        self.min_words_to_summarize = min_words_to_summarize
        self.max_summary_words = max_summary_words
        self.extractor = extractor or EmergencyTermExtractor()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.use_llm = use_llm if use_llm is not None else bool(self.api_key)
        self._client = client

    def _get_client(self):
        """Create and cache the Anthropic client."""
        if self._client is not None:
            return self._client

        if anthropic is None:
            raise RuntimeError("anthropic is not installed. Install it with: pip install anthropic")

        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _call_llm_summary(self, text: str) -> str:
        """Ask the LLM for a short summary of ``text``."""
        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=max(64, self.max_summary_words * 3),
            system=_SUMMARY_SYSTEM_PROMPT.format(max_words=self.max_summary_words),
            messages=[{"role": "user", "content": text}],
        )

        summary = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if not summary:
            raise RuntimeError("LLM summarization returned an empty response.")

        return summary

    def _ensure_emergency_terms_present(self, summary: str, emergency: EmergencyExtractionResult) -> str:
        """Append any matched emergency terms the summary dropped."""
        lower_summary = summary.lower()
        missing = [term for term in emergency.matched_terms if term not in lower_summary]

        if not missing:
            return summary

        logger.warning("Summary dropped emergency terms, re-appending: %s", missing)
        return f"{summary} ({', '.join(missing)})"

    def summarize(self, text: Optional[str]) -> SummaryResult:
        """Summarize an already-translated symptom message.

        Returns the message unchanged -- never guessed at or truncated --
        whenever summarization isn't worth doing, isn't safe, or isn't
        available.
        """
        normalized = (text or "").strip()
        emergency = self.extractor.extract(normalized)

        if not normalized:
            return SummaryResult(text="", emergency=emergency, was_summarized=False)

        if emergency.is_emergency:
            return SummaryResult(text=normalized, emergency=emergency, was_summarized=False)

        word_count = len(normalized.split())
        if word_count < self.min_words_to_summarize:
            return SummaryResult(text=normalized, emergency=emergency, was_summarized=False)

        if not self.use_llm:
            return SummaryResult(text=normalized, emergency=emergency, was_summarized=False)

        try:
            summary = self._call_llm_summary(normalized)
            summary = self._ensure_emergency_terms_present(summary, emergency)
            return SummaryResult(text=summary, emergency=emergency, was_summarized=True)
        except Exception as exc:
            logger.warning("LLM summarization failed; using original message. Error: %s", exc)
            return SummaryResult(text=normalized, emergency=emergency, was_summarized=False)


def summarize_message(text: Optional[str], **kwargs: Any) -> SummaryResult:
    """Convenience wrapper for a one-line summarization call."""
    return Summarizer(**kwargs).summarize(text)


__all__ = ["SummaryResult", "Summarizer", "summarize_message"]
