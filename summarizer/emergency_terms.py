"""Rule-based extraction of critical emergency/urgency terms.

This module answers one narrow question: does a message contain any of AfyaPlus's curated emergency phrases, and how severe are they? 
It is deliberately rule-based (no LLM calls) so it stays cheap, deterministic, and testable in isolation. 
Callers such as the summarizer use its output to make sure critical signals like "cannot breathe" survive being shortened, and to flag messages that may need to skip straight to urgent handling.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"high": 2, "medium": 1}


@dataclass(frozen=True)
class EmergencyMatch:
    """A single emergency phrase found in a message."""

    term: str
    severity: str  # "high" or "medium"


@dataclass(frozen=True)
class EmergencyExtractionResult:
    """The emergency phrases found in a message, ranked by severity."""

    matches: List[EmergencyMatch] = field(default_factory=list)

    @property
    def matched_terms(self) -> List[str]:
        """The matched phrases, without their severity."""
        return [match.term for match in self.matches]

    @property
    def highest_severity(self) -> Optional[str]:
        """The most severe match found, or None if nothing matched."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda match: _SEVERITY_RANK[match.severity]).severity

    @property
    def is_emergency(self) -> bool:
        """True when at least one high-severity term was found."""
        return self.highest_severity == "high"


class EmergencyTermExtractor:
    """Detects curated emergency/urgency phrases in a message."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        """Initialize the extractor.

        Args:
            data_dir: Directory containing emergency_terms.json. Defaults to
                this module's sibling ``data`` directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
        self._term_severity = self._load_term_severity()

    def _load_term_severity(self) -> Dict[str, str]:
        """Build a term -> severity map from emergency_terms.json.

        Terms explicitly listed under severity_keywords keep that rating.
        Every other term in emergency_terms/swahili_emergency_terms defaults
        to "high" -- the whole point of that list is to name the phrases
        that must never be silently dropped.
        """
        file_path = self.data_dir / "emergency_terms.json"

        if not file_path.exists():
            logger.warning("Emergency terms file not found: %s", file_path)
            return {}

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            logger.exception("Invalid or unreadable JSON file: %s", file_path)
            return {}

        severity_keywords = data.get("severity_keywords", {})
        term_severity: Dict[str, str] = {}

        for severity in ("medium", "high"):  # high last, so it wins if a term is in both
            for term in severity_keywords.get(severity, []):
                term_severity[str(term).lower().strip()] = severity

        uncategorized_terms = list(data.get("emergency_terms", [])) + list(data.get("swahili_emergency_terms", []))
        for term in uncategorized_terms:
            term_severity.setdefault(str(term).lower().strip(), "high")

        return term_severity

    def extract(self, text: str) -> EmergencyExtractionResult:
        """Find every curated emergency phrase present in ``text``."""
        if not text:
            return EmergencyExtractionResult()

        lower_text = text.lower()
        matches = []

        for term, severity in sorted(self._term_severity.items(), key=lambda item: len(item[0]), reverse=True):
            if not term:
                continue
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, lower_text):
                matches.append(EmergencyMatch(term=term, severity=severity))

        return EmergencyExtractionResult(matches=matches)


def extract_emergency_terms(text: str, **kwargs) -> EmergencyExtractionResult:
    """Convenience wrapper for a one-line extraction call."""
    return EmergencyTermExtractor(**kwargs).extract(text)


def merge_emergency_results(*results: EmergencyExtractionResult) -> EmergencyExtractionResult:
    """Union several extraction results, keeping each term's highest severity.

    Useful when a pipeline checks emergency terms in more than one version
    of a message (e.g. before and after translation) and wants a single
    combined signal rather than picking one version to trust.
    """
    best_severity: Dict[str, str] = {}

    for result in results:
        for match in result.matches:
            current = best_severity.get(match.term)
            if current is None or _SEVERITY_RANK[match.severity] > _SEVERITY_RANK[current]:
                best_severity[match.term] = match.severity

    merged = [EmergencyMatch(term=term, severity=severity) for term, severity in best_severity.items()]
    return EmergencyExtractionResult(matches=merged)


__all__ = [
    "EmergencyMatch",
    "EmergencyExtractionResult",
    "EmergencyTermExtractor",
    "extract_emergency_terms",
    "merge_emergency_results",
]
