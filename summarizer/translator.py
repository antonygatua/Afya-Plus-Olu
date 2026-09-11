"""Translation layer for mixed English/Swahili symptom messages.

This module normalizes short English, Swahili, and mixed-language symptom
messages into clearer English.

Translation strategy:

1. Normalize the original message.
2. Detect whether the original message contains Swahili markers.
3. If needed, try Google Cloud Translation on the original text.
4. If Google translation is unavailable or fails, use the local medical
   dictionary as a fallback.
5. Preserve the original text when no translation is required.

The local dictionary is intentionally a fallback rather than a preprocessing
step for Google Translate. This prevents partially translated sentences from
being sent to the external translation service.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from google.cloud import translate_v2 as translate
except ImportError:  # pragma: no cover
    translate = None

logger = logging.getLogger(__name__)


class MedicalTranslator:
    """Translate mixed English/Swahili symptom messages into English."""

    def __init__(self, data_dir: Optional[str] = None, target_language: str = "en", use_google_translate: Optional[bool] = None, project_id: Optional[str] = None) -> None:
        """Initialize the medical translator.
        Args:
            data_dir: Directory containing translation data files. If omitted, the module's local ``data`` directory is used.
            target_language: Target language code for translated text.
            use_google_translate: Explicitly enable or disable Google Cloud Translation. When omitted, Google translation is enabled when a Google Cloud project ID is available.
            project_id: Google Cloud project ID. Defaults to the ``GOOGLE_CLOUD_PROJECT`` environment variable.
        """
        self.base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.target_language = target_language
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.use_google_translate = (
            use_google_translate
            if use_google_translate is not None
            else bool(self.project_id)
        )

        self.medical_terms = self._load_json("swahili_medical_terms.json")
        marker_data = self._load_json("swahili_markers.json")
        self.swahili_markers = marker_data.get("markers", [])
        self._google_client = None

    def _load_json(self, file_name: str) -> Dict[str, Any]:
        """Load a JSON data file and return an empty dict on failure."""
        file_path = self.data_dir / file_name

        if not file_path.exists():
            logger.warning("Translation data file not found: %s", file_path)
            return {}

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            logger.exception("Invalid or unreadable JSON file: %s", file_path)
            return {}

        if not isinstance(data, dict):
            logger.warning("Translation data file must contain a JSON object: %s", file_path)
            return {}

        return data

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Normalize whitespace without changing the user's meaning."""
        if not text:
            return ""

        normalized = text.strip()
        return re.sub(r"\s+", " ", normalized)

    def _build_phrase_map(self) -> Dict[str, str]:
        """Build a normalized Swahili-to-English medical phrase map."""
        translations = self.medical_terms.get("translations", {})
        if not isinstance(translations, dict):
            logger.warning("The 'translations' value must be a JSON object.")
            return {}

        phrase_map: Dict[str, str] = {}
        for source, target in translations.items():
            source_key = str(source).lower().strip()
            target_value = str(target).lower().strip()
            if source_key and target_value:
                phrase_map[source_key] = target_value

        return phrase_map

    def _replace_local_terms(self, text: str) -> str:
        """Replace known Swahili medical phrases with English equivalents."""
        translated = self.normalize_text(text)
        phrase_map = self._build_phrase_map()

        if not phrase_map:
            return translated

        for phrase, replacement in sorted(
            phrase_map.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            pattern = rf"\b{re.escape(phrase)}\b"
            translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

        return self.normalize_text(translated)

    def _should_use_google_translate(self, text: str) -> bool:
        """Return True when the text appears to contain Swahili markers."""
        if not self.use_google_translate or not text:
            return False

        lower_text = text.lower()

        for marker in self.swahili_markers:
            marker = str(marker).lower().strip()

            if not marker:
                continue

            pattern = rf"\b{re.escape(marker)}\b"

            if re.search(pattern, lower_text):
                return True

        return False

    def _get_google_client(self):
        """Create and cache the Google Cloud Translation client."""
        if translate is None:
            raise RuntimeError(
                "google-cloud-translate is not installed. "
                "Install it with: pip install google-cloud-translate"
            )

        if self._google_client is None:
            self._google_client = translate.Client(project=self.project_id)

        return self._google_client

    def _google_translate_text(self, text: str) -> str:
        """Translate text using Google Cloud Translation."""
        client = self._get_google_client()
        result = client.translate(text, target_language=self.target_language)
        translated = result.get("translatedText")

        if not translated:
            raise RuntimeError("Google Cloud Translation returned an empty response.")

        return self.normalize_text(translated)

    def translate_to_english(self, text: Optional[str]) -> str:
        """Translate a symptom message into English.

        Google Cloud Translation is used for the original normalized message,
        while the local medical dictionary remains the fallback for known local
        symptom terms.
        """
        normalized = self.normalize_text(text)
        if not normalized:
            return ""

        if self._should_use_google_translate(normalized):
            try:
                translated = self._google_translate_text(normalized)
                logger.debug("Message translated successfully using Google Cloud.")
                return translated
            except Exception as exc:
                logger.warning(
                    "Google translation failed; using local medical dictionary fallback. Error: %s",
                    exc,
                )

        return self._replace_local_terms(normalized)


def translate_to_english(text: Optional[str], **kwargs: Any) -> str:
    """Convenience wrapper for a one-line translation call."""
    translator = MedicalTranslator(**kwargs)
    return translator.translate_to_english(text)


__all__ = ["MedicalTranslator", "translate_to_english"]
