"""Local translation tests for the AfyaPlus translator.

These tests validate the local medical dictionary logic without depending on
Google Cloud Translation.
"""

from summarizer.translator import MedicalTranslator


def test_local_translation_basic_symptoms():
    translator = MedicalTranslator(use_google_translate=False)

    text = "Nina homa na maumivu ya kifua"
    result = translator.translate_to_english(text)

    assert "fever" in result.lower()
    assert "chest pain" in result.lower()


def test_local_translation_breathing_problem():
    translator = MedicalTranslator(use_google_translate=False)

    text = "Siwezi kupumua na kikohozi"
    result = translator.translate_to_english(text)

    assert "cannot breathe" in result.lower()
    assert "cough" in result.lower()


def test_local_translation_feeling_variants():
    translator = MedicalTranslator(use_google_translate=False)

    text = "Nahisi maumivu ya kichwa"
    result = translator.translate_to_english(text)

    assert "i feel" in result.lower()
    assert "headache" in result.lower()


def test_local_translation_preserves_english_input():
    translator = MedicalTranslator(use_google_translate=False)

    text = "I have fever and chest pain"
    result = translator.translate_to_english(text)

    assert result.lower() == text.lower()
