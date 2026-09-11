"""Tests for rule-based emergency-term extraction."""

from summarizer.emergency_terms import EmergencyTermExtractor


def test_detects_high_severity_phrase():
    extractor = EmergencyTermExtractor()

    result = extractor.extract("I have severe chest pain and cannot breathe")

    assert "chest pain" in result.matched_terms
    assert "cannot breathe" in result.matched_terms
    assert result.is_emergency
    assert result.highest_severity == "high"


def test_detects_medium_severity_symptom_only():
    extractor = EmergencyTermExtractor()

    result = extractor.extract("I have a fever and a cough")

    assert result.matched_terms == ["fever", "cough"] or set(result.matched_terms) == {"fever", "cough"}
    assert not result.is_emergency
    assert result.highest_severity == "medium"


def test_uncategorized_emergency_term_defaults_to_high():
    extractor = EmergencyTermExtractor()

    result = extractor.extract("He is unconscious after the accident")

    assert "unconscious" in result.matched_terms
    assert result.is_emergency


def test_no_match_returns_empty_result():
    extractor = EmergencyTermExtractor()

    result = extractor.extract("I would like to book an appointment next week")

    assert result.matches == []
    assert result.highest_severity is None
    assert not result.is_emergency


def test_missing_data_file_fails_soft(tmp_path):
    extractor = EmergencyTermExtractor(data_dir=str(tmp_path))

    result = extractor.extract("severe chest pain")

    assert result.matches == []
