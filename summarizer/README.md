# AfyaPlus Summarizer Module

## Overview

The AfyaPlus Summarizer module prepares a raw, possibly mixed Swahili/English user message before it is handed to the health assistant (`assistant/remote.py`), which generates the response the patient actually sees.

It translates the message to English, checks it for curated emergency/urgency phrases, and shrinks it with an LLM call when doing so is safe and worth it — all while guaranteeing that critical signals like "cannot breathe" are never silently dropped.

### Pipeline

```text
Raw Message
    ↓
Translate to English (Google Cloud Translation, or local dictionary fallback)
    ↓
Detect emergency terms — checked on BOTH the raw text and the translated text
    ↓
 ┌── Emergency detected ─────────────┐
 ↓                                   ↓
Skip summarization              Not an emergency
(send translated text as-is)         ↓
                                Long enough to be worth summarizing?
                                      ↓
                                Summarize with an LLM
                                      ↓
                                Re-check: did the summary keep every
                                emergency term? Re-append any it dropped.
                                      ↓
Final Text
    ↓
Sent to the health assistant (assistant/remote.py)
```

---

## Responsibilities

The summarizer is responsible for:

* Normalizing and translating Swahili/mixed-language messages to English
* Detecting curated emergency/urgency phrases via cheap, rule-based matching
* Shrinking long, non-urgent messages with an LLM before they reach the health assistant
* Guaranteeing emergency phrases are never silently dropped, even if that means skipping summarization entirely
* Falling back safely to the untouched message whenever translation or summarization can't be trusted

The summarizer does **not** perform diagnosis, triage decisions, or generate responses to the user. Its responsibility is limited to preparing a message for the model that does.

---

## Key Design Decisions

### Emergencies Always Win

If either the raw message or its translated form matches a high-severity emergency term, summarization is skipped entirely and the translated (but not summarized) text is sent through untouched. This trades a small, occasional cost increase for the guarantee that urgent language is never compressed, reworded, or delayed by an extra LLM round-trip.

### Checking Both Raw and Translated Text

Emergency-term detection runs on the original message *and* the translated message, then the two results are merged. This guards against the local dictionary fallback leaving part of a message untranslated — a phrase like `"kuanguka"` (collapsed) has no entry in the local Swahili-to-English dictionary, so it would never trigger detection if only the translated text were checked. Checking the raw text directly catches it anyway.

### Summarization Is a Cost Tool, Not a Compression Requirement

Empty messages, short messages, and emergencies are all returned unchanged. Summarization only runs when a message is long enough that shrinking it is actually worth the extra LLM call — a short message costs more to summarize than it saves.

### Verify, Don't Just Prompt

After the LLM produces a summary, every matched emergency term is checked against it directly. Any term the summary dropped is re-appended rather than trusted to have survived on prompt instructions alone.

### Fail Safe, Not Fail Silent

Any failure in translation or summarization — network error, missing API key, empty LLM response — falls back to the original or translated message rather than guessing at a truncated or partial result.

---

## Project Structure

```text
AfyaPlus/
└── summarizer/
    ├── __init__.py          # Public entrypoint: prepare_message()
    ├── translator.py        # Swahili/English translation
    ├── emergency_terms.py   # Rule-based emergency/urgency term detection
    ├── summarizer.py        # LLM-backed message summarization
    ├── pipeline.py          # Orchestrates translate → detect → summarize
    ├── data/
    │   ├── emergency_terms.json        # Curated emergency terms + severity levels
    │   ├── swahili_medical_terms.json  # Swahili → English phrase dictionary
    │   └── swahili_markers.json        # Markers used to detect Swahili text
    └── tests/
        ├── local.py                 # Local-dictionary translation tests
        ├── google.py                # Placeholder for future Google Cloud tests
        ├── test_emergency_terms.py  # Emergency-term extraction tests
        ├── test_summarizer.py       # Summarization tests (fake LLM client)
        └── test_pipeline.py         # End-to-end pipeline tests
```

---

## Usage

### Prepare a Message

```python
from summarizer import prepare_message

result = prepare_message("Nina maumivu ya kifua na siwezi kupumua")

print(result.final_text)             # text to hand to the health assistant
print(result.emergency.is_emergency) # True
print(result.was_summarized)         # False -- emergencies skip summarization
```

### Using the Pieces Individually

```python
from summarizer.translator import MedicalTranslator
from summarizer.emergency_terms import EmergencyTermExtractor
from summarizer.summarizer import Summarizer

translator = MedicalTranslator()
extractor = EmergencyTermExtractor()
summarizer = Summarizer(extractor=extractor)

english_text = translator.translate_to_english(raw_message)
emergency = extractor.extract(english_text)
summary = summarizer.summarize(english_text)
```

---

## Configuration

* `ANTHROPIC_API_KEY` — enables LLM-backed summarization. Without it, `Summarizer` defaults to `use_llm=False` and returns messages unchanged.
* `GOOGLE_CLOUD_PROJECT` — enables Google Cloud Translation. Without it, `MedicalTranslator` falls back to the local Swahili dictionary.

Both integrations are optional at runtime and degrade gracefully instead of failing.

---

## Testing

```bash
source afyaplus_venv/bin/activate
python -m pytest summarizer/tests/ -v
```

`local.py` and `google.py` don't match pytest's default `test_*.py` discovery pattern, so a bare `pytest summarizer/tests/` won't collect them automatically — run them by explicit path, or add a `pytest.ini` extending `python_files`. All tests run offline with no real network or API calls (translation tests use `use_google_translate=False`; summarization tests inject a fake Anthropic client).

---

## Scope

This module currently supports:

* Swahili-to-English translation, with a local dictionary fallback
* Rule-based emergency/urgency term detection with severity ranking
* LLM-backed message summarization with emergency-term preservation
* End-to-end pipeline orchestration via `prepare_message()`

---

## Future Work

Potential future enhancements include:

* Closing local-dictionary gaps like the `"kuanguka"` case surfaced during testing, so translation itself catches more emergency phrasing
* Structured logging/telemetry on `was_summarized` and `is_emergency` for cost and safety monitoring in production
* Configurable summarization word budgets per downstream model
* Keeping this module's `emergency_terms.json` and the tokenizer's forced vocabulary (`shared_vocab_path` in `Tokenizer`) from drifting apart as both evolve

---

## Development Notes

Each stage — translate, detect, summarize — is independently testable and safe to run without the others' external dependencies. Emergency detection is intentionally kept rule-based and LLM-free, since it stays cheap, deterministic, and is the one signal every other stage defers to before deciding whether it's safe to shorten a message.
