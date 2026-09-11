import sys
from pathlib import Path

# Add the AfyaPlus root to the import path so `summarizer`/`assistant`
# resolve as packages no matter where this script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from summarizer.pipeline import MessagePipeline
from summarizer.translator import MedicalTranslator
from summarizer.summarizer import Summarizer
from assistant.local import LocalHealthAssistant


if __name__ == "__main__":
    # Fully local, zero-cost setup: no Google Cloud Translation (local
    # Swahili dictionary fallback only) and no Anthropic summarization call.
    pipeline = MessagePipeline(
        translator=MedicalTranslator(use_google_translate=False),
        summarizer=Summarizer(use_llm=False),
    )
    assistant = LocalHealthAssistant()

    # "Help!!" should trip emergency detection; "kwa siku tatu" ("for three
    # days") is a known gap in the local dictionary and will stay untranslated.
    raw_message = "Help!! Nina maumivu ya kichwa kwa siku tatu."

    # Translate, check for emergency terms (on both raw and translated text),
    # and summarize if it's safe and worth it.
    prepared = pipeline.prepare(raw_message)

    print("Translated message:", prepared.translated_text)
    print("Is emergency:", prepared.emergency.is_emergency)
    print("Was summarized:", prepared.was_summarized)

    # Always send final_text, not translated_text -- it's the field that
    # reflects summarization (and the emergency-skip decision) correctly.
    result = assistant.ask(prepared.final_text)

    print("--- Patient ---")
    print(f"Patient: {raw_message}")
    print("--- Local Health Assistant ---")
    print(result.message)
