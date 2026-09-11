"""Manual smoke test for the local Ollama-backed health assistant.

Run from anywhere:

    source afyaplus_venv/bin/activate
    python3 sandbox/try_local.py

Requires Ollama running locally with the model pulled (`ollama pull llama3.2`)
and the `openai` package installed (`pip install openai`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.local import LocalHealthAssistant

if __name__ == "__main__":
    assistant = LocalHealthAssistant()
    question = "I have had a mild headache for two days, what should I do?"

    result = assistant.ask(question)

    print("--- Local Health Assistant ---")
    print(f"Patient: {question}")
    print(f"Assistant: {result.message}")
    print("--- Usage ---")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"Total tokens: {result.total_tokens}")
