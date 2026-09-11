# AfyaPlus Assistant Module

## Overview

The AfyaPlus Assistant module answers the patient directly. It's the final step in the message pipeline: `summarizer` translates, checks for emergencies, and shrinks a raw message; `assistant` takes what `summarizer` produced, gets a response, and (optionally) records what that response actually cost.

### Pipeline

```text
Raw Message
    ↓
summarizer.pipeline.prepare_message()   (translate → detect → summarize)
    ↓
Final Text
    ↓
assistant.router.Assistant.ask()
    ↓
 ┌── ANTHROPIC_API_KEY set? ──────────────┐
 ↓                                        ↓
remote.HealthAssistant (claude-opus-5)    local.LocalHealthAssistant (Ollama)
    ↓                                        ↓
 Refusal? ──raise, never reroute            Reply
    ↓
 Other failure? ──fall back to local──────→ Reply
    ↓
Reply
    ↓
assistant.cost.CostTracker.record()   (optional, labeled by was_summarized)
```

`assistant.handler.MessageHandler` wires the whole top row into one call — see Usage below.

---

## Responsibilities

The assistant module is responsible for:

* Answering a prepared patient message using Anthropic (`remote.py`) or a local Ollama model (`local.py`), behind one shared interface
* Picking which backend to use, and falling back from remote to local on failure — never the other way around
* Treating a safety-classifier refusal as a decision to respect, not a failure to route around
* Turning real token usage into dollar-cost estimates (`cost.py`), so "does summarizing save money?" has an actual answer instead of an assumption
* Composing preparation (`summarizer`), answering, and cost tracking into one entrypoint (`handler.py`)

This module does **not** translate, detect emergencies, or summarize — that's `summarizer`'s job. It also doesn't decide *what* the assistant should say beyond the system prompt; prompt engineering for clinical safety lives in `remote.DEFAULT_SYSTEM_PROMPT` / `local.DEFAULT_SYSTEM_PROMPT` (shared between both backends) and should be reviewed as a distinct concern from the plumbing here.

---

## Key Design Decisions

### Remote and Local Share One Interface

`HealthAssistant.ask()` and `LocalHealthAssistant.ask()` both take a string and return the same `AssistantResponse(message, prompt_tokens, completion_tokens)`. Callers, tests, and `Assistant`/`MessageHandler` don't need to know or care which one actually answered.

### Fallback Is One-Directional: Remote → Local, Never Local → Remote

If the real assistant fails (network error, empty response, anything but a refusal), `Assistant` retries on the free local model automatically. It never falls back the other way — a local Ollama outage should not silently start spending real money on Anthropic's API without being asked to.

### A Refusal Is a Decision, Not an Outage

Claude Opus 5's safety classifiers can decline a request (`stop_reason: "refusal"`). `remote.py` raises this as a distinct `AssistantRefusalError` rather than a generic failure, and `router.py` explicitly never falls back to local on a refusal — rerouting a declined request to an unfiltered local model would defeat the point of the refusal.

### Server-Side Fallback Is Enabled by Default

`HealthAssistant` sends `output_config={"effort": "medium"}` plus Anthropic's server-side `fallbacks="default"` on every request, so a policy decline on `claude-opus-5` is automatically retried on Anthropic's recommended fallback model before it ever reaches `AssistantRefusalError`. This is separate from (and runs before) `router.py`'s own remote→local fallback.

### `last_model` Exists Because Fallback Is Invisible Otherwise

`Assistant.ask()` can silently answer from either backend. Cost tracking needs to know which one actually ran to price it correctly, so `Assistant.last_model` is set after every call and `MessageHandler` prefers it over a static `.model` attribute when recording cost.

### Cost Tracking Is Opt-In and Label-Driven

`CostTracker` doesn't track anything automatically — you call `record()` with a label. `MessageHandler` labels every call `"summarized"` or `"not_summarized"` from `PipelineResult.was_summarized`, so `tracker.summary("summarized")` vs `tracker.summary("not_summarized")` gives a direct, measured comparison instead of a guess.

### Fail Loud, Not Fail Silent (at the Backend Level)

`HealthAssistant.ask()` and `LocalHealthAssistant.ask()` both raise on failure rather than swallowing errors — the same convention as `translator.py` and `summarizer.py`. Fallback and recovery are `router.py`'s job, one layer up, not the individual backends'.

---

## Project Structure

```text
AfyaPlus/
└── assistant/
    ├── __init__.py       # Public entrypoint: Assistant, ask, MessageHandler, handle_message
    ├── remote.py         # HealthAssistant -- Anthropic (claude-opus-5), the real backend
    ├── local.py          # LocalHealthAssistant -- Ollama, the free/offline backend
    ├── router.py         # Assistant -- picks remote/local, handles fallback
    ├── cost.py           # CostTracker -- turns token usage into dollar estimates
    ├── handler.py        # MessageHandler -- prepare -> answer -> (optional) cost-track
    └── tests/
        ├── test_remote.py
        ├── test_local.py
        ├── test_router.py
        ├── test_cost.py
        └── test_handler.py
```

---

## Usage

### The One-Call Version

```python
from assistant import handle_message
from assistant.cost import CostTracker

tracker = CostTracker()
result = handle_message("Nina maumivu ya kifua na siwezi kupumua", cost_tracker=tracker)

print(result.final_text)             # what was actually sent to the assistant
print(result.emergency.is_emergency) # True
print(result.response.message)       # the assistant's reply

print(tracker.summary("not_summarized").total_cost)
```

### Using the Pieces Individually

```python
from assistant.router import Assistant
from assistant.remote import HealthAssistant, AssistantRefusalError
from assistant.local import LocalHealthAssistant

assistant = Assistant()  # auto-picks remote or local based on ANTHROPIC_API_KEY
result = assistant.ask("I have had a headache for three days.")
print(assistant.last_model)  # which backend actually answered
```

---

## Configuration

* `ANTHROPIC_API_KEY` — when set, `Assistant` defaults to the real backend (`claude-opus-5`). When unset, it defaults to the local Ollama backend instead of failing.
* Ollama must be running locally (`ollama serve`, or as a system service) with the target model pulled (`ollama pull llama3.2` by default) for the local backend to work.
* The `openai` package is required for `local.py` (Ollama exposes an OpenAI-compatible API); `anthropic` is required for `remote.py`.

---

## Testing

```bash
source afyaplus_venv/bin/activate
python -m pytest assistant/tests/ -v
```

All tests run offline with fake clients/backends injected via each class's `client=`/`remote=`/`local=`/`assistant=` parameter — no real Anthropic or Ollama calls, no cost. For a real end-to-end smoke test against a live local Ollama server, see `sandbox/try_local.py` and `sandbox/try_local_pipeline.py`.

---

## Scope

This module currently supports:

* Anthropic-backed patient responses (`remote.py`)
* Ollama-backed local/offline patient responses (`local.py`)
* Automatic backend selection and one-directional fallback (`router.py`)
* Dollar-cost tracking, grouped by label (`cost.py`)
* End-to-end composition with `summarizer` (`handler.py`)

---

## Future Work

Potential future enhancements include:

* Automatic cost recording wired into a real request path (currently opt-in via `cost_tracker=`), with per-route or per-day aggregation
* Structured logging of `AssistantRefusalError` occurrences for safety monitoring, not just a warning log
* A local backend refusal-equivalent — Ollama models have no safety classifier layer today, so nothing currently distinguishes "the local model gave a bad answer" from "it gave a fine one"
* Exercising the real Anthropic and Ollama backends in a way that's excluded from routine test runs (marked/opt-in integration tests), rather than only via the manual `sandbox/` scripts

---

## Development Notes

Every external call (`remote.py`, `local.py`) is injectable via a `client`/`remote`/`local` constructor parameter specifically so `router.py` and `handler.py` can be tested as compositions without ever making a real network call. Keep that pattern when adding new backends or pipeline stages -- it's what let `test_router.py` and `test_handler.py` verify fallback and cost-labeling behavior deterministically instead of depending on a live API key or a running Ollama server.
