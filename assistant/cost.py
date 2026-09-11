"""Turns real Anthropic token usage into dollar-cost estimates.

This is what actually answers the question the summarizer pipeline exists
to act on: does translating, detecting emergencies, and summarizing before
the health assistant answers actually save money? Record each real
AssistantResponse (from assistant/remote.py) here, tagged with whatever
label matters -- e.g. "summarized" vs "not_summarized", sourced from
summarizer.PipelineResult.was_summarized -- and compare the two groups
with summary() instead of assuming the pipeline pays for itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from assistant.remote import AssistantResponse


@dataclass(frozen=True)
class ModelPricing:
    """Dollar cost per 1,000,000 input/output tokens for one model."""

    input_per_million: float
    output_per_million: float


# Keep this in sync with the models actually used in this codebase
# (assistant/remote.py, summarizer/summarizer.py) rather than trying to
# track every Anthropic model in existence.
PRICING_PER_MILLION_TOKENS: Dict[str, ModelPricing] = {
    "claude-opus-5": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-haiku-4-5": ModelPricing(input_per_million=1.00, output_per_million=5.00),
}


@dataclass(frozen=True)
class CallCost:
    """The dollar cost of a single call, and the usage it was computed from."""

    model: str
    label: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass(frozen=True)
class CostSummary:
    """Aggregate token/cost stats over a set of recorded calls."""

    call_count: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    total_cost: float

    @property
    def avg_cost_per_call(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_cost / self.call_count

    def estimate_monthly_cost(self, daily_messages: float, days_per_month: int = 30) -> float:
        """Project monthly cost from this summary's average cost per call."""
        return self.avg_cost_per_call * daily_messages * days_per_month


class CostTracker:
    """Records the dollar cost of real assistant calls, grouped by label.

    Typical use: call record() once per assistant.remote.HealthAssistant
    call with a label like "summarized" or "not_summarized" (taken from
    summarizer.PipelineResult.was_summarized), then compare
    summary("summarized") against summary("not_summarized") to see whether
    the summarizer pipeline is actually paying for itself.
    """

    def __init__(self, pricing: Optional[Dict[str, ModelPricing]] = None) -> None:
        """Initialize the tracker.

        Args:
            pricing: Model -> ModelPricing lookup. Defaults to
                PRICING_PER_MILLION_TOKENS.
        """
        self.pricing = pricing if pricing is not None else PRICING_PER_MILLION_TOKENS
        self._calls: Dict[Optional[str], List[CallCost]] = defaultdict(list)

    def record(self, response: AssistantResponse, model: str, label: Optional[str] = None) -> CallCost:
        """Record one assistant call's token usage and compute its cost.

        Args:
            response: The AssistantResponse returned by HealthAssistant.ask()
                (or LocalHealthAssistant.ask(), though local calls are free
                and typically don't need tracking here).
            model: The model that produced this response -- must be a key
                in `pricing`, since cost can't be computed without knowing
                the rate.
            label: An optional tag to group calls by (e.g. "summarized" /
                "not_summarized", or a route name). Pass the same label
                consistently to compare groups later with summary(label).
        """
        if model not in self.pricing:
            raise ValueError(
                f"No pricing configured for model {model!r}. "
                f"Pass a `pricing` dict covering it, or add it to PRICING_PER_MILLION_TOKENS."
            )

        rate = self.pricing[model]
        input_cost = (response.prompt_tokens / 1_000_000) * rate.input_per_million
        output_cost = (response.completion_tokens / 1_000_000) * rate.output_per_million

        call_cost = CallCost(
            model=model,
            label=label,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
        )

        self._calls[label].append(call_cost)
        return call_cost

    def summary(self, label: Optional[str] = None) -> CostSummary:
        """Aggregate stats over calls recorded under `label`.

        Pass None (the default) to summarize calls recorded with no label.
        """
        calls = self._calls.get(label, [])

        if not calls:
            return CostSummary(call_count=0, avg_prompt_tokens=0.0, avg_completion_tokens=0.0, total_cost=0.0)

        return CostSummary(
            call_count=len(calls),
            avg_prompt_tokens=sum(c.prompt_tokens for c in calls) / len(calls),
            avg_completion_tokens=sum(c.completion_tokens for c in calls) / len(calls),
            total_cost=sum(c.total_cost for c in calls),
        )

    def labels(self) -> List[Optional[str]]:
        """Every label that has at least one recorded call."""
        return list(self._calls.keys())


__all__ = [
    "ModelPricing",
    "PRICING_PER_MILLION_TOKENS",
    "CallCost",
    "CostSummary",
    "CostTracker",
]
