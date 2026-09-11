"""Tests for cost tracking -- no API calls, just arithmetic over token counts."""

import pytest

from assistant.cost import CostTracker, ModelPricing
from assistant.remote import AssistantResponse


def test_record_computes_cost_from_pricing_table():
    tracker = CostTracker()
    response = AssistantResponse(message="ok", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    call_cost = tracker.record(response, model="claude-opus-5")

    assert call_cost.input_cost == pytest.approx(5.00)
    assert call_cost.output_cost == pytest.approx(25.00)
    assert call_cost.total_cost == pytest.approx(30.00)


def test_record_rejects_unknown_model():
    tracker = CostTracker()
    response = AssistantResponse(message="ok", prompt_tokens=100, completion_tokens=50)

    with pytest.raises(ValueError):
        tracker.record(response, model="some-unpriced-model")


def test_custom_pricing_overrides_default_table():
    tracker = CostTracker(pricing={"local-llama": ModelPricing(input_per_million=0.0, output_per_million=0.0)})
    response = AssistantResponse(message="ok", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    call_cost = tracker.record(response, model="local-llama")

    assert call_cost.total_cost == 0.0


def test_summary_aggregates_calls_under_the_same_label():
    tracker = CostTracker()
    tracker.record(
        AssistantResponse(message="a", prompt_tokens=1_000_000, completion_tokens=0),
        model="claude-haiku-4-5",
        label="summarized",
    )
    tracker.record(
        AssistantResponse(message="b", prompt_tokens=3_000_000, completion_tokens=0),
        model="claude-haiku-4-5",
        label="summarized",
    )

    result = tracker.summary("summarized")

    assert result.call_count == 2
    assert result.avg_prompt_tokens == 2_000_000
    assert result.total_cost == pytest.approx(4.00)  # (1M + 3M) / 1M * $1.00
    assert result.avg_cost_per_call == pytest.approx(2.00)


def test_summary_of_unrecorded_label_is_empty():
    tracker = CostTracker()

    result = tracker.summary("never_recorded")

    assert result.call_count == 0
    assert result.total_cost == 0.0
    assert result.avg_cost_per_call == 0.0


def test_summarized_vs_not_summarized_cost_comparison():
    tracker = CostTracker()

    # A long message sent to the expensive model without summarizing first.
    tracker.record(
        AssistantResponse(message="a", prompt_tokens=2_000_000, completion_tokens=500_000),
        model="claude-opus-5",
        label="not_summarized",
    )
    # The same conversation, but summarized down before reaching the assistant.
    tracker.record(
        AssistantResponse(message="b", prompt_tokens=200_000, completion_tokens=500_000),
        model="claude-opus-5",
        label="summarized",
    )

    not_summarized = tracker.summary("not_summarized")
    summarized = tracker.summary("summarized")

    assert summarized.total_cost < not_summarized.total_cost
    assert set(tracker.labels()) == {"not_summarized", "summarized"}


def test_estimate_monthly_cost_projects_from_average():
    tracker = CostTracker()
    tracker.record(
        AssistantResponse(message="a", prompt_tokens=1_000_000, completion_tokens=1_000_000),
        model="claude-opus-5",
    )

    monthly = tracker.summary().estimate_monthly_cost(daily_messages=10, days_per_month=30)

    assert monthly == pytest.approx(30.00 * 10 * 30)
