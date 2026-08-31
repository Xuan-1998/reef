"""Tracked OpenAI-compatible model calls and Pi trajectory usage."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from reef.harness.model_binding import ModelBinding, ModelBindingError


class TokenUsage(TypedDict):
    requests: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class ModelPrice:
    """Standard-processing USD rates observed for one pinned model."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    source: str
    observed_at: str = "2026-08-30"

    def estimate(self, usage: Mapping[str, int]) -> float:
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = int(usage.get("cached_input_tokens", 0))
        uncached_tokens = max(0, input_tokens - cached_tokens)
        output_tokens = int(usage.get("output_tokens", 0))
        return (
            uncached_tokens * self.input_per_million
            + cached_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


TASK_MODEL_PRICE = ModelPrice(
    input_per_million=0.40,
    cached_input_per_million=0.10,
    output_per_million=1.60,
    source="https://developers.openai.com/api/docs/models/gpt-4.1-mini",
)
REFLECTION_MODEL_PRICE = ModelPrice(
    input_per_million=1.25,
    cached_input_per_million=0.125,
    output_per_million=10.00,
    source="https://developers.openai.com/api/docs/models/gpt-5",
)


class UsageLedger:
    """Thread-safe token totals for model calls made outside Pi."""

    def __init__(self, price: ModelPrice) -> None:
        self.price = price
        self._usage: TokenUsage = empty_usage()
        self._lock = threading.Lock()

    def add_openai_response(self, response: Mapping[str, Any]) -> None:
        raw = response.get("usage")
        if not isinstance(raw, Mapping):
            return
        input_tokens = _integer(raw.get("prompt_tokens", raw.get("input_tokens", 0)))
        output_tokens = _integer(raw.get("completion_tokens", raw.get("output_tokens", 0)))
        input_details = raw.get("prompt_tokens_details", raw.get("input_tokens_details", {}))
        output_details = raw.get("completion_tokens_details", raw.get("output_tokens_details", {}))
        cached = _integer(input_details.get("cached_tokens", 0)) if isinstance(input_details, Mapping) else 0
        reasoning = _integer(output_details.get("reasoning_tokens", 0)) if isinstance(output_details, Mapping) else 0
        self.add(
            {
                "requests": 1,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning,
            }
        )

    def add(self, usage: Mapping[str, int]) -> None:
        with self._lock:
            for key in self._usage:
                self._usage[key] += int(usage.get(key, 0))

    def snapshot(self) -> TokenUsage:
        with self._lock:
            return dict(self._usage)  # type: ignore[return-value]

    @property
    def total_cost(self) -> float:
        return self.price.estimate(self.snapshot())

    @property
    def total_tokens_in(self) -> int:
        return self.snapshot()["input_tokens"]

    @property
    def total_tokens_out(self) -> int:
        return self.snapshot()["output_tokens"]


class TrackedChatModel:
    """A GEPA-compatible callable backed by Reef's model binding."""

    def __init__(self, binding: ModelBinding, *, price: ModelPrice) -> None:
        self.binding = binding
        self.usage = UsageLedger(price)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        response = self.binding.complete({"messages": messages})
        self.usage.add_openai_response(response)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc
        if not isinstance(content, str):
            raise ModelBindingError("model endpoint returned non-text content")
        return content

    @property
    def total_cost(self) -> float:
        return self.usage.total_cost

    @property
    def total_tokens_in(self) -> int:
        return self.usage.total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self.usage.total_tokens_out


def empty_usage() -> TokenUsage:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def trajectory_usage(trajectory: Sequence[Mapping[str, Any]]) -> TokenUsage:
    """Sum Pi assistant-message usage without trusting its model price table."""
    total = empty_usage()
    for event in trajectory:
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, Mapping) else event
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        total["requests"] += 1
        # Pi reports uncached input and cache reads separately. Store their sum
        # as total input so ModelPrice can apply the cached discount correctly.
        cached = _integer(usage.get("cacheRead", 0))
        total["input_tokens"] += _integer(usage.get("input", 0)) + cached
        total["cached_input_tokens"] += cached
        total["output_tokens"] += _integer(usage.get("output", 0))
        total["reasoning_tokens"] += _integer(usage.get("reasoning", 0))
    return total


def sum_usage(items: Sequence[Mapping[str, int]]) -> TokenUsage:
    total = empty_usage()
    for item in items:
        for key in total:
            total[key] += int(item.get(key, 0))
    return total


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
