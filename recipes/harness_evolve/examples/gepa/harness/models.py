"""Tracked OpenAI-compatible model calls and Pi trajectory usage."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import dspy
from gepa.lm import LM as GEPALM

from reef.harness.model_binding import ModelBinding, ModelBindingError


class TokenUsage(TypedDict):
    requests: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


class CostGuard(Protocol):
    """Minimal boundary shared by direct model calls and Pi episodes."""

    def before_call(self) -> None: ...

    def record_call(self, observed_cost_usd: float) -> None: ...


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
    source="https://developers.openai.com/api/docs/models/gpt-5.1",
)


class UsageLedger:
    """Thread-safe token totals for model calls made outside Pi."""

    def __init__(self, price: ModelPrice, path: Path | None = None) -> None:
        self.price = price
        self.path = Path(path).resolve() if path is not None else None
        self._usage: TokenUsage = empty_usage()
        self._lock = threading.Lock()
        if self.path is not None and self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping) or value.get("pricing") != asdict(price):
                raise ValueError(f"usage ledger has invalid or mismatched pricing: {self.path}")
            usage = value.get("usage")
            if not isinstance(usage, Mapping):
                raise ValueError(f"usage ledger has no usage object: {self.path}")
            loaded = empty_usage()
            if set(usage) != set(loaded):
                raise ValueError(f"usage ledger has unexpected token fields: {self.path}")
            for key in loaded:
                raw_count = usage[key]
                if not isinstance(raw_count, int) or raw_count < 0:
                    raise ValueError(f"usage ledger has invalid token counts: {self.path}")
                loaded[key] = raw_count
            self._usage = loaded
        elif self.path is not None:
            self._persist_locked()

    def add_openai_response(self, response: Mapping[str, Any]) -> TokenUsage:
        usage = empty_usage()
        usage["requests"] = 1
        raw = response.get("usage")
        if isinstance(raw, Mapping):
            input_details = raw.get("prompt_tokens_details", raw.get("input_tokens_details", {}))
            output_details = raw.get("completion_tokens_details", raw.get("output_tokens_details", {}))
            usage["input_tokens"] = _integer(raw.get("prompt_tokens", raw.get("input_tokens", 0)))
            usage["cached_input_tokens"] = (
                _integer(input_details.get("cached_tokens", 0)) if isinstance(input_details, Mapping) else 0
            )
            usage["output_tokens"] = _integer(raw.get("completion_tokens", raw.get("output_tokens", 0)))
            usage["reasoning_tokens"] = (
                _integer(output_details.get("reasoning_tokens", 0)) if isinstance(output_details, Mapping) else 0
            )
        self.add(usage)
        return usage

    def add(self, usage: Mapping[str, int]) -> None:
        with self._lock:
            for key in self._usage:
                increment = int(usage.get(key, 0))
                if increment < 0:
                    raise ValueError("token usage increments must be non-negative")
                self._usage[key] += increment
            self._persist_locked()

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

    def _persist_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"schema_version": 1, "pricing": asdict(self.price), "usage": self._usage},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class TrackedChatModel:
    """A GEPA-compatible callable backed by Reef's model binding."""

    def __init__(
        self,
        binding: ModelBinding,
        *,
        price: ModelPrice,
        spend_guard: CostGuard | None = None,
        usage_path: Path | None = None,
    ) -> None:
        self.binding = binding
        self.usage = UsageLedger(price, usage_path)
        self._spend_guard = spend_guard

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        if self._spend_guard is not None:
            self._spend_guard.before_call()
        response = self.binding.complete({"messages": messages})
        usage = self.usage.add_openai_response(response)
        if self._spend_guard is not None:
            self._spend_guard.record_call(self.usage.price.estimate(usage))
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


class TrackedDSPyLM(dspy.LM):
    """The official DSPy solver LM with persistent usage and a spend guard."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        price: ModelPrice,
        spend_guard: CostGuard | None = None,
        usage_path: Path | None = None,
    ) -> None:
        super().__init__(
            f"openai/{model}",
            api_key=api_key,
            api_base=f"{base_url.rstrip('/')}/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=True,
        )
        self.usage = UsageLedger(price, usage_path)
        self._spend_guard = spend_guard

    def forward(self, prompt=None, messages=None, **kwargs):
        if self._spend_guard is not None:
            self._spend_guard.before_call()
        response = super().forward(prompt=prompt, messages=messages, **kwargs)
        if getattr(response, "cache_hit", False):
            return response
        payload = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(payload, Mapping):
            raise ModelBindingError("DSPy returned a non-mapping model response")
        usage = self.usage.add_openai_response(payload)
        if self._spend_guard is not None:
            self._spend_guard.record_call(self.usage.price.estimate(usage))
        return response


class TrackedGEPALM(GEPALM):
    """Upstream GEPA LM transport with Reef's persistent accounting hooks."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        price: ModelPrice,
        spend_guard: CostGuard | None = None,
        usage_path: Path | None = None,
    ) -> None:
        super().__init__(
            f"openai/{model}",
            api_key=api_key,
            api_base=f"{base_url.rstrip('/')}/v1",
        )
        self.usage = UsageLedger(price, usage_path)
        self._spend_guard = spend_guard
        self._tracking_lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if self._spend_guard is not None:
            self._spend_guard.before_call()
        with self._tracking_lock:
            before_in = self.total_tokens_in
            before_out = self.total_tokens_out
            before_cost = self.total_cost
            result = super().__call__(prompt)
            usage = self._record_upstream_usage(before_in, before_out, requests=1)
            observed_cost = max(0.0, self.total_cost - before_cost)
        if self._spend_guard is not None:
            self._spend_guard.record_call(max(observed_cost, self.usage.price.estimate(usage)))
        return result

    def batch_complete(
        self,
        messages_list: list[list[dict[str, Any]]],
        max_workers: int = 10,
        **kwargs: Any,
    ) -> list[str]:
        if self._spend_guard is not None:
            for _ in messages_list:
                self._spend_guard.before_call()
        with self._tracking_lock:
            before_in = self.total_tokens_in
            before_out = self.total_tokens_out
            before_cost = self.total_cost
            results = super().batch_complete(messages_list, max_workers=max_workers, **kwargs)
            usage = self._record_upstream_usage(before_in, before_out, requests=len(results))
            observed_cost = max(0.0, self.total_cost - before_cost)
        if self._spend_guard is not None and results:
            per_request = max(observed_cost, self.usage.price.estimate(usage)) / len(results)
            for _ in results:
                self._spend_guard.record_call(per_request)
        return results

    def _record_upstream_usage(self, before_in: int, before_out: int, *, requests: int) -> TokenUsage:
        usage = empty_usage()
        usage["requests"] = requests
        usage["input_tokens"] = max(0, self.total_tokens_in - before_in)
        usage["output_tokens"] = max(0, self.total_tokens_out - before_out)
        self.usage.add(usage)
        return usage


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
