"""Pinned inputs shared by the GEPA reproduction runners and tests."""

from __future__ import annotations

from dataclasses import dataclass

REEF_COMMIT = "8e2fcc30f81bc476e5f98e7dcaa37c2d879d8201"
GEPA_COMMIT = "92dadfffbe98c8ecf508179a1cab09c1bb85cd32"
PI_VERSION = "0.84.2"
TASK_MODEL = "gpt-4.1-mini-2025-04-14"
REFLECTION_MODEL = "gpt-5-2025-08-07"
OPENAI_BASE_URL = "https://api.openai.com"
SEARCH_BUDGET = 150
EXPERIMENT_SEEDS = (0, 1, 2)
AIME_SPLIT_SIZES = {"train": 45, "validation": 45, "test": 150}


@dataclass(frozen=True)
class ExperimentConfig:
    """One reproducible GEPA search configuration.

    Credentials are deliberately absent. Live runners resolve a named
    environment variable at execution time and never serialize its value.
    """

    task_model: str = TASK_MODEL
    reflection_model: str = REFLECTION_MODEL
    base_url: str = OPENAI_BASE_URL
    max_metric_calls: int = SEARCH_BUDGET
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    api_key_env: str = "OPENAI_API_KEY"

    def __post_init__(self) -> None:
        if not self.task_model or not self.reflection_model:
            raise ValueError("task and reflection model names must be non-empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not self.api_key_env:
            raise ValueError("api_key_env must be non-empty")
