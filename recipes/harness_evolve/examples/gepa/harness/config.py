"""Pinned inputs shared by the GEPA reproduction runners and tests."""

from __future__ import annotations

from dataclasses import dataclass

REEF_COMMIT = "6a5c88f0dceaa5113b3fcf75c87385e0bb3d6253"
GEPA_COMMIT = "67da814e33328e6714c3636428d03c86adb66cd7"
PI_VERSION = "0.84.2"
TASK_MODEL = "gpt-4.1-mini-2025-04-14"
REFLECTION_MODEL = "gpt-5.1-2025-11-13"
OPENAI_BASE_URL = "https://api.openai.com"
SEARCH_BUDGET = 500
EXPERIMENT_SEEDS = (0,)
TASK_TEMPERATURE = 1.0
TASK_MAX_TOKENS = 32_000
MAX_WORKERS = 32
AIME_TRAIN_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"
AIME_TEST_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"
AIME_SPLIT_SIZES = {"train": 45, "validation": 45, "test": 30}
AIME_DATASET_SHA256 = "0ee1433b0a5ecc4e7875004af026662a9137eb6ff30b8ffb081f139713e9c2e9"


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
