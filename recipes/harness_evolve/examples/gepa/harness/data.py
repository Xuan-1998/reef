"""Pinned dataset and seed composition helpers."""

from __future__ import annotations

from typing import cast

from .adapter import AIMEExample
from .config import AIME_SPLIT_SIZES

RULES_SEED = (
    "You are a helpful assistant. Solve the math problem carefully and put the final answer "
    "on its own line in exactly the format '### <answer>'."
)
SKILL_SEED = """# AIME solver

Solve each competition-math problem from first principles. Check the result,
then put only the final value after `###` on the last answer line.
"""


def load_aime_splits() -> tuple[list[AIMEExample], list[AIMEExample], list[AIMEExample]]:
    """Load the exact splits supplied by the pinned upstream GEPA package."""
    from gepa.examples.aime import init_dataset

    trainset, valset, testset = init_dataset()
    observed = {"train": len(trainset), "validation": len(valset), "test": len(testset)}
    if observed != AIME_SPLIT_SIZES:
        raise RuntimeError(f"upstream AIME split sizes changed: observed {observed}, expected {AIME_SPLIT_SIZES}")
    return cast(list[AIMEExample], trainset), cast(list[AIMEExample], valset), cast(list[AIMEExample], testset)


def rules_seed() -> dict[str, str]:
    return {"rules": RULES_SEED}


def multi_node_seed() -> dict[str, str]:
    return {"rules": RULES_SEED, "skill": SKILL_SEED}
