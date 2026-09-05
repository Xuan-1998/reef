"""Frozen-provider comparison: did test-time training pay for itself?

The issue's last scope item: run the same CORAL task with a frozen provider
under an equal attempt/token budget, and compare. Both arms produce a result
bundle (:func:`reef_coral.bundle.build_result_bundle`); this module holds the
comparison down to like-for-like budgets and states plainly when budgets
were NOT equal, instead of normalizing the difference away.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

#: Budgets are "equal" when the smaller arm spent at least this fraction of
#: the larger. Attempt counts must match exactly; tokens get this tolerance
#: because rollout lengths are stochastic even under identical settings.
DEFAULT_TOKEN_TOLERANCE = 0.10


def _best_score(bundle: Mapping[str, Any]) -> float | None:
    best = bundle.get("best_attempt")
    return None if best is None else best["score"]


def _scores(bundle: Mapping[str, Any]) -> list[float]:
    return [s["score"] for s in bundle["score_over_time"] if s["score"] is not None]


def _total_tokens(bundle: Mapping[str, Any]) -> int:
    accounting = bundle["token_accounting"]
    return accounting["prompt_tokens"] + accounting["completion_tokens"]


def compare_to_frozen(
    adaptive: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    token_tolerance: float = DEFAULT_TOKEN_TOLERANCE,
) -> dict[str, Any]:
    """Compare the adaptive (training) arm against the frozen-provider arm.

    Returns a JSON-serializable verdict:

    - ``budget``: attempt counts, token totals, and whether they qualify as
      equal. When they don't, ``comparable`` is False and the score deltas
      are still reported but flagged.
    - ``best_delta`` / ``final_delta``: adaptive minus frozen, on the best
      and the last scored attempt.
    - ``score_curves``: both curves, for plotting score-over-time side by side.
    - ``training_signal``: revision evidence that the adaptive arm actually
      trained (more than one revision served), since an adaptive arm that
      never updated is just a noisy frozen arm.
    """
    attempts_equal = adaptive["attempt_count"] == frozen["attempt_count"]
    tokens_adaptive, tokens_frozen = _total_tokens(adaptive), _total_tokens(frozen)
    larger = max(tokens_adaptive, tokens_frozen)
    tokens_equal = larger == 0 or (min(tokens_adaptive, tokens_frozen) / larger) >= (1 - token_tolerance)

    adaptive_scores, frozen_scores = _scores(adaptive), _scores(frozen)
    best_adaptive, best_frozen = _best_score(adaptive), _best_score(frozen)

    adaptive_revisions = adaptive.get("model_revisions_served", [])

    return {
        "budget": {
            "attempts": {"adaptive": adaptive["attempt_count"], "frozen": frozen["attempt_count"]},
            "tokens": {"adaptive": tokens_adaptive, "frozen": tokens_frozen},
            "attempts_equal": attempts_equal,
            "tokens_equal": tokens_equal,
        },
        "comparable": attempts_equal and tokens_equal,
        "best_delta": None if best_adaptive is None or best_frozen is None else best_adaptive - best_frozen,
        "final_delta": None if not adaptive_scores or not frozen_scores else adaptive_scores[-1] - frozen_scores[-1],
        "score_curves": {"adaptive": adaptive_scores, "frozen": frozen_scores},
        "training_signal": {
            "adaptive_revisions_served": adaptive_revisions,
            "weights_updated": len(adaptive_revisions) > 1,
        },
    }
