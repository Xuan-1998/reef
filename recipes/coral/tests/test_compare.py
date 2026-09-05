"""Frozen-provider comparison (issue #3's last scope item)."""

from __future__ import annotations

from reef_coral.compare import compare_to_frozen


def _bundle(scores, *, tokens=1000, revisions=("rel-0",)):
    return {
        "attempt_count": len(scores),
        "score_over_time": [
            {"agent_id": "a", "commit_hash": f"c{i}", "parent_hash": None, "score": s, "status": "improved"}
            for i, s in enumerate(scores)
        ],
        "best_attempt": None
        if not [s for s in scores if s is not None]
        else {
            "agent_id": "a",
            "commit_hash": "cbest",
            "score": max(s for s in scores if s is not None),
            "parent_hash": None,
        },
        "model_revisions_served": list(revisions),
        "token_accounting": {
            "inference_calls": len(scores),
            "prompt_tokens": tokens,
            "completion_tokens": tokens // 10,
        },
        "attempts": [],
        "replay": {},
    }


def test_equal_budget_comparison_reports_deltas():
    adaptive = _bundle([0.2, 0.5, 0.8], revisions=("rel-0", "rel-1", "rel-2"))
    frozen = _bundle([0.2, 0.3, 0.4])
    verdict = compare_to_frozen(adaptive, frozen)
    assert verdict["comparable"] is True
    assert verdict["best_delta"] == 0.8 - 0.4
    assert verdict["final_delta"] == 0.8 - 0.4
    assert verdict["training_signal"]["weights_updated"] is True


def test_unequal_attempts_flagged_not_normalized():
    adaptive = _bundle([0.9, 0.9, 0.9, 0.9])
    frozen = _bundle([0.4])
    verdict = compare_to_frozen(adaptive, frozen)
    assert verdict["comparable"] is False
    assert verdict["budget"]["attempts_equal"] is False
    # deltas still present, consumer decides what they mean
    assert verdict["best_delta"] == 0.5


def test_token_budget_tolerance():
    adaptive = _bundle([0.5], tokens=1000)
    close = _bundle([0.4], tokens=950)
    far = _bundle([0.4], tokens=500)
    assert compare_to_frozen(adaptive, close)["budget"]["tokens_equal"] is True
    assert compare_to_frozen(adaptive, far)["budget"]["tokens_equal"] is False


def test_frozen_adaptive_arm_is_called_out():
    adaptive = _bundle([0.5, 0.6], revisions=("rel-0",))  # never trained
    frozen = _bundle([0.5, 0.6])
    verdict = compare_to_frozen(adaptive, frozen)
    assert verdict["training_signal"]["weights_updated"] is False


def test_unscored_runs_yield_null_deltas():
    verdict = compare_to_frozen(_bundle([None]), _bundle([None]))
    assert verdict["best_delta"] is None
    assert verdict["final_delta"] is None
