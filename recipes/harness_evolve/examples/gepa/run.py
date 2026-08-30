"""Run the pinned upstream, frozen, rules-only, and multi-node cells."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult

from harness.adapter import MULTI_NODE_COMPONENTS, ReefCompositionAdapter, ReefRulesAdapter, score_aime_answer
from harness.budget import ObservedCostLedger
from harness.callbacks import EvidenceCallback
from harness.config import AIME_SPLIT_SIZES, GEPA_COMMIT, PI_VERSION, REEF_COMMIT, ExperimentConfig
from harness.data import RULES_SEED, load_aime_splits, multi_node_seed, rules_seed
from harness.models import REFLECTION_MODEL_PRICE, TASK_MODEL_PRICE, TrackedChatModel
from harness.publication import publish_candidate
from harness.reporting import write_aggregate_report, write_search_report
from harness.search import run_sealed_search
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
CELLS = ("reference", "frozen", "rules", "multi")


class ExactAIMEEvaluator:
    def __call__(self, data: dict[str, Any], response: str) -> EvaluationResult:
        score = score_aime_answer(data["answer"], response)
        feedback = (
            f"Correct: the final answer exactly matched {data['answer']!r}."
            if score
            else f"Incorrect: the final answer line must be exactly {data['answer']!r}."
        )
        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=(*CELLS, "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ExperimentConfig().seeds))
    parser.add_argument("--budget", type=int, default=ExperimentConfig().max_metric_calls)
    parser.add_argument("--pi-binary", default=os.environ.get("REEF_PI_BINARY", "pi"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-observed-cost-usd",
        type=float,
        default=None,
        help="Required for live runs; stops before a new call after recorded cost reaches this value",
    )
    parser.add_argument("--smoke", action="store_true", help="Use two examples per split and an eight-call budget")
    parser.add_argument("--dry-run", action="store_true", help="Validate pins and print the plan without model calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(max_metric_calls=args.budget, seeds=tuple(args.seeds))
    reef_source = verify_reef_pin()
    verify_gepa_pin()
    selected_cells = CELLS if args.cell == "all" else (args.cell,)
    if any(cell != "reference" for cell in selected_cells):
        verify_pi_pin(args.pi_binary)

    output_root = args.output_dir or HERE / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = {
        "cells": selected_cells,
        "seeds": config.seeds,
        "budget": 8 if args.smoke else config.max_metric_calls,
        "output_dir": str(output_root),
        "task_model": config.task_model,
        "reflection_model": config.reflection_model,
        "api_key_env": config.api_key_env,
        "max_observed_cost_usd": args.max_observed_cost_usd,
        "pins": {
            "reef_base_commit": REEF_COMMIT,
            "reef_source_commit": reef_source["commit"],
            "reef_source_dirty": reef_source["dirty"],
            "gepa_commit": GEPA_COMMIT,
            "pi_version": PI_VERSION if any(cell != "reference" for cell in selected_cells) else None,
        },
        "smoke": args.smoke,
        "planned_task_evaluations": planned_task_evaluations(
            selected_cells,
            len(config.seeds),
            8 if args.smoke else config.max_metric_calls,
            2 if args.smoke else AIME_SPLIT_SIZES["test"],
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if args.max_observed_cost_usd is None or args.max_observed_cost_usd <= 0:
        raise SystemExit("--max-observed-cost-usd must be positive for a live run; no model calls were made")
    if reef_source["dirty"]:
        raise SystemExit("tracked Reef source has uncommitted changes; no model calls were made")
    api_key = os.environ.get(config.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{config.api_key_env} is not set; no live model calls were made")
    trainset, valset, testset = load_aime_splits()
    budget = config.max_metric_calls
    if args.smoke:
        trainset, valset, testset = trainset[:2], valset[:2], testset[:2]
        budget = 8

    output_root.mkdir(parents=True, exist_ok=True)
    ledger = ObservedCostLedger(output_root / "observed-cost.json", args.max_observed_cost_usd)
    (output_root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_dataset_manifest(output_root / "dataset-manifest.json", trainset, valset, testset)
    for cell in selected_cells:
        for seed in config.seeds:
            cell_dir = output_root / cell / f"seed-{seed}"
            cell_dir.mkdir(parents=True)
            if (cell_dir / "done.json").exists():
                print(f"skip completed cell {cell} seed {seed}: {cell_dir}")
                continue
            if cell == "reference":
                run_reference(config, trainset, valset, testset, budget, seed, cell_dir, api_key, ledger)
            elif cell == "frozen":
                run_frozen(config, testset, seed, cell_dir, api_key, args.pi_binary, ledger)
            else:
                run_reef_search(
                    config,
                    trainset,
                    valset,
                    testset,
                    budget,
                    seed,
                    cell_dir,
                    api_key,
                    args.pi_binary,
                    cell,
                    ledger,
                )
    # Full examples (including held-out answers) are retained only after every
    # requested search cell has finished or was already marked complete.
    write_dataset_artifact(output_root / "dataset.json", trainset, valset, testset)
    write_aggregate_report(output_dir=output_root, cells=selected_cells, seeds=config.seeds)


def run_reference(config, trainset, valset, testset, budget, seed, output_dir, api_key, ledger) -> None:
    task = TrackedChatModel(
        ModelBinding(config.base_url, config.task_model, api_key=api_key),
        price=TASK_MODEL_PRICE,
        spend_guard=ledger,
    )
    reflection = TrackedChatModel(
        ModelBinding(config.base_url, config.reflection_model, api_key=api_key),
        price=REFLECTION_MODEL_PRICE,
        spend_guard=ledger,
    )
    adapter = DefaultAdapter(model=task, evaluator=ExactAIMEEvaluator())
    callback = EvidenceCallback(output_dir / "events.jsonl")
    outcome = run_sealed_search(
        seed_candidate={"system_prompt": RULES_SEED},
        trainset=trainset,
        valset=valset,
        testset=testset,
        adapter=adapter,
        reflection_lm=reflection,
        max_metric_calls=budget,
        seed=seed,
        run_dir=output_dir / "search",
        callbacks=[callback],
    )
    write_search_report(
        output_dir=output_dir,
        cell="reference",
        seed=seed,
        outcome=outcome,
        config=report_config(config, budget, seed, "reference"),
        task_usage=task.usage.snapshot(),
        reflection_usage=reflection.usage.snapshot(),
        task_price=TASK_MODEL_PRICE,
        reflection_price=REFLECTION_MODEL_PRICE,
    )
    mark_done(output_dir)


def run_frozen(config, testset, seed, output_dir, api_key, pi_binary, ledger) -> None:
    adapter = ReefCompositionAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding(config.base_url, config.task_model, api_key=api_key),
        components=MULTI_NODE_COMPONENTS,
        binary=pi_binary,
        spend_guard=ledger,
    )
    candidate = multi_node_seed()
    started = datetime.now(timezone.utc)
    evaluated = adapter.evaluate(list(testset), candidate, capture_traces=False)
    score = sum(evaluated.scores) / len(evaluated.scores)
    usage = adapter.usage.snapshot()
    summary = {
        "cell": "frozen",
        "seed": seed,
        "test_score": score,
        "test_scores": evaluated.scores,
        "usage": {"task": usage},
        "estimated_cost_usd": TASK_MODEL_PRICE.estimate(usage),
        "pricing": asdict(TASK_MODEL_PRICE),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "config.json").write_text(
        json.dumps(report_config(config, 0, seed, "frozen"), indent=2, sort_keys=True) + "\n"
    )
    publish_candidate(
        adapter=adapter,
        candidate=candidate,
        output_dir=output_dir,
        scenario=f"gepa-frozen-{seed}",
        metadata={"cell": "frozen", "seed": seed, "test_score": score},
    )
    mark_done(output_dir)


def run_reef_search(
    config, trainset, valset, testset, budget, seed, output_dir, api_key, pi_binary, cell, ledger
) -> None:
    binding = ModelBinding(config.base_url, config.task_model, api_key=api_key)
    if cell == "rules":
        adapter = ReefRulesAdapter(
            descriptor=get_adapter("pi"), task_model=binding, binary=pi_binary, spend_guard=ledger
        )
        candidate = rules_seed()
    else:
        adapter = ReefCompositionAdapter(
            descriptor=get_adapter("pi"),
            task_model=binding,
            components=MULTI_NODE_COMPONENTS,
            binary=pi_binary,
            spend_guard=ledger,
        )
        candidate = multi_node_seed()
    reflection = TrackedChatModel(
        ModelBinding(config.base_url, config.reflection_model, api_key=api_key),
        price=REFLECTION_MODEL_PRICE,
        spend_guard=ledger,
    )
    callback = EvidenceCallback(output_dir / "events.jsonl")
    outcome = run_sealed_search(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        testset=testset,
        adapter=adapter,
        reflection_lm=reflection,
        max_metric_calls=budget,
        seed=seed,
        run_dir=output_dir / "search",
        callbacks=[callback],
    )
    write_search_report(
        output_dir=output_dir,
        cell=cell,
        seed=seed,
        outcome=outcome,
        config=report_config(config, budget, seed, cell),
        task_usage=adapter.usage.snapshot(),
        reflection_usage=reflection.usage.snapshot(),
        task_price=TASK_MODEL_PRICE,
        reflection_price=REFLECTION_MODEL_PRICE,
    )
    selected = outcome.result.candidates[outcome.promotion.candidate_idx]
    publish_candidate(
        adapter=adapter,
        candidate=selected,
        output_dir=output_dir,
        scenario=f"gepa-{cell}-{seed}",
        metadata={
            "cell": cell,
            "seed": seed,
            "promotion": asdict(outcome.promotion),
            "frozen_test_score": outcome.frozen_test_score,
            "selected_test_score": outcome.selected_test_score,
        },
    )
    mark_done(output_dir)


def report_config(config: ExperimentConfig, budget: int, seed: int, cell: str) -> dict[str, Any]:
    return {
        "cell": cell,
        "seed": seed,
        "max_metric_calls": budget,
        "task_model": config.task_model,
        "reflection_model": config.reflection_model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "gepa_commit": GEPA_COMMIT,
        "pi_version": PI_VERSION,
    }


def planned_task_evaluations(cells, seed_count: int, budget: int, test_size: int) -> int:
    per_seed = 0
    for cell in cells:
        per_seed += test_size if cell == "frozen" else budget + 2 * test_size
    return per_seed * seed_count


def verify_gepa_pin() -> None:
    direct_url = importlib.metadata.distribution("gepa").read_text("direct_url.json")
    if not direct_url:
        raise SystemExit("GEPA is not a direct source install; install this example's pinned dependency")
    metadata = json.loads(direct_url)
    installed = metadata.get("vcs_info", {}).get("commit_id")
    if installed != GEPA_COMMIT:
        raise SystemExit(f"GEPA commit is {installed!r}, expected {GEPA_COMMIT}")


def verify_reef_pin() -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    dirty = bool(_command_output(["git", "status", "--porcelain", "--untracked-files=no"]))
    base_ok = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REEF_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).returncode
    if base_ok != 0:
        raise SystemExit(f"current Reef source does not descend from pinned base {REEF_COMMIT}")
    return {"commit": commit, "dirty": dirty}


def verify_pi_pin(binary: str) -> None:
    resolved = shutil.which(binary) if not Path(binary).is_absolute() else binary
    if not resolved:
        raise SystemExit(f"Pi binary {binary!r} was not found; install version {PI_VERSION} or set REEF_PI_BINARY")
    completed = subprocess.run([resolved, "--version"], check=True, capture_output=True, text=True)
    installed = completed.stdout.strip()
    if installed != PI_VERSION:
        raise SystemExit(f"Pi version is {installed!r}, expected {PI_VERSION}")


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"runtime check failed for {command[0]!r}: {exc}") from exc


def write_dataset_manifest(path: Path, trainset, valset, testset) -> None:
    splits = {"train": trainset, "validation": valset, "test": testset}
    serialized = json.dumps(splits, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    payload = {
        "source": "gepa.examples.aime.init_dataset()",
        "gepa_commit": GEPA_COMMIT,
        "sha256": digest,
        "sizes": {name: len(items) for name, items in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dataset_artifact(path: Path, trainset, valset, testset) -> None:
    splits = {"train": trainset, "validation": valset, "test": testset}
    path.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mark_done(output_dir: Path) -> None:
    (output_dir / "done.json").write_text('{"complete": true}\n', encoding="utf-8")


if __name__ == "__main__":
    main()
