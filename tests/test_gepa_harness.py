"""Deterministic coverage for the GEPA harness-evolution reproduction."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "recipes" / "gepa" / "examples" / "aime_harness_evolve"


@pytest.fixture
def config_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.config")


@pytest.fixture
def adapter_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.adapter")


@pytest.fixture
def search_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.search")


@pytest.fixture
def runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("run", None)
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("run")


@pytest.fixture
def data_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.data")


@pytest.fixture
def evidence_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.evidence")


@pytest.fixture
def models_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.models")


@pytest.fixture
def reference_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.reference")


@pytest.fixture
def budget_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.budget")


@pytest.fixture
def heldout_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.heldout")


@pytest.fixture
def publication_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.publication")


@pytest.fixture
def reporting_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.reporting")


def test_reproduction_defaults_are_exact_and_secret_free(config_module):
    config = config_module.ExperimentConfig()

    assert config_module.REEF_COMMIT == "6a5c88f0dceaa5113b3fcf75c87385e0bb3d6253"
    assert config_module.GEPA_COMMIT == "67da814e33328e6714c3636428d03c86adb66cd7"
    assert config_module.PI_VERSION == "0.84.2"
    assert config_module.TASK_MODEL == "gpt-4.1-mini-2025-04-14"
    assert config_module.REFLECTION_MODEL == "gpt-5.1-2025-11-13"
    assert config_module.SEARCH_BUDGET == 500
    assert config_module.EXPERIMENT_SEEDS == (0,)
    assert config_module.TASK_TEMPERATURE == 1.0
    assert config_module.TASK_MAX_TOKENS == 32_000
    assert config_module.MAX_WORKERS == 32
    assert config_module.AIME_SPLIT_SIZES == {"train": 45, "validation": 45, "test": 30}
    assert config_module.AIME_DATASET_SHA256 == "0ee1433b0a5ecc4e7875004af026662a9137eb6ff30b8ffb081f139713e9c2e9"
    assert set(asdict(config)) == {
        "task_model",
        "reflection_model",
        "base_url",
        "max_metric_calls",
        "seeds",
        "api_key_env",
    }


def test_launcher_requires_the_locked_example_environment():
    launcher = (EXAMPLE_DIR / "run.sh").read_text(encoding="utf-8")

    assert "uv run" in launcher
    assert "--locked" in launcher


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_metric_calls": 0}, "positive"),
        ({"seeds": (0, 0)}, "unique"),
        ({"base_url": "api.openai.com"}, "HTTP"),
    ],
)
def test_reproduction_config_rejects_ambiguous_inputs(config_module, overrides, message):
    with pytest.raises(ValueError, match=message):
        config_module.ExperimentConfig(**overrides)


def test_rules_adapter_renders_binding_and_scores_a_realistic_pi_trace(adapter_module):
    from reef.harness.adapters import get_adapter
    from reef.harness.episode import EpisodeResult
    from reef.harness.model_binding import ModelBinding

    calls = []

    class Guard:
        def __init__(self):
            self.events = []

        def before_call(self):
            self.events.append("before")

        def record_call(self, observed_cost_usd):
            self.events.append(("record", observed_cost_usd))

    def run(descriptor, files, prompt, **kwargs):
        calls.append((descriptor, files, prompt, kwargs))
        return EpisodeResult(
            exit_code=0,
            stdout="",
            stderr="",
            trajectory=(
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Reasoning\n### 42"}],
                    }
                },
            ),
            residue=(),
        )

    binding = ModelBinding("http://model.test", "task-model", api_key="dummy")
    guard = Guard()
    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=binding,
        binary="fake-pi",
        episode_runner=run,
        spend_guard=guard,
    )

    evaluated = adapter.evaluate(
        [{"input": "What is six times seven?", "answer": "### 42"}],
        {"rules": "Solve carefully and end with ### <answer>."},
        capture_traces=True,
    )

    assert evaluated.scores == [1.0]
    assert evaluated.num_metric_calls == 1
    assert evaluated.outputs[0]["assistant_response"] == "Reasoning\n### 42"
    assert evaluated.outputs[0]["usage"]["requests"] == 0
    assert evaluated.trajectories[0]["expected_answer"] == "### 42"
    _, files, prompt, kwargs = calls[0]
    assert prompt == "What is six times seven?"
    assert kwargs == {"binary": "fake-pi", "timeout": 600.0}
    assert files["pi-agent/AGENTS.md"] == "Solve carefully and end with ### <answer>.\n"
    models = json.loads(files["pi-agent/models.json"])
    assert models["providers"]["reef"]["baseUrl"] == "http://model.test/v1"
    assert models["providers"]["reef"]["models"] == [{"id": "task-model"}]
    assert guard.events == ["before", ("record", 0.0)]


def test_rules_adapter_evaluates_batch_with_configured_concurrency(adapter_module):
    import threading

    from reef.harness.adapters import get_adapter
    from reef.harness.episode import EpisodeResult
    from reef.harness.model_binding import ModelBinding

    barrier = threading.Barrier(2)
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def run(descriptor, files, prompt, **kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait(timeout=5)
            answer = "1" if prompt == "first" else "2"
            return EpisodeResult(
                exit_code=0,
                stdout="",
                stderr="",
                trajectory=({"role": "assistant", "content": f"### {answer}"},),
                residue=(),
            )
        finally:
            with lock:
                active -= 1

    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model"),
        max_workers=2,
        episode_runner=run,
    )

    evaluated = adapter.evaluate(
        [
            {"input": "first", "answer": "### 1"},
            {"input": "second", "answer": "### 2"},
        ],
        {"rules": "seed"},
        capture_traces=True,
    )

    assert maximum_active == 2
    assert evaluated.scores == [1.0, 1.0]
    assert [trajectory["input"] for trajectory in evaluated.trajectories] == ["first", "second"]


def test_rules_adapter_builds_component_specific_reflection_records(adapter_module):
    from gepa.core.adapter import EvaluationBatch

    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model"),
        episode_runner=lambda *args, **kwargs: None,
    )
    batch = EvaluationBatch(
        outputs=[
            {
                "assistant_response": "### 1",
                "exit_code": 0,
                "stderr": "",
                "residue": [],
                "usage": {
                    "requests": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                },
            }
        ],
        scores=[0.0],
        trajectories=[
            {
                "input": "task",
                "expected_answer": "### 2",
                "assistant_response": "### 1",
                "feedback": "expected 2",
                "exit_code": 0,
                "stderr": "",
                "residue": [],
                "events": [{"role": "assistant", "content": "### 1"}],
                "usage": {
                    "requests": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                },
            }
        ],
    )

    reflective = adapter.make_reflective_dataset({"rules": "seed"}, batch, ["rules"])

    assert reflective == {
        "rules": [
            {
                "Inputs": "task",
                "Generated Outputs": "### 1",
                "Feedback": "expected 2",
                "Component role": "global rules loaded for every episode",
                "Harness trajectory": [{"role": "assistant", "content": "### 1"}],
            }
        ]
    }


def test_rules_cell_driver_persists_usage_before_search(runner_module, tmp_path, monkeypatch):
    class SearchReached(RuntimeError):
        pass

    def stop_before_search(**kwargs):
        assert kwargs["adapter"].usage.path == (tmp_path / "task-usage.json").resolve()
        assert kwargs["adapter"].max_workers == runner_module.MAX_WORKERS
        assert kwargs["heldout_evaluator"].max_workers == runner_module.HELDOUT_WORKERS
        raise SearchReached

    monkeypatch.setattr(runner_module, "run_sealed_search", stop_before_search)

    with pytest.raises(SearchReached):
        runner_module.run_reef_search(
            runner_module.ExperimentConfig(seeds=(0,)),
            [{"input": "train", "answer": "### 1"}],
            [{"input": "validation", "answer": "### 1"}],
            [{"input": "test", "answer": "### 1"}],
            8,
            0,
            tmp_path,
            "dummy",
            "fake-pi",
            "rules",
            None,
            False,
            "identity",
        )


def test_dataset_loader_rejects_same_size_content_drift(data_module, monkeypatch):
    import datasets

    def load_dataset(name, *args, **kwargs):
        if name == "AI-MO/aimo-validation-aime":
            return [{"problem": f"source-{index}", "solution": "solution", "answer": 1} for index in range(90)]
        return [{"problem": f"test-{index}", "answer": 1} for index in range(30)]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    with pytest.raises(RuntimeError, match="content changed"):
        data_module.load_aime_splits()


def test_official_aime_metric_accepts_only_a_bare_integer(reference_module):
    class Prediction:
        answer = "42"

    example = {
        "input": "problem",
        "answer": "### 42",
        "additional_context": {"solution": "full solution"},
    }

    score, feedback = reference_module.math_metric(example, Prediction())

    assert score == 1.0
    assert "full step-by-step solution" in feedback
    Prediction.answer = "The answer is 42"
    score, feedback = reference_module.math_metric(example, Prediction())
    assert score == 0.0
    assert "valid integer and nothing else" in feedback


def test_run_identity_refuses_incompatible_resume(runner_module, tmp_path):
    path = tmp_path / "run-identity.json"
    digest = runner_module.ensure_run_identity(path, {"schema_version": 1, "smoke": False})

    assert runner_module.ensure_run_identity(path, {"schema_version": 1, "smoke": False}) == digest
    with pytest.raises(RuntimeError, match="does not match"):
        runner_module.ensure_run_identity(path, {"schema_version": 1, "smoke": True})


def test_completed_cell_requires_matching_identity_and_artifacts(runner_module, tmp_path):
    run_dir = tmp_path / "rules" / "seed-0"
    run_dir.mkdir(parents=True)
    for name in ("summary.json", "config.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    repository = run_dir / "artifacts.git"
    repository.mkdir()
    (run_dir / "publication.json").write_text(json.dumps({"repository": str(repository)}) + "\n")
    runner_module.mark_done(run_dir, "rules", 0, "identity")

    assert runner_module.completed_cell(run_dir, "rules", 0, "identity")
    with pytest.raises(RuntimeError, match="does not match"):
        runner_module.completed_cell(run_dir, "rules", 0, "different")
    (run_dir / "summary.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence changed"):
        runner_module.completed_cell(run_dir, "rules", 0, "identity")


def test_evidence_export_is_complete_scrubbed_and_checksummed(evidence_module, config_module, monkeypatch, tmp_path):
    assert evidence_module.SEEDS == config_module.EXPERIMENT_SEEDS
    output_root = tmp_path / "full"
    output_root.mkdir()
    identity = "identity-sha"
    root_payloads = {
        "run-identity.json": {"sha256": identity},
        "plan.json": {"kind": "full"},
        "dataset-manifest.json": {"sha256": "dataset"},
        "dataset.json": {"train": []},
        "observed-cost.json": {"observed_cost_usd": 1.0},
        "results.json": {"cells": {}},
    }
    for name, payload in root_payloads.items():
        (output_root / name).write_text(json.dumps(payload) + "\n")
    secret = "sk-test-1234567890abcdef"
    for cell in evidence_module.CELLS:
        for seed in evidence_module.SEEDS:
            run_dir = output_root / cell / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text("{}\n")
            (run_dir / "config.json").write_text("{}\n")
            (run_dir / "events.jsonl").write_text(json.dumps({"authorization": f"Bearer {secret}"}) + "\n")
            search_dir = run_dir / "search"
            search_dir.mkdir()
            (search_dir / "candidates.json").write_text("{}\n")
            (search_dir / "gepa_state.bin").write_bytes(b"excluded")
            if cell != "reference":
                (run_dir / "publication.json").write_text("{}\n")
                tree = run_dir / "published-composition"
                tree.mkdir()
                (tree / "AGENTS.md").write_text("provider-free\n")
            marker = {
                "complete": True,
                "cell": cell,
                "seed": seed,
                "run_identity_sha256": identity,
                "files": evidence_module._cell_evidence_hashes(run_dir),
            }
            (run_dir / "done.json").write_text(json.dumps(marker) + "\n")

    archive_path = tmp_path / "gepa-evidence.tar.gz"
    exported = evidence_module.export_evidence(output_root, archive_path, api_key=secret)

    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == exported["archive_sha256"]
    assert archive_path.with_name(f"{archive_path.name}.sha256").is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        assert "gepa-evidence/evidence-manifest.json" in names
        assert not any(name.endswith("gepa_state.bin") for name in names)
        events = archive.extractfile("gepa-evidence/reference/seed-0/events.jsonl").read().decode()
        assert secret not in events
        assert "Bearer" not in events

    orphan_archive = tmp_path / "orphan-recovery.tar.gz"
    orphan_sidecar = orphan_archive.with_name(f"{orphan_archive.name}.sha256")
    orphan_sidecar.write_text("interrupted\n")
    evidence_module.export_evidence(output_root, orphan_archive, api_key=secret)
    assert orphan_archive.is_file()
    assert orphan_sidecar.is_file()

    failed_archive = tmp_path / "failed.tar.gz"

    def fail_archive(*args, **kwargs):
        raise RuntimeError("archive interrupted")

    with monkeypatch.context() as failure:
        failure.setattr(evidence_module.tarfile, "open", fail_archive)
        with pytest.raises(RuntimeError, match="archive interrupted"):
            evidence_module.export_evidence(output_root, failed_archive, api_key=secret)
    assert not failed_archive.exists()
    assert not failed_archive.with_name(f"{failed_archive.name}.sha256").exists()

    failed_pair = tmp_path / "failed-pair.tar.gz"
    replace = Path.replace

    def interrupt_archive_publish(path, target):
        if Path(target) == failed_pair:
            raise RuntimeError("archive publication interrupted")
        return replace(path, target)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "replace", interrupt_archive_publish)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            evidence_module.export_evidence(output_root, failed_pair, api_key=secret)
    assert not failed_pair.exists()
    assert not failed_pair.with_name(f"{failed_pair.name}.sha256").exists()

    (output_root / "reference" / "seed-0" / "summary.json").write_text('{"changed": true}\n')
    with pytest.raises(RuntimeError, match="files changed after completion"):
        evidence_module.export_evidence(output_root, tmp_path / "changed.tar.gz", api_key=secret)


def test_evidence_export_refuses_missing_runs(evidence_module, tmp_path):
    output_root = tmp_path / "incomplete"
    output_root.mkdir()
    (output_root / "run-identity.json").write_text('{"sha256":"identity"}\n')

    with pytest.raises(RuntimeError, match="not complete"):
        evidence_module.export_evidence(output_root, tmp_path / "evidence.tar.gz", api_key="")


@pytest.mark.parametrize(
    ("response", "expected", "score"),
    [
        ("work\n### 17", "### 17", 1.0),
        ("### 17\nmore text", "### 17", 1.0),
        ("The answer is 17", "### 17", 0.0),
        ("### 16", "### 17", 0.0),
    ],
)
def test_aime_scorer_requires_an_exact_marker_line(adapter_module, response, expected, score):
    assert adapter_module.score_aime_answer(expected, response) == score


def gepa_result(*, candidates, scores, fronts):
    from gepa.core.result import GEPAResult

    return GEPAResult(
        candidates=candidates,
        parents=[[None], *[[0] for _ in candidates[1:]]],
        val_aggregate_scores=scores,
        val_subscores=[{} for _ in candidates],
        per_val_instance_best_candidates=fronts,
        discovery_eval_counts=list(range(len(candidates))),
    )


def test_reference_driver_matches_current_official_aime_config(runner_module, monkeypatch, tmp_path):
    from gepa.core.adapter import EvaluationBatch
    from gepa.core.result import GEPAResult

    observed = {}

    class Usage:
        def snapshot(self):
            return {
                "requests": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }

    class Model:
        usage = Usage()

    class HeldoutAdapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            return EvaluationBatch(outputs=[{"candidate": candidate}], scores=[float(candidate == "better")])

    result = GEPAResult(
        candidates=[{"current_candidate": runner_module.OFFICIAL_SEED_PROMPT}, {"current_candidate": "better"}],
        parents=[[None], [0]],
        val_aggregate_scores=[0.0, 1.0],
        val_subscores=[{}, {}],
        per_val_instance_best_candidates={"validation": {1}},
        discovery_eval_counts=[0, 2],
        total_metric_calls=8,
        _str_candidate_key="current_candidate",
    )

    def optimize_anything(**kwargs):
        observed["optimization"] = kwargs
        return result

    monkeypatch.setattr(runner_module, "TrackedDSPyLM", lambda *args, **kwargs: Model())
    monkeypatch.setattr(runner_module, "TrackedGEPALM", lambda *args, **kwargs: Model())
    monkeypatch.setattr(runner_module.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(runner_module, "optimize_anything", optimize_anything)
    monkeypatch.setattr(runner_module, "OfficialAIMEAdapter", HeldoutAdapter)
    monkeypatch.setattr(runner_module, "write_search_report", lambda **kwargs: observed.setdefault("report", kwargs))
    monkeypatch.setattr(runner_module, "mark_done", lambda *args: observed.setdefault("done", args))

    runner_module.run_reference(
        runner_module.ExperimentConfig(seeds=(0,)),
        [{"input": "train", "answer": "### 1"}],
        [{"input": "validation", "answer": "### 1"}],
        [{"input": "test", "answer": "### 1"}],
        8,
        0,
        tmp_path,
        "dummy",
        None,
        32,
        "identity",
    )

    call = observed["optimization"]
    config = call["config"]
    assert call["seed_candidate"] == runner_module.OFFICIAL_SEED_PROMPT
    assert config.engine == "gepa"
    assert config.max_evals == 8
    assert config.max_concurrency == 32
    assert config.engine_config["engine"] == {
        "seed": 0,
        "track_best_outputs": True,
        "parallel": True,
        "max_workers": 32,
        "cache_evaluation": True,
    }
    assert config.engine_config["reflection"]["reflection_lm"] is not None
    assert observed["report"]["outcome"].selected_test_score == 1.0
    assert observed["done"][-1] == "identity"


def test_pareto_specialists_are_retained_without_bypassing_the_promotion_gate(search_module):
    result = gepa_result(
        candidates=[{"rules": "seed"}, {"rules": "algebra specialist"}, {"rules": "geometry specialist"}],
        scores=[0.5, 0.5, 0.5],
        fronts={"algebra": {1}, "geometry": {2}},
    )

    assert search_module.pareto_candidate_indices(result) == (1, 2)
    decision = search_module.decide_promotion(result)
    assert decision.selected is False
    assert decision.candidate_idx == 0
    assert "no candidate strictly improved" in decision.reason


def test_promotion_gate_selects_only_a_strict_validation_improvement(search_module):
    result = gepa_result(
        candidates=[{"rules": "seed"}, {"rules": "better"}],
        scores=[0.25, 0.75],
        fronts={"one": {1}, "two": {1}},
    )

    decision = search_module.decide_promotion(result)

    assert decision.selected is True
    assert decision.candidate_idx == 1
    assert decision.seed_score == 0.25
    assert decision.candidate_score == 0.75


def test_test_split_is_unsealed_only_after_upstream_search_returns(search_module, monkeypatch, tmp_path):
    from gepa.core.adapter import EvaluationBatch

    events = []
    trainset = [{"input": "train", "answer": "### 1"}]
    valset = [{"input": "validation", "answer": "### 2"}]
    testset = [{"input": "sealed-test", "answer": "### 3"}]

    def optimize(**kwargs):
        events.append(("optimize", kwargs["trainset"], kwargs["valset"]))
        assert all(example["input"] != "sealed-test" for example in kwargs["trainset"] + kwargs["valset"])
        return gepa_result(
            candidates=[{"rules": "seed"}, {"rules": "better"}],
            scores=[0.0, 1.0],
            fronts={0: {1}},
        )

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            events.append(("evaluate", batch, candidate))
            return EvaluationBatch(
                outputs=[{} for _ in batch],
                scores=[1.0 if candidate["rules"] == "better" else 0.0 for _ in batch],
            )

    monkeypatch.setattr(search_module.gepa, "optimize", optimize)
    outcome = search_module.run_sealed_search(
        seed_candidate={"rules": "seed"},
        trainset=trainset,
        valset=valset,
        testset=testset,
        adapter=Adapter(),
        reflection_lm=None,
        custom_candidate_proposer=lambda *_: {"rules": "better"},
        max_metric_calls=8,
        seed=0,
        run_dir=tmp_path / "run",
    )

    assert [event[0] for event in events] == ["optimize", "evaluate", "evaluate"]
    assert events[1][1] == testset
    assert events[2][1] == testset
    assert outcome.promotion.selected is True
    assert outcome.frozen_test_score == 0.0
    assert outcome.selected_test_score == 1.0


def test_search_forwards_official_gepa_semantics(search_module, monkeypatch, tmp_path):
    from gepa.core.adapter import EvaluationBatch

    policies = []

    def optimize(**kwargs):
        policies.append((kwargs["skip_perfect_score"], kwargs["frontier_type"], kwargs["cache_evaluation"]))
        return gepa_result(candidates=[{"rules": "seed"}], scores=[0.0], fronts={0: {0}})

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            return EvaluationBatch(outputs=[{} for _ in batch], scores=[0.0 for _ in batch])

    monkeypatch.setattr(search_module.gepa, "optimize", optimize)
    common = {
        "seed_candidate": {"rules": "seed"},
        "trainset": [{"input": "train", "answer": "### 1"}],
        "valset": [{"input": "validation", "answer": "### 2"}],
        "testset": [{"input": "test", "answer": "### 3"}],
        "adapter": Adapter(),
        "reflection_lm": None,
        "max_metric_calls": 8,
        "seed": 0,
    }

    search_module.run_sealed_search(**common, run_dir=tmp_path / "full")
    search_module.run_sealed_search(
        **common,
        run_dir=tmp_path / "smoke",
        skip_perfect_score=False,
    )

    assert policies == [(False, "hybrid", True), (False, "hybrid", True)]


def test_smoke_policy_reflects_on_a_perfect_training_minibatch(search_module, tmp_path):
    from gepa.core.adapter import EvaluationBatch

    reflection_prompts = []

    class PerfectAdapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            trajectories = [{"input": item["input"]} for item in batch] if capture_traces else None
            return EvaluationBatch(
                outputs=[{"rules": candidate["rules"]} for _ in batch],
                scores=[1.0 for _ in batch],
                trajectories=trajectories,
                objective_scores=[{"score": 1.0} for _ in batch],
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {
                component: [{"Feedback": "refine even though this sample passed"}]
                for component in components_to_update
            }

    def reflection_lm(prompt):
        reflection_prompts.append(prompt)
        return "refined rules"

    outcome = search_module.run_sealed_search(
        seed_candidate={"rules": "seed rules"},
        trainset=[{"input": f"train-{index}", "answer": "### 1"} for index in range(3)],
        valset=[{"input": "validation", "answer": "### 1"}],
        testset=[{"input": "test", "answer": "### 1"}],
        adapter=PerfectAdapter(),
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        seed=0,
        run_dir=tmp_path / "perfect-smoke",
        skip_perfect_score=False,
    )

    assert reflection_prompts
    assert outcome.result.candidates == [{"rules": "seed rules"}]


def test_pinned_gepa_checkpoint_resumes_without_replaying_work(search_module, tmp_path):
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch

    class DeterministicAdapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            improved = candidate["rules"] == "improved"
            trajectories = [{"input": item["input"]} for item in batch] if capture_traces else None
            return EvaluationBatch(
                outputs=[{"improved": improved} for _ in batch],
                scores=[1.0 if improved else 0.0 for _ in batch],
                trajectories=trajectories,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {component: [{"Feedback": "use improved"}] for component in components_to_update}

    adapter = DeterministicAdapter()
    dataset = [{"input": "deterministic"}]
    run_dir = tmp_path / "resume"

    def proposer(candidate, reflective, components):
        return {components[0]: "improved"}

    first = optimize(
        seed_candidate={"rules": "seed"},
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=None,
        custom_candidate_proposer=proposer,
        reflection_minibatch_size=1,
        max_metric_calls=4,
        run_dir=str(run_dir),
        seed=0,
        skip_perfect_score=False,
    )
    resumed = optimize(
        seed_candidate={"rules": "seed"},
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=None,
        custom_candidate_proposer=proposer,
        reflection_minibatch_size=1,
        max_metric_calls=0,
        run_dir=str(run_dir),
        seed=0,
        skip_perfect_score=False,
    )

    assert first.num_candidates == 2
    assert first.best_candidate == {"rules": "improved"}
    assert resumed.candidates == first.candidates
    assert resumed.parents == first.parents
    assert resumed.total_metric_calls == first.total_metric_calls


def test_heldout_checkpoint_resumes_without_repeating_completed_examples(heldout_module, tmp_path):
    from gepa.core.adapter import EvaluationBatch

    calls = []

    class InterruptingAdapter:
        interrupted = False

        def evaluate(self, batch, candidate, capture_traces=False):
            assert len(batch) == 1
            assert capture_traces is False
            calls.append(batch[0]["input"])
            if batch[0]["input"] == "two" and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("simulated interruption")
            return EvaluationBatch(
                outputs=[{"candidate": candidate["rules"], "input": batch[0]["input"]}],
                scores=[float(batch[0]["answer"])],
                num_metric_calls=1,
            )

    batch = [
        {"input": "one", "answer": "1"},
        {"input": "two", "answer": "0"},
        {"input": "three", "answer": "1"},
    ]
    evaluator = heldout_module.CheckpointedHeldoutEvaluator(InterruptingAdapter(), tmp_path / "checkpoints")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluator.evaluate("selected", batch, {"rules": "candidate"})
    assert calls == ["one", "two"]

    resumed = evaluator.evaluate("selected", batch, {"rules": "candidate"})

    assert calls == ["one", "two", "two", "three"]
    assert resumed.scores == [1.0, 0.0, 1.0]
    assert resumed.num_metric_calls == 3
    assert evaluator.evaluate("selected", batch, {"rules": "candidate"}).scores == resumed.scores
    assert calls == ["one", "two", "two", "three"]
    with pytest.raises(RuntimeError, match="identity changed"):
        evaluator.evaluate("selected", batch, {"rules": "different candidate"})


def test_heldout_checkpoint_can_record_official_failure_scores(heldout_module, budget_module, tmp_path):
    from gepa.core.adapter import EvaluationBatch

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            if batch[0]["input"] == "fails":
                raise RuntimeError("provider failure")
            return EvaluationBatch(outputs=[{"answer": "1"}], scores=[1.0])

    evaluator = heldout_module.CheckpointedHeldoutEvaluator(
        Adapter(),
        tmp_path / "checkpoints",
        max_workers=2,
        failure_score=0.0,
    )
    result = evaluator.evaluate(
        "official",
        [{"input": "works", "answer": "1"}, {"input": "fails", "answer": "1"}],
        "prompt",
    )

    assert result.scores == [1.0, 0.0]
    assert result.outputs[1] == {"error": "RuntimeError"}

    class StopAdapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            raise budget_module.SpendCapReached("stop")

    stop_evaluator = heldout_module.CheckpointedHeldoutEvaluator(
        StopAdapter(),
        tmp_path / "stop-checkpoints",
        max_workers=2,
        failure_score=0.0,
    )
    with pytest.raises(budget_module.SpendCapReached, match="stop"):
        stop_evaluator.evaluate("official", [{"input": "one", "answer": "1"}], "prompt")
    assert not (tmp_path / "stop-checkpoints" / "official" / "example-0000.json").exists()


def test_multi_node_candidate_renders_rules_and_skill_as_one_composition(adapter_module):
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    adapter = adapter_module.ReefCompositionAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model"),
        components=adapter_module.MULTI_NODE_COMPONENTS,
        episode_runner=lambda *args, **kwargs: None,
    )

    files = adapter.render_candidate({"rules": "Global reasoning rules.", "skill": "# AIME skill\n\nProcedure."})

    assert files["pi-agent/AGENTS.md"] == "Global reasoning rules.\n"
    assert files["pi-agent/skills/aime-solver/SKILL.md"] == "# AIME skill\n\nProcedure.\n"
    assert "providers" not in json.loads(files["pi-agent/models.json"])


def test_multi_node_reflection_records_name_the_selected_node_role(adapter_module):
    from gepa.core.adapter import EvaluationBatch

    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    adapter = adapter_module.ReefCompositionAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model"),
        components=adapter_module.MULTI_NODE_COMPONENTS,
        episode_runner=lambda *args, **kwargs: None,
    )
    trajectory = {
        "input": "task",
        "expected_answer": "### 2",
        "assistant_response": "### 1",
        "feedback": "expected 2",
        "exit_code": 0,
        "stderr": "",
        "residue": [],
        "events": [],
        "usage": {
            "requests": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        },
    }
    batch = EvaluationBatch(outputs=[{}], scores=[0.0], trajectories=[trajectory])

    reflective = adapter.make_reflective_dataset(
        {"rules": "seed rules", "skill": "seed skill"},
        batch,
        ["skill"],
    )

    assert set(reflective) == {"skill"}
    assert reflective["skill"][0]["Component role"] == "skill node 'aime-solver'"


def test_pinned_gepa_round_robin_evolves_both_nodes_in_the_complete_tree(adapter_module):
    from gepa import optimize

    from reef.harness.adapters import get_adapter
    from reef.harness.episode import EpisodeResult
    from reef.harness.model_binding import ModelBinding

    seen_compositions = []

    def run(descriptor, files, prompt, **kwargs):
        rules = files.get("pi-agent/AGENTS.md", "")
        skill = files.get("pi-agent/skills/aime-solver/SKILL.md", "")
        seen_compositions.append((rules, skill))
        correct = "improved rules" in rules if prompt == "rules task" else "improved skill" in skill
        answer = "### 1" if correct else "### 0"
        return EpisodeResult(
            exit_code=0,
            stdout="",
            stderr="",
            trajectory=({"role": "assistant", "content": answer},),
            residue=(),
        )

    adapter = adapter_module.ReefCompositionAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model"),
        components=adapter_module.MULTI_NODE_COMPONENTS,
        binary="fake-pi",
        episode_runner=run,
    )
    dataset = [
        {"input": "rules task", "answer": "### 1"},
        {"input": "skill task", "answer": "### 1"},
    ]

    def proposer(candidate, reflective, components):
        component = components[0]
        return {component: f"improved {component}"}

    result = optimize(
        seed_candidate={"rules": "seed rules", "skill": "seed skill"},
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=None,
        custom_candidate_proposer=proposer,
        candidate_selection_strategy="current_best",
        module_selector="round_robin",
        reflection_minibatch_size=2,
        max_metric_calls=14,
        seed=0,
        skip_perfect_score=False,
    )

    assert result.num_candidates == 3
    assert result.best_candidate == {"rules": "improved rules", "skill": "improved skill"}
    assert result.val_aggregate_scores == [0.0, 0.5, 1.0]
    assert any("improved rules" in rules and "seed skill" in skill for rules, skill in seen_compositions)
    assert any("improved rules" in rules and "improved skill" in skill for rules, skill in seen_compositions)


def test_pi_usage_and_price_estimate_include_cached_and_reasoning_tokens(models_module):
    usage = models_module.trajectory_usage(
        [
            {
                "message": {
                    "role": "assistant",
                    "usage": {"input": 60, "cacheRead": 40, "output": 20, "reasoning": 5},
                }
            }
        ]
    )
    price = models_module.ModelPrice(
        input_per_million=1.0,
        cached_input_per_million=0.5,
        output_per_million=2.0,
        source="test",
    )

    assert usage == {
        "requests": 1,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 5,
    }
    assert price.estimate(usage) == pytest.approx(0.00012)


def test_tracked_chat_model_retains_api_usage_without_serializing_a_key(models_module):
    class Guard:
        def __init__(self):
            self.before = 0
            self.costs = []

        def before_call(self):
            self.before += 1

        def record_call(self, observed_cost_usd):
            self.costs.append(observed_cost_usd)

    class Binding:
        def complete(self, body):
            assert body == {"messages": [{"role": "user", "content": "prompt"}]}
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            }

    guard = Guard()
    model = models_module.TrackedChatModel(Binding(), price=models_module.REFLECTION_MODEL_PRICE, spend_guard=guard)

    assert model("prompt") == "answer"
    assert model.usage.snapshot() == {
        "requests": 1,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }
    assert guard.before == 1
    assert guard.costs == [pytest.approx(models_module.REFLECTION_MODEL_PRICE.estimate(model.usage.snapshot()))]


def test_tracked_dspy_lm_accounts_provider_calls_but_not_cache_hits(models_module, monkeypatch, tmp_path):
    from dspy.clients import lm as dspy_lm_module
    from dspy.clients.cache import Cache

    class Guard:
        def __init__(self):
            self.events = []

        def before_call(self):
            self.events.append("before")

        def record_call(self, observed_cost_usd):
            self.events.append(("record", observed_cost_usd))

    class Response:
        choices = []
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }

        def model_dump(self):
            return {"usage": self.usage}

        def __getitem__(self, key):
            if key == "choices":
                return self.choices
            raise KeyError(key)

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(models_module.dspy, "cache", Cache(False, True, None))
    monkeypatch.setattr(dspy_lm_module, "litellm_completion", completion)
    guard = Guard()
    path = tmp_path / "task-usage.json"
    model = models_module.TrackedDSPyLM(
        "task-model",
        api_key="must-not-persist",
        base_url="https://model.test",
        temperature=1.0,
        max_tokens=32_000,
        price=models_module.TASK_MODEL_PRICE,
        spend_guard=guard,
        usage_path=path,
    )

    model.forward(prompt="same prompt")
    model.forward(prompt="same prompt")

    assert len(calls) == 1
    request = calls[0]["request"]
    assert request["model"] == "openai/task-model"
    assert request["api_base"] == "https://model.test/v1"
    assert request["temperature"] == 1.0
    assert request["max_tokens"] == 32_000
    assert model.usage.snapshot() == {
        "requests": 1,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }
    assert guard.events == [
        "before",
        ("record", pytest.approx(models_module.TASK_MODEL_PRICE.estimate(model.usage.snapshot()))),
        "before",
    ]
    assert "must-not-persist" not in path.read_text(encoding="utf-8")


def test_usage_ledger_persists_tokens_and_rejects_pricing_drift(models_module, tmp_path):
    path = tmp_path / "task-usage.json"
    usage = {
        "requests": 1,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }
    ledger = models_module.UsageLedger(models_module.TASK_MODEL_PRICE, path)
    ledger.add(usage)

    resumed = models_module.UsageLedger(models_module.TASK_MODEL_PRICE, path)

    assert resumed.snapshot() == usage
    assert resumed.total_cost == pytest.approx(models_module.TASK_MODEL_PRICE.estimate(usage))
    with pytest.raises(ValueError, match="mismatched pricing"):
        models_module.UsageLedger(models_module.REFLECTION_MODEL_PRICE, path)


def test_observed_cost_ledger_persists_and_stops_before_the_next_call(budget_module, tmp_path):
    path = tmp_path / "observed-cost.json"
    ledger = budget_module.ObservedCostLedger(path, 1.0)

    ledger.before_call()
    ledger.record_call(0.6)
    resumed = budget_module.ObservedCostLedger(path, 1.0)
    resumed.before_call()
    resumed.record_call(0.5)

    assert resumed.observed_cost_usd == pytest.approx(1.1)
    assert resumed.completed_calls == 2
    with pytest.raises(budget_module.SpendCapReached, match="no new model call"):
        resumed.before_call()


def test_publication_uses_reef_release_identities_and_excludes_transient_model_binding(
    adapter_module, publication_module, monkeypatch, tmp_path
):
    from reef.artifact import ArtifactRef
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    observed = {}

    class Backend:
        def __init__(self, scenario, repository, **kwargs):
            observed["scenario"] = scenario
            observed["repository"] = repository

        def fork(self, metadata):
            observed["fork_metadata"] = metadata
            return ArtifactRef("parent-content", "parent-release", None)

        def metadata(self):
            return None

        def publish(self, artifact, expected_parent):
            observed["expected_parent"] = expected_parent
            observed["models"] = (artifact.local_path / "pi-agent" / "models.json").read_text()
            observed["metadata"] = dict(artifact.metadata)
            return ArtifactRef("selected-content", "selected-release", expected_parent.release_id)

    monkeypatch.setattr(publication_module, "GitLFSRepositoryBackend", Backend)
    adapter = adapter_module.ReefCompositionAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model", api_key="must-not-publish"),
        components=adapter_module.MULTI_NODE_COMPONENTS,
        episode_runner=lambda *args, **kwargs: None,
    )

    published = publication_module.publish_candidate(
        adapter=adapter,
        candidate={"rules": "rules", "skill": "skill"},
        output_dir=tmp_path / "result",
        scenario="gepa-test",
        metadata={"score": 1.0},
    )

    assert published.content_id == "selected-content"
    assert published.release_id == "selected-release"
    assert published.parent_release_id == "parent-release"
    assert json.loads(observed["models"]) == {}
    published_text = "".join(
        path.read_text() for path in (tmp_path / "result" / "published-composition").rglob("*") if path.is_file()
    )
    assert "must-not-publish" not in published_text
    assert observed["metadata"]["score"] == 1.0
    assert len(observed["metadata"]["reproduction_candidate_sha256"]) == 64
    assert len(observed["metadata"]["reproduction_render_sha256"]) == 64


def test_real_reef_git_lfs_publication_smoke(adapter_module, publication_module, tmp_path):
    if shutil.which("git-lfs") is None:
        pytest.skip("git-lfs is required for the durable Reef artifact smoke test")
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model", api_key="transient"),
        episode_runner=lambda *args, **kwargs: None,
    )

    published = publication_module.publish_candidate(
        adapter=adapter,
        candidate={"rules": "published rules"},
        output_dir=tmp_path / "durable",
        scenario="gepa-durable-smoke",
        metadata={"kind": "test"},
    )

    assert published.content_id.startswith("content:")
    assert len(published.release_id) == 40
    assert len(published.parent_release_id) == 40
    assert Path(published.repository).is_dir()
    assert (tmp_path / "durable" / "publication.json").is_file()

    (tmp_path / "durable" / "publication.json").unlink()
    recovered = publication_module.publish_candidate(
        adapter=adapter,
        candidate={"rules": "published rules"},
        output_dir=tmp_path / "durable",
        scenario="gepa-durable-smoke",
        metadata={"kind": "test"},
    )

    assert recovered.content_id == published.content_id
    assert recovered.release_id == published.release_id
    assert (tmp_path / "durable" / "publication.json").is_file()


def test_publication_recovers_from_pending_fork(adapter_module, publication_module, monkeypatch, tmp_path):
    from reef.artifact import ArtifactRef
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    state = {"metadata": None, "current": None, "publish_calls": 0}

    class Backend:
        def __init__(self, scenario, repository, **kwargs):
            Path(repository).mkdir(parents=True, exist_ok=True)

        def metadata(self):
            return state["metadata"]

        def current(self):
            return state["current"]

        def fork(self, metadata):
            state["metadata"] = dict(metadata)
            state["current"] = ArtifactRef("pending-content", "pending-release", "initial-release")
            return state["current"]

        def publish(self, artifact, expected_parent):
            state["publish_calls"] += 1
            if state["publish_calls"] == 1:
                raise RuntimeError("interrupted after fork")
            state["metadata"] = dict(artifact.metadata)
            state["current"] = ArtifactRef("published-content", "published-release", expected_parent.release_id)
            return state["current"]

    monkeypatch.setattr(publication_module, "GitLFSRepositoryBackend", Backend)
    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model", api_key="transient"),
        episode_runner=lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="interrupted after fork"):
        publication_module.publish_candidate(
            adapter=adapter,
            candidate={"rules": "published rules"},
            output_dir=tmp_path / "pending",
            scenario="gepa-pending-smoke",
            metadata={"kind": "test"},
        )
    recovered = publication_module.publish_candidate(
        adapter=adapter,
        candidate={"rules": "published rules"},
        output_dir=tmp_path / "pending",
        scenario="gepa-pending-smoke",
        metadata={"kind": "test"},
    )

    assert recovered.content_id == "published-content"
    assert recovered.release_id == "published-release"
    assert state["publish_calls"] == 2
    assert state["metadata"]["publication_state"] == "published"


def test_publication_tree_is_staged_atomically(adapter_module, publication_module, monkeypatch, tmp_path):
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model", api_key="transient"),
        episode_runner=lambda *args, **kwargs: None,
    )

    def interrupt_tree(root, files):
        next(iter(files.items()))
        (root / "partial.txt").write_text("partial")
        raise RuntimeError("interrupted tree render")

    monkeypatch.setattr(publication_module, "_write_tree", interrupt_tree)
    with pytest.raises(RuntimeError, match="interrupted tree render"):
        publication_module.publish_candidate(
            adapter=adapter,
            candidate={"rules": "published rules"},
            output_dir=tmp_path / "atomic",
            scenario="gepa-atomic-smoke",
            metadata={"kind": "test"},
        )

    assert not (tmp_path / "atomic" / "published-composition").exists()


def test_completed_cells_are_aggregated_without_hiding_negative_results(reporting_module, tmp_path):
    for seed, frozen_score, selected_score, promoted, cost in [
        (0, 0.2, 0.4, True, 1.5),
        (1, 0.5, 0.3, True, 2.5),
        (2, 0.1, 0.1, False, 1.0),
    ]:
        run_dir = tmp_path / "rules" / f"seed-{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "frozen_test_score": frozen_score,
                    "selected_test_score": selected_score,
                    "promotion": {"selected": promoted},
                    "estimated_cost_usd": {"total": cost},
                    "wall_time_s": 10 + seed,
                }
            )
        )

    reporting_module.write_aggregate_report(output_dir=tmp_path, cells=("rules",), seeds=(0, 1, 2))

    aggregate = json.loads((tmp_path / "results.json").read_text())
    rules = aggregate["cells"]["rules"]
    assert rules["frozen_test_score_mean"] == pytest.approx(0.8 / 3)
    assert rules["selected_test_score_mean"] == pytest.approx(0.8 / 3)
    assert rules["test_delta_mean"] == pytest.approx(0.0)
    assert rules["promotion_rate"] == pytest.approx(2 / 3)
    assert rules["estimated_cost_usd_total"] == pytest.approx(5.0)
    assert rules["runs"][1]["test_delta"] == pytest.approx(-0.2)
