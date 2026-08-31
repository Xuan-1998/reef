"""Deterministic coverage for the GEPA harness-evolution reproduction."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "recipes" / "harness_evolve" / "examples" / "gepa"


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
def models_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.models")


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

    assert config_module.REEF_COMMIT == "8e2fcc30f81bc476e5f98e7dcaa37c2d879d8201"
    assert config_module.GEPA_COMMIT == "92dadfffbe98c8ecf508179a1cab09c1bb85cd32"
    assert config_module.PI_VERSION == "0.84.2"
    assert config_module.TASK_MODEL == "gpt-4.1-mini-2025-04-14"
    assert config_module.REFLECTION_MODEL == "gpt-5-2025-08-07"
    assert config_module.SEARCH_BUDGET == 150
    assert config_module.EXPERIMENT_SEEDS == (0, 1, 2)
    assert config_module.AIME_SPLIT_SIZES == {"train": 45, "validation": 45, "test": 150}
    assert set(asdict(config)) == {
        "task_model",
        "reflection_model",
        "base_url",
        "max_metric_calls",
        "seeds",
        "api_key_env",
    }


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
    adapter = adapter_module.ReefRulesAdapter(
        descriptor=get_adapter("pi"),
        task_model=binding,
        binary="fake-pi",
        episode_runner=run,
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

    model = models_module.TrackedChatModel(Binding(), price=models_module.REFLECTION_MODEL_PRICE)

    assert model("prompt") == "answer"
    assert model.usage.snapshot() == {
        "requests": 1,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }


def test_publication_uses_reef_versions_and_excludes_transient_model_binding(
    adapter_module, publication_module, monkeypatch, tmp_path
):
    from reef.artifacts.artifact import ArtifactRef
    from reef.harness.adapters import get_adapter
    from reef.harness.model_binding import ModelBinding

    observed = {}

    class Backend:
        def __init__(self, scenario, repository, **kwargs):
            observed["scenario"] = scenario
            observed["repository"] = repository

        def fork(self, metadata):
            observed["fork_metadata"] = metadata
            return ArtifactRef("parent", "parent-version", None)

        def publish(self, artifact, expected_parent):
            observed["expected_parent"] = expected_parent
            observed["models"] = (artifact.local_path / "pi-agent" / "models.json").read_text()
            observed["metadata"] = dict(artifact.metadata)
            return ArtifactRef("selected", "selected-version", expected_parent.version)

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

    assert published.artifact_version == "selected-version"
    assert published.parent_artifact_version == "parent-version"
    assert json.loads(observed["models"]) == {}
    published_text = "".join(
        path.read_text() for path in (tmp_path / "result" / "published-composition").rglob("*") if path.is_file()
    )
    assert "must-not-publish" not in published_text
    assert observed["metadata"] == {"score": 1.0}


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

    assert len(published.artifact_version) == 40
    assert len(published.parent_artifact_version) == 40
    assert Path(published.repository).is_dir()
    assert (tmp_path / "durable" / "publication.json").is_file()


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
