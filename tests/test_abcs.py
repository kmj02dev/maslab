from copy import deepcopy

import pytest

from maslab import (
    Benchmark,
    Experiment,
    ExperimentConfig,
    GeminiAPIModel,
    HiddenBench,
    HuggingfaceModel,
    Model,
    NvidiaBuildAPIModel,
    Response,
    SessionConfig,
)
from maslab.testing import check_benchmark, check_model_backend


class IncompleteModel(Model):
    pass


class CompleteModel(Model):
    def respond(self, messages):
        return Response(
            prompt=deepcopy(list(messages)),
            content="answer",
        )


class IncompleteBenchmark(Benchmark):
    pass


class CompleteBenchmark(Benchmark):
    def build_prompts(self, task, prompt, num_agent=None):
        return ["system prompt"]

    def compute_metrics(self, task, history, consensus_threshold=1.0):
        return {"num_entries": len(history)}


class StructuralModel:
    name = "structural-model"

    def respond(self, messages):
        return Response(prompt=deepcopy(list(messages)), content="answer")


class StructuralBenchmark:
    tasks = [{"id": "task"}]
    seed = 0

    def build_prompts(self, task, prompt, num_agent=None):
        return ["system prompt"]

    def compute_metrics(self, task, history, consensus_threshold=1.0):
        return {}

    def build_result(self, task, history, metrics=None, opinion_history=None):
        return {}

    def aggregate_metrics(self, results):
        return {}


def test_incomplete_model_subclass_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract method"):
        IncompleteModel("incomplete")


def test_incomplete_benchmark_subclass_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract methods"):
        IncompleteBenchmark([])


def test_complete_subclasses_work_with_experiment_and_checkers():
    model = CompleteModel("complete")
    benchmark = CompleteBenchmark([{"id": "task"}])

    assert check_model_backend(model)
    assert check_benchmark(benchmark)

    result = Experiment(
        model,
        benchmark,
        prompt="system prompt",
        config=ExperimentConfig(session=SessionConfig(num_rounds=0)),
    ).run()

    assert result.results[0].metrics == {"num_entries": 1}


def test_experiment_rejects_structural_non_subclasses():
    benchmark = CompleteBenchmark([{"id": "task"}])

    with pytest.raises(TypeError, match="model must inherit"):
        Experiment(StructuralModel(), benchmark, prompt="system prompt")
    with pytest.raises(TypeError, match="benchmark must inherit"):
        Experiment(
            CompleteModel("complete"),
            StructuralBenchmark(),
            prompt="system prompt",
        )


def test_checkers_reject_structural_non_subclasses():
    with pytest.raises(AssertionError, match="inherit from maslab.Model"):
        check_model_backend(StructuralModel())
    with pytest.raises(AssertionError, match="inherit from maslab.Benchmark"):
        check_benchmark(StructuralBenchmark())


def test_built_in_components_inherit_from_abstract_bases():
    assert issubclass(HuggingfaceModel, Model)
    assert issubclass(NvidiaBuildAPIModel, Model)
    assert issubclass(GeminiAPIModel, Model)
    assert issubclass(HiddenBench, Benchmark)
