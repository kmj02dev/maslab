"""Import boundaries and public exports for the separated packages."""

import importlib
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def run_isolated(source):
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("entrypoint", [
    "import maslab",
    "from maslab.core import Agent, Model, Response, MajorityVote",
    "from maslab.core.agents.agent import Agent; from maslab.core.multiagents import ParallelMultiagent; from maslab.core.aggregators import LLMAggregate",
    "from maslab.core.model import Model; from maslab.core.types import ChatMessage, Response, Usage",
])
def test_core_runs_with_experiments_adapters_and_optional_dependencies_blocked(entrypoint):
    code = """
import importlib.abc
from typing import get_type_hints

blocked = (
    "experiments", "benchmarks", "maslab.experiments", "maslab.utils",
    "maslab.models", "maslab.benchmarks", "maslab.core.config",
    "requests", "torch", "transformers", "accelerate",
)

class BlockNonCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Core imported a non-core module: " + fullname)

sys.meta_path.insert(0, BlockNonCore())
ENTRYPOINT
import maslab
from maslab.core import (
    Agent, Model, Response, LLMAggregate,
    MajorityVote, ParallelMultiagent, Pipeline, SequentialMultiagent, Transform,
)

class Echo(Model):
    def respond(self, messages):
        return Response(prompt=list(messages), content=self.name)

class Exclaim(Transform):
    def transform(self, response):
        response.content += "!"
        return response

assert "Experiment" not in dir(maslab)
assert maslab.Agent is Agent
assert get_type_hints(Agent.__init__)["model"] is Model
assert get_type_hints(Model.respond)["return"] is Response
assert get_type_hints(Agent.query)["return"] is Response
assert get_type_hints(Transform.transform)["return"] is Response
assert get_type_hints(SequentialMultiagent.query)["return"] is Response
assert get_type_hints(Pipeline.query)["return"] is Response
assert get_type_hints(ParallelMultiagent.query)["return"] == list[Response]
first = Agent("first", Echo("A"), use_context=False)
second = Agent("second", Echo("B"), use_context=False)
assert Pipeline([SequentialMultiagent([first, second]), Exclaim()]).query("question").content == "B!"
responses = ParallelMultiagent([first, second]).query("question")
assert all(isinstance(response, Response) for response in responses)
assert [response.content for response in responses] == ["A", "B"]
assert MajorityVote()(responses) == "A"
aggregator = LLMAggregate(Echo("final"))
assert aggregator(responses) == "final"
assert len(aggregator.history()) == 1
assert not any(
    loaded == name or loaded.startswith(name + ".")
    for loaded in sys.modules for name in blocked
)
print("isolated core passed")
"""
    assert "isolated core passed" in run_isolated(code.replace("ENTRYPOINT", entrypoint))


def test_direct_model_import_loads_only_the_selected_adapter():
    run_isolated("""
        import importlib.abc
        from types import SimpleNamespace

        class BlockUnselected(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = ("requests", "torch", "transformers", "maslab.experiments",
                           "maslab.benchmarks", "maslab.models.gemini_api",
                           "maslab.models.nvidia_build_api")
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("Unselected dependency loaded: " + fullname)

        sys.meta_path.insert(0, BlockUnselected())
        assert "maslab.models.huggingface" not in sys.modules
        from maslab.models import HuggingfaceModel
        model = HuggingfaceModel(
            "fake",
            tokenizer=SimpleNamespace(pad_token_id=0),
            model_instance=SimpleNamespace(device=None),
        )
        from maslab.models import HuggingfaceModel
        assert isinstance(model, HuggingfaceModel)
        assert "maslab.models.gemini_api" not in sys.modules
        assert "maslab.models.nvidia_build_api" not in sys.modules
    """)




@pytest.mark.parametrize(("public_module", "implementation_module", "names"), [
    ("maslab", "maslab.core", ["Agent", "Model", "Response", "Transform", "Pipeline"]),
    ("maslab.core", "maslab.core.transforms", ["Transform"]),
    ("maslab.core", "maslab.core.multiagents.pipeline", ["Pipeline"]),
    ("maslab.core", "maslab.core.types", ["ChatMessage", "Response", "Usage"]),
    ("maslab.core.agents", "maslab.core.agents.agent", ["Agent"]),
    ("maslab.core", "maslab.core.aggregators", ["Aggregate", "MajorityVote", "LLMAggregate"]),
    ("maslab.core", "maslab.core.multiagents", ["Multiagent", "SequentialMultiagent", "ParallelMultiagent"]),
])
def test_package_exports_resolve_to_the_implementation_classes(public_module, implementation_module, names):
    public = importlib.import_module(public_module)
    implementation = importlib.import_module(implementation_module)
    for name in names:
        assert getattr(public, name) is getattr(implementation, name)


def test_removed_compatibility_modules_cannot_be_imported():
    run_isolated("""
        import importlib

        removed_modules = (
            "maslab.base",
            "maslab.config",
            "maslab.types",
            "maslab.experiment",
            "maslab.prompts",
            "maslab.multiagent",
            "maslab.agents",
            "maslab.agents.agent",
            "maslab.agents._history",
            "maslab.multiagents",
            "maslab.multiagents.multiagent",
            "maslab.multiagents.sequential_multiagent",
            "maslab.multiagents.parallel_multiagent",
            "maslab.aggregators",
            "maslab.aggregators.aggregator",
            "maslab.aggregators.majority_vote",
            "maslab.aggregators.llm_aggregator",
        )
        for module_name in removed_modules:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                assert error.name == module_name or module_name.startswith(error.name + "."), error
            else:
                raise AssertionError("Compatibility module is still importable: " + module_name)

        from maslab import Agent, ParallelMultiagent, MajorityVote
        from maslab.core import Agent as CoreAgent
        assert Agent is CoreAgent
    """)


def test_public_packages_support_discovery_and_reject_unknown_names():
    for module_name in ("maslab", "maslab.core", "maslab.models"):
        module = importlib.import_module(module_name)
        assert set(module.__all__).issubset(dir(module))
        with pytest.raises(AttributeError):
            getattr(module, "missing_public_name")


def test_experiment_features_are_not_part_of_maslab():
    run_isolated("""
        import importlib
        import maslab
        for name in ("Experiment", "Session", "Config", "Benchmark", "HiddenBench", "PromptContext"):
            assert not hasattr(maslab, name)
        for name in ("maslab.experiments", "maslab.benchmarks"):
            try:
                importlib.import_module(name)
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError(name + " is still part of the library")
        import maslab.testing
    """)
