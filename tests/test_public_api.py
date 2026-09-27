import importlib

import pytest

import maslab


def test_public_api_exposes_framework_entry_points():
    assert maslab.__version__ == "0.1.0"
    assert maslab.Agent
    assert maslab.Response.__name__ == "Response"
    for module_name in ("maslab", "maslab.core", "maslab.core.types"):
        module = importlib.import_module(module_name)
        assert module.Response is maslab.Response
        assert not hasattr(module, "Answer")
    assert maslab.Multiagent
    assert maslab.SequentialMultiagent
    assert maslab.ParallelMultiagent
    assert maslab.Pipeline
    assert maslab.Aggregate
    assert maslab.ConcatAggregate
    assert maslab.LLMAggregate
    assert not hasattr(maslab, "Aggregator")
    assert maslab.Model
    removed_names = (
        "Model" + "Backend",
        "Benchmark" + "Protocol",
        "Prompt" + "Renderer",
    )
    assert not any(hasattr(maslab, name) for name in removed_names)


def test_parallel_and_aggregator_package_imports_match_public_exports():
    from maslab.core.aggregators import Aggregate, LLMAggregate, ConcatAggregate
    from maslab.core.multiagents import ParallelMultiagent

    assert Aggregate is maslab.Aggregate
    assert Aggregate.__name__ == "Aggregate"
    assert Aggregate.__module__ == "maslab.core.aggregators.aggregate"
    for module_name in ("maslab", "maslab.core", "maslab.core.aggregators"):
        module = importlib.import_module(module_name)
        for removed_name in ("Aggregator", "LLMAggregator", "MajorityVote", "PeerAggregate"):
            assert removed_name not in module.__all__
            assert not hasattr(module, removed_name)
    assert LLMAggregate is maslab.LLMAggregate
    assert LLMAggregate.__name__ == "LLMAggregate"
    assert LLMAggregate.__module__ == "maslab.core.aggregators.llm_aggregate"
    assert issubclass(LLMAggregate, Aggregate)
    assert issubclass(ConcatAggregate, Aggregate)
    assert ConcatAggregate is maslab.ConcatAggregate
    assert ConcatAggregate.__module__ == "maslab.core.aggregators.concat_aggregate"
    assert ParallelMultiagent is maslab.ParallelMultiagent


def test_generation_config_has_no_class_module_or_public_export():
    for module_name in ("maslab", "maslab.core"):
        module = importlib.import_module(module_name)
        assert "GenerationConfig" not in module.__all__
        assert not hasattr(module, "GenerationConfig")
    with pytest.raises(ModuleNotFoundError, match="maslab.core.config"):
        importlib.import_module("maslab.core.config")


def test_registry_is_removed():
    for module_name in ("maslab", "maslab.models"):
        module = importlib.import_module(module_name)
        for name in ("create_model", "register_model_backend", "get_model", "get_model_backend", "MODEL_BACKENDS", "LEGACY_MODEL_BACKENDS"):
            assert not hasattr(module, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("maslab.models.registry")


def test_quickstart_runs_using_only_library_api():
    from pathlib import Path
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "examples/quickstart.py")],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Mesh answers:" in result.stdout and "Pipeline answer:" in result.stdout
