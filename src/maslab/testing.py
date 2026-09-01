"""Validation helpers for third-party MASLab components."""

from copy import deepcopy

from .base import Benchmark, Model
from .types import Response


def check_model_backend(backend):
    """Exercise a backend once and assert the public response contract.

    This invokes the backend. Callers should use a test model or mocked
    transport when checking a remote provider.
    """
    if not isinstance(backend, Model):
        raise AssertionError("model backend must inherit from maslab.Model")
    if not isinstance(getattr(backend, "name", None), str):
        raise AssertionError("model backend must expose a string name")
    messages = [{"role": "user", "content": "MASLab contract check"}]
    before = deepcopy(messages)
    response = backend.respond(messages)
    if messages != before:
        raise AssertionError("model backend must not mutate input messages")
    if not isinstance(response, Response):
        raise AssertionError("model backend must return maslab.Response")
    if not isinstance(response.content, str):
        raise AssertionError("model response content must be a string")
    return True


def check_benchmark(benchmark):
    """Assert the minimum dataset and prompt-building benchmark contract."""
    if not isinstance(benchmark, Benchmark):
        raise AssertionError("benchmark must inherit from maslab.Benchmark")
    if not isinstance(benchmark.tasks, list):
        raise AssertionError("benchmark.tasks must be a list")
    if benchmark.tasks:
        prompts = benchmark.build_prompts(
            benchmark.tasks[0],
            lambda **context: "MASLab contract prompt",
        )
        if not isinstance(prompts, list) or not prompts:
            raise AssertionError("benchmark.build_prompts() must return a non-empty list")
        if not all(isinstance(prompt, str) for prompt in prompts):
            raise AssertionError("benchmark prompts must be strings")
    return True
