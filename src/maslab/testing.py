"""Validation helpers for third-party MASLab components."""

from copy import deepcopy

from .core import Model, Response


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


