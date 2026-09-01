"""Provider-level model backend registry."""

import os


LEGACY_MODEL_BACKENDS = {
    "google/gemma-4-31b-it": ("nvidia", "NVIDIA_API_KEY"),
    "gemini-3.5-flash-lite": ("gemini", "GEMINI_API_KEY"),
    "Qwen/Qwen2.5-3B-Instruct": ("huggingface", None),
}

MODEL_BACKENDS = {}


def register_model_backend(name, factory, *, overwrite=False):
    """Register a backend factory under a provider-level name."""
    if not name or not isinstance(name, str):
        raise ValueError("backend name must be a non-empty string")
    if name in MODEL_BACKENDS and not overwrite:
        raise ValueError(f"Model backend is already registered: {name}")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    MODEL_BACKENDS[name] = factory


def create_model(backend, name, **kwargs):
    """Create a model by backend name without constraining the model ID."""
    try:
        factory = MODEL_BACKENDS[backend]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_BACKENDS))
        raise ValueError(
            f"Unknown model backend {backend!r}; choose from: {choices}"
        ) from error
    return factory(name=name, **kwargs)


def get_model_backend(name):
    try:
        return LEGACY_MODEL_BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {name}") from error


def get_model(name, *, backend=None, api_key=None, **kwargs):
    """Create a model, retaining model-ID inference for the original presets."""
    credential_env = None
    if backend is None:
        backend, credential_env = get_model_backend(name)
    if credential_env and api_key is None:
        api_key = os.getenv(credential_env)
        if not api_key:
            raise EnvironmentError(f"Missing environment variable: {credential_env}")
    if api_key is not None:
        kwargs["api_key"] = api_key
    return create_model(backend, name, **kwargs)
