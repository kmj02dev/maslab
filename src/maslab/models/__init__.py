"""Built-in model adapters and backend registry."""

from .gemini_api import GeminiAPIModel
from .huggingface import HuggingfaceModel
from .nvidia_build_api import NvidiaBuildAPIModel
from .registry import (
    LEGACY_MODEL_BACKENDS,
    MODEL_BACKENDS,
    create_model,
    get_model,
    get_model_backend,
    register_model_backend,
)

register_model_backend("huggingface", HuggingfaceModel)
register_model_backend("nvidia", NvidiaBuildAPIModel)
register_model_backend("gemini", GeminiAPIModel)

__all__ = [
    "GeminiAPIModel",
    "HuggingfaceModel",
    "LEGACY_MODEL_BACKENDS",
    "MODEL_BACKENDS",
    "NvidiaBuildAPIModel",
    "create_model",
    "get_model",
    "get_model_backend",
    "register_model_backend",
]
