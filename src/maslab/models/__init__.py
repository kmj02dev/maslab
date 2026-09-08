"""Optional model adapters, loaded only when selected."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maslab.models.gemini_api import (
        GeminiAPIModel,
    )
    from maslab.models.huggingface import (
        HuggingfaceModel,
    )
    from maslab.models.nvidia_build_api import (
        NvidiaBuildAPIModel,
    )

_LAZY_EXPORTS = {
    "GeminiAPIModel": "maslab.models.gemini_api",
    "HuggingfaceModel": "maslab.models.huggingface",
    "NvidiaBuildAPIModel": "maslab.models.nvidia_build_api",
}

__all__ = [
    "GeminiAPIModel",
    "HuggingfaceModel",
    "NvidiaBuildAPIModel",
]


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
