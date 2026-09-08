"""Agent composition core with optional model adapters."""

from importlib import import_module
from typing import TYPE_CHECKING

from importlib.metadata import PackageNotFoundError, version

from .core import (
    Agent, Aggregate, ChatMessage, LLMAggregate, MajorityVote,
    Model, Multiagent, MeshMultiagent, ParallelMultiagent, Pipeline, Response, SequentialMultiagent, Transform, Prefix, Suffix, Wrap, Usage,
)

try:
    __version__ = version("maslab")
except PackageNotFoundError:
    __version__ = "0.1.0"

if TYPE_CHECKING:
    from maslab.utils import dialog
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
    "dialog": "maslab.utils",
    "GeminiAPIModel": "maslab.models.gemini_api",
    "HuggingfaceModel": "maslab.models.huggingface",
    "NvidiaBuildAPIModel": "maslab.models.nvidia_build_api",
}

__all__ = [
    "Agent",
    "Aggregate",
    "ChatMessage",
    "GeminiAPIModel",
    "HuggingfaceModel",
    "LLMAggregate",
    "MajorityVote",
    "Model",
    "Multiagent",
    "NvidiaBuildAPIModel",
    "ParallelMultiagent",
    "MeshMultiagent",
    "Pipeline",
    "Response",
    "SequentialMultiagent",
    "Transform",
    "Prefix",
    "Suffix",
    "Wrap",
    "Usage",
    "dialog",
    "__version__",
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
