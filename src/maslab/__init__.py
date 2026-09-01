"""Composable primitives for multi-agent decision-making experiments."""

from importlib.metadata import PackageNotFoundError, version

from .base import (
    Agent,
    Benchmark,
    CheckpointStore,
    Config,
    Message,
    Model,
    Response,
    Session,
)
from .benchmarks import HiddenBench
from .config import (
    AgentConfig,
    ExperimentConfig,
    GenerationConfig,
    SessionConfig,
)
from .experiment import Experiment
from .models import (
    GeminiAPIModel,
    HuggingfaceModel,
    NvidiaBuildAPIModel,
    create_model,
    register_model_backend,
)
from .prompts import (
    DecisionPolicy,
    DecisionPrompt,
    JsonDecision,
    PromptContext,
    PromptPhase,
    PromptSet,
    format_messages,
    generic_debate_prompt,
    generic_decision_prompt,
    generic_prompts,
    render_prompt,
)
from .types import ChatMessage, RunResult, SessionState, TaskResult, Usage

try:
    __version__ = version("maslab")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "Benchmark",
    "ChatMessage",
    "CheckpointStore",
    "Config",
    "DecisionPolicy",
    "DecisionPrompt",
    "Experiment",
    "ExperimentConfig",
    "GeminiAPIModel",
    "GenerationConfig",
    "HiddenBench",
    "HuggingfaceModel",
    "JsonDecision",
    "Message",
    "Model",
    "NvidiaBuildAPIModel",
    "PromptContext",
    "PromptPhase",
    "PromptSet",
    "Response",
    "RunResult",
    "Session",
    "SessionConfig",
    "SessionState",
    "TaskResult",
    "Usage",
    "create_model",
    "format_messages",
    "generic_debate_prompt",
    "generic_decision_prompt",
    "generic_prompts",
    "render_prompt",
    "register_model_backend",
    "__version__",
]
