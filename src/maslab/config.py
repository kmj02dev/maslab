"""Configuration objects for reusable MASLab experiments."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Provider-independent text generation settings."""

    reasoning: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    max_retries: int = 2
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Settings applied when an experiment creates its agents."""

    num_agents: int | None = None
    context_policy: str = "append"
    use_context_in_debate: bool = True
    use_context_in_decision: bool = False
    # Backward-compatible alias for use_context_in_debate.
    use_context: bool | None = None


@dataclass(frozen=True)
class SessionConfig:
    """Settings that control one multi-agent session."""

    num_rounds: int = 3
    debate_topology: str = "sequential"
    initial_decision_visibility: str = "broadcast"
    consensus_threshold: float = 1.0
    early_stop_on_consensus: bool = False
    checkpoint_path: str | Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level, serializable experiment settings."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    random_state: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        checkpoint_path = value["session"].get("checkpoint_path")
        if checkpoint_path is not None:
            value["session"]["checkpoint_path"] = str(checkpoint_path)
        return value
