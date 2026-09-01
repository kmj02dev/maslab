"""Public value types returned by MASLab."""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass
class Response:
    prompt: list[ChatMessage]
    content: str
    reasoning: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_time: float | None = None


@dataclass(frozen=True)
class Message:
    speaker: str
    content: str
    content_tokens: int
    round_idx: int | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    generation_time: float | None = None


@dataclass(frozen=True)
class TaskResult:
    task: Any
    metrics: dict[str, Any] | None
    decisions: dict[str, dict[Any, Any]]
    history: list[dict[str, Any]]
    opinion_history: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskResult":
        return cls(
            task=value["task"],
            metrics=value.get("metrics"),
            decisions=value.get("decisions", {}),
            history=value.get("history", []),
            opinion_history=value.get("opinion_history"),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "task": self.task,
            "metrics": self.metrics,
            "decisions": self.decisions,
            "history": self.history,
        }
        if self.opinion_history is not None:
            value["opinion_history"] = self.opinion_history
        return value


@dataclass(frozen=True)
class RunResult:
    config: dict[str, Any]
    metrics: dict[str, Any]
    results: list[TaskResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "metrics": self.metrics,
            "results": [result.to_dict() for result in self.results],
        }

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path


@dataclass
class SessionState:
    history: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    opinion_history: list[dict[str, Any]] = field(default_factory=list)
    consensus_round: int | None = None
    stopped_early: bool = False
    total_rounds: int = 0
    next_round: int = 0
    global_context: list[Message] = field(default_factory=list)
