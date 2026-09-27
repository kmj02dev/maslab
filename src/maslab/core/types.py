"""Values shared by models, agents, groups, and aggregators."""

from dataclasses import dataclass
from typing import TypedDict


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
    agent_id: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    generation_time: float | None = None
