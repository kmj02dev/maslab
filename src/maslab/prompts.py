"""Prompt lifecycle and decision parsing primitives."""

from dataclasses import dataclass, field
import inspect
import json
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .types import Message


PromptPhase = Literal["initial", "debate", "decision"]


@dataclass(frozen=True)
class PromptContext:
    """Typed context supplied to every MASLab prompt renderer."""

    task: Any
    phase: PromptPhase
    round_idx: int
    agent_idx: int
    agent_id: str
    visible_messages: tuple[Message, ...] = ()
    options: tuple[str, ...] | None = None
    agent: Any = None
    variables: Mapping[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments understood by pre-PromptContext renderers."""
        values = dict(self.variables)
        values.update({
            "task": self.task,
            "phase": self.phase,
            "round_idx": self.round_idx,
            "agent_idx": self.agent_idx,
            "agent_id": self.agent_id,
            "agent": self.agent,
            "visible_messages": self.visible_messages,
            "received_messages": list(self.visible_messages),
            "options": self.options,
        })
        return values


def render_prompt(renderer, context: PromptContext) -> str:
    """Render a typed, legacy callable, Jinja-compatible, or static prompt."""
    if isinstance(renderer, str):
        rendered = renderer
    elif hasattr(renderer, "render"):
        rendered = renderer.render(**context.to_kwargs())
    elif callable(renderer):
        try:
            parameters = inspect.signature(renderer).parameters
        except (TypeError, ValueError):
            rendered = renderer(context)
        else:
            if any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                rendered = renderer(**context.to_kwargs())
            elif len(parameters) == 1:
                name = next(iter(parameters))
                if name in {"context", "prompt_context"} or name not in context.to_kwargs():
                    rendered = renderer(context)
                else:
                    rendered = renderer(**{name: context.to_kwargs()[name]})
            else:
                values = context.to_kwargs()
                rendered = renderer(**{
                    name: values[name]
                    for name in parameters
                    if name in values
                })
    else:
        raise TypeError("prompt must be a string, callable, or provide render()")

    if not isinstance(rendered, str):
        raise TypeError("prompt renderer must return a string")
    rendered = rendered.strip()
    if not rendered:
        raise ValueError(f"{context.phase} prompt must not be empty")
    return rendered


def format_messages(messages: Sequence[Message]) -> str:
    return "\n\n".join(
        f"{f'Round {message.round_idx} - ' if message.round_idx is not None else ''}"
        f"Agent {message.speaker}: {message.content}"
        for message in messages
    )


def generic_debate_prompt(context: PromptContext) -> str:
    transcript = format_messages(context.visible_messages)
    if not transcript:
        return (
            "No other agent messages are available yet. Re-evaluate your "
            "evidence and state your current position."
        )
    return (
        f"Other agents' messages:\n{transcript}\n\n"
        "Compare their evidence with yours and state your current position."
    )


def generic_decision_prompt(context: PromptContext) -> str:
    transcript = format_messages(context.visible_messages)
    instruction = "Provide your final decision."
    return f"Discussion:\n{transcript}\n\n{instruction}" if transcript else instruction


def _identity(value: str):
    return value


@runtime_checkable
class DecisionPolicy(Protocol):
    def render(self, context: PromptContext) -> str:
        ...

    def parse(self, content: str):
        ...


@dataclass(frozen=True)
class DecisionPrompt:
    """Pair one decision renderer with its response parser."""

    renderer: Any = generic_decision_prompt
    parser: Callable[[str], Any] = _identity

    def render(self, context: PromptContext) -> str:
        return render_prompt(self.renderer, context)

    def parse(self, content: str):
        return self.parser(content)


class JsonDecision:
    """Render and validate a JSON-object decision."""

    def __init__(self, fields: Mapping[str, type], renderer=None):
        if not fields:
            raise ValueError("JSON decision fields must not be empty")
        self.fields = dict(fields)
        self.renderer = renderer

    def render(self, context: PromptContext) -> str:
        if self.renderer is not None:
            return render_prompt(self.renderer, context)

        transcript = format_messages(context.visible_messages)
        schema = {
            name: f"<{expected_type.__name__}>"
            for name, expected_type in self.fields.items()
        }
        parts = []
        if transcript:
            parts.append(f"Discussion:\n{transcript}")
        if context.options:
            parts.append(f"Valid options: {', '.join(context.options)}")
        parts.append(
            "Return only one JSON object matching this schema: "
            f"{json.dumps(schema)}"
        )
        return "\n\n".join(parts)

    def parse(self, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("JSON decision response must be a string")
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace == -1 or last_brace <= first_brace:
            raise ValueError("Decision response does not contain a JSON object")
        candidate = content[first_brace:last_brace + 1]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as error:
            if not error.msg.startswith("Invalid control character"):
                raise ValueError(
                    "Decision response contains invalid JSON"
                ) from error
            try:
                value = json.loads(candidate, strict=False)
            except json.JSONDecodeError as fallback_error:
                raise ValueError(
                    "Decision response contains invalid JSON"
                ) from fallback_error
        if not isinstance(value, dict):
            raise ValueError("Decision response must contain a JSON object")
        for name, expected_type in self.fields.items():
            if name not in value:
                raise ValueError(f"Decision response is missing field: {name}")
            if not isinstance(value[name], expected_type):
                raise ValueError(
                    f"Decision field {name!r} must be {expected_type.__name__}"
                )
        return value


@dataclass(frozen=True)
class PromptSet:
    """All prompt behavior required by one experiment."""

    initial: Any
    debate: Any = generic_debate_prompt
    decision: DecisionPolicy = field(default_factory=DecisionPrompt)


def generic_prompts(initial) -> PromptSet:
    return PromptSet(initial=initial)
