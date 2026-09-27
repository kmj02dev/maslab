"""Single-agent conversations and execution history."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..model import Model
from ..types import ChatMessage, Response
from .._history import _query_entry


class Agent:
    CONTEXT_POLICIES = {"append", "replace", "last", "last_conversation"}

    def __init__(
        self,
        id: str,
        model: Model,
        system_prompt: str = "",
        context_policy: str = "append",
    ):
        if context_policy not in self.CONTEXT_POLICIES:
            choices = ", ".join(sorted(self.CONTEXT_POLICIES))
            raise ValueError(f"context_policy must be one of: {choices}")
        self.id = id
        self.model = model
        self.context: list[ChatMessage] = []
        self.context_policy = context_policy
        self._history: list[dict[str, Any]] = []

        if system_prompt:
            self.context.append({
                "role": "system",
                "content": system_prompt,
            })

    def generate(
        self,
        message: str = "Continue.",
        use_context: bool = True,
    ) -> Response:
        """Generate a response tagged with this agent ID without recording a query."""
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean")
        messages = deepcopy(self.context if use_context else self.system_context())
        messages.append({
            "role": "user",
            "content": message,
        })
        return replace(self.model.respond(messages), agent_id=self.id)

    def query(
        self,
        message: str = "Continue.",
        use_context: bool = True,
        update_context: bool = True,
    ) -> Response:
        """Return the response and record the completed conversation."""
        response = self.generate(message, use_context=use_context)
        if update_context:
            self.context.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.content},
            ])
            self.manage_context()
        self._history.append(_query_entry(self.id, message, response))
        return response

    def history(self) -> list[dict[str, Any]]:
        """Return independent snapshots of all completed query calls."""
        return deepcopy(self._history)

    def system_context(self) -> list[ChatMessage]:
        if self.context and self.context[0].get("role") == "system":
            return [deepcopy(self.context[0])]
        return []

    def clear_context(self):
        self.context = self.system_context()

    def manage_context(self):
        if self.context_policy == "replace":
            self.clear_context()
        elif self.context_policy == "last":
            self.context = self.system_context() + self.context[-1:]
        elif self.context_policy == "last_conversation":
            self.context = self.system_context() + self.context[-2:]
