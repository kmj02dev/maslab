"""Provider-independent model invocation contract."""

from abc import ABC, abstractmethod
from typing import Sequence

from .types import ChatMessage, Response


class Model(ABC):
    """A named model that implements the message-to-response contract."""

    def __init__(self, name):
        self.name = name

    @staticmethod
    def build_messages(prompt: str, system_prompt: str = None):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return messages

    @abstractmethod
    def respond(self, messages: Sequence[ChatMessage]) -> Response:
        raise NotImplementedError()
