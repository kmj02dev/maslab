"""Common conversation contract for multiagent groups."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, TypeVar

from ..types import Response


ResponseT = TypeVar("ResponseT", bound=Response | list[Response])


class Multiagent(ABC, Generic[ResponseT]):
    """A group returning one response or an ordered collection of responses.

    Collection-producing subclasses set ``returns_multiple = True``. A
    collection must be reduced with an Aggregate before a single-answer
    participant can consume it.
    ``accepts_prompts`` separately marks support for per-participant string lists.
    """

    returns_multiple = False
    accepts_multiple = False
    accepts_prompts = False

    def __init__(self, *, id: str):
        if not isinstance(id, str) or not id.strip():
            raise ValueError("multiagent id must be a non-empty string")
        self.id = id
        self._history: list[dict[str, Any]] = []

    @abstractmethod
    def query(
        self,
        message: str = "Continue.",
        use_context: bool = True,
        update_context: bool = True,
    ) -> ResponseT:
        """Execute and record one group conversation, returning its responses."""
        raise NotImplementedError

    def history(self) -> list[dict[str, Any]]:
        """Return independent snapshots of group calls, including their steps."""
        return deepcopy(self._history)
