"""Contract for reducing independent agent responses to one answer."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..types import Response


class Aggregate(ABC):
    """Apply a reduction to response texts or raw model responses.

    Both ``aggregator(responses)`` and ``aggregator.aggregate(responses)``
    return the final answer text. Implement ``aggregate`` in custom reducers.
    """

    def __call__(self, responses: Sequence[str | Response]) -> str:
        return self.aggregate(responses)

    @abstractmethod
    def aggregate(self, responses: Sequence[str | Response]) -> str:
        """Reduce a non-empty sequence of responses to one answer."""
        raise NotImplementedError

    @staticmethod
    def _contents(responses: Sequence[str | Response]) -> list[str]:
        if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
            raise TypeError("responses must be a sequence of strings or Response objects")
        if not responses:
            raise ValueError("at least one response is required")
        contents = []
        for response in responses:
            content = response.content if isinstance(response, Response) else response
            if not isinstance(content, str):
                raise TypeError("each response must be a string or Response with string content")
            contents.append(content)
        return contents
