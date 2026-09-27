"""Contract for reducing independent agent responses to one response."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..types import Response


class Aggregate(ABC):
    """Reduce an iterable of model responses to one response.

    Both aggregator(responses) and aggregator.aggregate(responses)
    return the final Response. Implement aggregate in custom reducers.
    """

    def __call__(self, responses: Iterable[Response]) -> Response:
        return self.aggregate(responses)

    @abstractmethod
    def aggregate(self, responses: Iterable[Response]) -> Response:
        """Reduce a non-empty iterable of Response objects to one Response."""
        raise NotImplementedError

    @staticmethod
    def _contents(responses: Iterable[Response]) -> list[str]:
        """Validate and collect contents in one pass, including from generators."""
        if isinstance(responses, (str, bytes)):
            raise TypeError("responses must be an iterable of Response objects")
        try:
            iterator = iter(responses)
        except TypeError as error:
            raise TypeError("responses must be an iterable of Response objects") from error
        contents = []
        for response in iterator:
            if not isinstance(response, Response) or not isinstance(response.content, str):
                raise TypeError("each response must be a Response with string content")
            contents.append(response.content)
        if not contents:
            raise ValueError("at least one response is required")
        return contents
