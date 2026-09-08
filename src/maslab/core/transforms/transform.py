"""Independent response processing between query steps."""

from abc import ABC, abstractmethod

from ..types import Response


class Transform(ABC):
    """Convert one response into another without calling a model.

    Pipeline supplies a deep copy of the preceding response and
    records the input and output separately from model usage. Implementations
    must return a Response with string content. They may edit the supplied copy
    or construct a new response, for example with dataclasses.replace().
    """

    @abstractmethod
    def transform(self, response: Response) -> Response:
        """Return the response to pass to the next step."""
        raise NotImplementedError
