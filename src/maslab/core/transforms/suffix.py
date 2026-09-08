"""Append text to response content."""

from dataclasses import replace

from .transform import Transform
from ..types import Response


class Suffix(Transform):
    """Append text to response content. Whitespace is preserved exactly."""

    def __init__(self, suffix: str):
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string")
        self.suffix = suffix

    def transform(self, response: Response) -> Response:
        return replace(response, content=response.content + self.suffix)
