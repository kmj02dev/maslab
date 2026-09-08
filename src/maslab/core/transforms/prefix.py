"""Prepend text to response content."""

from dataclasses import replace

from .transform import Transform
from ..types import Response


class Prefix(Transform):
    """Prepend text to response content. Whitespace is preserved exactly."""

    def __init__(self, prefix: str):
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        self.prefix = prefix

    def transform(self, response: Response) -> Response:
        return replace(response, content=self.prefix + response.content)
