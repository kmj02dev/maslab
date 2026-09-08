"""Surround response content with text."""

from dataclasses import replace

from .transform import Transform
from ..types import Response


class Wrap(Transform):
    """Add a prefix and suffix, preserving whitespace exactly."""

    def __init__(self, prefix: str, suffix: str):
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string")
        self.prefix = prefix
        self.suffix = suffix

    def transform(self, response: Response) -> Response:
        return replace(response, content=self.prefix + response.content + self.suffix)
