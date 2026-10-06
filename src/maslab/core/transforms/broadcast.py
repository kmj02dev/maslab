"""Expand one response into independent per-participant prompts."""

from copy import deepcopy

from .transform import Transform
from ..types import Response


def _response(source: Response, content: str) -> Response:
    if not isinstance(source, Response) or not isinstance(source.content, str):
        raise TypeError("input must be a Response with string content")
    return Response(prompt=deepcopy(source.prompt), content=content)


class Broadcast(Transform):
    """Copy content count times without duplicating model usage or identity."""

    returns_multiple = True
    returns_prompts = True

    def __init__(self, count: int):
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        self.count = count

    def transform(self, response: Response) -> list[Response]:
        if not isinstance(response, Response) or not isinstance(response.content, str):
            raise TypeError("input must be a Response with string content")
        return [_response(response, response.content) for _ in range(self.count)]


class WrapBroadcast(Transform):
    """Wrap content with shared strings or equally sized per-output lists.

    None is an empty string. Without lists, produce one response. Lists must
    be non-empty; when both sides are lists their lengths must match.
    """

    returns_multiple = True
    returns_prompts = True

    def __init__(self, prefix: str | list[str] | None = None,
                 suffix: str | list[str] | None = None):
        self.prefix = self._validate(prefix, "prefix")
        self.suffix = self._validate(suffix, "suffix")
        sizes = [len(v) for v in (self.prefix, self.suffix) if isinstance(v, tuple)]
        if len(set(sizes)) > 1:
            raise ValueError("prefix and suffix lists must have equal lengths")
        self.count = sizes[0] if sizes else 1

    @staticmethod
    def _validate(value, name):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{name} list must not be empty")
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"{name} list must contain only strings")
            return tuple(value)
        raise TypeError(f"{name} must be str, list[str], or None")

    def transform(self, response: Response) -> list[Response]:
        if not isinstance(response, Response) or not isinstance(response.content, str):
            raise TypeError("input must be a Response with string content")
        return [
            _response(response,
                      (self.prefix[i] if isinstance(self.prefix, tuple) else self.prefix)
                      + response.content
                      + (self.suffix[i] if isinstance(self.suffix, tuple) else self.suffix))
            for i in range(self.count)
        ]
