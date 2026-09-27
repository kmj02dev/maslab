"""Join response contents into numbered agent sections without model inference."""

from collections.abc import Iterable

from ..types import Response
from .aggregate import Aggregate


class ConcatAggregate(Aggregate):
    """Preserve each content under [agent N], numbered from one in input order.

    Sections are separated by a blank line. No responses are filtered or
    deduplicated, and input contents are preserved exactly. Input prompts and
    usage remain with their source responses; this step adds no inference cost.
    """

    def aggregate(self, responses: Iterable[Response]) -> Response:
        contents = self._contents(responses)
        content = "\n\n".join(
            f"[agent {index}]\n{text}" for index, text in enumerate(contents, start=1)
        )
        return Response(prompt=[], content=content)
