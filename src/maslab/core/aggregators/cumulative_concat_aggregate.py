"""Accumulate response rounds without adding model inference costs."""

from collections.abc import Iterable
from copy import deepcopy

from ..types import Response
from .aggregate import Aggregate


class CumulativeConcatAggregate(Aggregate):
    """Append one round per successful call and format all recorded responses.

    State persists across Pipeline.query calls until reset(). Use a separate
    instance for each task and concurrent branch. Contents, including duplicates
    and empty strings, are preserved. Missing agent IDs use positional labels.
    Returned responses contain no source usage; history retains source metadata.
    """

    def __init__(self):
        self._rounds: list[list[Response]] = []

    def aggregate(self, responses: Iterable[Response]) -> Response:
        # Materialize once so generators are supported; validate before mutation.
        if isinstance(responses, (str, bytes)):
            raise TypeError("responses must be an iterable of Response objects")
        current = list(responses)
        self._contents(current)
        current = deepcopy(current)
        rounds = [*self._rounds, current]
        sections = []
        for round_number, responses_in_round in enumerate(rounds, start=1):
            for position, response in enumerate(responses_in_round, start=1):
                identity = (response.agent_id if response.agent_id is not None
                            else f"agent {position}")
                sections.append(f"[round {round_number} | {identity}]\n{response.content}")
        result = Response(prompt=[], content="\n\n".join(sections))
        self._rounds.append(current)
        return result

    def history(self) -> list[list[Response]]:
        """Return independent response snapshots grouped by aggregate call."""
        return deepcopy(self._rounds)

    def reset(self) -> None:
        """Discard all rounds; the next successful call starts at round one."""
        self._rounds.clear()
