"""Deterministic voting over exact response texts."""

from collections import Counter
from collections.abc import Sequence

from ..types import Response
from .aggregate import Aggregate


class MajorityVote(Aggregate):
    """Return the most frequent text, breaking ties by first input occurrence.

    Votes use exact, case-sensitive string equality. A strict majority is not
    required; callers can extract labels before voting on free-form answers.
    """

    def aggregate(self, responses: Sequence[str | Response]) -> str:
        contents = self._contents(responses)
        return Counter(contents).most_common(1)[0][0]
