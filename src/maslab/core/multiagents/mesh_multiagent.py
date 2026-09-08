"""Synchronous all-to-all response exchange between parallel rounds."""

import json
from typing import Sequence

from ..agents import Agent
from ..types import Response
from .multiagent import Multiagent
from .parallel_multiagent import ParallelMultiagent


class MeshMultiagent(ParallelMultiagent):
    """Run parallel rounds, broadcasting every previous response to all agents.

    ``loop`` counts all rounds, including the initial question. Later messages
    are JSON objects containing the original question and the previous round's
    responses in participant order (including the recipient's own response).
    Only the last round is broadcast; agent context follows normal query policy.
    Returns the final round's responses. History and usage cover every round.
    Shared model backends must support concurrent calls, as in ParallelMultiagent.
    """

    def __init__(
        self,
        agents: Sequence[Agent | Multiagent[Response]],
        loop: int = 1,
        *,
        id: str = "mesh",
        max_workers: int | None = None,
    ):
        super().__init__(agents, id=id, max_workers=max_workers)
        if type(loop) is not int or loop < 1:
            raise ValueError("loop must be a positive integer")
        self.loop = loop

    def query(
        self,
        message: str = "Continue.",
        use_context: bool | None = None,
        update_context: bool = True,
    ) -> list[Response]:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if use_context is not None and not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean or None")
        if not isinstance(update_context, bool):
            raise TypeError("update_context must be a boolean")
        self._validate_independent_branches()
        records = []
        current_message = message
        responses = []
        for round_index in range(1, self.loop + 1):
            group = ParallelMultiagent(self.agents, max_workers=self.max_workers)
            try:
                responses = group.query(current_message, use_context, update_context)
            except Exception:
                for step in group.history()[-1]["steps"]:
                    records.append({**step, "loop": round_index})
                self._record(message, records, failed=True)
                self._history[-1]["content"] = [
                    step["content"] for step in records
                    if step["loop"] == round_index and step["status"] == "completed"
                ]
                raise
            for step in group.history()[-1]["steps"]:
                records.append({**step, "loop": round_index})
            current_message = json.dumps({
                "question": message,
                "responses": [
                    {"agent_id": agent.id, "content": response.content}
                    for agent, response in zip(self.agents, responses)
                ],
            }, ensure_ascii=False)
        self._record(message, records, failed=False)
        self._history[-1]["content"] = [response.content for response in responses]
        return responses
