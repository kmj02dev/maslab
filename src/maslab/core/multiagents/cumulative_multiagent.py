"""Sequential queries with an explicit cumulative response history."""

import json
from typing import Sequence

from ..agents import Agent
from .._execution import _finish_sequence, _query_step
from ..types import Response
from .multiagent import Multiagent
from .sequential_multiagent import SequentialMultiagent


class CumulativeMultiagent(SequentialMultiagent):
    """Send every preceding participant output to each query in order.

    The first participant receives the original message. Subsequent inputs are
    JSON strings containing only ``responses``, without repeating the question.
    Response entries contain one-based ``round``, ``agent_id`` and ``content``.
    They include the recipient's own earlier outputs and outputs from earlier
    participants in the current round. Each query starts a new response list.

    Context flags follow the normal agent API. Use both flags as False to send
    system instructions plus the shared response list without personal-context
    duplication. History is always recorded. Nested groups contribute their
    final response to the response list and retain their full execution history.
    The returned Response contains the last output and usage for the whole run.
    """

    def __init__(
        self,
        agents: Sequence[Agent | Multiagent[Response]],
        loop: int = 1,
        *,
        id: str = "cumulative",
    ):
        super().__init__(agents, loop=loop, id=id)

    def _validate_agents(self):
        super()._validate_agents()
        if any(isinstance(agent, Multiagent) and agent.accepts_multiple for agent in self.agents):
            raise TypeError("cumulative participants must accept a string message")

    def query(
        self,
        message: str = "Continue.",
        use_context: bool = True,
        update_context: bool = True,
    ) -> Response:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean")
        if not isinstance(update_context, bool):
            raise TypeError("update_context must be a boolean")
        self._validate_agents()
        records = []
        responses = []
        current_response = None
        try:
            for loop_idx in range(1, self.loop + 1):
                for step_idx, agent in enumerate(self.agents, start=1):
                    current_message = (
                        json.dumps({"responses": responses}, ensure_ascii=False)
                        if responses else message
                    )
                    current_response = _query_step(
                        agent, current_message, records,
                        loop_idx=loop_idx, step_idx=step_idx,
                        use_context=use_context, update_context=update_context,
                    )
                    responses.append({
                        "round": loop_idx,
                        "agent_id": agent.id,
                        "content": current_response.content,
                    })
        except Exception as error:
            _finish_sequence(self._history, self.id, message, records, current_response, error=error)
            raise
        return _finish_sequence(self._history, self.id, message, records, current_response)
