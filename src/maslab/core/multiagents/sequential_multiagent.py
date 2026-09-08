"""Sequential execution of agent queries."""

from typing import Sequence

from ..agents import Agent
from .._execution import _finish_sequence, _query_step
from ..types import Response
from .multiagent import Multiagent


class SequentialMultiagent(Multiagent[Response]):
    """Query each participant in order for complete loop passes.

    Each response's content becomes the next query's message, including across
    loop boundaries. Participants must be agents or single-response multiagents.
    Use Pipeline to interleave queries with response transformations.
    """

    def __init__(
        self,
        agents: Sequence[Agent | Multiagent[Response]],
        loop: int = 1,
        *,
        id: str = "sequential",
    ):
        super().__init__(id=id)
        self.agents = tuple(agents)
        self._validate_agents()
        if type(loop) is not int or loop < 1:
            raise ValueError("loop must be a positive integer")
        self.loop = loop

    def _validate_agents(self):
        if not self.agents:
            raise ValueError("a sequential multiagent requires at least one agent")
        if any(not isinstance(agent, (Agent, Multiagent)) for agent in self.agents):
            raise TypeError("participants must be Agent or Multiagent instances; use Pipeline for Transform steps")
        if any(isinstance(agent, Multiagent) and agent.returns_multiple for agent in self.agents):
            raise TypeError("reduce parallel responses with an Aggregate before sequential handoff")

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
        current_response = None
        try:
            for loop_idx in range(1, self.loop + 1):
                for step_idx, agent in enumerate(self.agents, start=1):
                    current_message = message if current_response is None else current_response.content
                    current_response = _query_step(
                        agent, current_message, records,
                        loop_idx=loop_idx, step_idx=step_idx,
                        use_context=use_context, update_context=update_context,
                    )
        except Exception as error:
            _finish_sequence(self._history, self.id, message, records, current_response, error=error)
            raise
        return _finish_sequence(self._history, self.id, message, records, current_response)
