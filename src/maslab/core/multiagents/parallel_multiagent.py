"""Independent concurrent execution of agent conversations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Sequence

from ..agents import Agent
from ..aggregators import Aggregate
from .._history import _query_entry, _message_snapshot
from ..types import Response
from .multiagent import Multiagent


class ParallelMultiagent(Multiagent[list[Response]]):
    """Send the same message to all participants and return answers in input order.

    Collection-aware branches can instead receive a Response list, copied per
    branch, for an explicit Aggregate-first pipeline. No current sibling output
    is forwarded. Participants' own context policies
    remain effective. Branches must use distinct Agent/Multiagent/Transform/Aggregate instances;
    shared model backends must support concurrent ``respond`` calls.
    Completed work is recorded even when another branch fails. All submitted
    branches finish before the first failure in participant order is raised.
    """

    returns_multiple = True

    def __init__(
        self,
        agents: Sequence[Agent | Multiagent[Response]],
        *,
        id: str = "parallel",
        max_workers: int | None = None,
    ):
        super().__init__(id=id)
        self.agents = tuple(agents)
        if not self.agents:
            raise ValueError("a parallel multiagent requires at least one agent")
        if any(not isinstance(agent, (Agent, Multiagent)) for agent in self.agents):
            raise TypeError("participants must be Agent or Multiagent instances")
        if any(isinstance(agent, Multiagent) and agent.returns_multiple for agent in self.agents):
            raise TypeError("parallel participants must each return one answer; aggregate nested collections first")
        if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
            raise ValueError("max_workers must be a positive integer or None")
        self.max_workers = max_workers
        self._validate_independent_branches()

    @property
    def accepts_multiple(self):
        return all(isinstance(agent, Multiagent) and agent.accepts_multiple for agent in self.agents)

    def _validate_independent_branches(self):
        seen = set()
        for branch in self.agents:
            branch_ids = set()
            pending = [branch]
            while pending:
                participant = pending.pop()
                identity = id(participant)
                if identity in branch_ids:
                    continue
                branch_ids.add(identity)
                if isinstance(participant, Multiagent):
                    pending.extend(getattr(participant, "steps", getattr(participant, "agents", ())))
                elif isinstance(participant, Aggregate) and isinstance(getattr(participant, "agent", None), Agent):
                    pending.append(participant.agent)
            if seen.intersection(branch_ids):
                raise ValueError("parallel branches must not share Agent, Multiagent, or Transform instances")
            seen.update(branch_ids)

    def _record(self, message: str | list[Response], steps: list[dict[str, Any]], *, failed: bool):
        usage = {
            "input_tokens": sum(step["usage"]["input_tokens"] for step in steps),
            "output_tokens": sum(step["usage"]["output_tokens"] for step in steps),
            "total_tokens": sum(step["usage"]["total_tokens"] for step in steps),
        }
        times = [
            step["usage"]["generation_time"]
            for step in steps
            if step["usage"]["generation_time"] is not None
        ]
        usage["generation_time"] = sum(times) if times else None
        self._history.append({
            "agent_id": self.id,
            "message": _message_snapshot(message),
            "prompt": [{"role": "user", "content": message}] if isinstance(message, str) else [],
            "content": [step["content"] for step in steps if step["status"] == "completed"],
            "reasoning": None,
            "usage": usage,
            "status": "failed" if failed else "completed",
            "steps": steps,
        })

    def query(
        self,
        message: str | list[Response] = "Continue.",
        use_context: bool = True,
        update_context: bool = True,
    ) -> list[Response]:
        if not isinstance(message, str) and not (
            self.accepts_multiple and isinstance(message, list) and message
            and all(isinstance(r, Response) and isinstance(r.content, str) for r in message)
        ):
            raise TypeError("message must be a string, or a Response list for collection-aware branches")
        if not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean")
        if not isinstance(update_context, bool):
            raise TypeError("update_context must be a boolean")
        self._validate_independent_branches()
        responses = {}
        steps = {}
        errors = {}
        starts = [len(agent._history) for agent in self.agents]
        with ThreadPoolExecutor(max_workers=self.max_workers or len(self.agents)) as executor:
            futures = {
                executor.submit(
                    agent.query,
                    deepcopy(message),
                    use_context=use_context,
                    update_context=update_context,
                ): index
                for index, agent in enumerate(self.agents)
            }
            for future in as_completed(futures):
                index = futures[future]
                agent = self.agents[index]
                try:
                    response = future.result()
                    if not isinstance(response, Response):
                        raise TypeError("parallel participants must return a single Response")
                    responses[index] = response
                    step = _query_entry(agent.id, message, response)
                    if isinstance(agent, Multiagent) and len(agent._history) > starts[index]:
                        step["steps"] = deepcopy(agent._history[-1].get("steps", []))
                except Exception as error:
                    errors[index] = error
                    if isinstance(agent, Multiagent) and len(agent._history) > starts[index]:
                        step = deepcopy(agent._history[-1])
                    else:
                        # No model response or actual prompt is available on failure.
                        step = _query_entry(agent.id, message, Response(prompt=[], content=""))
                    step.update({
                        "status": "failed",
                        "error": {"type": type(error).__name__, "message": str(error)},
                    })
                step.update({"kind": "query", "step": index + 1})
                steps[index] = step
        self._record(message, [steps[index] for index in range(len(self.agents))], failed=bool(errors))
        if errors:
            raise errors[min(errors)]
        return [responses[index] for index in range(len(self.agents))]
