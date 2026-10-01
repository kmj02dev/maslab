"""Ordered composition of queries and response transformations."""

from copy import deepcopy
from typing import Any, Sequence

from ..agents import Agent
from ..aggregators import Aggregate
from .._execution import _finish_sequence, _query_step
from .._history import _response_snapshot, _query_entry
from .multiagent import Multiagent
from ..transforms import Transform
from ..types import Response


class Pipeline(Multiagent[Response | list[Response]]):
    """Compose queries, single-response transforms, and collection reductions.

    Response lists pass intact to Aggregate or collection-aware groups/pipelines.
    Aggregate responses pass to subsequent queries or transforms.
    A leading Aggregate accepts a response list as the pipeline input. The final
    step determines whether query returns one Response or an ordered list.
    """

    def __init__(
        self,
        steps: Sequence[Agent | Multiagent | Transform | Aggregate],
        loop: int = 1,
        *,
        id: str = "pipeline",
    ):
        super().__init__(id=id)
        self.steps = tuple(steps)
        if type(loop) is not int or loop < 1:
            raise ValueError("loop must be a positive integer")
        self.loop = loop
        self._validate_steps()

    @property
    def returns_multiple(self):
        return isinstance(self.steps[-1], (Multiagent, Transform)) and self.steps[-1].returns_multiple

    @property
    def accepts_multiple(self):
        first = self.steps[0]
        return isinstance(first, Aggregate) or (
            isinstance(first, Multiagent) and first.accepts_multiple
        )

    @property
    def accepts_prompts(self):
        first = self.steps[0]
        return isinstance(first, Multiagent) and first.accepts_prompts

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("a pipeline requires at least one step")
        if any(not isinstance(step, (Agent, Multiagent, Transform, Aggregate)) for step in self.steps):
            raise TypeError("participants must be Agent, Multiagent, Transform, or Aggregate instances")
        if isinstance(self.steps[0], Transform):
            raise ValueError("the first step must be a query participant or Aggregate")
        self._validate_flow(self.accepts_multiple)

    def _validate_flow(self, multiple):
        for _ in range(self.loop):
            for step in self.steps:
                if isinstance(step, Aggregate):
                    if not multiple:
                        raise TypeError("Aggregate steps require a response list")
                    multiple = False
                elif isinstance(step, Transform):
                    if multiple:
                        raise TypeError("reduce response lists with an Aggregate before a Transform")
                    multiple = step.returns_multiple
                else:
                    if multiple and not (isinstance(step, Multiagent) and step.accepts_multiple):
                        raise TypeError("reduce response lists with an Aggregate before single-message queries")
                    multiple = isinstance(step, Multiagent) and step.returns_multiple

    @staticmethod
    def _run_aggregate(aggregate, responses, records, loop_idx, step_idx):
        record = _query_entry(type(aggregate).__name__, responses, Response(prompt=[], content=""))
        record.update(kind="aggregate", name=type(aggregate).__name__, loop=loop_idx,
                      step=step_idx, input=[_response_snapshot(r) for r in responses],
                      output=None, status="failed", steps=[])
        records.append(record)
        try:
            response = aggregate.aggregate(deepcopy(responses))
            if not isinstance(response, Response) or not isinstance(response.content, str):
                raise TypeError("Aggregate.aggregate() must return a Response with string content")
            response = deepcopy(response)
            record.update(_response_snapshot(response), agent_id=type(aggregate).__name__)
            record.update(output=_response_snapshot(response), status="completed")
            # Text-only reductions have no agent turns; model reductions expose their call.
            agent = getattr(aggregate, "agent", None)
            if isinstance(agent, Agent):
                record["steps"] = deepcopy(agent.history()[-1:])
            return response
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise

    @staticmethod
    def _run_transform(
        transform: Transform,
        response: Response,
        records: list[dict[str, Any]],
        loop_idx: int,
        step_idx: int,
    ) -> Response | list[Response]:
        record = {
            "kind": "transform",
            "name": type(transform).__name__,
            "loop": loop_idx,
            "step": step_idx,
            "input": _response_snapshot(response),
            "output": None,
            "status": "failed",
        }
        records.append(record)
        try:
            transformed = transform.transform(deepcopy(response))
            if transform.returns_multiple:
                if not isinstance(transformed, list) or not transformed or not all(
                    isinstance(item, Response) and isinstance(item.content, str) for item in transformed
                ):
                    raise TypeError("Transform must return a non-empty list of Response objects with string content")
            elif not isinstance(transformed, Response):
                raise TypeError("Transform.transform() must return a Response")
            elif not isinstance(transformed.content, str):
                raise TypeError("Transform output content must be a string")
            # A transform may retain its result. Keep subsequent execution isolated.
            transformed = deepcopy(transformed)
            record["output"] = _response_snapshot(transformed)
            record["status"] = "completed"
            return transformed
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise

    def query(
        self,
        message: str | list[str] | list[Response] = "Continue.",
        use_context: bool = True,
        update_context: bool = True,
    ) -> Response | list[Response]:
        prompts = isinstance(message, list) and bool(message) and all(isinstance(item, str) for item in message)
        if prompts and not self.accepts_prompts:
            raise TypeError("first step does not accept per-participant prompts")
        if not isinstance(message, str) and not prompts and not (
            isinstance(message, list) and message
            and all(isinstance(r, Response) and isinstance(r.content, str) for r in message)
        ):
            raise TypeError("message must be a string or a non-empty list of strings or Response objects")
        if not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean")
        if not isinstance(update_context, bool):
            raise TypeError("update_context must be a boolean")
        self._validate_steps()
        response_list = isinstance(message, list) and not prompts
        self._validate_flow(response_list)
        records = []
        current_response = deepcopy(message) if response_list else None
        try:
            for loop_idx in range(1, self.loop + 1):
                for step_idx, step in enumerate(self.steps, start=1):
                    if isinstance(step, Aggregate):
                        current_response = self._run_aggregate(
                            step, current_response, records, loop_idx, step_idx,
                        )
                    elif isinstance(step, Transform):
                        assert current_response is not None
                        current_response = self._run_transform(
                            step, current_response, records, loop_idx, step_idx,
                        )
                    else:
                        current_message = (message if current_response is None else
                                           current_response if isinstance(current_response, list)
                                           else current_response.content)
                        current_response = _query_step(
                            step, current_message, records,
                            loop_idx=loop_idx, step_idx=step_idx,
                            use_context=use_context, update_context=update_context,
                            allow_multiple=True,
                        )
        except Exception as error:
            _finish_sequence(self._history, self.id, message, records, current_response, error=error)
            raise
        return _finish_sequence(self._history, self.id, message, records, current_response)
