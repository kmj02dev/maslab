"""Ordered composition of queries and response transformations."""

from copy import deepcopy
from typing import Any, Sequence

from ..agents import Agent
from .._execution import _finish_sequence, _query_step
from .._history import _response_snapshot
from .multiagent import Multiagent
from ..transforms import Transform
from ..types import Response


class Pipeline(Multiagent[Response]):
    """Execute queries and transforms in order for complete loop passes.

    The first step must return one response from a query. Transforms receive a
    copy of the preceding Response; query steps receive its content. Final
    processed content crosses loop boundaries and becomes the pipeline answer.
    A pipeline can itself participate in a pipeline or multiagent query.
    """

    def __init__(
        self,
        steps: Sequence[Agent | Multiagent[Response] | Transform],
        loop: int = 1,
        *,
        id: str = "pipeline",
    ):
        super().__init__(id=id)
        self.steps = tuple(steps)
        self._validate_steps()
        if type(loop) is not int or loop < 1:
            raise ValueError("loop must be a positive integer")
        self.loop = loop

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("a pipeline requires at least one query step")
        if any(not isinstance(step, (Agent, Multiagent, Transform)) for step in self.steps):
            raise TypeError("participants must be Agent, Multiagent, or Transform instances")
        if any(isinstance(step, Multiagent) and step.returns_multiple for step in self.steps):
            raise TypeError("reduce parallel responses with an Aggregate before pipeline handoff")
        if isinstance(self.steps[0], Transform):
            raise ValueError("the first step must be an Agent or single-response Multiagent")

    @staticmethod
    def _run_transform(
        transform: Transform,
        response: Response,
        records: list[dict[str, Any]],
        loop_idx: int,
        step_idx: int,
    ) -> Response:
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
            if not isinstance(transformed, Response):
                raise TypeError("Transform.transform() must return a Response")
            if not isinstance(transformed.content, str):
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
        message: str = "Continue.",
        use_context: bool | None = None,
        update_context: bool = True,
    ) -> Response:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if use_context is not None and not isinstance(use_context, bool):
            raise TypeError("use_context must be a boolean or None")
        if not isinstance(update_context, bool):
            raise TypeError("update_context must be a boolean")
        self._validate_steps()
        records = []
        current_response = None
        try:
            for loop_idx in range(1, self.loop + 1):
                for step_idx, step in enumerate(self.steps, start=1):
                    if isinstance(step, Transform):
                        assert current_response is not None
                        current_response = self._run_transform(
                            step, current_response, records, loop_idx, step_idx,
                        )
                    else:
                        current_message = message if current_response is None else current_response.content
                        current_response = _query_step(
                            step, current_message, records,
                            loop_idx=loop_idx, step_idx=step_idx,
                            use_context=use_context, update_context=update_context,
                        )
        except Exception as error:
            _finish_sequence(self._history, self.id, message, records, current_response, error=error)
            raise
        return _finish_sequence(self._history, self.id, message, records, current_response)
