"""Shared query recording and result assembly for ordered execution."""

from copy import deepcopy
from typing import Any

from .agents import Agent
from ._history import _query_entry
from .multiagents.multiagent import Multiagent
from .types import Response


def _query_step(
    participant: Agent | Multiagent[Response],
    message: str,
    records: list[dict[str, Any]],
    *,
    loop_idx: int,
    step_idx: int,
    use_context: bool | None,
    update_context: bool,
) -> Response:
    history_start = len(participant._history)
    try:
        response = participant.query(
            message, use_context=use_context, update_context=update_context,
        )
        if not isinstance(response, Response):
            raise TypeError("query participants must return a single Response")
    except Exception:
        if isinstance(participant, Multiagent) and len(participant._history) > history_start:
            record = deepcopy(participant._history[-1])
            record.update({"kind": "query", "loop": loop_idx, "step": step_idx})
            records.append(record)
        raise
    record = _query_entry(participant.id, message, response)
    if isinstance(participant, Multiagent) and len(participant._history) > history_start:
        record["steps"] = deepcopy(participant._history[-1].get("steps", []))
    record.update({"kind": "query", "loop": loop_idx, "step": step_idx})
    records.append(record)
    return response


def _finish_sequence(
    history: list[dict[str, Any]],
    id: str,
    message: str,
    records: list[dict[str, Any]],
    current_response: Response | None,
    *,
    error: Exception | None = None,
) -> Response:
    def total(field):
        values = [
            record["usage"][field]
            for record in records
            if record["kind"] == "query" and record["usage"][field] is not None
        ]
        return sum(values) if values else None

    content = current_response.content if current_response is not None else ""
    reasoning = current_response.reasoning if current_response is not None else None
    if error is not None and records:
        last = records[-1]
        if last["kind"] == "query" and last["status"] == "failed":
            # A nested executor can complete partial work without returning a response.
            content = last["content"]
            reasoning = last.get("reasoning")
    response = Response(
        prompt=[{"role": "user", "content": message}],
        content=content,
        reasoning=reasoning,
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        generation_time=total("generation_time"),
    )
    entry = _query_entry(id, message, response)
    entry["steps"] = records
    if error is not None:
        entry.update({
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        })
    history.append(entry)
    return response
