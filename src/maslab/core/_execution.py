"""Shared query recording and result assembly for ordered execution."""

from copy import deepcopy
from typing import Any

from .agents import Agent
from ._history import _query_entry
from .multiagents.multiagent import Multiagent
from .types import Response


def _query_step(
    participant: Agent | Multiagent[Response],
    message: str | list[str] | list[Response],
    records: list[dict[str, Any]],
    *,
    loop_idx: int,
    step_idx: int,
    use_context: bool,
    update_context: bool,
    allow_multiple: bool = False,
) -> Response | list[Response]:
    history_start = len(participant._history)
    try:
        response = participant.query(
            deepcopy(message), use_context=use_context, update_context=update_context,
        )
        multiple = (allow_multiple and isinstance(response, list) and bool(response)
                    and all(isinstance(item, Response) and isinstance(item.content, str)
                            for item in response))
        if not isinstance(response, Response) and not multiple:
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
        record["usage"] = deepcopy(participant._history[-1]["usage"])
    record.update({"kind": "query", "loop": loop_idx, "step": step_idx})
    records.append(record)
    return response


def _finish_sequence(
    history: list[dict[str, Any]],
    id: str,
    message: str | list[str] | list[Response],
    records: list[dict[str, Any]],
    current_response: Response | list[Response] | None,
    *,
    error: Exception | None = None,
) -> Response | list[Response]:
    def total(field):
        values = [
            record["usage"][field]
            for record in records
            if record["kind"] in {"query", "aggregate"} and record["usage"][field] is not None
        ]
        return sum(values) if values else None

    if isinstance(current_response, list):
        response = deepcopy(current_response)
        entry = _query_entry(id, message, response)
        entry["usage"] = {field: total(field) for field in
                          ("input_tokens", "output_tokens", "total_tokens", "generation_time")}
        entry["steps"] = records
        if error is not None:
            if records and records[-1]["kind"] == "query" and records[-1]["status"] == "failed":
                entry["content"] = deepcopy(records[-1]["content"])
                entry["reasoning"] = deepcopy(records[-1].get("reasoning"))
            entry.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        history.append(entry)
        return response

    content = current_response.content if current_response is not None else ""
    reasoning = current_response.reasoning if current_response is not None else None
    if error is not None and records:
        last = records[-1]
        if last["kind"] == "query" and last["status"] == "failed":
            # A nested executor can complete partial work without returning a response.
            content = last["content"]
            reasoning = last.get("reasoning")
    response = Response(
        prompt=[{"role": "user", "content": message}] if isinstance(message, str) else [],
        content=content,
        reasoning=reasoning,
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        generation_time=total("generation_time"),
        agent_id=current_response.agent_id if current_response is not None else None,
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
