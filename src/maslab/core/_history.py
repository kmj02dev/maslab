"""Shared query-history snapshot construction."""

from copy import deepcopy
from typing import Any

from .types import Response


def _response_snapshot(response: Response | list[Response]) -> dict[str, Any]:
    """Copy response details with all model metrics nested under usage."""
    if isinstance(response, list):
        snapshots = [_response_snapshot(item) for item in response]
        times = [item["usage"]["generation_time"] for item in snapshots
                 if item["usage"]["generation_time"] is not None]
        return {
            "prompt": [],
            "content": [item["content"] for item in snapshots],
            "reasoning": None,
            "usage": {
                **{field: sum(item["usage"][field] for item in snapshots)
                   for field in ("input_tokens", "output_tokens", "total_tokens")},
                "generation_time": sum(times) if times else None,
            },
        }
    input_tokens = response.input_tokens or 0
    output_tokens = response.output_tokens or 0
    return {
        "agent_id": response.agent_id,
        "prompt": deepcopy(response.prompt),
        "content": response.content,
        "reasoning": response.reasoning,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "generation_time": response.generation_time,
        },
    }


def _query_entry(agent_id: str, message: str | list[Response], response: Response | list[Response]) -> dict[str, Any]:
    """Snapshot one completed query without exposing mutable model responses."""
    return {
        **_response_snapshot(response),
        "agent_id": agent_id,
        "message": _message_snapshot(message),
        "status": "completed",
    }


def _message_snapshot(message: str | list[Response]):
    """Keep collection inputs JSON-serializable and independent of callers."""
    return ([_response_snapshot(item) for item in message]
            if isinstance(message, list) else message)
