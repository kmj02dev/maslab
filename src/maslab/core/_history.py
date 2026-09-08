"""Shared query-history snapshot construction."""

from copy import deepcopy
from typing import Any

from .types import Response


def _response_snapshot(response: Response) -> dict[str, Any]:
    """Copy response details with all model metrics nested under usage."""
    input_tokens = response.input_tokens or 0
    output_tokens = response.output_tokens or 0
    return {
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


def _query_entry(agent_id: str, message: str, response: Response) -> dict[str, Any]:
    """Snapshot one completed query without exposing mutable model responses."""
    return {
        "agent_id": agent_id,
        "message": message,
        **_response_snapshot(response),
        "status": "completed",
    }
