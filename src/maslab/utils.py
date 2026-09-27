"""Text views of recorded agent conversations."""

from collections.abc import Iterable, Mapping
from typing import Any


def dialog(history: Iterable[Mapping[str, Any]]) -> str:
    """Return completed outputs as ``[loop i | agent_id]\ncontent`` blocks.

    Accepts Agent, SequentialMultiagent, ParallelMultiagent, and Pipeline
    histories, including snapshots loaded from JSON. Nested groups are expanded
    in recorded order; parallel branches follow participant order, not completion
    time. Group summaries, transforms, prompts, reasoning, usage, and failed
    leaf calls are omitted. Completed steps of failed groups remain visible.

    Content is preserved verbatim. The input is not modified, nothing is printed,
    and an empty history returns an empty string. Use ``print(dialog(history))``
    to display it, or pass ``history[-1:]`` to show only the latest run.
    """
    turns: list[str] = []

    def visit(entries):
        for entry in entries:
            if entry.get("kind") == "transform":
                continue
            if "steps" in entry:
                visit(entry["steps"])
            elif entry.get("status") == "completed":
                loop = entry.get("loop", 1)
                turns.append(f"[loop {loop} | {entry['agent_id']}]\n{entry['content']}")

    visit(history)
    return "\n\n".join(turns)
