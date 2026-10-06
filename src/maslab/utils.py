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
    Entries without a loop number inherit their enclosing group's loop number;
    explicit nested loop numbers take precedence. The fallback is loop 1.

    Content is preserved verbatim. The input is not modified, nothing is printed,
    and an empty history returns an empty string. Use ``print(dialog(history))``
    to display it, or pass ``history[-1:]`` to show only the latest run.
    """
    turns: list[str] = []

    def visit(entries, inherited_loop=1):
        for entry in entries:
            if entry.get("kind") == "transform":
                continue
            loop = entry.get("loop", inherited_loop)
            if "steps" in entry:
                visit(entry["steps"], loop)
            elif entry.get("status") == "completed":
                turns.append(f"[loop {loop} | {entry['agent_id']}]\n{entry['content']}")

    visit(history)
    return "\n\n".join(turns)
