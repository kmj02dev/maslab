import json
from threading import Barrier

import pytest

from maslab import Agent, MeshMultiagent, Response
from conftest import FixedModel


def test_round_barrier_broadcast_order_usage_and_final_results():
    barrier = Barrier(3)

    class RoundModel(FixedModel):
        def respond(self, messages):
            barrier.wait(timeout=5)
            response = super().respond(messages)
            response.content = f"{response.content}:{len(self.calls)}"
            return response

    agents = [Agent(str(i), RoundModel(str(i)), use_context=False) for i in range(3)]
    group = MeshMultiagent(agents, loop=3)
    results = group.query("question", update_context=False)
    assert [r.content for r in results] == ["0:3", "1:3", "2:3"]
    for agent in agents:
        assert agent.model.calls[0][-1]["content"] == "question"
        for round_index in [1, 2]:
            payload = json.loads(agent.model.calls[round_index][-1]["content"])
            assert payload == {"question": "question", "responses": [
                {"agent_id": str(i), "content": f"{i}:{round_index}"} for i in range(3)
            ]}
    history = group.history()
    assert len(history) == 1
    assert len(history[0]["steps"]) == 9
    assert [s["loop"] for s in history[0]["steps"]] == [1]*3 + [2]*3 + [3]*3
    assert history[0]["content"] == [r.content for r in results]
    assert history[0]["usage"]["total_tokens"] == 18
    history[0]["steps"].clear()
    assert len(group.history()[0]["steps"]) == 9


def test_failed_round_preserves_completed_work_and_stops():
    class Failing(FixedModel):
        def respond(self, messages):
            if self.calls:
                raise RuntimeError("failed round two")
            return super().respond(messages)

    agents = [Agent("a", FixedModel("a")), Agent("b", Failing("b"))]
    group = MeshMultiagent(agents, loop=3, max_workers=1)
    with pytest.raises(RuntimeError, match="round two"):
        group.query("question", update_context=False)
    h = group.history()[0]
    assert h["status"] == "failed"
    assert len(h["steps"]) == 4
    assert h["content"] == ["a"]
    assert h["usage"]["total_tokens"] == 6
    assert len(agents[0].history()) == 2


@pytest.mark.parametrize("loop", [0, -1, True, 1.5])
def test_invalid_loop(loop):
    with pytest.raises(ValueError):
        MeshMultiagent([Agent("a", FixedModel("a"))], loop=loop)


def test_one_round_and_repeated_queries_start_from_new_question():
    agent = Agent("a", FixedModel("a"), use_context=False)
    group = MeshMultiagent([agent])
    assert isinstance(group.query("first")[0], Response)
    group.query("second")
    assert agent.model.calls[-1][-1]["content"] == "second"
    assert len(group.history()) == 2
    assert group.returns_multiple
