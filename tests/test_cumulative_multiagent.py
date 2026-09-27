import json

import pytest

from maslab import (
    Agent, CumulativeMultiagent, ParallelMultiagent, ConcatAggregate, Pipeline,
    Response, SequentialMultiagent, dialog,
)
from conftest import FixedModel


class NumberedModel(FixedModel):
    def respond(self, messages):
        response = super().respond(messages)
        response.content = f"{self.content}:{len(self.calls)}"
        return response


def test_every_preceding_output_is_visible_including_same_round_and_self():
    agents = [Agent(f"agent_{i}", NumberedModel(f"의견 {i}"), f"private-{i}") for i in range(4)]
    group = CumulativeMultiagent(agents, loop=2)
    result = group.query('질문 "Q"', use_context=False, update_context=False)
    responses = []
    for round_idx in range(1, 3):
        for agent in agents:
            prompt = agent.model.calls[round_idx - 1]
            assert len(prompt) == 2
            assert prompt[0] == {"role": "system", "content": agent.context[0]["content"]}
            if not responses:
                assert prompt[-1]["content"] == '질문 "Q"'
            else:
                assert json.loads(prompt[-1]["content"]) == {"responses": responses}
            responses.append({
                "round": round_idx, "agent_id": agent.id,
                "content": f"{agent.model.content}:{round_idx}",
            })
    assert result.content == "의견 3:2"
    assert result.input_tokens == result.output_tokens == 8
    assert result.generation_time == pytest.approx(0.08)
    assert [len(a.context) for a in agents] == [1] * 4
    assert [len(a.history()) for a in agents] == [2] * 4
    steps = group.history()[0]["steps"]
    assert [(s["loop"], s["step"]) for s in steps] == [(r, a) for r in [1, 2] for a in range(1, 5)]
    assert group.history()[0]["usage"]["total_tokens"] == 16
    assert dialog(group.history()).count("| agent_") == 8
    saved = group.history()
    saved[0]["steps"][0]["prompt"].clear()
    assert group.history()[0]["steps"][0]["prompt"]


def test_repeated_queries_reset_responses_and_do_not_reuse_private_context_when_disabled():
    agent = Agent("a", FixedModel("answer"), "private")
    agent.query("outside discussion")
    group = CumulativeMultiagent([agent], loop=2)
    group.query("first", use_context=False, update_context=False)
    before = agent.history()
    group.query("second", use_context=False, update_context=False)
    assert agent.model.calls[-2][-1]["content"] == "second"
    assert len(agent.model.calls[-2]) == 2
    assert agent.history()[:3] == before
    assert len(group.history()) == 2


def test_context_flags_follow_existing_query_contract():
    agent = Agent("a", FixedModel("answer"), "private")
    group = CumulativeMultiagent([agent], loop=2)
    group.query("go")
    assert [len(p) for p in agent.model.calls] == [2, 4]
    assert len(agent.context) == 5
    group.query("new", use_context=False, update_context=False)
    assert len(agent.context) == 5
    assert all(len(p) == 2 for p in agent.model.calls[-2:])


def test_pipeline_composition_keeps_nested_history_and_counts_usage_once():
    a, b, c = [Agent(label, FixedModel(label)) for label in "abc"]
    inner = SequentialMultiagent([a, b], id="inner")
    cumulative = CumulativeMultiagent([inner, c], loop=2)
    pipeline = Pipeline([cumulative, Agent("editor", FixedModel("final"))])
    result = pipeline.query("go", use_context=False, update_context=False)
    assert isinstance(result, Response)
    assert result.content == "final"
    assert result.input_tokens == result.output_tokens == 7
    assert json.loads(c.model.calls[0][-1]["content"])["responses"] == [
        {"round": 1, "agent_id": "inner", "content": "b"},
    ]
    assert len(cumulative.history()[0]["steps"][0]["steps"]) == 2
    assert dialog(pipeline.history()).splitlines()[0] == "[loop 1 | a]"
    assert pipeline.history()[0]["usage"]["total_tokens"] == 14


def test_nested_failure_retains_completed_work_and_next_query_starts_fresh():
    class FailsOnce(FixedModel):
        def respond(self, messages):
            if not self.calls:
                self.calls.append(messages)
                raise RuntimeError("temporary error")
            return super().respond(messages)

    a, b, c = Agent("a", FixedModel("a")), Agent("b", FailsOnce("b")), Agent("c", FixedModel("c"))
    inner = Pipeline([a, b], id="inner")
    group = CumulativeMultiagent([inner, c])
    with pytest.raises(RuntimeError, match="temporary"):
        group.query("failed", use_context=False, update_context=False)
    entry = group.history()[0]
    assert entry["status"] == "failed"
    assert entry["content"] == "a"
    assert entry["usage"]["total_tokens"] == 2
    assert entry["steps"][0]["status"] == "failed"
    assert c.model.calls == []
    group.query("retry", use_context=False, update_context=False)
    assert a.model.calls[-1][-1]["content"] == "retry"
    assert len(group.history()) == 2


@pytest.mark.parametrize("loop", [0, -1, True, 1.5])
def test_invalid_loop(loop):
    with pytest.raises(ValueError, match="positive integer"):
        CumulativeMultiagent([Agent("a", FixedModel())], loop=loop)


def test_invalid_participants_and_inputs_fail_before_inference():
    agent = Agent("a", FixedModel())
    with pytest.raises(ValueError, match="at least one"):
        CumulativeMultiagent([])
    with pytest.raises(TypeError, match="participants"):
        CumulativeMultiagent([object()])
    with pytest.raises(TypeError):
        CumulativeMultiagent([ParallelMultiagent([agent])])
    with pytest.raises(TypeError, match="string message"):
        CumulativeMultiagent([Pipeline([ConcatAggregate(), agent])])
    group = CumulativeMultiagent([agent])
    for kwargs in [{"message": None}, {"use_context": 1}, {"update_context": "false"}]:
        with pytest.raises(TypeError):
            group.query(**kwargs)
    assert agent.model.calls == []
    assert group.history() == []


def test_public_exports_are_the_same_class():
    from maslab.core import CumulativeMultiagent as CoreCumulative
    from maslab.core.multiagents import CumulativeMultiagent as PackageCumulative
    assert CoreCumulative is PackageCumulative is CumulativeMultiagent
