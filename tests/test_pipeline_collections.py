from copy import deepcopy
import json

import pytest

from maslab import (
    Agent, Aggregate, LLMAggregate, ConcatAggregate, MeshMultiagent,
    ParallelMultiagent, Pipeline, Response, SequentialMultiagent,
    Suffix, Wrap, dialog,
)
from conftest import FixedModel


def test_pipeline_reduces_collections_and_counts_only_generated_usage():
    agents = [Agent(str(i), FixedModel(text)) for i, text in enumerate(["A", "B", "A"])]
    last = Agent("last", FixedModel("final"))
    pipeline = Pipeline([ParallelMultiagent(agents), ConcatAggregate(), Wrap(prefix="", suffix="!"), last], loop=2)
    result = pipeline.query("question", use_context=False, update_context=False)
    assert not pipeline.returns_multiple
    assert result.content == "final"
    assert result.agent_id == "last"
    assert result.input_tokens == result.output_tokens == 8
    assert result.generation_time == pytest.approx(0.08)
    assert [prompt[-1]["content"] for prompt in last.model.calls] == ["[agent 1]\nA\n\n[agent 2]\nB\n\n[agent 3]\nA!"] * 2
    record = pipeline.history()[0]["steps"][1]
    assert record["usage"]["total_tokens"] == 0
    assert record["agent_id"] == "ConcatAggregate"
    assert record["output"]["agent_id"] is None
    assert [item["agent_id"] for item in record["message"]] == ["0", "1", "2"]
    assert all(not agent.context for agent in [*agents, last])
    assert "ConcatAggregate" not in dialog(pipeline.history())
    json.dumps(pipeline.history())


def test_pipeline_can_return_lists_and_nested_usage_includes_all_mesh_rounds():
    agents = [Agent(str(i), FixedModel(str(i))) for i in range(2)]
    inner = Pipeline([MeshMultiagent(agents, loop=3)])
    assert inner.returns_multiple
    with pytest.raises(TypeError, match="Aggregate"):
        SequentialMultiagent([inner])
    outer = Pipeline([inner, ConcatAggregate()])
    combined = outer.query("question")
    assert combined.input_tokens == 6
    assert combined.agent_id is None
    assert inner.history()[0]["usage"]["total_tokens"] == 12
    result = inner.query("another")
    assert [response.agent_id for response in result] == ["0", "1"]
    result[0].content = "changed"
    assert inner.history()[-1]["content"] == ["0", "1"]


def test_pipeline_llm_aggregate_usage_and_actual_judge_dialog():
    judge = LLMAggregate(FixedModel("verdict"), id="judge")
    pipeline = Pipeline([ParallelMultiagent([Agent("a", FixedModel()), Agent("b", FixedModel())]), judge])
    result = pipeline.query("question")
    assert result.input_tokens == result.output_tokens == 3
    assert result.agent_id == "judge"
    assert result.generation_time == pytest.approx(0.03)
    assert len(judge.history()) == 1
    record = pipeline.history()[0]["steps"][-1]
    assert record["agent_id"] == "LLMAggregate"
    assert record["output"]["agent_id"] == record["steps"][0]["agent_id"] == "judge"
    assert [item["agent_id"] for item in record["input"]] == ["a", "b"]
    assert "[loop 1 | judge]\nverdict" in dialog(pipeline.history())


def test_formatted_mesh_uses_all_previous_responses_and_keeps_own_context():
    class RoundModel(FixedModel):
        def respond(self, messages):
            response = super().respond(messages)
            response.content = f"{self.content}-round-{len(self.calls)}"
            return response

    agents = [Agent(str(i), RoundModel(f"answer-{i}")) for i in range(3)]
    review = MeshMultiagent([
        Pipeline([ConcatAggregate(), Suffix("\nOriginal question: Q"), agent], id=f"review-{i}")
        for i, agent in enumerate(agents)
    ], loop=2)
    pipeline = Pipeline([MeshMultiagent(agents, loop=1), review])
    results = pipeline.query("Q")
    assert [r.content for r in results] == [f"answer-{i}-round-3" for i in range(3)]
    assert [r.agent_id for r in results] == ["0", "1", "2"]
    for i, agent in enumerate(agents):
        assert len(agent.model.calls) == 3
        for round_index in [1, 2]:
            prompt = agent.model.calls[round_index]
            message = prompt[-1]["content"]
            assert message.endswith("Original question: Q")
            for j in range(3):
                assert f"answer-{j}-round-{round_index}" in message
                assert f"answer-{j}-round-{round_index + 1}" not in message
            assert any(m["role"] == "assistant" and m["content"] == f"answer-{i}-round-{round_index}"
                       for m in prompt[:-1])
    assert pipeline.history()[0]["usage"]["total_tokens"] == 18
    assert len([line for line in dialog(pipeline.history()).splitlines() if line.startswith("[")]) == 9
    assert "ConcatAggregate" not in dialog(pipeline.history())
    json.dumps(pipeline.history())


def test_single_agent_formatting_and_context_flags():
    agent = Agent("a", FixedModel("answer"), "system")
    pipeline = Pipeline([
        MeshMultiagent([agent]),
        MeshMultiagent([Pipeline([ConcatAggregate(), agent])]),
    ])
    pipeline.query("Q", use_context=False, update_context=False)
    assert agent.model.calls[-1][-1]["content"] == "[agent 1]\nanswer"
    assert len(agent.model.calls[-1]) == 2
    assert agent.context == [{"role": "system", "content": "system"}]


def test_collection_inputs_are_copied_and_their_usage_is_not_charged_again():
    class Mutating(Aggregate):
        def aggregate(self, responses):
            next(iter(responses)).content = "changed"
            return Response(prompt=[], content="message")

    inputs = [Response(prompt=[], content="original", input_tokens=999)]
    other = Agent("other", FixedModel("done"))
    group = ParallelMultiagent([
        Pipeline([Mutating(), Agent("first", FixedModel())]),
        Pipeline([ConcatAggregate(), other]),
    ], max_workers=1)
    assert group.accepts_multiple
    group.query(inputs)
    assert inputs[0].content == "original"
    assert other.model.calls[0][-1]["content"] == "[agent 1]\noriginal"
    assert group.history()[0]["usage"]["total_tokens"] == 4
    json.dumps(group.history())


def test_collection_failure_keeps_partial_second_round_usage_and_dialog():
    class Failing(FixedModel):
        def respond(self, messages):
            if self.calls:
                raise RuntimeError("review failed")
            return super().respond(messages)

    agents = [Agent("a", FixedModel("A")), Agent("b", Failing("B"))]
    pipeline = Pipeline([
        MeshMultiagent(agents, max_workers=1),
        MeshMultiagent([Pipeline([ConcatAggregate(), agent]) for i, agent in enumerate(agents)], max_workers=1),
    ])
    with pytest.raises(RuntimeError, match="review failed"):
        pipeline.query("Q")
    entry = pipeline.history()[0]
    assert entry["status"] == "failed"
    assert entry["usage"]["total_tokens"] == 6
    assert entry["content"] == ["A"]
    assert dialog([entry]) == (
        "[loop 1 | a]\nA\n\n[loop 1 | b]\nB\n\n[loop 1 | a]\nA"
    )


def test_invalid_handoffs_fail_before_inference():
    agent = Agent("a", FixedModel())
    with pytest.raises(TypeError, match="Aggregate"):
        Pipeline([ParallelMultiagent([agent]), agent])
    with pytest.raises(TypeError, match="list"):
        Pipeline([agent, ConcatAggregate()])
    pipeline = Pipeline([ConcatAggregate(), agent])
    with pytest.raises(TypeError, match="list"):
        pipeline.query("not a response list")
    with pytest.raises(TypeError):
        pipeline.query(["text instead of Response"])
    assert agent.model.calls == []


def test_failed_aggregate_preserves_inputs_costs_and_history():
    class Broken(Aggregate):
        def aggregate(self, responses):
            next(iter(responses)).content = "mutated"
            raise RuntimeError("bad reduction")

    first = ParallelMultiagent([Agent("a", FixedModel("A"))])
    last = Agent("last", FixedModel())
    pipeline = Pipeline([first, Broken(), last])
    with pytest.raises(RuntimeError, match="bad reduction"):
        pipeline.query("Q")
    history = pipeline.history()[0]
    assert history["content"] == ["A"]
    assert history["usage"]["total_tokens"] == 2
    assert history["steps"][-1]["input"][0]["content"] == "A"
    assert history["steps"][-1]["output"] is None
    assert last.model.calls == []
    json.dumps(history)
