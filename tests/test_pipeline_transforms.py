from copy import deepcopy
from dataclasses import replace
import json

import pytest

from maslab import (
    Agent, Aggregate, Broadcast, ConcatAggregate, CumulativeConcatAggregate,
    LLMAggregate, ParallelMultiagent, Pipeline, Transform, WrapBroadcast, dialog,
)
from conftest import FixedModel


class RoundModel(FixedModel):
    def respond(self, messages):
        response = super().respond(messages)
        return replace(response, content=f"{self.content}-{len(self.calls)}",
                       reasoning="original reasoning")


@pytest.mark.parametrize("loops", [1, 2, 4])
def test_round_transforms_preserve_final_answers_and_accumulate_only_between_rounds(loops):
    agents = [Agent(name, RoundModel(name)) for name in ("A", "B")]
    memory = CumulativeConcatAggregate()
    pipeline = Pipeline(
        [ParallelMultiagent(agents)], loop=loops,
        transforms=[memory, WrapBroadcast(prefix=["task-A\n", "task-B\n"], suffix="\nreview")],
    )
    responses = pipeline.query(["task-A", "task-B"], use_context=False, update_context=False)

    assert pipeline.returns_multiple
    assert [r.content for r in responses] == [f"{a.id}-{loops}" for a in agents]
    assert [r.agent_id for r in responses] == ["A", "B"]
    assert all(r.reasoning == "original reasoning" and r.input_tokens == 1 for r in responses)
    assert len(memory.history()) == loops - 1
    for agent, response in zip(agents, responses):
        assert len(agent.model.calls) == loops
        assert response.prompt == agent.model.calls[-1]
        assert agent.context == []
        assert agent.model.calls[0][-1]["content"] == f"task-{agent.id}"
        for index, call in enumerate(agent.model.calls[1:], start=1):
            message = call[-1]["content"]
            assert message.startswith(f"task-{agent.id}\n") and message.endswith("\nreview")
            for prior in range(1, index + 1):
                assert f"A-{prior}" in message and f"B-{prior}" in message
            assert f"A-{index + 1}" not in message

    history = json.loads(json.dumps(pipeline.history()))
    before = deepcopy(history)
    assert history[0]["content"] == [r.content for r in responses]
    assert history[0]["usage"]["total_tokens"] == loops * 4
    assert dialog(history) == "\n\n".join(
        f"[loop {i} | {name}]\n{name}-{i}"
        for i in range(1, loops + 1) for name in ("A", "B")
    )
    assert history == before
    responses[0].content = "changed"
    assert pipeline.history()[0]["content"][0] == f"A-{loops}"


def test_return_shape_and_nested_usage_follow_steps_not_between_round_transforms():
    inner = Pipeline([ParallelMultiagent([Agent("A", FixedModel("answer"))])],
                     loop=3, transforms=[ConcatAggregate()])
    outer = Pipeline([inner, ConcatAggregate()])
    assert inner.returns_multiple
    assert outer.query("question").content == "[agent 1]\nanswer"
    assert outer.history()[0]["usage"]["total_tokens"] == 6
    assert [r.agent_id for r in inner.query("again")] == ["A"]
    assert "[loop 1 | A]" in dialog(inner.history()[-1:])
    assert "[loop 3 | A]" in dialog(inner.history()[-1:])


def test_model_aggregate_runs_between_rounds_and_keeps_usage_and_loop_labels():
    judge = LLMAggregate(FixedModel("advice"), id="judge")
    pipeline = Pipeline([ParallelMultiagent([Agent("A", FixedModel("answer"))])],
                        loop=3, transforms=[judge])
    responses = pipeline.query("question")
    assert [r.content for r in responses] == ["answer"]
    assert len(judge.history()) == 2
    assert pipeline.history()[0]["usage"]["total_tokens"] == 10
    assert dialog(pipeline.history()) == (
        "[loop 1 | A]\nanswer\n\n[loop 1 | judge]\nadvice\n\n"
        "[loop 2 | A]\nanswer\n\n[loop 2 | judge]\nadvice\n\n"
        "[loop 3 | A]\nanswer"
    )


@pytest.mark.parametrize("failure", ["aggregate", "transform"])
def test_between_round_failure_preserves_completed_calls_and_stops(failure):
    class BrokenAggregate(Aggregate):
        def aggregate(self, responses):
            responses[0].content = "mutated"
            raise RuntimeError("cannot prepare next round")

    class BrokenTransform(Transform):
        def transform(self, response):
            response.content = "mutated"
            raise RuntimeError("cannot prepare next round")

    agent = Agent("A", FixedModel("answer"))
    transforms = ([BrokenAggregate()] if failure == "aggregate"
                  else [ConcatAggregate(), BrokenTransform()])
    pipeline = Pipeline([ParallelMultiagent([agent])], loop=4, transforms=transforms)
    with pytest.raises(RuntimeError, match="cannot prepare"):
        pipeline.query("question")
    assert len(agent.model.calls) == 1
    entry = pipeline.history()[0]
    assert entry["status"] == "failed"
    assert entry["usage"]["total_tokens"] == 2
    assert entry["steps"][-1]["status"] == "failed"
    assert entry["steps"][-1]["output"] is None
    assert "mutated" not in json.dumps(entry)
    assert dialog([entry]) == "[loop 1 | A]\nanswer"


def test_later_round_query_failure_keeps_partial_outputs_with_correct_loop():
    class FailReview(RoundModel):
        def respond(self, messages):
            if self.calls:
                raise RuntimeError("review failed")
            return super().respond(messages)

    agents = [Agent("A", RoundModel("A")), Agent("B", FailReview("B"))]
    memory = CumulativeConcatAggregate()
    pipeline = Pipeline([ParallelMultiagent(agents)], loop=4, transforms=[memory])
    with pytest.raises(RuntimeError, match="review failed"):
        pipeline.query("question")
    entry = pipeline.history()[0]
    assert entry["status"] == "failed"
    assert entry["content"] == ["A-2"]
    assert entry["usage"]["total_tokens"] == 6
    assert len(memory.history()) == 1
    assert dialog([entry]) == (
        "[loop 1 | A]\nA-1\n\n[loop 1 | B]\nB-1\n\n[loop 2 | A]\nA-2"
    )


def test_transforms_are_validated_but_unused_handoffs_are_skipped_for_one_loop():
    agent = Agent("A", FixedModel("answer"))
    for invalid in (object(), agent):
        with pytest.raises(TypeError, match="transforms"):
            Pipeline([agent], transforms=[invalid])
    with pytest.raises(TypeError, match="Aggregate"):
        Pipeline([agent], loop=2, transforms=[Broadcast(2)])
    with pytest.raises(TypeError, match="Aggregate"):
        Pipeline([ParallelMultiagent([agent])], loop=2, transforms=[Broadcast(2)])
    assert agent.model.calls == []
    pipeline = Pipeline([agent], loop=1, transforms=[Broadcast(2)])
    assert not pipeline.returns_multiple
    assert pipeline.query("question").content == "answer"
    assert len(pipeline.history()[0]["steps"]) == 1


@pytest.mark.parametrize("shared", [CumulativeConcatAggregate(), Broadcast(1)])
def test_parallel_rejects_shared_between_round_state(shared):
    branches = [Pipeline([Agent(name, FixedModel())], loop=2,
                         transforms=[shared, ConcatAggregate()] if isinstance(shared, Transform)
                         else [shared]) for name in ("A", "B")]
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent(branches)


def test_parallel_finds_agents_inside_between_round_aggregates():
    judge = LLMAggregate(FixedModel())
    branch = Pipeline([Agent("A", FixedModel())], loop=2,
                      transforms=[Broadcast(1), judge])
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([branch, judge.agent])
