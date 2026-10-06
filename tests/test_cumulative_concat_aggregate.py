import importlib

import pytest

from maslab import (
    Agent, CumulativeConcatAggregate, Model, ParallelMultiagent, Pipeline, Response,
)


def test_rounds_snapshots_reset_and_exports():
    aggregate = CumulativeConcatAggregate()
    original = Response([{"role": "user", "content": "question"}], " A ",
                        agent_id="researcher", input_tokens=5, output_tokens=2)
    first = aggregate(iter([original]))
    original.content = "changed"
    original.prompt[0]["content"] = "changed"
    history = aggregate.history()
    history[0][0].content = "changed again"
    history.clear()
    second = aggregate([Response([], ""), Response([], " A ", agent_id="reviewer")])
    assert first.content == "[round 1 | researcher]\n A "
    assert second.content == (
        "[round 1 | researcher]\n A \n\n"
        "[round 2 | agent 1]\n\n\n[round 2 | reviewer]\n A "
    )
    assert aggregate.history()[0][0].prompt[0]["content"] == "question"
    assert aggregate.history()[0][0].input_tokens == 5
    assert second.prompt == [] and second.agent_id is None
    assert second.input_tokens is None and second.output_tokens is None
    assert second.generation_time is None and second.reasoning is None
    aggregate.reset()
    assert aggregate.history() == []
    assert aggregate([Response([], "new")]).content == "[round 1 | agent 1]\nnew"
    for name in ("maslab", "maslab.core", "maslab.core.aggregators"):
        assert getattr(importlib.import_module(name), "CumulativeConcatAggregate") is CumulativeConcatAggregate


@pytest.mark.parametrize("value,error", [
    ([], ValueError), ("", TypeError), ("bad", TypeError), (None, TypeError),
    ([Response([], "valid"), "bad"], TypeError), ([Response([], 12)], TypeError),
    (Response([], 12), TypeError),
])
def test_invalid_input_does_not_change_state(value, error):
    aggregate = CumulativeConcatAggregate()
    aggregate([Response([], "first")])
    before = aggregate.history()
    with pytest.raises(error):
        aggregate(value)
    assert aggregate.history() == before


def test_iterator_failure_does_not_commit_partial_round():
    def broken():
        yield Response([], "partial")
        raise RuntimeError("iterator failed")

    aggregate = CumulativeConcatAggregate()
    with pytest.raises(RuntimeError, match="iterator failed"):
        aggregate(broken())
    assert aggregate.history() == []


def test_pipeline_accumulation_and_usage_across_queries():
    class Fixed(Model):
        def respond(self, messages):
            return Response(list(messages), "answer", input_tokens=3, output_tokens=2)

    agents = [Agent(str(i), Fixed("mock")) for i in range(2)]
    aggregate = CumulativeConcatAggregate()
    pipeline = Pipeline([ParallelMultiagent(agents), aggregate], loop=3)
    response = pipeline.query("question", use_context=False, update_context=False)
    assert len(aggregate.history()) == 3
    assert "[round 2 | 1]" in agents[0].history()[2]["message"]
    assert "[round 3 | 1]" in response.content
    assert response.input_tokens == 18 and response.output_tokens == 12
    assert pipeline.history()[-1]["usage"]["total_tokens"] == 30
    pipeline.query("next", use_context=False, update_context=False)
    assert len(aggregate.history()) == 6
    aggregate.reset()
    assert aggregate.history() == []


def test_single_and_multiple_inputs_share_memory():
    aggregate = CumulativeConcatAggregate()
    source = Response([], "one", agent_id="a")
    aggregate(source)
    source.content = "changed"
    aggregate([Response([], "two"), Response([], "three")])
    result = aggregate(Response([], "four"))
    assert [len(batch) for batch in aggregate.history()] == [1, 2, 1]
    assert "one" in result.content and "changed" not in result.content
    assert all(word in result.content for word in ("two", "three", "four"))


def test_shared_single_response_aggregate_in_pipeline():
    class Fixed(Model):
        def respond(self, messages):
            return Response(list(messages), self.name, input_tokens=3, output_tokens=2)

    a, b = Agent("a", Fixed("A")), Agent("b", Fixed("B"))
    memory = CumulativeConcatAggregate()
    pipeline = Pipeline([a, memory, b, memory], loop=2)
    result = pipeline.query("start", use_context=False, update_context=False)
    assert [batch[0].content for batch in memory.history()] == ["A", "B", "A", "B"]
    assert "[round 2 | b]\nB" in a.history()[1]["message"]
    assert "[round 4 | b]\nB" in result.content
    assert result.input_tokens == 12 and result.output_tokens == 8
    assert pipeline.history()[-1]["usage"]["total_tokens"] == 20
    import json
    json.dumps(pipeline.history())
    assert isinstance(pipeline.history()[-1]["steps"][1]["input"], list)


def test_list_only_aggregate_and_leading_input_contracts():
    from maslab import ConcatAggregate

    class Fixed(Model):
        def respond(self, messages):
            return Response(list(messages), "answer")

    with pytest.raises(TypeError):
        Pipeline([Agent("a", Fixed("mock")), ConcatAggregate()])
    with pytest.raises(TypeError):
        ConcatAggregate()(Response([], "answer"))
    memory = CumulativeConcatAggregate()
    pipeline = Pipeline([memory])
    with pytest.raises(TypeError):
        pipeline.query("not a response")
    assert memory.history() == []
    assert "answer" in pipeline.query([Response([], "answer")]).content
