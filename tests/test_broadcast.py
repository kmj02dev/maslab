import json

import pytest

from maslab import (Agent, Broadcast, WrapBroadcast, Response, Pipeline,
                    ParallelMultiagent, ConcatAggregate, Transform)
from conftest import FixedModel


@pytest.mark.parametrize("prefix,suffix,expected", [
    (None, None, ["X"]), ("a", "b", ["aXb"]),
    (["a", "b"], "!", ["aX!", "bX!"]),
    ("!", ["a", "b"], ["!Xa", "!Xb"]),
    (["a", "b"], ["c", "d"], ["aXc", "bXd"]),
])
def test_wrap(prefix, suffix, expected):
    assert [r.content for r in WrapBroadcast(prefix, suffix).transform(Response([], "X"))] == expected


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        Broadcast(count)


@pytest.mark.parametrize("prefix,suffix,error", [([], None, ValueError),
    (None, [], ValueError), (["a"], ["b", "c"], ValueError),
    ([1], None, TypeError), (None, 1, TypeError), (("a",), None, TypeError)])
def test_invalid_wrappers(prefix, suffix, error):
    with pytest.raises(error):
        WrapBroadcast(prefix, suffix)


@pytest.mark.parametrize("transform", [Broadcast(2), WrapBroadcast(["a", "b"])])
def test_independence_and_usage(transform):
    source = Response([{"role": "user", "content": "original"}], "X", input_tokens=5, agent_id="a")
    results = transform.transform(source)
    results[0].prompt[0]["content"] = "changed"
    assert results[1].prompt == source.prompt
    assert all(r.input_tokens is None and r.agent_id is None for r in results)


def test_wrapper_copies_configuration():
    prefixes = ["a", "b"]
    transform = WrapBroadcast(prefixes)
    prefixes.clear()
    assert len(transform.transform(Response([], "X"))) == 2


def test_loop_distribution_history_and_usage():
    agents = [Agent(str(i), FixedModel(str(i))) for i in range(2)]
    pipeline = Pipeline([ParallelMultiagent(agents), ConcatAggregate(),
                         WrapBroadcast(["task A\n", "task B\n"], "\nreview")], loop=2)
    results = pipeline.query(["task A", "task B"], use_context=False, update_context=False)
    assert pipeline.returns_multiple
    for i, agent in enumerate(agents):
        assert agent.model.calls[1][-1]["content"] == ["task A\n", "task B\n"][i] + "[agent 1]\n0\n\n[agent 2]\n1\nreview"
    assert len(results) == 2
    assert pipeline.history()[0]["usage"]["input_tokens"] == 4
    assert all(r.input_tokens is None for r in results)
    json.dumps(pipeline.history())


def test_nested_output_and_aggregate():
    inner = Pipeline([Agent("source", FixedModel("X")), Broadcast(2)])
    agents = [Agent(str(i), FixedModel()) for i in range(2)]
    outer = Pipeline([inner, Pipeline([ParallelMultiagent(agents), ConcatAggregate()])])
    outer.query("start")
    assert all(a.model.calls[0][-1]["content"] == "X" for a in agents)
    reduced = Pipeline([inner, ConcatAggregate()]).query("start")
    assert reduced.content == "[agent 1]\nX\n\n[agent 2]\nX"


def test_invalid_flow_and_count():
    with pytest.raises(TypeError):
        Pipeline([Agent("a", FixedModel()), Broadcast(2), Agent("b", FixedModel())])
    target = Agent("target", FixedModel())
    pipeline = Pipeline([Agent("a", FixedModel()), Broadcast(2), ParallelMultiagent([target])])
    with pytest.raises(ValueError, match="count"):
        pipeline.query("start")
    assert not target.model.calls
    assert pipeline.history()[0]["status"] == "failed"


@pytest.mark.parametrize("output", [[], ["bad"], Response([], "bad"), [Response([], 42)]])
def test_invalid_transform_output_recorded(output):
    class Broken(Transform):
        returns_multiple = True
        def transform(self, response):
            return output
    pipeline = Pipeline([Agent("a", FixedModel()), Broken()])
    with pytest.raises(TypeError):
        pipeline.query("start")
    assert pipeline.history()[0]["steps"][-1]["status"] == "failed"


def test_parallel_distributes_responses_and_pipeline_preserves_them():
    agents = [Agent(str(i), FixedModel()) for i in range(2)]
    parallel = ParallelMultiagent(agents)
    inputs = [Response([], "A"), Response([], "B")]
    Pipeline([parallel]).query(inputs, use_context=False, update_context=False)
    assert [a.model.calls[0][-1]["content"] for a in agents] == ["A", "B"]
    assert [r["content"] for r in parallel.history()[0]["message"]] == ["A", "B"]
    with pytest.raises(ValueError, match="count"):
        parallel.query([Response([], "only one")])
    assert all(len(a.model.calls) == 1 for a in agents)
