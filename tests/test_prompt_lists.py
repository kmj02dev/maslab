import json

import pytest

from maslab import Agent, ConcatAggregate, ParallelMultiagent, Pipeline, Response, Wrap
from conftest import FixedModel


def group():
    return ParallelMultiagent([Agent(str(i), FixedModel(str(i))) for i in range(2)])


def test_prompt_distribution_and_history_snapshots():
    parallel = group()
    prompts = ["private A", "private B"]
    answers = parallel.query(prompts, use_context=False, update_context=False)
    assert [r.content for r in answers] == ["0", "1"]
    assert [a.model.calls[0][-1]["content"] for a in parallel.agents] == prompts
    assert [s["message"] for s in parallel.history()[0]["steps"]] == prompts
    prompts[0] = "mutated"
    assert parallel.history()[0]["message"] == ["private A", "private B"]
    json.dumps(parallel.history())


@pytest.mark.parametrize("message,error", [([], TypeError), (["one"], ValueError),
    (["a", "b", "c"], ValueError), (["a", Response(prompt=[], content="b")], TypeError),
    ([1, 2], TypeError)])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_inputs_do_not_call_models(message, error, nested):
    parallel = group()
    target = Pipeline([parallel, ConcatAggregate()]) if nested else parallel
    with pytest.raises(error):
        target.query(message)
    assert all(not a.model.calls for a in parallel.agents)


def test_nested_pipeline_and_loop_broadcast():
    parallel = group()
    pipeline = Pipeline([Pipeline([parallel, ConcatAggregate()]), Wrap("Review:\n", "")], loop=2)
    assert pipeline.accepts_prompts and not pipeline.accepts_multiple
    pipeline.query(["A", "B"], use_context=False, update_context=False)
    for i, agent in enumerate(parallel.agents):
        assert agent.model.calls[0][-1]["content"] == ["A", "B"][i]
        assert agent.model.calls[1][-1]["content"] == "Review:\n[agent 1]\n0\n\n[agent 2]\n1"
    json.dumps(pipeline.history())


@pytest.mark.parametrize("steps", [[Agent("a", FixedModel())], [ConcatAggregate()]])
def test_unsupported_first_step(steps):
    with pytest.raises(TypeError, match="per-participant"):
        Pipeline(steps).query(["A", "B"])


def test_failure_history_retains_individual_messages():
    class FailingModel(FixedModel):
        def respond(self, messages):
            raise RuntimeError("failed")

    parallel = ParallelMultiagent([Agent("a", FixedModel()), Agent("b", FailingModel())])
    pipeline = Pipeline([parallel, ConcatAggregate()])
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.query(["A", "B"])
    assert [s["message"] for s in parallel.history()[0]["steps"]] == ["A", "B"]
    assert pipeline.history()[0]["status"] == "failed"
    json.dumps(pipeline.history())
