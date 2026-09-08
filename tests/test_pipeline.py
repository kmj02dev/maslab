from dataclasses import replace

import pytest

from maslab import Agent, Multiagent, ParallelMultiagent, Pipeline, Response, SequentialMultiagent, Transform

from conftest import FixedModel


class Append(Transform):
    def __init__(self, suffix="!"):
        self.suffix = suffix

    def transform(self, response):
        return replace(response, content=response.content + self.suffix)


class EchoModel(FixedModel):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def respond(self, messages):
        self.content = self.label + ":" + messages[-1]["content"]
        return super().respond(messages)


def test_sequential_accepts_only_query_participants_and_pipeline_accepts_transforms():
    agent = Agent("agent", FixedModel("A"))
    transform = Append()
    for participants in ([transform], [agent, transform], [transform, agent]):
        with pytest.raises(TypeError, match="Pipeline"):
            SequentialMultiagent(agents=participants)
    assert agent.model.calls == []

    sequential = SequentialMultiagent(agents=[agent])
    pipeline = Pipeline(steps=[sequential, transform])
    assert isinstance(pipeline, Multiagent)
    assert not isinstance(pipeline, SequentialMultiagent)
    assert sequential.agents == (agent,)
    assert pipeline.steps == (sequential, transform)
    assert not hasattr(sequential, "steps")
    assert not hasattr(pipeline, "agents")
    assert pipeline.query("go").content == "A!"
    assert sequential.history()[0]["content"] == "A"
    assert all(record["kind"] == "query" for record in sequential.history()[0]["steps"])
    sequential.agents = (agent, transform)
    with pytest.raises(TypeError, match="Pipeline"):
        sequential.query("stop")
    assert len(agent.model.calls) == 1
    assert len(sequential.history()) == 1


def test_pipeline_and_sequential_nest_both_directions_and_propagate_context_flags():
    agents = [Agent(label, EchoModel(label), "system") for label in "ABCD"]
    first, second, third, last = agents
    team = SequentialMultiagent(agents=[first, second], loop=2)
    pipeline = Pipeline(steps=[team, Append("!"), third, Append("?")], loop=2)
    outer = SequentialMultiagent(agents=[pipeline, last])

    response = outer.query("go", use_context=False, update_context=False)

    assert response.content == "D:C:B:A:B:A:C:B:A:B:A:go!?!?"
    assert [len(agent.model.calls) for agent in agents] == [4, 4, 2, 1]
    assert all(agent.context == [{"role": "system", "content": "system"}] for agent in agents)
    assert response.input_tokens == response.output_tokens == 11
    assert response.generation_time == pytest.approx(0.11)
    entry = outer.history()[0]
    assert entry["usage"]["total_tokens"] == 22
    assert entry["steps"][0]["agent_id"] == "pipeline"
    assert entry["steps"][0]["steps"][1]["output"]["content"] == "B:A:B:A:go!"
    assert entry["steps"][0]["steps"][0]["steps"][0]["message"] == "go"


def test_parallel_finds_shared_agents_through_pipelines_and_sequential_groups():
    shared = Agent("shared", FixedModel())
    team = SequentialMultiagent([shared])
    pipeline = Pipeline([team, Append()])
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([pipeline, shared])
    other = Agent("other", FixedModel())
    group = ParallelMultiagent([pipeline, other])
    team.agents = (other,)
    with pytest.raises(ValueError, match="must not share"):
        group.query("go")
    assert shared.model.calls == other.model.calls == []


def test_query_failure_after_transform_preserves_processed_content_and_stops_pipeline():
    class FailingModel(FixedModel):
        def respond(self, messages):
            raise RuntimeError("query failed")

    first = Agent("first", FixedModel("A"))
    last = Agent("last", FixedModel("B"))
    pipeline = Pipeline([first, Append(), Agent("broken", FailingModel()), last], loop=2)
    with pytest.raises(RuntimeError, match="query failed"):
        pipeline.query("go")
    entry = pipeline.history()[0]
    assert entry["status"] == "failed"
    assert entry["content"] == "A!"
    assert entry["usage"]["total_tokens"] == 2
    assert entry["steps"][1]["output"]["content"] == "A!"
    assert len(first.model.calls) == 1
    assert last.model.calls == []


def test_pipeline_validates_steps_and_query_arguments_before_model_calls():
    with pytest.raises(ValueError, match="at least one"):
        Pipeline([])
    with pytest.raises(TypeError, match="participants"):
        Pipeline([object()])
    agent = Agent("agent", FixedModel())
    pipeline = Pipeline([agent, Append()])
    for kwargs in ({"message": None}, {"use_context": "false"}, {"update_context": "false"}):
        with pytest.raises(TypeError):
            pipeline.query(**kwargs)
    assert agent.model.calls == []
    assert pipeline.history() == []
