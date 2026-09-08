from dataclasses import replace
import json

import pytest

from maslab import Agent, ParallelMultiagent, Pipeline, Response, SequentialMultiagent, Transform
from maslab.core.transforms import Transform as PackageTransform
from maslab.core.transforms.transform import Transform as ImplementationTransform

from conftest import FixedModel


class Append(Transform):
    def __init__(self, suffix):
        self.suffix = suffix
        self.calls = 0

    def transform(self, response):
        self.calls += 1
        return replace(response, content=response.content + self.suffix)


class EchoModel(FixedModel):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def respond(self, messages):
        self.content = self.label + ":" + messages[-1]["content"]
        self.response = super().respond(messages)
        return self.response


def test_transform_is_an_abstract_public_interface():
    assert Transform is PackageTransform is ImplementationTransform
    with pytest.raises(TypeError, match="abstract"):
        Transform()
    assert Append("!").transform(Response(prompt=[], content="A")).content == "A!"


def test_explicit_steps_transform_handoffs_consecutive_results_and_loop_boundaries():
    first = Agent("first", EchoModel("A"), "system A")
    second = Agent("second", EchoModel("B"), "system B")
    x, y, z = Append("|x"), Append("|y"), Append("|z")
    group = Pipeline(steps=[first, x, y, second, z], loop=2)

    response = group.query("go", use_context=False)

    first_pass = "B:A:go|x|y|z"
    assert first.model.calls[1][-1]["content"] == first_pass
    assert second.model.calls[0][-1]["content"] == "A:go|x|y"
    assert response.content == "B:A:" + first_pass + "|x|y|z"
    assert first.context[-1]["content"] == "A:" + first_pass
    assert second.context[-1]["content"] == response.content.removesuffix("|z")
    assert x.calls == y.calls == z.calls == 2
    assert response.input_tokens == response.output_tokens == 4
    assert response.generation_time == pytest.approx(0.04)
    entry = group.history()[0]
    assert entry["usage"]["total_tokens"] == 8
    assert [step["kind"] for step in entry["steps"]] == [
        "query", "transform", "transform", "query", "transform",
    ] * 2
    assert [(step["loop"], step["step"]) for step in entry["steps"]] == [
        (loop, index) for loop in (1, 2) for index in range(1, 6)
    ]
    transformed = entry["steps"][1]
    assert transformed["name"] == "Append"
    assert transformed["input"]["content"] == "A:go"
    assert transformed["output"]["content"] == "A:go|x"
    assert "usage" not in transformed
    assert transformed["input"]["usage"] == transformed["output"]["usage"]
    assert json.loads(json.dumps(entry)) == entry


def test_transform_copies_protect_originals_and_preserve_final_content_and_reasoning():
    class Rewrite(Transform):
        def transform(self, response):
            self.result = response
            response.prompt[0]["content"] = "changed prompt"
            response.content = "rewritten"
            response.reasoning = "formatted reasoning"
            response.input_tokens = response.output_tokens = 999
            response.generation_time = 999
            return response

    agent = Agent("raw", EchoModel("A"), "original system")
    rewrite = Rewrite()
    group = Pipeline([agent, rewrite])
    response = group.query("go")

    assert response.content == "rewritten"
    assert response.reasoning == "formatted reasoning"
    assert response.prompt == [{"role": "user", "content": "go"}]
    assert response.input_tokens == response.output_tokens == 1
    assert response.generation_time == 0.01
    assert agent.model.response.content == "A:go"
    assert agent.model.response.prompt[0]["content"] == "original system"
    assert agent.context[-1]["content"] == "A:go"
    assert agent.history()[0]["content"] == "A:go"
    detail = group.history()
    record = detail[0]["steps"][1]
    assert record["input"]["prompt"][0]["content"] == "original system"
    assert record["output"]["prompt"][0]["content"] == "changed prompt"
    assert record["output"]["usage"]["total_tokens"] == 1998
    assert set(record["output"]["usage"]).isdisjoint(record["output"])
    rewrite.result.content = "later change"
    rewrite.result.prompt.clear()
    record["input"]["prompt"].clear()
    record["output"]["content"] = "history edit"
    response.content = "caller edit"
    saved = group.history()[0]
    assert saved["content"] == "rewritten"
    assert saved["steps"][1]["input"]["prompt"]
    assert saved["steps"][1]["output"]["content"] == "rewritten"
    assert saved["usage"]["total_tokens"] == 2


@pytest.mark.parametrize("result_kind", ["exception", "none", "string", "list", "invalid_content"])
def test_transform_failure_preserves_costs_and_original_response_and_stops_execution(result_kind):
    class Broken(Transform):
        def transform(self, response):
            response.prompt.clear()
            response.content = "mutated copy"
            if result_kind == "exception":
                raise ValueError("cannot transform")
            return {
                "none": None, "string": "wrong", "list": [response],
                "invalid_content": Response(prompt=[], content=1),
            }[result_kind]

    first, last = Agent("first", EchoModel("A")), Agent("last", FixedModel())
    group = Pipeline([first, Broken(), last], loop=2)
    error_type = ValueError if result_kind == "exception" else TypeError
    with pytest.raises(error_type):
        group.query("go")

    assert len(first.model.calls) == len(first.history()) == 1
    assert last.model.calls == []
    assert first.model.response.content == "A:go"
    assert first.model.response.prompt
    entry = group.history()[0]
    assert entry["status"] == "failed"
    assert entry["content"] == "A:go"
    assert entry["usage"]["total_tokens"] == 2
    assert entry["usage"]["generation_time"] == 0.01
    assert len(entry["steps"]) == 2
    failed = entry["steps"][-1]
    assert failed["kind"] == "transform"
    assert failed["status"] == "failed"
    assert failed["input"]["content"] == "A:go"
    assert failed["output"] is None
    assert failed["error"]["type"] == error_type.__name__


@pytest.mark.parametrize("group_class", [SequentialMultiagent, ParallelMultiagent, Pipeline])
def test_nested_transform_failure_preserves_trace_and_counts_model_usage_once(group_class):
    class Fail(Transform):
        def transform(self, response):
            raise RuntimeError("transform unavailable")

    first = Agent("first", FixedModel("A"))
    last = Agent("last", FixedModel("B"))
    inner = Pipeline([first, Append("!"), Fail()])
    middle = SequentialMultiagent([inner])
    outer = group_class([middle, last])
    with pytest.raises(RuntimeError, match="transform unavailable"):
        outer.query("go")

    entry = outer.history()[0]
    expected_calls = 2 if group_class is ParallelMultiagent else 1
    assert entry["usage"]["total_tokens"] == 2 * expected_calls
    assert entry["usage"]["generation_time"] == pytest.approx(0.01 * expected_calls)
    assert len(last.model.calls) == expected_calls - 1
    failed = entry["steps"][0]["steps"][0]["steps"][-1]
    assert failed["kind"] == "transform"
    assert failed["input"]["content"] == "A!"
    assert failed["status"] == "failed"
    assert middle.history()[0]["content"] == "A!"


@pytest.mark.parametrize("metered", [False, True])
def test_parallel_nested_transforms_preserve_outputs_and_known_usage(metered):
    class MaybeMetered(FixedModel):
        def respond(self, messages):
            return super().respond(messages) if metered else Response(prompt=list(messages), content="A")

    inner = Pipeline([Agent("a", MaybeMetered("A")), Append("!")])
    outer = Pipeline([inner, Append("?")])
    parallel = ParallelMultiagent([outer, Agent("b", FixedModel("B"))])
    responses = parallel.query("go")
    assert [response.content for response in responses] == ["A!?", "B"]
    entry = parallel.history()[0]
    assert entry["usage"]["total_tokens"] == (4 if metered else 2)
    assert entry["usage"]["generation_time"] == pytest.approx(0.02 if metered else 0.01)
    assert responses[0].generation_time == (0.01 if metered else None)
    assert entry["steps"][0]["steps"][0]["steps"][1]["output"]["content"] == "A!"


def test_parallel_rejects_shared_transforms_and_follows_changed_pipeline_steps():
    first = Agent("a", FixedModel("A"))
    second = Agent("b", FixedModel("B"))
    shared = Append("!")
    left = Pipeline([first, shared, shared])
    right = Pipeline([second, shared])
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([left, right])
    right.steps = (second, Append("?"))
    group = ParallelMultiagent([left, right])
    assert [response.content for response in group.query()] == ["A!!", "B?"]
    first_calls, second_calls = len(first.model.calls), len(second.model.calls)
    right.steps = (second, shared)
    with pytest.raises(ValueError, match="must not share"):
        group.query()
    assert len(first.model.calls) == first_calls
    assert len(second.model.calls) == second_calls


def test_transform_placement_and_runtime_plan_validation_happen_before_queries():
    agent = Agent("a", FixedModel())
    transform = Append("!")
    for steps in ([transform], [transform, agent]):
        with pytest.raises(ValueError, match="first step"):
            Pipeline(steps)
    with pytest.raises(TypeError, match="participants"):
        ParallelMultiagent([transform])
    with pytest.raises(TypeError, match="Aggregate"):
        Pipeline([ParallelMultiagent([agent]), transform])
    group = Pipeline([agent, transform])
    group.steps = (transform, agent)
    with pytest.raises(ValueError, match="first step"):
        group.query("go")
    assert agent.model.calls == []
    assert group.history() == []
