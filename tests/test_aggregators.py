from collections.abc import Iterable
from copy import deepcopy
import json
from typing import get_type_hints

import pytest

from maslab import Aggregate, ConcatAggregate, LLMAggregate, Response

from conftest import FixedModel


def test_aggregate_is_an_extensible_callable_abstract_base():
    with pytest.raises(TypeError, match="abstract"):
        Aggregate()

    class Join(Aggregate):
        def aggregate(self, responses):
            return Response(prompt=[], content=", ".join(self._contents(responses)))

    for method in (Aggregate.aggregate, Aggregate.__call__, ConcatAggregate.aggregate, LLMAggregate.aggregate):
        assert get_type_hints(method)["responses"] == Iterable[Response]
        assert get_type_hints(method)["return"] is Response
    responses = [Response(prompt=[], content=text) for text in ("A", "B")]
    assert Join()(iter(responses)).content == "A, B"
    assert Join().aggregate(iter(responses)).content == "A, B"
    assert not hasattr(Join(), "aggregate_response")


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (["A"], "[agent 1]\nA"),
        (["A", "B", "A"], "[agent 1]\nA\n\n[agent 2]\nB\n\n[agent 3]\nA"),
        (["", "B"], "[agent 1]\n\n\n[agent 2]\nB"),
        (["  A\n", "\nB  "], "[agent 1]\n  A\n\n\n[agent 2]\n\nB  "),
        (["한글\n## Heading", '{"vote":"A"}'], '[agent 1]\n한글\n## Heading\n\n[agent 2]\n{"vote":"A"}'),
    ],
)
def test_concat_preserves_contents_duplicates_and_order(contents, expected):
    responses = (Response(prompt=[], content=text) for text in contents)
    result = ConcatAggregate()(responses)
    assert isinstance(result, Response)
    assert result.content == expected
    assert result.prompt == []
    assert result.agent_id is None
    assert result.reasoning is None
    assert result.input_tokens is result.output_tokens is result.generation_time is None


def test_aggregators_consume_a_one_pass_iterable_without_requiring_a_sequence():
    class OnePass:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            return iter([Response(prompt=[], content="A"), Response(prompt=[], content="B")])

        def __len__(self):
            raise AssertionError("input must not require a length")

    model = FixedModel("combined")
    responses = OnePass()
    assert ConcatAggregate()(responses).content == "[agent 1]\nA\n\n[agent 2]\nB"
    assert responses.iterations == 1
    responses = OnePass()
    assert LLMAggregate(model)(responses).content == "combined"
    assert responses.iterations == 1
    assert json.loads(model.calls[0][-1]["content"]) == ["A", "B"]


def test_concat_preserves_source_responses_and_does_not_recharge_their_usage():
    responses = [
        Response(
            prompt=[{"role": "user", "content": "original"}],
            content="A", reasoning="private reasoning", agent_id="source-a",
            input_tokens=50, output_tokens=10, generation_time=1.5,
        ),
        Response(prompt=[], content="B", agent_id="source-b"),
    ]
    before = deepcopy(responses)
    result = ConcatAggregate().aggregate(responses)
    assert responses == before
    assert result.agent_id is None
    assert result.content == "[agent 1]\nA\n\n[agent 2]\nB"
    assert result.input_tokens is result.output_tokens is result.generation_time is None
    result.prompt.append({"role": "user", "content": "changed"})
    result.content = "changed"
    assert responses == before


@pytest.mark.parametrize("factory", [
    lambda: [],
    lambda: iter(()),
    lambda: "A",
    lambda: b"A",
    lambda: None,
    lambda: Response(prompt=[], content="A"),
    lambda: ["A"],
    lambda: [1],
    lambda: [Response(prompt=[], content="A"), "B"],
    lambda: [Response(prompt=[], content=1)],
    lambda: (item for item in [Response(prompt=[], content="A"), object()]),
])
def test_built_in_aggregators_reject_invalid_inputs_before_model_calls(factory):
    model = FixedModel()
    for aggregator in (ConcatAggregate(), LLMAggregate(model)):
        with pytest.raises((TypeError, ValueError)):
            aggregator(factory())
    assert model.calls == []


def test_llm_aggregate_preserves_candidate_boundaries_and_uses_one_model_call():
    model = FixedModel("synthesized answer")
    aggregator = LLMAggregate(model, "Judge the proposed answers to question X.", id="judge")
    candidates = [
        Response(prompt=[], content='A: "quoted"\nsecond line'),
        Response(prompt=[], content="B"),
    ]

    result = aggregator(response for response in candidates)
    assert isinstance(result, Response)
    assert result.content == "synthesized answer"
    assert result.agent_id == "judge"
    assert len(model.calls) == 1
    assert model.calls[0][0]["content"] == "Judge the proposed answers to question X."
    assert json.loads(model.calls[0][1]["content"]) == [response.content for response in candidates]
    entry = aggregator.history()[0]
    assert entry["agent_id"] == "judge"
    assert entry["prompt"] == model.calls[0]
    assert entry["usage"]["total_tokens"] == 2
    assert set(entry["usage"]).isdisjoint(entry)


def test_llm_aggregate_calls_are_independent_and_history_is_a_snapshot():
    model = FixedModel("answer")
    aggregator = LLMAggregate(model)
    first = aggregator.aggregate(Response(prompt=[], content=text) for text in ("first", "second"))
    aggregator([Response(prompt=[], content="third")])
    assert len(model.calls) == 2
    assert all(len(prompt) == 2 for prompt in model.calls)
    assert json.loads(model.calls[-1][-1]["content"]) == ["third"]
    first.prompt.clear()
    history = aggregator.history()
    history[0]["usage"]["total_tokens"] = 999
    history[0]["prompt"].clear()

    assert len(aggregator.agent.context) == 1
    assert len(aggregator.history()) == 2
    assert aggregator.history()[0]["prompt"]
    assert aggregator.history()[0]["usage"]["total_tokens"] == 2


def test_llm_aggregate_validates_model_and_instructions():
    with pytest.raises(TypeError, match="Model"):
        LLMAggregate(object())
    with pytest.raises(ValueError, match="system_prompt"):
        LLMAggregate(FixedModel(), "")
