import json

import pytest

from maslab import Aggregate, LLMAggregate, MajorityVote, Response

from conftest import FixedModel


def test_aggregate_is_an_extensible_callable_abstract_base():
    with pytest.raises(TypeError, match="abstract"):
        Aggregate()

    class Join(Aggregate):
        def aggregate(self, responses):
            return ", ".join(self._contents(responses))

    assert Join()(["A", "B"]) == "A, B"
    assert Join().aggregate(["A", "B"]) == "A, B"


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (["A", "B", "A"], "A"),
        (["B", "A", "A", "B"], "B"),
        (["A", "B", "C", "A"], "A"),
        (["a", "A", "A"], "A"),
        ([" A ", "A"], " A "),
        ([""], ""),
    ],
)
def test_majority_vote_uses_exact_text_and_first_occurrence_for_ties(responses, expected):
    assert MajorityVote()(responses) == expected


def test_majority_vote_accepts_raw_and_text_responses_without_mutating_them():
    response = Response(prompt=[], content="A", input_tokens=50)
    responses = [response, "B", "A"]

    assert MajorityVote().aggregate(responses) == "A"
    assert responses == [response, "B", "A"]
    assert response.input_tokens == 50


@pytest.mark.parametrize("responses", [[], "A", None, [1], ["A", object()]])
def test_built_in_aggregators_reject_invalid_inputs_before_model_calls(responses):
    model = FixedModel()
    for aggregator in (MajorityVote(), LLMAggregate(model)):
        with pytest.raises((TypeError, ValueError)):
            aggregator(responses)
    assert model.calls == []


def test_llm_aggregate_preserves_candidate_boundaries_and_uses_one_model_call():
    model = FixedModel("synthesized answer")
    aggregator = LLMAggregate(model, "Judge the proposed answers to question X.", id="judge")
    candidates = ['A: "quoted"\nsecond line', Response(prompt=[], content="B")]

    assert aggregator(candidates) == "synthesized answer"

    assert len(model.calls) == 1
    assert model.calls[0][0]["content"] == "Judge the proposed answers to question X."
    assert json.loads(model.calls[0][1]["content"]) == [candidates[0], "B"]
    entry = aggregator.history()[0]
    assert entry["agent_id"] == "judge"
    assert entry["prompt"] == model.calls[0]
    assert entry["usage"]["total_tokens"] == 2
    assert set(entry["usage"]).isdisjoint(entry)


def test_llm_aggregate_calls_are_independent_and_history_is_a_snapshot():
    model = FixedModel("answer")
    aggregator = LLMAggregate(model)
    first = aggregator.aggregate_response(["first", "second"])
    aggregator(["third"])
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
