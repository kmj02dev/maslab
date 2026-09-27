import json
from threading import Barrier, Event

import pytest

from maslab import (
    Agent, LLMAggregate, ConcatAggregate, Multiagent, ParallelMultiagent,
    Response, SequentialMultiagent,
)

from conftest import FixedModel


def make_agent(label):
    return Agent(label, FixedModel(label), f"system {label}")


def test_parallel_calls_overlap_but_results_and_history_follow_input_order():
    barrier = Barrier(3)
    finished = [Event() for _ in range(3)]
    completion_order = []

    class CoordinatedModel(FixedModel):
        def __init__(self, index):
            super().__init__(str(index))
            self.index = index

        def respond(self, messages):
            barrier.wait(timeout=5)
            if self.index < 2:
                assert finished[self.index + 1].wait(timeout=5)
            response = super().respond(messages)
            completion_order.append(self.index)
            finished[self.index].set()
            return response

    agents = [Agent(str(index), CoordinatedModel(index)) for index in range(3)]
    group = ParallelMultiagent(agents)

    responses = group.query("same question")
    assert all(isinstance(response, Response) for response in responses)
    assert [response.content for response in responses] == ["0", "1", "2"]
    assert [response.agent_id for response in responses] == ["0", "1", "2"]
    assert all(response.input_tokens == response.output_tokens == 1 for response in responses)
    assert not hasattr(group, "chat")
    assert not hasattr(group, "chat_response")
    assert completion_order == [2, 1, 0]
    assert isinstance(group, Multiagent)
    entry = group.history()[0]
    assert entry["content"] == ["0", "1", "2"]
    assert [step["agent_id"] for step in entry["steps"]] == ["0", "1", "2"]
    assert [step["step"] for step in entry["steps"]] == [1, 2, 3]
    assert entry["usage"] == {
        "input_tokens": 3, "output_tokens": 3, "total_tokens": 6,
        "generation_time": pytest.approx(0.03),
    }
    for agent, step in zip(agents, entry["steps"]):
        assert agent.model.calls[0] == [{"role": "user", "content": "same question"}]
        assert step["message"] == "same question"
        assert set(step["usage"]).isdisjoint(step)
    assert set(entry["usage"]).isdisjoint(entry)


def test_parallel_retains_private_context_and_propagates_per_call_overrides():
    first, second = make_agent("A"), make_agent("B")
    first.query("private question")
    group = ParallelMultiagent([first, second])

    assert [response.content for response in group.query("shared question")] == ["A", "B"]
    assert [message["content"] for message in first.model.calls[-1]] == [
        "system A", "private question", "A", "shared question",
    ]
    assert [message["content"] for message in second.model.calls[-1]] == [
        "system B", "shared question",
    ]
    contexts = [list(first.context), list(second.context)]
    group.query("another question", use_context=False, update_context=False)
    assert [first.context, second.context] == contexts
    assert all(len(agent.model.calls[-1]) == 2 for agent in (first, second))
    assert len(group.history()) == 2
    assert len(first.history()) == 3
    assert len(second.history()) == 2


def test_parallel_responses_and_returned_history_do_not_alias_saved_details():
    group = ParallelMultiagent([make_agent("A"), make_agent("B")], max_workers=1)

    responses = group.query()
    assert all(isinstance(response, Response) for response in responses)
    assert responses[0].prompt[-1]["content"] == "Continue."
    responses[0].prompt.clear()
    responses[0].content = "changed"
    detail = group.history()
    detail[0]["content"].clear()
    detail[0]["steps"][0]["prompt"].clear()
    detail[0]["usage"]["total_tokens"] = 999

    assert group.history()[0]["content"] == ["A", "B"]
    assert group.history()[0]["steps"][0]["prompt"]
    assert group.history()[0]["usage"]["total_tokens"] == 4
    assert json.loads(json.dumps(group.history())) == group.history()


def test_parallel_can_execute_independent_sequential_groups():
    team = SequentialMultiagent([make_agent("A"), make_agent("B")], loop=2)
    group = ParallelMultiagent([team, make_agent("C")])

    assert [response.content for response in group.query("go")] == ["B", "C"]
    entry = group.history()[0]
    assert len(entry["steps"][0]["steps"]) == 4
    assert entry["steps"][0]["steps"][0]["message"] == "go"
    assert entry["steps"][1]["message"] == "go"
    assert entry["usage"]["total_tokens"] == 10
    assert entry["usage"]["generation_time"] == pytest.approx(0.05)


def test_parallel_failure_records_successes_and_all_failures_before_raising():
    class FailingModel(FixedModel):
        def respond(self, messages):
            raise RuntimeError(self.content)

    first = Agent("broken-1", FailingModel("first failure"))
    success = make_agent("B")
    last = Agent("broken-2", FailingModel("last failure"))
    group = ParallelMultiagent([first, success, last])

    with pytest.raises(RuntimeError, match="first failure"):
        group.query("go")

    assert len(success.history()) == 1
    assert first.history() == last.history() == []
    entry = group.history()[0]
    assert entry["status"] == "failed"
    assert entry["content"] == ["B"]
    assert entry["usage"]["total_tokens"] == 2
    assert [step["status"] for step in entry["steps"]] == ["failed", "completed", "failed"]
    assert entry["steps"][0]["error"] == {"type": "RuntimeError", "message": "first failure"}
    assert entry["steps"][2]["error"]["message"] == "last failure"


def test_parallel_failure_preserves_nested_partial_work_and_usage():
    class FailingModel(FixedModel):
        def respond(self, messages):
            raise RuntimeError("unavailable")

    inner = SequentialMultiagent([make_agent("A"), Agent("broken", FailingModel())])
    group = ParallelMultiagent([inner, make_agent("B")])

    with pytest.raises(RuntimeError, match="unavailable"):
        group.query("go")

    entry = group.history()[0]
    assert entry["content"] == ["B"]
    assert entry["steps"][0]["steps"][0]["content"] == "A"
    assert entry["usage"]["total_tokens"] == 4
    assert entry["usage"]["generation_time"] == 0.02


def test_parallel_missing_metrics_remain_unknown_for_generation_time():
    class UnmeteredModel(FixedModel):
        def respond(self, messages):
            return Response(prompt=list(messages), content="A")

    group = ParallelMultiagent([Agent("A", UnmeteredModel())])
    assert [response.content for response in group.query()] == ["A"]
    assert group.history()[0]["usage"] == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "generation_time": None,
    }


def test_parallel_rejects_state_shared_across_branches_but_allows_repetition_within_one():
    agent = make_agent("A")
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([agent, agent])
    inner = SequentialMultiagent([agent, agent])
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([inner, agent])
    other = SequentialMultiagent([agent])
    with pytest.raises(ValueError, match="must not share"):
        ParallelMultiagent([inner, other])
    assert [response.content for response in ParallelMultiagent([inner, make_agent("B")]).query()] == ["A", "B"]


def test_parallel_revalidates_nested_participants_before_starting_workers():
    first, second = make_agent("A"), make_agent("B")
    inner = SequentialMultiagent([first])
    group = ParallelMultiagent([inner, second])
    inner.agents = (second,)
    with pytest.raises(ValueError, match="must not share"):
        group.query()
    assert first.model.calls == second.model.calls == []


@pytest.mark.parametrize("max_workers", [0, -1, True, 1.5, "2"])
def test_parallel_rejects_invalid_worker_counts(max_workers):
    with pytest.raises(ValueError, match="max_workers"):
        ParallelMultiagent([make_agent("A")], max_workers=max_workers)


def test_parallel_validates_inputs_before_model_calls():
    with pytest.raises(ValueError, match="at least one"):
        ParallelMultiagent([])
    with pytest.raises(TypeError, match="participants"):
        ParallelMultiagent([object()])
    agent = make_agent("A")
    group = ParallelMultiagent([agent])
    for kwargs in ({"message": None}, {"use_context": "false"}, {"update_context": "false"}):
        with pytest.raises(TypeError):
            group.query(**kwargs)
    assert agent.model.calls == []
    assert group.history() == []


def test_parallel_collections_require_explicit_reduction_before_single_answer_handoff():
    group = ParallelMultiagent([make_agent("A")])
    with pytest.raises(TypeError, match="Aggregate"):
        SequentialMultiagent([group, make_agent("B")])
    with pytest.raises(TypeError, match="aggregate"):
        ParallelMultiagent([group])


def test_parallel_results_feed_both_aggregator_implementations():
    group = ParallelMultiagent([make_agent("A"), make_agent("B"), make_agent("A")])
    responses = group.query("question")
    assert ConcatAggregate()(responses).content == "[agent 1]\nA\n\n[agent 2]\nB\n\n[agent 3]\nA"
    judge = FixedModel("final answer")
    aggregator = LLMAggregate(judge)
    assert aggregator(responses).content == "final answer"
    assert json.loads(judge.calls[0][-1]["content"]) == [response.content for response in responses]
    assert group.history()[0]["usage"]["total_tokens"] == 6
    assert aggregator.history()[0]["usage"]["total_tokens"] == 2
