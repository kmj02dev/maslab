import pytest

from maslab import Agent, Multiagent, ParallelMultiagent, Pipeline, Response, SequentialMultiagent

from conftest import FixedModel


class EchoModel(FixedModel):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def respond(self, messages):
        self.content = f"{self.label}:{messages[-1]['content']}"
        return super().respond(messages)


def make_agent(label):
    return Agent(label, EchoModel(label), f"system {label}")


def test_single_and_multiagent_share_the_user_query_history_interface():
    single = make_agent("a")
    group = SequentialMultiagent([make_agent("a")])

    for participant in (single, group):
        assert participant.history() == []
        response = participant.query()
        assert isinstance(response, Response)
        assert response.content == "a:Continue."
        assert response.input_tokens == response.output_tokens == 1
        assert participant.history()[-1]["content"] == response.content
        assert not hasattr(participant, "chat")
        assert not hasattr(participant, "chat_response")
        assert participant.history()[-1]["usage"]["total_tokens"] == 2
    assert isinstance(group, Multiagent)


def test_loop_runs_every_agent_and_hands_off_across_passes():
    first, second = make_agent("a"), make_agent("b")
    group = SequentialMultiagent([first, second], loop=5)

    assert group.query("go debate!").content == "b:a:b:a:b:a:b:a:b:a:go debate!"
    assert len(first.model.calls) == len(second.model.calls) == 5
    detail = group.history()
    assert len(detail) == 1
    assert detail[0]["message"] == "go debate!"
    assert detail[0]["usage"]["total_tokens"] == 20
    steps = detail[0]["steps"]
    assert len(steps) == 10
    assert steps[0]["message"] == "go debate!"
    assert steps[1]["message"] == "a:go debate!"
    assert steps[2]["message"] == "b:a:go debate!"
    assert [(step["loop"], step["step"]) for step in steps] == [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 1),
        (3, 2), (4, 1), (4, 2), (5, 1), (5, 2),
    ]


def test_repeated_calls_retain_history_and_read_context_by_default():
    first, second = make_agent("a"), make_agent("b")
    group = SequentialMultiagent([first, second])
    first.query("outside group")

    assert group.query("first").content == "b:a:first"
    assert group.query("second").content == "b:a:second"
    assert len(first.model.calls[-1]) == 6
    assert len(second.model.calls[-1]) == 4
    assert len(group.history()) == 2
    assert group.history()[0]["usage"]["total_tokens"] == 4
    assert group.history()[1]["steps"][0]["message"] == "second"


def test_context_overrides_and_update_context_propagate_to_children():
    first, second = make_agent("a"), make_agent("b")
    group = SequentialMultiagent([first, second], loop=2)

    group.query("start", use_context=False, update_context=False)

    for agent in (first, second):
        assert len(agent.context) == 1
        assert all(len(prompt) == 2 for prompt in agent.model.calls)
        assert len(agent.history()) == 2
        assert not hasattr(agent, "use_context")


def test_nested_multiagents_preserve_input_order_details_and_usage():
    inner = SequentialMultiagent([make_agent("a"), make_agent("b")], id="inner")
    outer = SequentialMultiagent([inner, make_agent("c")], loop=2, id="outer")

    assert outer.query("go").content == "c:b:a:c:b:a:go"
    detail = outer.history()[0]
    assert detail["agent_id"] == "outer"
    assert detail["usage"] == {
        "input_tokens": 6,
        "output_tokens": 6,
        "total_tokens": 12,
        "generation_time": pytest.approx(0.06),
    }
    pending = [detail]
    while pending:
        entry = pending.pop()
        assert set(entry["usage"]).isdisjoint(entry)
        pending.extend(entry.get("steps", []))
    assert len(detail["steps"]) == 4
    assert detail["steps"][0]["agent_id"] == "inner"
    assert detail["steps"][0]["steps"][0]["agent_id"] == "a"
    assert detail["steps"][0]["steps"][1]["message"] == "a:go"
    assert len(inner.history()) == 2


def test_history_is_a_deep_snapshot_of_all_nested_details():
    first = make_agent("a")
    inner = SequentialMultiagent([first])
    outer = SequentialMultiagent([inner])
    outer.query("go")
    saved = outer.history()
    saved[0]["steps"][0]["steps"][0]["prompt"].clear()
    saved[0]["usage"]["total_tokens"] = 999
    saved.clear()

    assert outer.history()[0]["usage"]["total_tokens"] == 2
    assert outer.history()[0]["steps"][0]["steps"][0]["prompt"]
    assert first.history()[0]["prompt"]


def test_nested_failure_preserves_successful_steps_without_running_later_agents():
    class FailingModel(FixedModel):
        def respond(self, messages):
            raise RuntimeError("model unavailable")

    first, last = make_agent("a"), make_agent("c")
    inner = SequentialMultiagent([first, Agent("broken", FailingModel())])
    outer = SequentialMultiagent([inner, last])

    with pytest.raises(RuntimeError, match="unavailable"):
        outer.query("go")

    assert len(first.model.calls) == 1
    assert last.model.calls == []
    assert outer.history()[0]["status"] == "failed"
    assert outer.history()[0]["usage"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "generation_time": 0.01,
    }
    assert outer.history()[0]["steps"][0]["status"] == "failed"
    assert inner.history()[0]["steps"][0]["content"] == "a:go"


@pytest.mark.parametrize("include_known_usage", [False, True])
def test_nested_usage_aggregation_handles_missing_model_metrics(include_known_usage):
    class UnmeteredModel(FixedModel):
        def respond(self, messages):
            response = super().respond(messages)
            response.input_tokens = None
            response.output_tokens = None
            response.generation_time = None
            return response

    inner = SequentialMultiagent([Agent("unmetered", UnmeteredModel())])
    agents = [inner, make_agent("metered")] if include_known_usage else [inner]
    outer = SequentialMultiagent(agents)

    response = outer.query("go")

    count = 1 if include_known_usage else 0
    expected_time = 0.01 if include_known_usage else None
    assert outer.history()[0]["usage"] == {
        "input_tokens": count,
        "output_tokens": count,
        "total_tokens": count * 2,
        "generation_time": expected_time,
    }
    assert response.input_tokens == count
    assert response.output_tokens == count
    assert response.generation_time == expected_time


@pytest.mark.parametrize("group_class", [SequentialMultiagent, Pipeline])
@pytest.mark.parametrize("loop", [0, -1, True, 1.5, "5"])
def test_invalid_loop_is_rejected_before_any_model_call(loop, group_class):
    agent = make_agent("a")
    with pytest.raises(ValueError, match="positive integer"):
        group_class([agent], loop=loop)
    assert agent.model.calls == []


def test_invalid_participants_and_messages_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SequentialMultiagent([])
    with pytest.raises(TypeError, match="participants"):
        SequentialMultiagent([object()])
    agent = make_agent("a")
    group = SequentialMultiagent([agent])
    with pytest.raises(TypeError, match="message"):
        group.query(None)
    with pytest.raises(TypeError, match="use_context"):
        group.query("go", use_context="false")
    assert agent.model.calls == []
    assert group.history() == []


@pytest.mark.parametrize("group_class", [SequentialMultiagent, ParallelMultiagent, Pipeline])
def test_groups_execute_participant_query_overrides_once(group_class):
    class CustomAgent(Agent):
        def __init__(self):
            super().__init__("custom", FixedModel("answer"))
            self.query_calls = []

        def query(self, message="Continue.", use_context=True, update_context=True) -> Response:
            self.query_calls.append((message, use_context, update_context))
            return super().query(message, use_context=use_context, update_context=update_context)

    agent = CustomAgent()
    group = group_class([agent])
    result = group.query("question", use_context=False, update_context=False)
    response = result[0] if isinstance(result, list) else result

    assert isinstance(response, Response)
    assert response.content == "answer"
    assert response.input_tokens == response.output_tokens == 1
    assert agent.query_calls == [("question", False, False)]
    assert len(agent.model.calls) == len(agent.history()) == len(group.history()) == 1
    assert agent.context == []


@pytest.mark.parametrize("kind", ["agent", "sequential", "parallel", "mesh", "pipeline"])
def test_none_context_flag_is_rejected_at_every_query_entry(kind):
    from maslab import MeshMultiagent
    agent = make_agent("a")
    participant = agent if kind == "agent" else {
        "sequential": SequentialMultiagent,
        "parallel": ParallelMultiagent,
        "mesh": MeshMultiagent,
        "pipeline": Pipeline,
    }[kind]([agent])
    with pytest.raises(TypeError, match="use_context"):
        participant.query("hello", use_context=None)
    assert agent.model.calls == []
    assert participant.history() == []
