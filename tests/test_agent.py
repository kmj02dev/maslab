import pytest

from maslab import Agent, ParallelMultiagent, Response

from conftest import FixedModel


def test_generate_does_not_mutate_context():
    agent = Agent("0", FixedModel(), "system")

    response = agent.generate("hello")

    assert response.content
    assert response.agent_id == "0"
    assert agent.context == [{"role": "system", "content": "system"}]


def test_query_records_complete_conversation():
    agent = Agent("0", FixedModel(), "system")

    response = agent.query("hello")

    assert isinstance(response, Response)
    assert response.content == '{"vote": "A"}'

    assert agent.context == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": '{"vote": "A"}'},
    ]


def test_last_conversation_policy_preserves_system_prompt():
    agent = Agent(
        "0",
        FixedModel(),
        "system",
        context_policy="last_conversation",
    )

    agent.query("first")
    agent.query("second")

    assert agent.context[0] == {"role": "system", "content": "system"}
    assert [message["role"] for message in agent.context[1:]] == [
        "user",
        "assistant",
    ]
    assert agent.context[1]["content"] == "second"


def test_query_without_message_uses_default_request_and_records_details():
    model = FixedModel("answer")
    agent = Agent("writer", model, "Write an article.")

    response = agent.query()
    assert response.content == "answer"
    assert response.agent_id == "writer"
    assert response.prompt == model.calls[0]
    assert response.input_tokens == response.output_tokens == 1
    assert response.generation_time == 0.01
    assert len(model.calls) == len(agent.history()) == 1
    detail = agent.history()

    assert detail[0]["message"] == "Continue."
    assert detail[0]["agent_id"] == "writer"
    assert detail[0]["content"] == "answer"
    assert detail[0]["prompt"] == model.calls[0]
    assert detail[0]["usage"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "generation_time": 0.01,
    }
    assert set(detail[0]["usage"]).isdisjoint(detail[0])


def test_context_is_selected_per_call_without_changing_default():
    model = FixedModel("answer")
    agent = Agent("0", model, "system")
    agent.query("first")
    agent.query("second", use_context=False)
    agent.query("third")
    agent.query("fourth", use_context=False)

    assert [message["content"] for message in model.calls[1]] == ["system", "second"]
    assert [message["content"] for message in model.calls[2]] == [
        "system", "first", "answer", "second", "answer", "third",
    ]
    assert [message["content"] for message in model.calls[3]] == ["system", "fourth"]
    assert not hasattr(agent, "use_context")
    assert len(agent.history()) == 4


def test_history_is_independent_of_context_policy_and_returned_responses():
    agent = Agent("0", FixedModel("answer"), "system", context_policy="replace")
    response = agent.query("first")
    agent.query("second", update_context=False)

    assert isinstance(response, Response)
    assert agent.context == [{"role": "system", "content": "system"}]
    response.prompt.clear()
    response.content = "changed"
    response.agent_id = "changed"
    detail = agent.history()
    detail[0]["prompt"].clear()
    detail[0]["usage"]["total_tokens"] = 999
    detail.clear()

    assert len(agent.history()) == 2
    assert agent.history()[0]["content"] == "answer"
    assert agent.history()[0]["agent_id"] == "0"
    assert agent.history()[0]["prompt"]
    assert agent.history()[0]["usage"]["total_tokens"] == 2


def test_generate_uses_explicit_context_without_recording_a_query():
    model = FixedModel("answer")
    agent = Agent("0", model, "system")
    agent.query("first")
    before = agent.history()
    response = agent.generate("stateless", use_context=False)

    assert isinstance(response, Response)
    assert [message["content"] for message in model.calls[-1]] == ["system", "stateless"]
    assert agent.history() == before


def test_failed_query_keeps_completed_history_and_context():
    class FailingModel(FixedModel):
        def respond(self, messages):
            if self.calls:
                raise RuntimeError("model unavailable")
            return super().respond(messages)

    agent = Agent("0", FailingModel())
    agent.query("first")
    before = agent.history()
    context = list(agent.context)

    with pytest.raises(RuntimeError, match="unavailable"):
        agent.query("second")

    assert agent.history() == before
    assert agent.context == context


def test_invalid_query_input_does_not_call_model():
    model = FixedModel()
    agent = Agent("0", model)
    with pytest.raises(TypeError, match="message"):
        agent.query(None)
    with pytest.raises(TypeError, match="use_context"):
        agent.query("hello", use_context="false")
    assert model.calls == []


def test_constructor_no_longer_accepts_context_flag():
    with pytest.raises(TypeError, match="use_context"):
        Agent("a", FixedModel(), use_context=False)


@pytest.mark.parametrize("use_context", [False, True])
@pytest.mark.parametrize("update_context", [False, True])
def test_context_reading_and_writing_are_independent(use_context, update_context):
    model = FixedModel("answer")
    agent = Agent("a", model, "system")
    agent.query("first")
    before = list(agent.context)
    agent.query("second", use_context=use_context, update_context=update_context)
    assert model.calls[-1] == (before if use_context else before[:1]) + [
        {"role": "user", "content": "second"}]
    assert agent.context == before + ([
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"}] if update_context else [])
    assert len(agent.history()) == 2


def test_generate_rejects_none_without_model_call():
    model = FixedModel()
    with pytest.raises(TypeError, match="use_context"):
        Agent("a", model).generate("hello", use_context=None)
    assert model.calls == []


def test_response_agent_id_is_optional_and_round_trips_with_existing_arguments():
    from dataclasses import asdict

    response = Response([], "answer", "reason", 3, 2, 0.5)
    assert response.agent_id is None
    response.agent_id = "writer"
    assert Response(**asdict(response)) == response
    assert asdict(response)["agent_id"] == "writer"
    assert FixedModel().respond([{"role": "user", "content": "Q"}]).agent_id is None


def test_agents_tag_shared_backend_responses_without_mutating_them():
    class ReusingModel(FixedModel):
        def __init__(self):
            super().__init__()
            self.response = Response(
                prompt=[{"role": "user", "content": "Q"}],
                content="answer", reasoning="reason", input_tokens=3,
                output_tokens=2, generation_time=0.5, agent_id="backend",
            )

        def respond(self, messages):
            return self.response

    model = ReusingModel()
    agents = [Agent("first", model), Agent("second", model)]
    responses = ParallelMultiagent(agents).query("Q", use_context=False, update_context=False)
    assert [response.agent_id for response in responses] == ["first", "second"]
    assert model.response.agent_id == "backend"
    assert responses[0] is not responses[1]
    for agent, response in zip(agents, responses):
        assert response is not model.response
        assert response.content == "answer"
        assert response.reasoning == "reason"
        assert response.input_tokens == 3 and response.output_tokens == 2
        assert response.generation_time == 0.5
        assert agent.history()[0]["agent_id"] == agent.id
        assert agent.context == []
