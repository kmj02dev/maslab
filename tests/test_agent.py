import pytest

from maslab import Agent, Response

from conftest import FixedModel


def test_generate_does_not_mutate_context():
    agent = Agent("0", FixedModel(), "system")

    response = agent.generate("hello")

    assert response.content
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
    agent = Agent("writer", model, "Write an article.", use_context=False)

    response = agent.query()
    assert response.content == "answer"
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


def test_constructor_context_setting_and_per_call_override_are_independent():
    model = FixedModel("answer")
    agent = Agent("0", model, "system", use_context=False)
    agent.query("first")
    agent.query("second")
    agent.query("third", use_context=True)
    agent.query("fourth")

    assert [message["content"] for message in model.calls[1]] == ["system", "second"]
    assert [message["content"] for message in model.calls[2]] == [
        "system", "first", "answer", "second", "answer", "third",
    ]
    assert [message["content"] for message in model.calls[3]] == ["system", "fourth"]
    assert agent.use_context is False
    assert len(agent.history()) == 4


def test_history_is_independent_of_context_policy_and_returned_responses():
    agent = Agent("0", FixedModel("answer"), "system", context_policy="replace")
    response = agent.query("first")
    agent.query("second", update_context=False)

    assert isinstance(response, Response)
    assert agent.context == [{"role": "system", "content": "system"}]
    response.prompt.clear()
    response.content = "changed"
    detail = agent.history()
    detail[0]["prompt"].clear()
    detail[0]["usage"]["total_tokens"] = 999
    detail.clear()

    assert len(agent.history()) == 2
    assert agent.history()[0]["content"] == "answer"
    assert agent.history()[0]["prompt"]
    assert agent.history()[0]["usage"]["total_tokens"] == 2


def test_generate_honors_constructor_context_without_recording_a_query():
    model = FixedModel("answer")
    agent = Agent("0", model, "system", use_context=False)
    agent.query("first")
    before = agent.history()
    response = agent.generate("stateless")

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
