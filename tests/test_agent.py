from maslab import Agent

from conftest import FixedModel


def test_generate_does_not_mutate_context():
    agent = Agent("0", FixedModel(), "system")

    response = agent.generate("hello")

    assert response.content
    assert agent.context == [{"role": "system", "content": "system"}]


def test_chat_records_complete_conversation():
    agent = Agent("0", FixedModel(), "system")

    agent.chat("hello")

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

    agent.chat("first")
    agent.chat("second")

    assert agent.context[0] == {"role": "system", "content": "system"}
    assert [message["role"] for message in agent.context[1:]] == [
        "user",
        "assistant",
    ]
    assert agent.context[1]["content"] == "second"
