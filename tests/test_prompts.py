import pytest

from maslab import (
    JsonDecision,
    Message,
    PromptContext,
    generic_debate_prompt,
    render_prompt,
)


def prompt_context(**changes):
    values = {
        "task": {"id": "task"},
        "phase": "debate",
        "round_idx": 1,
        "agent_idx": 0,
        "agent_id": "0",
    }
    values.update(changes)
    return PromptContext(**values)


def test_prompt_context_supports_typed_and_legacy_renderers():
    context = prompt_context(variables={"information": ["fact"]})

    typed = render_prompt(lambda context: context.variables["information"][0], context)
    legacy = render_prompt(lambda **values: values["information"][0], context)

    assert typed == "fact"
    assert legacy == "fact"


def test_generic_debate_prompt_is_not_empty_without_visible_messages():
    value = render_prompt(generic_debate_prompt, prompt_context())

    assert value
    assert "No other agent messages" in value


def test_json_decision_renders_and_validates_schema():
    decision = JsonDecision({"vote": str})
    context = prompt_context(
        phase="decision",
        options=("A", "B"),
        visible_messages=(Message("1", "I choose A", 3, 1),),
    )

    prompt = decision.render(context)

    assert "Valid options: A, B" in prompt
    assert decision.parse('analysis {"vote": "A"}') == {"vote": "A"}
    with pytest.raises(ValueError, match="missing field"):
        decision.parse('{"answer": "A"}')


def test_json_decision_accepts_unescaped_control_characters():
    decision = JsonDecision({"vote": str, "rationale": str})

    value = decision.parse(
        '{"vote": "A", "rationale": "first line\n\tsecond line"}'
    )

    assert value == {
        "vote": "A",
        "rationale": "first line\n\tsecond line",
    }


def test_json_decision_still_rejects_other_invalid_json():
    decision = JsonDecision({"vote": str})

    with pytest.raises(ValueError, match="invalid JSON"):
        decision.parse('{"vote": "A",}')


def test_render_prompt_rejects_empty_output():
    with pytest.raises(ValueError, match="must not be empty"):
        render_prompt(lambda context: "", prompt_context())
