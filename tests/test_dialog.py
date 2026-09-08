from copy import deepcopy
from dataclasses import replace
import json

import pytest

from maslab import Agent, ParallelMultiagent, Pipeline, SequentialMultiagent, Transform, dialog
from maslab.utils import dialog as display_dialog

from conftest import FixedModel


class MarkOutput(Transform):
    def transform(self, response):
        return replace(response, content=response.content + " [transformed]")


class FailingModel(FixedModel):
    def respond(self, messages):
        raise RuntimeError("model failed")


def test_single_agent_dialog_preserves_outputs_and_has_no_side_effects(capsys):
    agent = Agent("writer", FixedModel("  first line\nsecond line\n"), "system input")
    assert dialog is display_dialog
    assert dialog(agent.history()) == ""
    agent.query("first input")
    agent.query("second input")
    before = agent.history()
    calls = deepcopy(agent.model.calls)

    text = dialog(agent.history())

    turn = "[writer]\n  first line\nsecond line\n"
    assert text == turn + "\n\n" + turn
    assert dialog(agent.history()[-1:]) == turn
    assert agent.history() == before
    assert agent.model.calls == calls
    assert capsys.readouterr().out == ""


def test_nested_pipeline_dialog_from_json_keeps_turn_order_without_summary_duplicates():
    team = SequentialMultiagent([
        Agent("solver", FixedModel("draft")),
        Agent("reviewer", FixedModel("review")),
    ], loop=2)
    pipeline = Pipeline([team, MarkOutput(), Agent("editor", FixedModel("final")), MarkOutput()])
    pipeline.query("question one")
    pipeline.query("question two")
    history = json.loads(json.dumps(pipeline.history()))
    before = deepcopy(history)
    run = "[solver]\ndraft\n\n[reviewer]\nreview\n\n[solver]\ndraft\n\n[reviewer]\nreview\n\n[editor]\nfinal"

    assert dialog(history) == run + "\n\n" + run
    assert dialog(history[-1:]) == run
    assert history == before
    assert pipeline.history()[-1]["content"] == "final [transformed]"


def test_failed_parallel_group_keeps_completed_nested_outputs_in_participant_order():
    partial = Pipeline([
        Agent("first", FixedModel("finished before failure")), MarkOutput(),
        Agent("broken", FailingModel()),
    ], id="partial")
    parallel = ParallelMultiagent([
        partial, Agent("second", FixedModel("independent response")),
        Agent("failed-leaf", FailingModel()),
    ])
    with pytest.raises(RuntimeError, match="model failed"):
        parallel.query("input")

    assert dialog(parallel.history()) == (
        "[first]\nfinished before failure\n\n[second]\nindependent response"
    )
    empty_group = SequentialMultiagent([Agent("broken", FailingModel())])
    with pytest.raises(RuntimeError, match="model failed"):
        empty_group.query("input")
    assert dialog(empty_group.history()) == ""
