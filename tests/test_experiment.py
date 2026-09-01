import pytest

from maslab import (
    Agent,
    AgentConfig,
    Experiment,
    ExperimentConfig,
    HiddenBench,
    RunResult,
    Session,
    SessionConfig,
)

from conftest import FixedModel, hiddenbench_task


class RecordingModel(FixedModel):
    def respond(self, messages):
        system_prompt = messages[0]["content"]
        self.content = (
            '{"vote": "agent-0"}'
            if system_prompt == "system 0"
            else '{"vote": "agent-1"}'
        )
        return super().respond(messages)


def initial_prompt(**context):
    return "\n".join(context["information"])


@pytest.mark.parametrize("topology", ["sequential", "fully_connected", "cumulative"])
def test_session_run_dispatches_supported_topologies(topology):
    model = FixedModel()
    agents = [Agent("0", model, "system"), Agent("1", model, "system")]
    session = Session(
        hiddenbench_task(),
        agents,
        SessionConfig(num_rounds=1, debate_topology=topology),
    )

    history = session.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )

    assert history
    assert session.state.next_round == 2


def test_session_state_round_trip():
    task = hiddenbench_task()
    config = SessionConfig(num_rounds=0)
    session = Session(task, [Agent("0", FixedModel(), "system")], config)
    session.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )
    state = session.state_dict()

    restored = Session(task, [Agent("0", FixedModel(), "system")], config)
    restored.load_state_dict(state)

    assert restored.state_dict() == state


def test_experiment_returns_typed_in_memory_result(tmp_path):
    experiment = Experiment(
        model=FixedModel(),
        benchmark=HiddenBench([hiddenbench_task()]),
        prompt=initial_prompt,
        config=ExperimentConfig(
            agent=AgentConfig(use_context=False),
            session=SessionConfig(
                num_rounds=1,
                debate_topology="fully_connected",
            ),
            random_state=7,
        ),
    )

    result = experiment.run()

    assert isinstance(result, RunResult)
    assert result.config["random_state"] == 7
    assert result.results[0].metrics["overall"]["acc"] == 1.0
    assert result.metrics["num_tasks"] == 1
    assert len(experiment.sessions_) == 1
    output_path = result.save_json(tmp_path / "result.json")
    assert output_path.is_file()


def test_with_config_returns_a_new_experiment():
    experiment = Experiment(
        FixedModel(),
        HiddenBench([hiddenbench_task()]),
        initial_prompt,
    )

    changed = experiment.with_config(
        random_state=9,
        session__num_rounds=5,
        agent__use_context=False,
    )

    assert experiment.config.session.num_rounds == 3
    assert changed.config.random_state == 9
    assert changed.config.session.num_rounds == 5
    assert changed.config.agent.use_context is False


def test_session_checkpoint_uses_serializable_session_config(tmp_path):
    checkpoint_path = tmp_path / "state.json"
    task = hiddenbench_task()
    config = SessionConfig(num_rounds=0, checkpoint_path=checkpoint_path)
    session = Session(task, [Agent("0", FixedModel(), "system")], config)

    session.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )

    assert checkpoint_path.is_file()
    restored = Session(task, [Agent("0", FixedModel(), "system")], config)
    history = restored.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )
    assert history == session.history


def test_session_rejects_invalid_round_count():
    session = Session(
        hiddenbench_task(),
        [Agent("0", FixedModel(), "system")],
        SessionConfig(num_rounds=-1),
    )

    with pytest.raises(ValueError, match="num_rounds"):
        session.run(decision_message_callback=lambda **context: "decide")


@pytest.mark.parametrize(
    ("use_in_debate", "use_in_decision"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_debate_and_decision_context_settings_are_independent(
    use_in_debate,
    use_in_decision,
):
    model = FixedModel()
    experiment = Experiment(
        model=model,
        benchmark=HiddenBench([hiddenbench_task()]),
        config=ExperimentConfig(
            agent=AgentConfig(
                num_agents=1,
                use_context_in_debate=use_in_debate,
                use_context_in_decision=use_in_decision,
            ),
            session=SessionConfig(num_rounds=2),
        ),
    )

    experiment.run()

    second_debate_messages = model.calls[3]
    final_decision_messages = model.calls[4]
    assert len(second_debate_messages) == (4 if use_in_debate else 2)
    assert len(final_decision_messages) == (6 if use_in_decision else 2)


def test_fully_connected_broadcasts_independent_round_zero_decisions():
    model = RecordingModel()
    session = Session(
        hiddenbench_task(),
        [Agent("0", model, "system 0"), Agent("1", model, "system 1")],
        SessionConfig(
            num_rounds=1,
            debate_topology="fully_connected",
            initial_decision_visibility="broadcast",
        ),
    )

    session.run(use_context=False)

    agent_zero_debate = model.calls[2][-1]["content"]
    agent_one_debate = model.calls[3][-1]["content"]
    assert 'Agent 1: {"vote": "agent-1"}' in agent_zero_debate
    assert 'Agent 0: {"vote": "agent-0"}' in agent_one_debate
    assert 'Agent 0: {"vote": "agent-0"}' not in agent_zero_debate


def test_private_initial_decisions_still_produce_non_empty_debate_prompt():
    model = RecordingModel()
    session = Session(
        hiddenbench_task(),
        [Agent("0", model, "system 0"), Agent("1", model, "system 1")],
        SessionConfig(
            num_rounds=1,
            debate_topology="fully_connected",
            initial_decision_visibility="private",
        ),
    )

    session.run(use_context=False)

    assert "No other agent messages" in model.calls[2][-1]["content"]


def test_hiddenbench_default_prompt_set_returns_structured_decisions():
    result = Experiment(
        model=FixedModel(),
        benchmark=HiddenBench([hiddenbench_task()]),
        config=ExperimentConfig(session=SessionConfig(num_rounds=0)),
    ).run()

    assert result.results[0].decisions["0"]["0"] == {"vote": "A"}
    assert result.results[0].history[0]["decision"] == {"vote": "A"}


@pytest.mark.parametrize("topology", ["sequential", "fully_connected", "cumulative"])
@pytest.mark.parametrize("num_rounds", [0, 1, 2])
def test_decision_rounds_are_uniform_across_topologies(topology, num_rounds):
    num_agents = 2
    model = FixedModel()
    experiment = Experiment(
        model=model,
        benchmark=HiddenBench([hiddenbench_task()]),
        config=ExperimentConfig(
            agent=AgentConfig(num_agents=num_agents),
            session=SessionConfig(
                num_rounds=num_rounds,
                debate_topology=topology,
            ),
        ),
    )

    result = experiment.run()

    expected_rounds = list(range(num_rounds + 1))
    assert list(result.results[0].decisions) == [
        str(round_idx) for round_idx in expected_rounds
    ]
    assert [
        entry["round"] for entry in experiment.sessions_[0].opinion_history
    ] == expected_rounds
    assert len(model.calls) == num_agents * (1 + 2 * num_rounds)


def test_cumulative_early_stop_records_decisions_before_consensus():
    model = FixedModel()
    session = Session(
        hiddenbench_task(),
        [Agent("0", model, "system 0"), Agent("1", model, "system 1")],
        SessionConfig(
            num_rounds=3,
            debate_topology="cumulative",
            early_stop_on_consensus=True,
        ),
    )
    observed_decision_rounds = []

    def consensus_callback(*, round_idx, **kwargs):
        observed_decision_rounds.append([
            entry["round"]
            for entry in session.opinion_history
        ])
        return "A" if round_idx == 2 else None

    session.run(
        decision_message_callback=lambda **context: "decide",
        consensus_callback=consensus_callback,
        use_context=False,
    )

    assert session.total_rounds == 2
    assert session.consensus_round == 2
    assert session.stopped_early
    assert observed_decision_rounds == [[0, 1], [0, 1, 2]]
    assert [entry["round"] for entry in session.opinion_history] == [0, 1, 2]
    assert len(model.calls) == 2 * (1 + 2 * 2)


def test_cumulative_checkpoint_resume_does_not_repeat_decisions(tmp_path):
    checkpoint_path = tmp_path / "cumulative-state.json"
    task = hiddenbench_task()
    first_model = FixedModel()
    first_session = Session(
        task,
        [
            Agent("0", first_model, "system 0"),
            Agent("1", first_model, "system 1"),
        ],
        SessionConfig(
            num_rounds=1,
            debate_topology="cumulative",
            checkpoint_path=checkpoint_path,
        ),
    )
    first_session.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )

    resumed_model = FixedModel()
    resumed_session = Session(
        task,
        [
            Agent("0", resumed_model, "system 0"),
            Agent("1", resumed_model, "system 1"),
        ],
        SessionConfig(
            num_rounds=2,
            debate_topology="cumulative",
            checkpoint_path=checkpoint_path,
        ),
    )
    resumed_session.run(
        decision_message_callback=lambda **context: "decide",
        use_context=False,
    )

    decision_entries = [
        entry for entry in resumed_session.history if "decision" in entry
    ]
    for round_idx in (0, 1, 2):
        assert sum(
            entry["round"] == round_idx for entry in decision_entries
        ) == 2
    assert [
        entry["round"] for entry in resumed_session.opinion_history
    ] == [0, 1, 2]
    assert len(resumed_model.calls) == 4
    assert len(resumed_session.agents[0].context) == (
        len(first_session.agents[0].context) + 2
    )
