"""High-level experiment orchestration."""

from dataclasses import replace
from typing import Any

from .base import Agent, Benchmark, Model, Session
from .config import ExperimentConfig
from .prompts import (
    DecisionPolicy,
    DecisionPrompt,
    PromptSet,
    generic_debate_prompt,
    generic_decision_prompt,
)
from .types import RunResult, TaskResult


default_decision_prompt = generic_decision_prompt


class Experiment:
    """Compose a model, benchmark, prompts, and session into one runnable object."""

    def __init__(
        self,
        model: Model,
        benchmark: Benchmark,
        prompt=None,
        *,
        prompts: PromptSet | None = None,
        config: ExperimentConfig | None = None,
        debate_prompt=None,
        decision_prompt=None,
        decision_parser=None,
    ):
        if not isinstance(model, Model):
            raise TypeError("model must inherit from maslab.Model")
        if not isinstance(benchmark, Benchmark):
            raise TypeError("benchmark must inherit from maslab.Benchmark")
        if config is not None and not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        if prompts is not None and not isinstance(prompts, PromptSet):
            raise TypeError("prompts must be a PromptSet")
        if prompts is not None and not isinstance(prompts.decision, DecisionPolicy):
            raise TypeError("PromptSet.decision must implement DecisionPolicy")
        if prompts is not None and any(
            value is not None
            for value in (prompt, debate_prompt, decision_prompt, decision_parser)
        ):
            raise ValueError(
                "prompts cannot be combined with legacy prompt arguments"
            )

        default_factory = getattr(benchmark, "default_prompts", None)
        defaults = default_factory() if callable(default_factory) else None
        if prompts is None:
            initial = prompt if prompt is not None else getattr(defaults, "initial", None)
            if initial is None:
                raise ValueError("an initial prompt or PromptSet is required")
            debate = (
                debate_prompt
                if debate_prompt is not None
                else getattr(defaults, "debate", generic_debate_prompt)
            )
            if decision_prompt is not None or decision_parser is not None:
                if decision_parser is None:
                    decision = DecisionPrompt(
                        renderer=decision_prompt or generic_decision_prompt,
                    )
                else:
                    decision = DecisionPrompt(
                        renderer=decision_prompt or generic_decision_prompt,
                        parser=decision_parser,
                    )
            else:
                decision = getattr(defaults, "decision", DecisionPrompt())
            prompts = PromptSet(
                initial=initial,
                debate=debate,
                decision=decision,
            )

        if prompts.initial is None:
            raise ValueError("PromptSet.initial must not be None")
        if prompts.debate is None:
            raise ValueError("PromptSet.debate must not be None")
        if not isinstance(prompts.decision, DecisionPolicy):
            raise TypeError("PromptSet.decision must implement DecisionPolicy")

        self.model = model
        self.benchmark = benchmark
        self.prompts = prompts
        self.prompt = prompts.initial
        self.config = config or ExperimentConfig()
        self.debate_prompt = prompts.debate
        self.decision_prompt = prompts.decision.render
        self.decision_parser = prompts.decision.parse

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def with_config(self, **changes) -> "Experiment":
        """Return a copy with top-level or ``section__field`` config changes."""
        agent_changes = {}
        session_changes = {}
        top_level_changes = {}
        for name, value in changes.items():
            if name.startswith("agent__"):
                agent_changes[name.removeprefix("agent__")] = value
            elif name.startswith("session__"):
                session_changes[name.removeprefix("session__")] = value
            else:
                top_level_changes[name] = value

        config = replace(
            self.config,
            agent=replace(self.config.agent, **agent_changes),
            session=replace(self.config.session, **session_changes),
            **top_level_changes,
        )
        return type(self)(
            model=self.model,
            benchmark=self.benchmark,
            prompts=self.prompts,
            config=config,
        )

    def _build_agents(self, task):
        prompts = self.benchmark.build_prompts(
            task,
            self.prompt,
            num_agent=self.config.agent.num_agents,
        )
        return [
            Agent(
                str(agent_idx),
                self.model,
                system_prompt,
                context_policy=self.config.agent.context_policy,
            )
            for agent_idx, system_prompt in enumerate(prompts)
        ]

    def _consensus_callback(self):
        detector = getattr(self.benchmark, "detect_discussion_consensus", None)
        if detector is None:
            return None

        def detect(*, task, round_messages, **kwargs):
            options = task.get("possible_answers") if isinstance(task, dict) else None
            if options is None:
                return None
            return detector(round_messages, options)

        return detect

    def run(self) -> RunResult:
        """Run all benchmark tasks and return an in-memory typed result."""
        self.benchmark.seed = self.config.random_state
        self.sessions_ = []
        task_results = []
        for task in self.benchmark.tasks:
            agents = self._build_agents(task)
            session = Session(task, agents, self.config.session)
            self.sessions_.append(session)
            use_context_in_debate = self.config.agent.use_context_in_debate
            if self.config.agent.use_context is not None:
                use_context_in_debate = self.config.agent.use_context
            history = session.run(
                debate_message_callback=self.debate_prompt,
                decision_message_callback=self.decision_prompt,
                decision_parser=self.decision_parser,
                consensus_callback=self._consensus_callback(),
                use_context_in_debate=use_context_in_debate,
                use_context_in_decision=(
                    self.config.agent.use_context_in_decision
                ),
            )
            metrics = self.benchmark.compute_metrics(
                task,
                history,
                consensus_threshold=self.config.session.consensus_threshold,
            )
            value = self.benchmark.build_result(
                task,
                history,
                metrics=metrics,
                opinion_history=session.opinion_history,
            )
            task_results.append(TaskResult.from_dict(value))

        aggregate = self.benchmark.aggregate_metrics(
            result.to_dict() for result in task_results
        )
        self.result_ = RunResult(
            config=self.get_config(),
            metrics=aggregate,
            results=task_results,
        )
        return self.result_
