from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from pathlib import Path
import random
from typing import Sequence

from .config import GenerationConfig
from .prompts import (
    PromptContext,
    generic_debate_prompt,
    generic_decision_prompt,
    render_prompt as render_prompt_value,
)
from .types import ChatMessage, Message, Response, RunResult, SessionState, TaskResult

@dataclass
class Config:
    # experiment
    experiment_name: str = "debug"
    output_path: str = "outputs/test_result.json"
    checkpoint_path: str | None = None
    seed: int = 0

    # benchmark / dataset
    benchmark: str = "hiddenbench"
    dataset_path: str = "datasets/hiddenbench.json"
    num_tasks: int | None = None
    shuffle: bool = False

    # prompt
    prompt_path: str | None = None

    # model
    model: str = "google/gemma-4-31b-it"
    reasoning: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95

    # agents
    num_agents: int | None = None
    context_policy: str = "append"
    use_context: bool = False
    use_context_in_debate: bool | None = None
    use_context_in_decision: bool = False

    # session
    # Inclusive last round index; round 0 is independent inference.
    num_rounds: int = 3
    debate_topology: str = "sequential"
    initial_decision_visibility: str = "broadcast"

    # stopping / metrics
    consensus_threshold: float = 1.0
    early_stop_on_consensus: bool = False

    # robustness
    max_retries: int = 2
    timeout_seconds: int | None = None

class Model(ABC):
    def __init__(
        self,
        name,
        api_key=None,
        reasoning=False,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        max_retries=2,
        timeout_seconds=None,
        generation_config: GenerationConfig | None = None,
    ):
        if generation_config is not None:
            reasoning = generation_config.reasoning
            max_tokens = generation_config.max_tokens
            temperature = generation_config.temperature
            top_p = generation_config.top_p
            max_retries = generation_config.max_retries
            timeout_seconds = generation_config.timeout_seconds
        self.name = name
        self.api_key = api_key
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.generation_config = GenerationConfig(
            reasoning=reasoning,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def build_messages(prompt: str, system_prompt: str = None):
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return messages

    @abstractmethod
    def respond(self, messages: Sequence[ChatMessage]) -> Response:
        raise NotImplementedError()

class Agent:
    CONTEXT_POLICIES = {"append", "replace", "last", "last_conversation"}

    def __init__(
        self,
        id: str,
        model: Model,
        system_prompt: str = "",
        context_policy: str = "append",
    ):
        if context_policy not in self.CONTEXT_POLICIES:
            choices = ", ".join(sorted(self.CONTEXT_POLICIES))
            raise ValueError(f"context_policy must be one of: {choices}")
        self.id = id
        self.model = model
        self.context: list[ChatMessage] = []
        self.context_policy = context_policy

        if system_prompt:
            self.context.append({
                "role": "system",
                "content": system_prompt,
            })

    def generate(self, message: str, use_context: bool = True) -> Response:
        """Generate one response without changing the agent's context."""
        messages = deepcopy(self.context if use_context else self.system_context())
        messages.append({
            "role": "user",
            "content": message,
        })
        return self.model.respond(messages)

    def chat(
        self,
        message: str,
        use_context: bool = True,
        update_context: bool = True,
    ) -> Response:
        """Generate a response and, by default, record the full conversation."""
        response = self.generate(message, use_context=use_context)
        if not update_context:
            return response

        self.context.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response.content},
        ])
        self.manage_context()
        return response

    def system_context(self) -> list[ChatMessage]:
        if self.context and self.context[0].get("role") == "system":
            return [deepcopy(self.context[0])]
        return []

    def clear_context(self):
        self.context = self.system_context()

    def manage_context(self):
        if self.context_policy == "replace":
            self.clear_context()
        elif self.context_policy == "last":
            self.context = self.system_context() + self.context[-1:]
        elif self.context_policy == "last_conversation":
            self.context = self.system_context() + self.context[-2:]

class Session:
    def __init__(self, task, agents, config):
        self.task = task
        self.agents = agents
        self.config = config
        self.num_rounds = config.num_rounds
        self.checkpoint = (
            CheckpointStore(config.checkpoint_path, config)
            if config.checkpoint_path else None
        )
        self.state = SessionState()

    @property
    def history(self):
        return self.state.history

    @history.setter
    def history(self, value):
        self.state.history = value

    @property
    def logs(self):
        return self.state.logs

    @logs.setter
    def logs(self, value):
        self.state.logs = value

    @property
    def opinion_history(self):
        return self.state.opinion_history

    @opinion_history.setter
    def opinion_history(self, value):
        self.state.opinion_history = value

    @property
    def consensus_round(self):
        return self.state.consensus_round

    @consensus_round.setter
    def consensus_round(self, value):
        self.state.consensus_round = value

    @property
    def stopped_early(self):
        return self.state.stopped_early

    @stopped_early.setter
    def stopped_early(self, value):
        self.state.stopped_early = value

    @property
    def total_rounds(self):
        return self.state.total_rounds

    @total_rounds.setter
    def total_rounds(self, value):
        self.state.total_rounds = value

    def _task_id(self):
        if isinstance(self.task, dict):
            return self.task.get("id")
        return None

    def _usage_from_response(self, response):
        input_tokens = response.input_tokens or 0
        output_tokens = response.output_tokens or 0
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "generation_time": response.generation_time,
        }

    def log_response(
        self,
        *,
        round_idx,
        agent,
        message,
        response,
    ):
        entry = {
            "task_id": self._task_id(),
            "round": round_idx,
            "agent_id": agent.id,
            "message": message,
            "usage": self._usage_from_response(response),
            **asdict(response),
        }
        self.logs.append(entry)
        return entry

    def restore_checkpoint(self):
        if self.checkpoint is None:
            return 0, []

        state = self.checkpoint.task_state(self._task_id())
        if state is None:
            return 0, []

        start_round = state["next_round"]
        if start_round > self.num_rounds + 1:
            raise ValueError(
                "num_rounds cannot be less than the checkpoint's last completed round"
            )

        agent_contexts = state["agent_contexts"]
        if len(agent_contexts) != len(self.agents):
            raise ValueError("Checkpoint agent count does not match the current task")

        self.load_state_dict(state)
        if not any(entry.get("round") == 0 for entry in self.opinion_history):
            decisions = {
                entry["agent_id"]: entry.get("content")
                for entry in self.history
                if entry.get("round") == 0 and entry.get("agent_id") is not None
            }
            if decisions:
                self.opinion_history.insert(0, {
                    "round": 0,
                    "opinions": {},
                    "decisions": decisions,
                })
        return start_round, list(self.state.global_context)

    def _restore_global_context(self, messages):
        history_lookup = {}
        for entry in self.history:
            round_idx = entry.get("round")
            if round_idx in (None, 0):
                continue
            key = (str(entry.get("agent_id")), entry.get("content"))
            history_lookup.setdefault(key, []).append(round_idx)

        round_lookup = {}
        for entry in self.opinion_history:
            round_idx = entry.get("round")
            if round_idx in (None, 0):
                continue
            for speaker, content in (entry.get("opinions") or {}).items():
                round_lookup.setdefault((str(speaker), content), []).append(round_idx)

        restored_messages = []
        for message in messages:
            payload = dict(message)
            if payload.get("round_idx") is None:
                key = (str(payload.get("speaker")), payload.get("content"))
                rounds = history_lookup.get(key) or round_lookup.get(key)
                if rounds:
                    payload["round_idx"] = rounds.pop(0)
            restored_messages.append(Message(**payload))
        return restored_messages

    def checkpoint_state(self, next_round, latest_messages):
        return {
            "next_round": next_round,
            "history": self.history,
            "agent_contexts": [agent.context for agent in self.agents],
            "global_context": [asdict(message) for message in latest_messages],
            "opinion_history": self.opinion_history,
            "consensus_round": self.consensus_round,
            "stopped_early": self.stopped_early,
            "total_rounds": self.total_rounds,
        }

    def state_dict(self):
        """Return the JSON-serializable execution state for this session."""
        return self.checkpoint_state(
            self.state.next_round,
            self.state.global_context,
        )

    def load_state_dict(self, state):
        """Restore execution state produced by :meth:`state_dict`."""
        agent_contexts = state["agent_contexts"]
        if len(agent_contexts) != len(self.agents):
            raise ValueError("State agent count does not match the current session")
        for agent, context in zip(self.agents, agent_contexts):
            agent.context = deepcopy(context)

        self.history = deepcopy(state.get("history", []))
        self.logs = list(self.history)
        self.opinion_history = deepcopy(state.get("opinion_history", []))
        self.consensus_round = state.get("consensus_round")
        self.stopped_early = state.get("stopped_early", False)
        self.total_rounds = state.get("total_rounds", 0)
        self.state.next_round = state.get("next_round", 0)
        self.state.global_context = self._restore_global_context(
            state.get("global_context", [])
        )
        return self

    def save_checkpoint(self, next_round, latest_messages):
        self.state.next_round = next_round
        self.state.global_context = list(latest_messages)
        if self.checkpoint is None:
            return None
        return self.checkpoint.save_task(
            self._task_id(),
            self.checkpoint_state(next_round, latest_messages),
        )

    def _initial_debate_state(self, start_round, global_context):
        if start_round != 0 or global_context is not None:
            return start_round, list(global_context or [])
        return self.restore_checkpoint()

    def _render_debate_message(
        self,
        debate_message_callback,
        round_idx,
        agent_idx,
        agent,
        received_messages,
    ):
        context = self._prompt_context(
            phase="debate",
            round_idx=round_idx,
            agent_idx=agent_idx,
            agent=agent,
            received_messages=received_messages,
        )
        return render_prompt_value(
            debate_message_callback or generic_debate_prompt,
            context,
        )

    def _prompt_context(
        self,
        *,
        phase,
        round_idx,
        agent_idx,
        agent,
        received_messages,
    ):
        options = None
        if isinstance(self.task, dict) and self.task.get("possible_answers"):
            options = tuple(str(option) for option in self.task["possible_answers"])
        return PromptContext(
            task=self.task,
            phase=phase,
            round_idx=round_idx,
            agent_idx=agent_idx,
            agent_id=str(agent.id),
            agent=agent,
            visible_messages=tuple(received_messages),
            options=options,
        )

    def _render_decision_message(
        self,
        decision_message_callback,
        round_idx,
        agent_idx,
        agent,
        received_messages,
    ):
        context = self._prompt_context(
            phase="decision",
            round_idx=round_idx,
            agent_idx=agent_idx,
            agent=agent,
            received_messages=received_messages,
        )
        return render_prompt_value(
            decision_message_callback or generic_decision_prompt,
            context,
        )

    def _parse_decision(self, response, decision_parser):
        if decision_parser is None:
            return response.content
        return decision_parser(response.content)

    def _run_decision_round(
        self,
        round_idx,
        decision_message_callback,
        decision_parser,
        received_messages,
        use_context_in_decision=False,
    ):
        decisions = {}
        for agent_idx, agent in enumerate(self.agents):
            message = self._render_decision_message(
                decision_message_callback,
                round_idx,
                agent_idx,
                agent,
                received_messages,
            )
            response = agent.generate(
                message,
                use_context=use_context_in_decision,
            )
            decision = self._parse_decision(response, decision_parser)
            history_entry = self.log_response(
                round_idx=round_idx,
                agent=agent,
                message=message,
                response=response,
            )
            history_entry["decision"] = decision
            self.history.append(history_entry)
            decisions[agent.id] = decision
        return decisions

    def _record_initial_decision_round(
        self,
        decision_message_callback,
        decision_parser,
        use_context_in_decision,
    ):
        history_start = len(self.history)
        decisions = self._run_decision_round(
            0,
            decision_message_callback,
            decision_parser,
            [],
            use_context_in_decision,
        )
        self.opinion_history.append({
            "round": 0,
            "opinions": {},
            "decisions": decisions,
        })
        return [
            Message(
                str(entry["agent_id"]),
                entry["content"],
                entry.get("output_tokens")
                or (entry.get("usage") or {}).get("output_tokens")
                or 0,
                0,
            )
            for entry in self.history[history_start:]
        ]

    def _initial_debate_context(self, messages):
        visibility = getattr(
            self.config,
            "initial_decision_visibility",
            "broadcast",
        )
        return list(messages) if visibility == "broadcast" else []

    def _record_round_decisions(
        self,
        round_idx,
        round_history,
        latest_messages,
        decision_message_callback,
        decision_parser,
        use_context_in_decision,
    ):
        opinions = {
            entry["agent_id"]: entry["content"]
            for entry in round_history
        }
        decisions = self._run_decision_round(
            round_idx,
            decision_message_callback,
            decision_parser,
            latest_messages,
            use_context_in_decision,
        )
        self.opinion_history.append({
            "round": round_idx,
            "opinions": opinions,
            "decisions": decisions,
        })

    def sequential_debate(
        self,
        debate_message_callback=None,
        decision_message_callback=None,
        decision_parser=None,
        use_context=True,
        use_context_in_decision=False,
        start_round=0,
        global_context=None,
    ):
        start_round, global_context = self._initial_debate_state(
            start_round,
            global_context,
        )

        if start_round == 0:
            initial_messages = self._record_initial_decision_round(
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            global_context = self._initial_debate_context(initial_messages)
            self.save_checkpoint(1, global_context)
            start_round = 1

        for round_idx in range(start_round, self.num_rounds + 1):
            round_history = []

            for agent_idx, agent in enumerate(self.agents):
                message = self._render_debate_message(
                    debate_message_callback,
                    round_idx,
                    agent_idx,
                    agent,
                    global_context,
                )

                response = agent.chat(
                    message,
                    use_context=use_context,
                )

                global_context = [
                    Message(
                        agent.id,
                        response.content,
                        response.output_tokens or 0,
                        round_idx,
                    )
                ]

                history_entry = self.log_response(
                    round_idx=round_idx,
                    agent=agent,
                    message=message,
                    response=response,
                )
                self.history.append(history_entry)
                round_history.append(history_entry)

            self._record_round_decisions(
                round_idx,
                round_history,
                global_context,
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            self.total_rounds = round_idx
            self.save_checkpoint(round_idx + 1, global_context)

        return self.history
            
    def fully_connected_debate(
        self,
        debate_message_callback=None,
        decision_message_callback=None,
        decision_parser=None,
        use_context=True,
        use_context_in_decision=False,
        start_round=0,
        global_context=None,
    ):
        start_round, global_context = self._initial_debate_state(
            start_round,
            global_context,
        )

        if start_round == 0:
            initial_messages = self._record_initial_decision_round(
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            global_context = self._initial_debate_context(initial_messages)
            self.save_checkpoint(1, global_context)
            start_round = 1

        for round_idx in range(start_round, self.num_rounds + 1):
            previous_global_context = list(global_context)
            global_context = []
            round_history = []

            for agent_idx, agent in enumerate(self.agents):
                agent_global_context = [
                    message
                    for message in previous_global_context
                    if message.speaker != agent.id
                ]

                message = self._render_debate_message(
                    debate_message_callback,
                    round_idx,
                    agent_idx,
                    agent,
                    agent_global_context,
                )

                response = agent.chat(
                    message,
                    use_context=use_context,
                )

                global_context.append(
                    Message(
                        agent.id,
                        response.content,
                        response.output_tokens or 0,
                        round_idx,
                    )
                )

                history_entry = self.log_response(
                    round_idx=round_idx,
                    agent=agent,
                    message=message,
                    response=response,
                )
                self.history.append(history_entry)
                round_history.append(history_entry)

            self._record_round_decisions(
                round_idx,
                round_history,
                global_context,
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            self.total_rounds = round_idx
            self.save_checkpoint(round_idx + 1, global_context)

        return self.history

    def cumulative_debate(
        self,
        debate_message_callback=None,
        decision_message_callback=None,
        decision_parser=None,
        consensus_callback=None,
        use_context=True,
        use_context_in_decision=False,
        start_round=0,
        global_context=None,
    ):
        start_round, global_context = self._initial_debate_state(
            start_round,
            global_context,
        )

        if start_round == 0:
            initial_messages = self._record_initial_decision_round(
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            global_context = self._initial_debate_context(initial_messages)
            self.save_checkpoint(1, global_context)
            start_round = 1

        for round_idx in range(start_round, self.num_rounds + 1):
            round_history = []

            for agent_idx, agent in enumerate(self.agents):
                message = self._render_debate_message(
                    debate_message_callback,
                    round_idx,
                    agent_idx,
                    agent,
                    list(global_context),
                )

                response = agent.chat(
                    message,
                    use_context=use_context,
                )
                global_context.append(
                    Message(
                        agent.id,
                        response.content,
                        response.output_tokens or 0,
                        round_idx,
                    )
                )

                history_entry = self.log_response(
                    round_idx=round_idx,
                    agent=agent,
                    message=message,
                    response=response,
                )
                self.history.append(history_entry)
                round_history.append(history_entry)

            self.total_rounds = round_idx
            self._record_round_decisions(
                round_idx,
                round_history,
                global_context,
                decision_message_callback,
                decision_parser,
                use_context_in_decision,
            )
            self.save_checkpoint(round_idx + 1, global_context)

            if consensus_callback is not None:
                round_messages = [
                    Message(
                        entry["agent_id"],
                        entry["content"],
                        entry.get("output_tokens")
                        or (entry.get("usage") or {}).get("output_tokens")
                        or 0,
                        round_idx,
                    )
                    for entry in round_history
                ]
                consensus_vote = consensus_callback(
                    task=self.task,
                    round_idx=round_idx,
                    round_messages=round_messages,
                    global_context=list(global_context),
                )
                if consensus_vote is not None and self.consensus_round is None:
                    self.consensus_round = round_idx
                    if self.config.early_stop_on_consensus:
                        self.stopped_early = True
                        self.save_checkpoint(round_idx + 1, global_context)
                        break

        return self.history

    def run(
        self,
        *,
        debate_message_callback=None,
        decision_message_callback=None,
        decision_parser=None,
        consensus_callback=None,
        use_context_in_debate=True,
        use_context_in_decision=False,
        use_context=None,
    ):
        """Run the topology selected by the session configuration."""
        if not self.agents:
            raise ValueError("a session requires at least one agent")
        if type(self.num_rounds) is not int or self.num_rounds < 0:
            raise ValueError("num_rounds must be a non-negative integer")
        consensus_threshold = getattr(self.config, "consensus_threshold", 1.0)
        if not 0 < consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be in the interval (0, 1]")
        visibility = getattr(
            self.config,
            "initial_decision_visibility",
            "broadcast",
        )
        if visibility not in {"broadcast", "private"}:
            raise ValueError(
                "initial_decision_visibility must be 'broadcast' or 'private'"
            )
        topology = self.config.debate_topology.removesuffix("_debate")
        debates = {
            "sequential": self.sequential_debate,
            "fully_connected": self.fully_connected_debate,
            "cumulative": self.cumulative_debate,
        }
        try:
            debate = debates[topology]
        except KeyError as error:
            choices = ", ".join(sorted(debates))
            raise ValueError(f"debate_topology must be one of: {choices}") from error

        if use_context is not None:
            use_context_in_debate = use_context
        kwargs = {
            "debate_message_callback": debate_message_callback,
            "decision_message_callback": decision_message_callback,
            "decision_parser": decision_parser,
            "use_context": use_context_in_debate,
            "use_context_in_decision": use_context_in_decision,
        }
        if topology == "cumulative":
            kwargs["consensus_callback"] = consensus_callback
        return debate(**kwargs)

    def append_history(self, round: int, agent_id: str, message: str):
        self.history.append({
            "round": round,
            "agent_id": agent_id,
            "content": message,
        })

    def clear_history(self):
        self.history = []
        self.logs = []
        self.opinion_history = []
        self.consensus_round = None
        self.stopped_early = False
        self.total_rounds = 0
        self.state.next_round = 0
        self.state.global_context = []
        
class Benchmark(ABC):
    USAGE_METRIC_FIELDS = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "generation_time",
    }

    def __init__(self, tasks, count=None, shuffle=False, seed=0):
        self.seed = seed
        if isinstance(tasks, (str, os.PathLike)):
            loaded_tasks = self.load_tasks(tasks)
        else:
            loaded_tasks = list(tasks)

        if shuffle:
            random.Random(seed).shuffle(loaded_tasks)
        if count is not None:
            loaded_tasks = loaded_tasks[:count]
        self.tasks: list = loaded_tasks

    @classmethod
    def from_json(cls, path, count=None, shuffle=False, seed=0):
        return cls(
            cls.load_tasks(path),
            count=count,
            shuffle=shuffle,
            seed=seed,
        )

    @staticmethod
    def load_tasks(path):
        with Path(path).open("r", encoding="utf-8") as file:
            tasks = json.load(file)
        if not isinstance(tasks, list):
            raise ValueError("Benchmark JSON must contain a list of tasks")
        return tasks

    @staticmethod
    def render_prompt(prompt, context=None, **variables):
        if context is None:
            task = variables.pop("task", None)
            context = PromptContext(
                task=task,
                phase="initial",
                round_idx=0,
                agent_idx=variables.pop("agent_idx", 0),
                agent_id=str(variables.pop("agent_id", "0")),
                options=tuple(str(value) for value in variables.get("options", ()))
                or None,
                variables=variables,
            )
        return render_prompt_value(prompt, context)

    @abstractmethod
    def build_prompts(self, task, prompt, num_agent=None):
        raise NotImplementedError()

    def default_prompts(self):
        return None

    @abstractmethod
    def compute_metrics(self, task, history, consensus_threshold=1.0):
        raise NotImplementedError()

    def _sum_usage(self, entries):
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "generation_time": 0,
        }
        has_generation_time = False

        for entry in entries:
            entry_usage = entry.get("usage") or {}
            input_tokens = entry_usage.get("input_tokens", entry.get("input_tokens") or 0) or 0
            output_tokens = entry_usage.get("output_tokens", entry.get("output_tokens") or 0) or 0
            total_tokens = entry_usage.get("total_tokens", input_tokens + output_tokens) or 0
            generation_time = entry_usage.get("generation_time", entry.get("generation_time"))

            usage["input_tokens"] += input_tokens
            usage["output_tokens"] += output_tokens
            usage["total_tokens"] += total_tokens
            if generation_time is not None:
                usage["generation_time"] += generation_time
                has_generation_time = True

        if not has_generation_time:
            usage["generation_time"] = None
        return usage

    def _compact_history(self, history):
        compact_history = []
        for entry in history:
            usage = entry.get("usage") or {}
            compact_history.append({
                "round": entry.get("round"),
                "agent_id": entry.get("agent_id"),
                "prompt": entry.get("prompt"),
                "content": entry.get("content"),
                "reasoning": entry.get("reasoning"),
                "decision": entry.get("decision"),
                "input_tokens": entry.get("input_tokens", usage.get("input_tokens")),
                "output_tokens": entry.get("output_tokens", usage.get("output_tokens")),
                "generation_time": entry.get("generation_time", usage.get("generation_time")),
            })
        return compact_history

    def _decisions_by_round(self, history):
        decisions = {}
        for entry in history:
            round_idx = entry.get("round")
            if round_idx is None:
                continue

            round_key = str(round_idx)
            decisions.setdefault(round_key, {})[entry.get("agent_id")] = entry.get(
                "decision",
                entry.get("content"),
            )
        return decisions

    def build_result(self, task, history, metrics=None, opinion_history=None):
        result = {
            "task": task,
            "metrics": metrics,
            "decisions": self._decisions_by_round(history),
        }
        if opinion_history is not None:
            result["opinion_history"] = opinion_history
        result["history"] = self._compact_history(history)
        return result

    def _aggregate_metric_scope(self, scopes):
        scopes = [scope for scope in scopes if isinstance(scope, dict)]
        aggregate = {"num_tasks": len(scopes)}
        if not scopes:
            return aggregate

        for field in ("input_tokens", "output_tokens", "total_tokens"):
            total = sum((scope.get(field) or 0) for scope in scopes)
            aggregate[field] = total / len(scopes)

        generation_times = [
            scope.get("generation_time")
            for scope in scopes
            if scope.get("generation_time") is not None
        ]
        aggregate["generation_time"] = (
            sum(generation_times) / len(generation_times)
            if generation_times else None
        )

        numeric_fields = sorted({
            key
            for scope in scopes
            for key, value in scope.items()
            if (
                key not in self.USAGE_METRIC_FIELDS
                and type(value) in (int, float)
            )
        })
        for field in numeric_fields:
            values = [
                scope[field]
                for scope in scopes
                if type(scope.get(field)) in (int, float)
            ]
            if values:
                aggregate[field] = sum(values) / len(values)

        return aggregate

    def aggregate_metrics(self, results):
        results = list(results)
        metrics = [
            result.get("metrics")
            for result in results
            if isinstance(result.get("metrics"), dict)
        ]
        aggregate = {"num_tasks": len(results)}
        if not metrics:
            return aggregate

        overall_scopes = [
            metric["overall"]
            if isinstance(metric.get("overall"), dict)
            else metric
            for metric in metrics
        ]
        aggregate["overall"] = self._aggregate_metric_scope(overall_scopes)

        by_round = {}
        for metric in metrics:
            round_metrics = metric.get("by_round")
            if not isinstance(round_metrics, dict):
                continue
            for round_idx, scope in round_metrics.items():
                by_round.setdefault(str(round_idx), []).append(scope)

        if by_round:
            aggregate["by_round"] = {
                round_idx: self._aggregate_metric_scope(by_round[round_idx])
                for round_idx in sorted(by_round, key=lambda value: int(value))
            }

        return aggregate

    def save_result(self, path, results, config=None):
        if isinstance(results, RunResult):
            return results.save_json(path)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(results, TaskResult):
            results = [results.to_dict()]
        if isinstance(results, dict):
            results = [results]
        results = [
            result.to_dict() if isinstance(result, TaskResult) else result
            for result in results
        ]
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        elif is_dataclass(config):
            config = asdict(config)

        output = {
            "config": config or {},
            "metrics": self.aggregate_metrics(results),
            "results": results,
        }

        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

class CheckpointStore:
    VERSION = 1
    MUTABLE_CONFIG_FIELDS = {
        "checkpoint_path",
        "max_retries",
        "num_rounds",
        "num_tasks",
        "output_path",
        "timeout_seconds",
    }

    def __init__(self, path, config):
        self.path = Path(path)
        self.config = self._json_compatible(asdict(config))
        self.data = {
            "version": self.VERSION,
            "config": self.config,
            "tasks": {},
        }

        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("version") != self.VERSION:
                raise ValueError("Unsupported checkpoint version")
            if self._signature(self.data.get("config", {})) != self._signature(self.config):
                raise ValueError("Checkpoint is incompatible with current config")

    def _json_compatible(self, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                key: self._json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._json_compatible(item) for item in value]
        return value

    def _signature(self, config):
        return {
            key: value
            for key, value in config.items()
            if key not in self.MUTABLE_CONFIG_FIELDS
        }

    def task_state(self, task_id):
        return deepcopy(self.data["tasks"].get(str(task_id)))

    def save_task(self, task_id, state):
        self.data["config"] = self.config
        self.data["tasks"][str(task_id)] = deepcopy(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(f"{self.path}.tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, self.path)
        return self.path
