"""HiddenBench benchmark implementation."""

from collections import Counter
import json
import re

from ..base import Benchmark
from ..prompts import JsonDecision, PromptContext, PromptSet, generic_debate_prompt


class HiddenBench(Benchmark):

    @staticmethod
    def _initial_prompt(context: PromptContext) -> str:
        description = context.variables["description"]
        if isinstance(description, (list, tuple)):
            description = "\n".join(str(item) for item in description)
        information = "\n".join(
            f"- {item}" for item in context.variables["information"]
        )
        options = ", ".join(str(option) for option in context.options or ())
        return (
            f"{description}\n"
            f"Information:\n{information}\n"
            f"Options: {options}"
        )

    @classmethod
    def prompts(cls, initial=None) -> PromptSet:
        return PromptSet(
            initial=initial or cls._initial_prompt,
            debate=generic_debate_prompt,
            decision=JsonDecision({"vote": str}),
        )

    def default_prompts(self):
        return self.prompts()

    def build_prompts(self, task, prompt, num_agent=None):
        description = [task["description"]]
        shared_info = task["shared_information"]
        hidden_info = task["hidden_information"]
        options = task["possible_answers"]

        if num_agent is None:
            num_agent = len(hidden_info)

        if num_agent < 1:
            raise ValueError("num_agent must be at least 1")

        if num_agent > len(hidden_info):
            raise ValueError(
                f"num_agent={num_agent} exceeds available hidden information "
                f"({len(hidden_info)})"
            )

        prompts = []
        for agent_idx in range(num_agent):
            rendered_prompt = self.render_prompt(
                prompt,
                PromptContext(
                    task=task,
                    phase="initial",
                    round_idx=0,
                    agent_idx=agent_idx,
                    agent_id=str(agent_idx),
                    options=tuple(str(option) for option in options),
                    variables={
                        "description": description,
                        "information": shared_info + [hidden_info[agent_idx]],
                        "discussions": [],
                    },
                ),
            )
            prompts.append(rendered_prompt)

        return prompts

    def _mean(self, values):
        values = list(values)
        return sum(values) / len(values) if values else 0

    def _extract_vote(self, value):
        if isinstance(value, dict):
            if "vote" in value:
                return value.get("vote")
            value = value.get("content")
        if not isinstance(value, str):
            return None

        first_brace_idx = value.find("{")
        last_brace_idx = value.rfind("}")
        if first_brace_idx != -1 and last_brace_idx > first_brace_idx:
            try:
                parsed = json.loads(value[first_brace_idx:last_brace_idx + 1])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed.get("vote")

        match = re.search(r'"vote"\s*:\s*"([^"]+)"', value)
        if match:
            return match.group(1)
        return value

    def _normalize_vote(self, vote):
        if vote is None:
            return None
        normalized = str(vote).strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _votes(self, decisions):
        return [
            self._normalize_vote(self._extract_vote(decision))
            for decision in decisions.values()
        ]

    def _majority_vote(self, votes):
        votes = [vote for vote in votes if vote]
        if not votes:
            return None

        counts = Counter(votes)
        top_count = max(counts.values())
        top_votes = [vote for vote, count in counts.items() if count == top_count]
        if len(top_votes) != 1:
            return None
        return top_votes[0]

    def _consensus_vote(self, votes, consensus_threshold):
        total_votes = len(votes)
        votes = [vote for vote in votes if vote]
        if not votes or total_votes == 0:
            return None

        vote, count = Counter(votes).most_common(1)[0]
        if count / total_votes >= consensus_threshold:
            return vote
        return None

    def detect_discussion_consensus(self, messages, options):
        """Match HiddenBench's round-level unanimous preference detection.

        Discussion turns are intentionally free text.  A preference is accepted
        when an official preference phrase names an option, or when exactly one
        option is mentioned.  Every agent must yield the same non-empty option.
        """
        preferences = {}
        for message in messages:
            content = str(message.content).lower()
            detected_option = None
            for option in options:
                normalized_option = str(option).lower()
                patterns = (
                    f"vote for {normalized_option}",
                    f"choose {normalized_option}",
                    f"support {normalized_option}",
                    f"agree on {normalized_option}",
                    f"go with {normalized_option}",
                    f"prefer {normalized_option}",
                    f"decision is {normalized_option}",
                    f"answer is {normalized_option}",
                    f"should be {normalized_option}",
                    f"it's {normalized_option}",
                    f"it is {normalized_option}",
                )
                if any(pattern in content for pattern in patterns):
                    detected_option = option
                    break

            if detected_option is None:
                mentioned = [
                    option
                    for option in options
                    if str(option).lower() in content
                ]
                if len(mentioned) == 1:
                    detected_option = mentioned[0]

            preferences[message.speaker] = detected_option

        if not preferences or any(value is None for value in preferences.values()):
            return None
        normalized = {
            self._normalize_vote(value)
            for value in preferences.values()
        }
        if len(normalized) != 1:
            return None
        return next(iter(preferences.values()))

    def _decision_metrics(self, decisions, answer, consensus_threshold):
        answer = self._normalize_vote(answer)
        votes = self._votes(decisions)
        majority_vote = self._majority_vote(votes)
        consensus_vote = self._consensus_vote(votes, consensus_threshold)
        return {
            "acc": self._mean(vote == answer for vote in votes),
            "majority_acc": int(majority_vote == answer),
            "consensus_acc": int(consensus_vote == answer),
            "consensus_rate": int(consensus_vote is not None),
        }

    def _decisions_from_entries(self, entries):
        return {
            entry["agent_id"]: entry.get("decision", entry.get("content"))
            for entry in entries
        }

    def _latest_decisions(self, history):
        latest_entries = {}
        for entry in history:
            latest_entries[entry["agent_id"]] = entry
        return self._decisions_from_entries(latest_entries.values())

    def _round_indices(self, history):
        return sorted({
            entry["round"]
            for entry in history
            if entry.get("round") is not None
        })

    def _first_consensus(self, history, answer, consensus_threshold):
        answer = self._normalize_vote(answer)
        for round_idx in self._round_indices(history):
            if round_idx < 1:
                continue
            round_entries = [
                entry
                for entry in history
                if entry.get("round") == round_idx
            ]
            votes = self._votes(self._decisions_from_entries(round_entries))
            consensus_vote = self._consensus_vote(votes, consensus_threshold)
            if consensus_vote is not None:
                return {
                    "round": round_idx,
                    "correct": int(consensus_vote == answer),
                }
        return {
            "round": None,
            "correct": 0,
        }

    def compute_metrics(self, task, history, consensus_threshold=1.0):
        history = list(history)
        answer = task["correct_answer"]
        final_decisions = self._latest_decisions(history)
        final_metrics = self._decision_metrics(
            final_decisions,
            answer,
            consensus_threshold,
        )
        first_consensus = self._first_consensus(
            history,
            answer,
            consensus_threshold,
        )

        by_round = {}
        for round_idx in self._round_indices(history):
            round_entries = [
                entry
                for entry in history
                if entry.get("round") == round_idx
            ]
            round_decisions = self._decisions_from_entries(round_entries)
            by_round[str(round_idx)] = {
                **self._decision_metrics(round_decisions, answer, consensus_threshold),
                **self._sum_usage(round_entries),
            }

        return {
            "task_id": task.get("id"),
            "answer": answer,
            "normalized_answer": self._normalize_vote(answer),
            "overall": {
                **final_metrics,
                "first_consensus_round": first_consensus["round"],
                "first_consensus_acc": first_consensus["correct"],
                **self._sum_usage(history),
            },
            "by_round": by_round,
        }
