"""Run a current MASLab experiment without API keys or model downloads."""

from copy import deepcopy
import json

import maslab as ml


class DemoBackend(ml.Model):
    """Deterministic backend used only to make this example self-contained."""

    def __init__(self):
        super().__init__("demo-backend")

    def respond(self, messages):
        prompt = deepcopy(list(messages))
        visible_text = " ".join(message["content"] for message in prompt).lower()
        vote = (
            "banana"
            if "yellow" in visible_text or "curved" in visible_text
            else "apple"
        )
        is_decision = "return only one json object" in prompt[-1]["content"].lower()
        content = (
            json.dumps({"vote": vote})
            if is_decision
            else f"I support {vote} based on the facts visible to me."
        )
        return ml.Response(
            prompt=prompt,
            content=content,
            input_tokens=len(visible_text.split()),
            output_tokens=len(content.split()),
        )


def main() -> None:
    task = {
        "id": "fruit-1",
        "description": "Choose the fruit supported by the information.",
        "shared_information": ["The object is a fruit."],
        "hidden_information": [
            "The object is yellow.",
            "The object is curved.",
        ],
        "possible_answers": ["banana", "apple"],
        "correct_answer": "banana",
    }

    experiment = ml.Experiment(
        model=DemoBackend(),
        benchmark=ml.HiddenBench([task]),
        config=ml.ExperimentConfig(
            session=ml.SessionConfig(
                num_rounds=2,
                debate_topology="fully_connected",
            ),
            random_state=0,
        ),
    )

    result = experiment.run()
    task_result = result.results[0]

    print("MASLab version:", ml.__version__)
    print("Aggregate metrics:", result.metrics)
    print("Task metrics:", task_result.metrics["overall"])
    print("Decision rounds:", list(task_result.decisions))
    print("Decisions by round:", task_result.decisions)

    # Persist only when needed:
    # result.save_json("outputs/quickstart.json")


if __name__ == "__main__":
    main()
