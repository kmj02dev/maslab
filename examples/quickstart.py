"""Run MASLab composition examples without API keys or model downloads."""

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
    model = DemoBackend()
    agent = ml.Agent(
        "reader",
        model,
        system_prompt="Identify the fruit. It is yellow.",
    )
    response = agent.query(use_context=False)
    detail = agent.history()
    print("Single-agent answer:", response.content)
    print("Single-agent usage:", detail[-1]["usage"])

    agents = [
        ml.Agent("first", model, "The fruit is yellow."),
        ml.Agent("second", model, "The fruit is curved."),
    ]
    multiagent = ml.SequentialMultiagent(agents, loop=5)
    response = multiagent.query("go debate!", use_context=False)
    detail = multiagent.history()
    print("Multiagent answer:", response.content)
    print("Multiagent steps:", len(detail[-1]["steps"]))
    print("Multiagent usage:", detail[-1]["usage"])

    parallel = ml.ParallelMultiagent([
        ml.Agent("yellow", model, "The fruit is yellow."),
        ml.Agent("curved", model, "The fruit is curved."),
        ml.Agent("unknown", model, "Identify the fruit."),
    ])
    responses = parallel.query("Which fruit is supported by your facts?", use_context=False)
    print("Parallel answers:", [response.content for response in responses])
    print("Majority vote:", ml.MajorityVote()(responses))
    aggregator = ml.LLMAggregate(model)
    print("LLM aggregator (demo backend):", aggregator(responses))
    print("Parallel usage:", parallel.history()[-1]["usage"])
    print("Aggregation usage:", aggregator.history()[-1]["usage"])

    mesh = ml.MeshMultiagent([
        ml.Agent("first", model, "The fruit is yellow."),
        ml.Agent("second", model, "The fruit is curved."),
    ], loop=2)
    responses = mesh.query("Identify the fruit.", use_context=False, update_context=False)
    print("Mesh answers:", [response.content for response in responses])
    print("Mesh steps:", len(mesh.history()[-1]["steps"]))

    pipeline = ml.Pipeline([
        ml.Agent("writer", model, "The fruit is yellow."),
        ml.Suffix("\n\nReview this answer."),
        ml.Agent("reviewer", model),
    ])
    print("Pipeline answer:", pipeline.query("Identify the fruit.", use_context=False).content)
    print(ml.dialog(pipeline.history()))
    print("MASLab version:", ml.__version__)


if __name__ == "__main__":
    main()
