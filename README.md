# MASLab

MASLab is a composable Python library for multi-agent conversations and
decision-making experiments. It provides model backends, stateful agents,
sequential, parallel, and mesh multiagents, response-processing pipelines, and aggregates.

Single agents, multiagents, and pipelines share the same conversation API:

```python
response = agent.query()
detail = agent.history()

multiagent = SequentialMultiagent(agents, loop=5)
response = multiagent.query("go debate!")
detail = multiagent.history()
```

`query()` returns a `Response` for single agents, sequential groups, and pipelines. Read
`response.content` for the answer text and the response's fields for its prompt,
reasoning, token counts, and generation time. Parallel and mesh groups return an ordered
`list[Response]`, which an `Aggregate` can reduce to one answer text.
`history()` returns detailed snapshots of previous calls. Extend `Model`,
`Multiagent`, `Transform`, and `Aggregate` to implement custom behavior.

## Features

- Shared `query()` / `history()` API for agents and nested multiagents
- Sequential message handoff with an explicit number of complete passes
- Pipelines with explicit `Transform` steps for application-defined response processing
- Independent parallel agent calls with ordered results and partial failure traces
- Exact-text majority voting and model-backed response synthesis
- Local Hugging Face, NVIDIA Build API, and Gemini model backends
- Explicit abstract base classes for models and agent composition

## Package boundaries

MASLab includes `maslab.core`, optional `maslab.models`, `maslab.utils`, and
model contract helpers in `maslab.testing`. Experiments, benchmarks, and job
scripts are maintained separately in `D:/workspace/maslab-experiments`.
They are not distributed or re-exported by MASLab. The external project imports
MASLab; MASLab does not depend on the external project.

## Installation

MASLab requires Python 3.10 or later. The base installation has no third-party
runtime dependencies.

```bash
python3 -m pip install -e .
```

On externally managed Linux distributions, use a project virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Run the self-contained example without an API key or model download:

```bash
.venv/bin/python examples/quickstart.py
```

Install Gemini and NVIDIA HTTP API support with:

```bash
python3 -m pip install -e ".[api]"
```

Install Hugging Face support with:

```bash
python3 -m pip install -e ".[hf]"
```

The `hf` extra installs a general PyTorch dependency. For GPU workloads, install
a PyTorch build compatible with the local driver and CUDA environment. The
repository's `requirements.txt` remains a CUDA 12.8 environment specification;
it is not MASLab's portable runtime dependency set.

For development:

```bash
python3 -m pip install -e ".[test]"
python3 -m pytest
```

## Quick start

With an existing model backend, create an agent and inspect its answers:

```python
import maslab as ml

agent = ml.Agent(
    "researcher",
    model,
    system_prompt="Explain how information sharing improves group decisions.",
    use_context=False,
)

response = agent.query()
detail = agent.history()
print(response.content)
print(detail[-1]["prompt"])
print(detail[-1]["usage"])
```

Calling `query()` without a message sends `"Continue."` along with the agent's
system instructions. Pass a string to provide an explicit request instead.
`use_context` defaults to `True`; setting it to `False` excludes previous
conversation turns from model input while preserving the system instructions.
Query history is still recorded.

Compose multiple agents using the same interface:

```python
agents = [
    ml.Agent("researcher", model, "Find the relevant facts.", use_context=False),
    ml.Agent("reviewer", model, "Review the previous answer.", use_context=False),
]
multiagent = ml.SequentialMultiagent(agents, loop=5)

response = multiagent.query("go debate!")
detail = multiagent.history()
print(response.content)
print(detail[-1]["steps"])
```

The first agent receives the supplied message. Each later participant receives
the preceding response's content directly. `SequentialMultiagent(agents, loop=5)`
runs the entire agent list five times: two agents make ten query calls, with each
nested group performing its own calls. The final response of each pass becomes
the next pass's input. `loop` must be a positive integer.
The returned `Response` contains the final participant's `content` and `reasoning`,
the group input in `prompt`, and token counts and generation time summed across
query steps. Each participant's full response is recorded in the group's history.
The `agents` argument and attribute contain query participants only; use
`Pipeline(steps)` to insert response transformations.

Participants retain their own context settings and conversations between calls.
An explicit `multiagent.query(message, use_context=False)` overrides the context
setting for that call on every participant. `update_context=False` prevents
conversation updates while still recording execution history.

Multiagents can also be participants in another multiagent:

```python
team = ml.SequentialMultiagent(agents, loop=2, id="research-team")
editor = ml.Agent("editor", model, "Write a concise final answer.")
group = ml.SequentialMultiagent([team, editor])
response = group.query("Explain the tradeoffs of group decision-making.")
print(response.content)
```

Each `history()` entry describes one `query()` call and includes `agent_id`,
`message`, `prompt`, `content`, `reasoning`, `usage`, and `status`.
Token counts (`input_tokens`, `output_tokens`, `total_tokens`) and
`generation_time` are stored only inside `usage`, including in nested steps. Multiagent entries additionally
contain `steps` with one-based `step` indices; round-based groups also record `loop`.
Nested groups retain their steps inside the corresponding entry. Group usage
includes all child calls; each child's `prompt` records its actual model input.
Repeated calls append entries, and editing a returned history never changes
the original conversation or history. If a group call raises an exception, its
history retains completed steps with `status="failed"`; the exception is raised
to the caller and later participants are not run.

To inspect the final answer and full history, including nested steps:

```python
import json

print(response.content)
print(json.dumps(multiagent.history(), ensure_ascii=False, indent=2))
```

To display only agent outputs as a conversation:

```python
from maslab import dialog

print(dialog(agent.history()))
print(dialog(multiagent.history()))
print(dialog(pipeline.history()[-1:]))  # Latest run only
```

`dialog(history)` returns a string with `[agent_id]` labels and the original
response content. It also accepts histories loaded from JSON. Nested groups are
expanded without repeating group summaries; transforms, inputs, reasoning, and
usage are omitted. Completed steps remain visible if a later step fails.
Sequential calls and loops follow recorded order. Parallel branches use
participant order, not completion time. The function leaves history unchanged
and returns an empty string when there are no completed agent outputs.

For per-agent inputs and outputs from the latest sequential run:

```python
for step in multiagent.history()[-1]["steps"]:
    print(f"[Loop {step['loop']} / Step {step['step']}] {step['kind']}")
    print("Agent:", step["agent_id"])
    print("Input:", step["message"])
    print("Output:", step["content"])
    print("Usage:", step["usage"])
```

## Pipelines and response transformations

`Pipeline` composes queries and explicit response transformations.
`SequentialMultiagent` performs only sequential queries. Both support `query()`,
`history()`, and `loop`, and can be nested inside each other.

Place `Transform` implementations in a pipeline's ordered `steps` list:

```python
from dataclasses import replace
import json
from maslab import Pipeline, Response, Transform

class ExtractAnswer(Transform):
    def transform(self, response: Response) -> Response:
        return replace(response, content=json.loads(response.content)["answer"])

class FormatOutput(Transform):
    def transform(self, response: Response) -> Response:
        return replace(response, content=response.content.strip())

# Configure the researcher to return JSON with an "answer" field.
pipeline = Pipeline(
    steps=[researcher, ExtractAnswer(), reviewer, FormatOutput()],
    loop=5,
)
response = pipeline.query("Compare the evidence.")
print(response.content)
```

`Pipeline(steps, loop=...)` stores its participants in `steps`;
`SequentialMultiagent(agents, loop=...)` stores its participants in `agents`.
Move sequences containing transforms to `Pipeline`. The first pipeline step
must be an `Agent` or a `Multiagent` that returns one `Response`. Later steps may
be queries or transforms, including consecutive transforms and a final transform.
Every loop executes all steps. Query steps receive the preceding response's
content; transforms receive the whole response. The final processed content
becomes the next loop's input and, after the last loop, the pipeline's final answer.

`Transform.transform(Response) -> Response` processes a response without calling
a model. Extraction, normalization, and formatting policies belong in application
implementations of this interface. Use an agent step for model-based processing.
The executor supplies a deep copy and snapshots the returned response, so a
transform can edit its supplied copy without changing the originating agent's
context or history. Outputs must be `Response` instances with string `content`.

Pipeline history records distinguish `kind="query"` and `kind="transform"`.
Query records contain the usual `agent_id`, `message`, response details and `usage`.
Transform records contain `name`, `input`, `output`, and `status`, alongside the
one-based `loop` and `step` indices. Input/output snapshots contain `prompt`,
`content`, `reasoning`, and nested `usage`; these metrics describe the response
being processed and are not additional model costs. Group usage sums only query
records, including nested groups once. Editing response metrics in a transform
does not change the recorded model cost.

If a transform raises or returns an invalid result, its record has
`status="failed"`, `output=None`, and an `error` with the exception type and
message. Execution stops and the group preserves previous successful responses,
transforms, and model usage in its failed history entry.

`Pipeline` implements the `Multiagent[Response]` contract, independently of
`SequentialMultiagent`. A pipeline can contain sequential groups or other
pipelines, and can participate in sequential or parallel groups. Parallel
collections must be reduced with an `Aggregate` before entering a pipeline
as a single answer.

Parallel branches must have distinct agent, group, and transform instances;
sharing the same transform within one pipeline branch is allowed. This also
protects transforms that maintain their own state. Transforms cannot be direct
parallel participants because they process an existing response rather than a
query string.

To inspect transformations from the latest pipeline run:

```python
for step in pipeline.history()[-1]["steps"]:
    if step["kind"] == "transform":
        print(step["name"], step["status"])
        print("Before:", step["input"])
        print("After:", step["output"])  # None when the transform failed
```

Run `python ../../maslab-experiments/examples/transforms.py` for a complete example with before/after
output, loop handoffs, and usage totals. No model download or API key is required.

## Parallel execution and aggregation

`ParallelMultiagent` sends the same message to every participant without sharing
responses between them. `query()` returns `list[Response]` in participant order,
regardless of completion order. Each response contains that participant's answer
and execution metadata. Each call runs every participant once.

```python
parallel = ml.ParallelMultiagent(agents, max_workers=3)
responses = parallel.query("What conclusion is supported by the evidence?", use_context=False)
detail = parallel.history()

answer = ml.MajorityVote()(responses)
# Equivalent: ml.MajorityVote().aggregate(responses)
print([response.content for response in responses])
print(answer)
print(detail[-1]["usage"])
```

`MajorityVote` selects the most frequent exact response text. Comparisons preserve
case and whitespace; ties select the text occurring first in the input list.
A strict majority is not required. Extract decision labels before voting if
responses contain explanations that should not distinguish votes.

`LLMAggregate` instead makes one model call to synthesize the candidates:

```python
aggregator = ml.LLMAggregate(
    model,
    system_prompt="Evaluate these proposed answers to question X and return the best final answer.",
)
answer = aggregator(responses)
detail = aggregator.history()
print(answer)
print(detail[-1]["usage"])
```

Both aggregators accept a non-empty sequence of strings, raw `Response` objects,
or a mixture. `Aggregate` is an abstract base: implement `aggregate(responses)`
to define another reduction. `LLMAggregate.aggregate_response(responses)` returns
the raw synthesis response. Its calls exclude previous aggregation context while
retaining an independent execution history. Its usage measures only the synthesis
call; the parallel group's history records participant usage separately.

Parallel participants may be single agents, sequential groups, or pipelines,
provided that each branch returns one answer. Reduce a parallel collection
before handing it to another single-answer participant. For example:

```python
response = editor.query(ml.MajorityVote()(responses))
print(response.content)
```

Each participant retains its own context setting. The optional `use_context` and
`update_context` call arguments propagate to every branch. Branches must use
distinct Agent/Multiagent/Transform instances, including inside nested groups and
pipelines, so that their context and history cannot be modified by a sibling. Shared model
backends must support concurrent `respond()` calls. Calls run in threads;
`max_workers` limits concurrent branches and defaults to the participant count.
Actual speedup depends on the model backend and available resources.

The parallel history's `content` is a list of answer texts. `steps` preserves one
entry per participant with a one-based `step` index and its actual model prompt.
Nested groups and pipelines retain their own `steps`. Usage is summed once over
branches; `generation_time` is the sum of reported model times, not wall time.
If any branch fails, all submitted branches finish, the group records
`status="failed"` with successful answers and per-branch errors, and the first
exception in participant order is raised. Failed branches without a response
have an empty `prompt` and unknown model usage represented by zero token counts
and `generation_time=None`; nested partial work retains its reported usage.

## Model backends

Model ID and provider selection are independent:

```python
model = ml.NvidiaBuildAPIModel(
    "google/gemma-4-31b-it",
    api_key=os.environ["NVIDIA_API_KEY"],
)
```

Instantiate `HuggingfaceModel`, `NvidiaBuildAPIModel`, or `GeminiAPIModel` directly. Custom
backends must inherit from `Model` and implement `respond()`:

```python
from copy import deepcopy

from maslab import Model, Response


class MyBackend(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def respond(self, messages):
        return Response(
            prompt=deepcopy(messages),
            content="my response",
        )


model = MyBackend("my-model")
```

Check an implementation with:

```python
from maslab.testing import check_model_backend

check_model_backend(MyBackend("my-model"))
```

The check invokes the backend once. Mock network transport when checking a
remote provider.

## Agent state semantics

`Agent.generate()` returns a raw `Response` without changing context or query
history:

```python
response = agent.generate("One stateless request")
```

`Agent.query()` returns a `Response` and records both the user message and model
response by default:

```python
response = agent.query("Continue this conversation")
print(response.content)
print(response.input_tokens, response.output_tokens, response.generation_time)
detail = agent.history()
```

Context policies are `append`, `replace`, `last`, and `last_conversation`.
System prompts are preserved by all policies. These policies affect the context
sent to future model calls; they do not discard execution history. Both
`generate()` and `query()` use the constructor's `use_context` setting unless a
per-call override is supplied.

`chat()` and `chat_response()` have been replaced by `query()`. Read
`query(...).content` when answer text is needed. For parallel groups, use
`[response.content for response in responses]`.
`SequentialMultiagent.query()` and `Pipeline.query()` return one response with
token counts and generation time summed across query participants.
`ParallelMultiagent.query()` returns an ordered
list of participant responses; group usage is available in `history()`.
Custom `Multiagent` implementations override `query()` and return `Response` or
`list[Response]` according to their generic type. Nested
groups use the same `query()` interface internally.

## Project structure

```text
maslab/
├── src/maslab/
│   ├── core/
│   ├── models/
│   ├── utils.py
│   └── testing.py
├── examples/quickstart.py
└── tests/
```

## Known limitations

- Model calls are synchronous; streaming and async backends are not yet part of
  the public contract.


## License

MASLab is licensed under the MIT License. See [LICENSE](LICENSE).


### MeshMultiagent

`MeshMultiagent` executes synchronous parallel rounds. `loop` includes the first
round, where every participant receives the original question. In subsequent
rounds, each participant receives a JSON message with `question` and `responses`:
the original question and all previous-round responses (including its own), in
participant order, identified by `agent_id`. Agent context settings still apply.

```python
from maslab import MeshMultiagent

mesh = MeshMultiagent(agents, loop=3, max_workers=3)
responses = mesh.query("Compare the available options.", update_context=False)
detail = mesh.history()
```

The return value contains only the final round's responses in participant order.
History includes every call with its `loop` and `step`, and usage totals all
rounds. A failed round preserves completed work and prevents the next round from
starting. Participants must be distinct instances; shared model backends must
support concurrent calls. Use `max_workers=1` for a backend requiring serial calls.



External experiment documentation: [maslab-experiments](../../maslab-experiments/README.md).
