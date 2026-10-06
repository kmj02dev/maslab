# MASLab

MASLab is a composable Python library for multi-agent conversations and
decision-making experiments. It provides model backends, stateful agents,
sequential, cumulative, parallel, and mesh multiagents, response-processing pipelines, and aggregates.

Single agents, multiagents, and pipelines share the same conversation API:

```python
response = agent.query(use_context=False)
detail = agent.history()

multiagent = SequentialMultiagent(agents, loop=5)
response = multiagent.query("go debate!", use_context=False)
detail = multiagent.history()
```

`query()` returns a `Response` for single agents, sequential and cumulative groups; pipelines return
a `Response` or a final response list depending on their last step. Read
`response.content` for the answer text and the response's fields for its prompt,
reasoning, token counts, generation time, and originating agent ID (`response.agent_id`).
Parallel and mesh groups return an ordered
`list[Response]`, which an `Aggregate` can reduce to one `Response`.
`history()` returns detailed snapshots of previous calls. Extend `Model`,
`Multiagent`, `Transform`, and `Aggregate` to implement custom behavior.

## Features

- Shared `query()` / `history()` API for agents and nested multiagents
- Sequential message handoff with an explicit number of complete passes
- Cumulative handoff of all preceding discussion outputs, including the current round
- Pipelines with explicit `Transform` steps for application-defined response processing
- Independent parallel agent calls with ordered results and partial failure traces
- Numbered response concatenation and model-backed response synthesis
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
)

response = agent.query(use_context=False)
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
    ml.Agent("researcher", model, "Find the relevant facts."),
    ml.Agent("reviewer", model, "Review the previous answer."),
]
multiagent = ml.SequentialMultiagent(agents, loop=5)

response = multiagent.query("go debate!", use_context=False)
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
query steps. The final response's `agent_id` is preserved through sequential,
cumulative, and pipeline composition, including nested groups and wrappers.
Group history entries still use the group's own ID; nested steps identify the
individual participants. Each participant's full response is recorded in the
group's history.
The `agents` argument and attribute contain query participants only; use
`Pipeline(steps)` to insert response transformations.

Participants retain their own conversations between calls.
`multiagent.query(message, use_context=False)` excludes prior conversation turns
for that call on every participant. `update_context=False` prevents
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

`dialog(history)` returns a string with `[loop i | agent_id]` labels and the original
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

## Cumulative discussions

`CumulativeMultiagent` queries participants sequentially, passing every preceding
output instead of only the immediately preceding response:

```python
from maslab import CumulativeMultiagent, dialog

debate = CumulativeMultiagent(agents, loop=4)
response = debate.query(
    "lets start debate", use_context=False, update_context=False,
)
print(response.content)
print(dialog(debate.history()))
```

The first participant receives the original query message. Subsequent inputs
are JSON strings containing only a `responses` list; the original question
is not reinserted. Each response entry has `round` (one-based), `agent_id`, and the unchanged response `content`.
For four agents, the second agent in round two sees all four round-one outputs
plus the first output from round two. A participant's own earlier outputs are
included; current and future outputs are not. Each group `query()` starts a new
response list, while `history()` retains previous runs.

As with other groups, both context flags default to `True` and propagate to
participants. The example disables personal context reads and updates to avoid
duplicating the explicit response history. System prompts and execution history are
preserved. The returned `Response` uses the last participant's output and sums
usage across the run. Single-response groups and pipelines can be participants;
their final output enters the response history and their inner steps remain in history.
There are no built-in initial decisions, votes, or stopping rules.

## Pipelines and response transformations

### Broadcast transforms

`Broadcast(count)` and `WrapBroadcast(prefix=None, suffix=None)` transform one
`Response` into a non-empty `list[Response]` without calling a model.
`Broadcast` requires a positive integer count (not bool). `WrapBroadcast` accepts
`str | list[str] | None` on each side: strings are shared, None is empty, and list
length determines the output count. Two lists must have equal lengths; empty
lists and non-string elements are rejected. Without lists it produces one output.
Whitespace is preserved exactly. Each output has an independent copy of the
source prompt and new content; usage, reasoning, and agent identity are not copied.

```python
from maslab import Agent, ParallelMultiagent, Pipeline, ConcatAggregate, WrapBroadcast

tasks = ["Task with private fact A", "Task with private fact B"]
agents = [Agent(str(i), model) for i in range(len(tasks))]
pipeline = Pipeline([
    ParallelMultiagent(agents),
    ConcatAggregate(),
    WrapBroadcast(
        prefix=[task + "\n\nSolutions:\n" for task in tasks],
        suffix="\n\nReview the solutions and give an updated answer.",
    ),
], loop=2)
review_prompts = pipeline.query(tasks, use_context=False, update_context=False)
```

Transforms declare `returns_multiple=True` for list output. Pipeline passes
response lists unchanged, including across loop boundaries and nested pipelines.
ParallelMultiagent distributes response contents by index to ordinary participants,
with matching list and participant counts required. When all branches accept
response collections, whole-list delivery is preserved for compatibility.
Aggregates receive the full response list. Broadcast output cannot feed a single
Agent directly.
Transforms still cannot be the first step of a pipeline.

The final result above is the next review's prompt list, not the last round's
model answers. Final broadcast outputs remain `list[Response]`; response distribution belongs to ParallelMultiagent rather than Pipeline. Model usage remains in the
pipeline history without being duplicated in generated prompts.

`ParallelMultiagent.query()` accepts either a shared string or a non-empty
`list[str]` with exactly one prompt per participant, in participant order.
`Pipeline.query()` forwards this list when its first step supports
`accepts_prompts` (including nested pipelines starting with a parallel group).
Empty or mixed lists and unsupported first steps raise `TypeError`; prompt
count mismatches raise `ValueError` before any branch runs.

```python
parallel = ParallelMultiagent([Agent("a", model), Agent("b", model)])
pipeline = Pipeline([parallel, ConcatAggregate()])
result = pipeline.query(["Question with private fact A", "Question with private fact B"])
```

This does not change `list[Response]` input or its `accepts_multiple` capability.
String lists are used for the initial query only; subsequent steps and loop
iterations consume the preceding output normally. To preserve each participant's
original information during review with context disabled, place that information
in its `Agent.system_prompt`. Group history stores the input list, while branch
history records the individual prompt, including on failure.

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
Move sequences containing transforms or aggregates to `Pipeline`. Queries receive
preceding response content; transforms receive one whole response; aggregates receive
an ordered response list. A leading Aggregate accepts a list through `query(responses)`.
A Mesh/Parallel output can pass intact to an Aggregate or collection-aware nested group.
Lists are never implicitly converted to strings or mapped through single-response transforms.
Every loop executes all steps; handoffs, including loop boundaries, must match these types.
The final step determines whether the pipeline returns one Response or a list.

Use the optional `transforms` argument for Aggregate/Transform stages that prepare
the next loop's input. They run only between iterations, exactly `loop - 1` times,
and are skipped entirely with `loop=1`. The final iteration returns the last
`steps` output, retaining the agents' responses instead of returning review prompts:

```python
pipeline = ml.Pipeline(
    steps=[ml.ParallelMultiagent(agents)],
    loop=4,
    transforms=[
        ml.CumulativeConcatAggregate(),
        ml.WrapBroadcast(prefix=review_prefixes, suffix=review_suffixes),
    ],
)
responses = pipeline.query(initial_prompts)
```

Between-loop stages use the preceding loop number in history and continue its
step numbering. `dialog()` inherits a parent's loop number for entries that lack
one; explicit nested loop numbers still take precedence. Stateful aggregates
retain their normal lifetime across queries; create a new instance or reset the
aggregate for an independent task.

`Transform.transform(Response) -> Response` processes a response without calling
a model. Extraction, normalization, and formatting policies belong in application
implementations of this interface. Use an agent step for model-based processing.
The executor supplies a deep copy and snapshots the returned response, so a
transform can edit its supplied copy without changing the originating agent's
context or history. Outputs must be `Response` instances with string `content`.

Pipeline history records distinguish `kind="query"`, `kind="transform"`, and `kind="aggregate"`.
Query records contain the usual `agent_id`, `message`, response details and `usage`.
Transform records contain `name`, `input`, `output`, and `status`, alongside the
one-based `loop` and `step` indices. Input/output snapshots contain `agent_id`, `prompt`,
`content`, `reasoning`, and nested `usage`; these metrics describe the response
being processed and are not additional model costs. Group usage sums query records and the Aggregate
step's own inference usage, including nested groups once. Text-only aggregates have
zero inference usage; input response costs are not charged again. A list result retains
its final individual response metrics; group history contains the complete execution cost. Editing response metrics in a transform
does not change the recorded model cost.

If a transform raises or returns an invalid result, its record has
`status="failed"`, `output=None`, and an `error` with the exception type and
message. Execution stops and the group preserves previous successful responses,
transforms, and model usage in its failed history entry.

`Pipeline` implements the `Multiagent` contract independently of
`SequentialMultiagent`. Single-output pipelines can participate in sequential or
parallel groups; collection-output pipelines require an explicit Aggregate before
handoff to a single-message participant.

### Reviewing formatted Mesh responses

`ConcatAggregate()` combines all response contents in input order under
`[agent 1]`, `[agent 2]`, and subsequent numbered headings. It preserves every
response, including the recipient's own answer and duplicate contents.
Use a `Suffix` step to add the original question and review instructions.

```python
from maslab import Agent, ConcatAggregate, MeshMultiagent, Pipeline, Suffix, dialog

question = "What is 12 + 15?"
agents = [Agent(f"agent_{i + 1}", model) for i in range(3)]  # your model backend
review = MeshMultiagent([
    Pipeline([
        ConcatAggregate(),
        Suffix(f"\n\nReview the responses and revise your answer. Original question: {question}"),
        agent,
    ], id=f"review_{i + 1}")
    for i, agent in enumerate(agents)
], loop=1)

debate = Pipeline([MeshMultiagent(agents, loop=1), review])
responses = debate.query(question, use_context=False, update_context=False)
print([response.content for response in responses])
print(dialog(debate.history()))
```

This makes six model calls: three independent answers followed by three reviews.
For three or more total rounds, set `review.loop` to `rounds - 1`. Review branches
receive the same completed previous-round list, each as a separate copy. No
same-round sibling output is read. The example disables personal conversation
context because previous answers are already present in the formatted input.
Default string-input Mesh behavior is unchanged. Use `max_workers=1` on both
Mesh groups for backends that require serialized inference.

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

response = ml.ConcatAggregate()(responses)
# Equivalent: ml.ConcatAggregate().aggregate(responses)
print([response.content for response in responses])
print(response.content)
print(detail[-1]["usage"])
```

`ConcatAggregate` creates a single `Response` without model inference:

```text
[agent 1]
First response content

[agent 2]
Second response content
```

Numbers start at one and follow iteration order; they are positional labels, not
agent IDs from metadata. Contents, whitespace, duplicates, and empty response
contents are preserved. Sections are joined with two newline characters.
The new response has an empty `prompt`, `agent_id=None`, and no reasoning or generation metrics;
source prompts, reasoning, and costs remain with the original responses.

`LLMAggregate` instead makes one model call to synthesize the candidates:

```python
aggregator = ml.LLMAggregate(
    model,
    system_prompt="Evaluate these proposed answers to question X and return the best final answer.",
)
response = aggregator(responses)
detail = aggregator.history()
print(response.content)
print(detail[-1]["usage"])
```

The `Aggregate` contract is `Iterable[Response] -> Response`. Both built-in
aggregators accept non-empty lists, tuples, iterators, and generators of `Response`
objects, consuming them once in iteration order. Strings and mixed string/Response
inputs are rejected. An empty iterable raises `ValueError`; invalid items or
non-string `content` raise `TypeError` before any model call.

```python
response = ml.ConcatAggregate()(item for item in responses)
```

Implement `aggregate(responses)` to define a custom reduction. The protected
`_contents(responses)` helper validates this contract and collects contents in
one pass. `aggregator(responses)` delegates to `aggregator.aggregate(responses)`.
Pipeline's `query()` collection input remains a non-empty `list[Response]`.

`LLMAggregate.aggregate(responses)` returns
the raw synthesis response, including its model usage. Its calls exclude previous aggregation context while
retaining an independent execution history. Its usage measures only the synthesis
call; the parallel group's history records participant usage separately.

Parallel participants may be single agents, sequential groups, or pipelines,
provided that each branch returns one answer. Reduce a parallel collection
before handing it to another single-answer participant. For example:

```python
response = editor.query(ml.ConcatAggregate()(responses).content)
print(response.content)
```

The `use_context` and `update_context` call arguments both default to `True`
and propagate to every branch. Branches must use
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

`Agent.generate()` returns a `Response` with `agent_id` set to the calling
agent's `id`, without changing context or query history. `Agent.query()` uses
the same assignment. A new response is returned so a shared backend's response
object is not retagged in place. `Response.agent_id` is optional and defaults
to `None`, preserving existing constructors and direct model calls. Wrappers
preserve it; `LLMAggregate` returns its internal judge agent's ID.

```python
response = agent.query("Question")
print(response.agent_id)  # agent.id
```

For generation without recording a query:

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
`generate()` and `query()` accept `use_context=True` as a call argument.
`Agent` does not accept or store `use_context` in its constructor; `None` is not
a supported value. `query()` also accepts `update_context=True`: reading past
turns and saving new turns are independent. To disable both, pass
`use_context=False, update_context=False`. Execution history is still recorded.

`chat()` and `chat_response()` have been replaced by `query()`. Read
`query(...).content` when answer text is needed. For parallel groups, use
`[response.content for response in responses]`.
`SequentialMultiagent.query()` and single-output `Pipeline.query()` return one response with
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
rounds, each participant receives a JSON message containing only `responses`:
all previous-round responses (including its own), without the original question, in
participant order, identified by `agent_id`. The context arguments supplied to
`mesh.query()` apply to every participant.

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


## CumulativeConcatAggregate

`from maslab import CumulativeConcatAggregate` provides stateful concatenation.
Each successful aggregate call appends one deep-copied round and returns all
contents under `[round N | agent_id]` labels. Missing IDs use positional
`agent 1`, `agent 2` labels. Duplicates, empty contents and whitespace are retained.

`history()` returns independent `list[list[Response]]` snapshots, including
source metadata; it is not the query-history schema consumed by `dialog()`.
Use `dialog(pipeline.history())` for conversation display. `reset()` clears
all rounds. Invalid input or failing generators leave state unchanged.

Replace a ConcatAggregate step with this class to accumulate across iterations.
State also persists across query calls regardless of context flags: create an
instance per task or reset explicitly. Do not share it across concurrent runs.
Only responses passed to aggregate are stored; final parallel outputs after
the last aggregate are not captured automatically. There is no truncation.

Returned Response has an empty prompt and default metadata, so source model
costs are not charged again. Round numbers count aggregate calls.

### Single-response accumulation

CumulativeConcatAggregate accepts either one Response or a non-empty iterable
of Responses, through both aggregate() and the callable interface. Each call
adds one batch. It declares accepts_single=True; other Aggregate subclasses
default to False and keep their collection-only input contract.

```python
memory = CumulativeConcatAggregate()
pipeline = Pipeline([agent_a, memory, agent_b, memory], loop=2)
result = pipeline.query("Start", use_context=False, update_context=False)
```

The same memory instance retains outputs from all four agent calls. Pipeline
normalizes a single aggregate input to a one-element list for execution and
history snapshots; it does not clone the aggregate instance. No source costs
are charged twice. A leading Aggregate still requires list[Response] input to
Pipeline.query(); raw strings and standalone Response query inputs are unsupported.
