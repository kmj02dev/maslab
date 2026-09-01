# MASLab

MASLab is a composable Python library for multi-agent decision-making
experiments. It provides model backends, stateful agents, debate sessions,
benchmark evaluation, checkpointing, and typed in-memory results.

The public API is designed around one high-level operation:

```python
result = experiment.run()
```

Advanced users can compose `Agent` and `Session` directly or extend the
`Model` and `Benchmark` abstract base classes.

## Features

- In-memory-first experiment API
- Sequential, fully connected, and cumulative debate topologies
- Independent round-0 decisions and decisions after every debate round
- Local Hugging Face, NVIDIA Build API, and Gemini model backends
- Callable or Jinja-compatible prompt rendering
- HiddenBench benchmark implementation
- Typed run and task results with JSON serialization
- Versioned round-level checkpoints and session state restoration
- Provider-level model backend registry
- Explicit abstract base classes for model and benchmark extensions

## Installation

MASLab requires Python 3.10 or later.

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

The benchmark accepts tasks already present in memory. A `PromptSet` groups the
initial, debate, and decision behavior used throughout the experiment.

```python
import os

import maslab as ml


task = {
    "id": "example-1",
    "description": "Choose the supported answer.",
    "shared_information": ["The object is a fruit."],
    "hidden_information": [
        "It is yellow.",
        "It is curved.",
    ],
    "possible_answers": ["banana", "apple"],
    "correct_answer": "banana",
}


def initial_prompt(context: ml.PromptContext):
    information = "\n".join(context.variables["information"])
    options = ", ".join(context.options or ())
    return f"{information}\nChoose one of: {options}"


experiment = ml.Experiment(
    model=ml.GeminiAPIModel(
        name="gemini-3.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
    ),
    benchmark=ml.HiddenBench([task]),
    prompts=ml.HiddenBench.prompts(initial=initial_prompt),
    config=ml.ExperimentConfig(
        agent=ml.AgentConfig(
            use_context_in_debate=True,
            use_context_in_decision=False,
        ),
        session=ml.SessionConfig(
            num_rounds=2,
            debate_topology="fully_connected",
        ),
        random_state=0,
    ),
)

result = experiment.run()
print(result.metrics)
result.save_json("outputs/example.json")
```

MASLab does not automatically load `.env` files. Applications may use
`python-dotenv` themselves, while library callers can pass credentials through
their own secret-management system.

## Prompt lifecycle

Every renderer receives the same typed `PromptContext`, including its phase,
round, agent, visible messages, answer options, and benchmark-specific
variables. A complete prompt policy is expressed as:

```python
prompts = ml.PromptSet(
    initial=initial_prompt,
    debate=ml.generic_debate_prompt,
    decision=ml.JsonDecision({"vote": str}),
)
```

`JsonDecision` keeps the decision instruction and parser/schema validation
together. Parsed decisions are stored in task results and history entries.

HiddenBench provides a ready-to-use policy, so the initial prompt can also be
omitted:

```python
experiment = ml.Experiment(
    model=model,
    benchmark=ml.HiddenBench(tasks),
)
```

Older `prompt=`, `debate_prompt=`, `decision_prompt=`, and `decision_parser=`
arguments remain supported. Legacy `**kwargs` renderers and Jinja objects with
a `render()` method are adapted to `PromptContext` automatically.

## Model backends

Model ID and provider selection are independent:

```python
model = ml.create_model(
    "nvidia",
    "google/gemma-4-31b-it",
    api_key=os.environ["NVIDIA_API_KEY"],
)
```

The built-in backend names are `huggingface`, `nvidia`, and `gemini`. Custom
backends must inherit from `Model` and implement `respond()`:

```python
from copy import deepcopy

from maslab import Model, Response, register_model_backend


class MyBackend(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def respond(self, messages):
        return Response(
            prompt=deepcopy(messages),
            content="my response",
        )


register_model_backend("my-provider", MyBackend)
```

Check an implementation with:

```python
from maslab.testing import check_model_backend

check_model_backend(MyBackend("my-model"))
```

The check invokes the backend once. Mock network transport when checking a
remote provider.

## Benchmarks and data

Benchmarks use in-memory tasks by default:

```python
benchmark = ml.HiddenBench(tasks)
```

JSON is an optional loading adapter:

```python
benchmark = ml.HiddenBench.from_json(
    "hiddenbench.json",
    count=100,
    shuffle=True,
    seed=0,
)
```

A HiddenBench task must contain:

```json
{
  "id": "task-1",
  "description": "Task description",
  "shared_information": ["Shared fact"],
  "hidden_information": ["Agent 0 fact", "Agent 1 fact"],
  "possible_answers": ["Option A", "Option B"],
  "correct_answer": "Option A"
}
```

Custom benchmarks must inherit from `Benchmark` and implement `build_prompts()`
and `compute_metrics()`. The base class retains the common JSON loading, result
building, and metric aggregation behavior. Use
`maslab.testing.check_benchmark()` to validate the minimum task and prompt
contract of an implementation.

## Configuration

Configuration and runtime state are separate. Configuration objects are frozen
and serializable:

```text
ExperimentConfig
├── AgentConfig
└── SessionConfig
```

Create a modified experiment without mutating the original:

```python
extended = experiment.with_config(
    session__num_rounds=5,
    agent__use_context_in_debate=False,
    agent__use_context_in_decision=True,
    random_state=42,
)
```

`use_context_in_debate` controls whether a debate turn reads the agent's
previous debate conversation. Debate turns are still recorded, allowing
`use_context_in_decision=True` to read them independently. Decision calls read
context only when enabled and never append the decision prompt or response to
the agent conversation. The older `use_context` setting remains a
debate-only compatibility alias.

`GenerationConfig` can be passed to built-in model constructors:

```python
generation = ml.GenerationConfig(
    max_tokens=512,
    temperature=0,
)

model = ml.HuggingfaceModel(
    name="Qwen/Qwen2.5-3B-Instruct",
    generation_config=generation,
)
```

## Agent state semantics

`Agent.generate()` never changes context:

```python
response = agent.generate("One stateless request")
```

`Agent.chat()` records both the user message and model response by default:

```python
response = agent.chat("Continue this conversation")
```

Context policies are `append`, `replace`, `last`, and `last_conversation`.
System prompts are preserved by all policies.

## Session state and checkpoints

`Session.state` holds runtime state separately from `SessionConfig`. It can be
copied between equivalent sessions:

```python
state = session.state_dict()
restored_session.load_state_dict(state)
```

When `SessionConfig.checkpoint_path` is set, the same state is saved atomically
after completed rounds. Checkpoints include agent contexts, history, opinions,
consensus state, and topology-visible messages. Restoring a checkpoint resumes
at the next unfinished round, so completed debate and decision calls are not
repeated.

## Debate topologies

`num_rounds` is the number of debate rounds and also the last round index.
Round 0 is always an independent decision: agents decide without seeing one
another. Every debate round then ends with a new decision from every agent.
This decision timing is identical for all topologies; a topology controls only
how debate messages are delivered.

```text
num_rounds=0  -> decision rounds [0]
num_rounds=1  -> decision rounds [0, 1]
num_rounds=2  -> decision rounds [0, 1, 2]
```

For `N` agents and `R` debate rounds, a complete session makes
`N * (1 + 2R)` model calls: `N` initial decisions, then `N` debate calls and
`N` decision calls per round. If cumulative consensus stops a session at round
`K`, that round's decisions are recorded before stopping.

Initial decisions are broadcast to round 1 by default:

```python
session = ml.SessionConfig(
    initial_decision_visibility="broadcast",  # or "private"
)
```

- `sequential`: each agent receives the latest preceding response.
- `fully_connected`: each round reads a snapshot of the previous round.
- `cumulative`: agents share one append-only conversation.

Use the unified session entry point for low-level composition:

```python
history = session.run(
    decision_message_callback=decision_prompt,
    use_context_in_debate=True,
    use_context_in_decision=False,
)
```

## Results

`Experiment.run()` returns `RunResult`, containing typed `TaskResult` objects.
It has no file-system side effect until `save_json()` is called.

```python
result.metrics
result.results[0].metrics
result.results[0].history
result.to_dict()
result.save_json("result.json")
```

## Project structure

```text
maslab/
├── pyproject.toml
├── src/maslab/
│   ├── __init__.py       # Stable public imports
│   ├── base.py           # Agent, Session, Benchmark, checkpoints
│   ├── config.py         # Frozen configuration values
│   ├── experiment.py     # High-level Experiment API
│   ├── models/           # Built-in model adapters and registry
│   │   ├── __init__.py
│   │   ├── gemini_api.py
│   │   ├── huggingface.py
│   │   ├── nvidia_build_api.py
│   │   └── registry.py
│   ├── benchmarks/       # Built-in benchmarks
│   │   ├── __init__.py
│   │   └── hiddenbench.py
│   ├── prompts.py        # Prompt lifecycle and decision policies
│   ├── testing.py        # Model and benchmark validation helpers
│   ├── types.py          # Responses, state, and typed results
│   └── py.typed           # PEP 561 typing marker
└── tests/
```

## Compatibility

The original `Config` class, direct path construction such as
`HiddenBench("tasks.json")`, topology-specific session methods, and model-ID
presets remain available for 0.1 compatibility. New code should prefer
`ExperimentConfig`, `from_json()`, `Session.run()`, and provider-level
`create_model()`.

## Known limitations

- Model calls are synchronous; streaming and async backends are not yet part of
  the public contract.
- The backend registry is process-local.
- Checkpoint schema migration is not implemented beyond rejecting unsupported
  versions.
- `early_stop_on_consensus` currently applies to cumulative sessions with a
  consensus callback.

## License

MASLab is licensed under the MIT License. See [LICENSE](LICENSE).
