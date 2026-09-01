import json

from maslab import HiddenBench
from maslab.testing import check_benchmark

from conftest import hiddenbench_task


def prompt_renderer(**context):
    return " | ".join(context["information"])


def test_benchmark_has_its_own_module():
    assert HiddenBench.__module__ == "maslab.benchmarks.hiddenbench"


def test_benchmark_accepts_in_memory_tasks_and_callable_prompt():
    benchmark = HiddenBench([hiddenbench_task()])

    prompts = benchmark.build_prompts(benchmark.tasks[0], prompt_renderer)

    assert prompts == [
        "Shared fact | Private fact 1",
        "Shared fact | Private fact 2",
    ]
    assert check_benchmark(benchmark)


def test_benchmark_from_json_is_an_optional_loader(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([hiddenbench_task()]), encoding="utf-8")

    benchmark = HiddenBench.from_json(path)

    assert benchmark.tasks[0]["id"] == "task-1"


def test_hiddenbench_owns_its_prompt_set():
    prompts = HiddenBench.prompts()

    assert prompts.decision.parse('{"vote": "A"}') == {"vote": "A"}
