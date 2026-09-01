import maslab


def test_public_api_exposes_framework_entry_points():
    assert maslab.__version__ == "0.1.0"
    assert maslab.Experiment
    assert maslab.Model
    assert maslab.Benchmark
    assert maslab.PromptContext
    assert maslab.PromptPhase
    assert maslab.RunResult
    assert maslab.create_model
    removed_names = (
        "Model" + "Backend",
        "Benchmark" + "Protocol",
        "Prompt" + "Renderer",
    )
    assert not any(hasattr(maslab, name) for name in removed_names)
