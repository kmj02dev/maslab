import pytest

from maslab import (
    GeminiAPIModel,
    HuggingfaceModel,
    NvidiaBuildAPIModel,
    create_model,
    register_model_backend,
)
from maslab.testing import check_model_backend

from conftest import FixedModel


def test_each_model_has_its_own_module():
    assert GeminiAPIModel.__module__ == "maslab.models.gemini_api"
    assert HuggingfaceModel.__module__ == "maslab.models.huggingface"
    assert NvidiaBuildAPIModel.__module__ == "maslab.models.nvidia_build_api"


def test_model_backend_contract_check():
    assert check_model_backend(FixedModel())


def test_backend_registry_is_independent_of_model_id():
    backend_name = "test-fixed"
    register_model_backend(
        backend_name,
        lambda name, **kwargs: FixedModel(content=name),
        overwrite=True,
    )

    model = create_model(backend_name, "any/model-id")

    assert model.content == "any/model-id"


def test_backend_registry_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown model backend"):
        create_model("missing", "any/model-id")
