import pytest

from maslab import (
    GeminiAPIModel,
    HuggingfaceModel,
    Model,
    NvidiaBuildAPIModel,
    create_model,
    register_model_backend,
)
from maslab.testing import check_model_backend

from conftest import FixedModel


class ThinkingTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return 99 if token == "</think>" else self.unk_token_id

    def apply_chat_template(self, messages, **kwargs):
        return "rendered prompt"

    def __call__(self, prompt, *, return_tensors):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        return {"input_ids": [[1, 2]]}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        texts = {10: "<think> deliberate", 20: '{"vote": "A"}', 30: "plain"}
        return "".join(texts.get(int(token_id), "") for token_id in token_ids)


class ThinkingModel:
    device = None

    def __init__(self):
        self.last_generation_kwargs = None

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.last_generation_kwargs = kwargs
        return [[1, 2, 30]]


def test_each_model_has_its_own_module():
    assert GeminiAPIModel.__module__ == "maslab.models.gemini_api"
    assert HuggingfaceModel.__module__ == "maslab.models.huggingface"
    assert NvidiaBuildAPIModel.__module__ == "maslab.models.nvidia_build_api"


def test_gemini_backend_is_independent_from_nvidia_backend():
    assert issubclass(GeminiAPIModel, Model)
    assert not issubclass(GeminiAPIModel, NvidiaBuildAPIModel)


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


def test_huggingface_model_splits_native_thinking_from_final_content():
    model = HuggingfaceModel(
        "fake",
        tokenizer=ThinkingTokenizer(),
        model_instance=ThinkingModel(),
        reasoning=True,
    )

    content, reasoning = model._decode_response([10, 99, 20])

    assert reasoning == "deliberate"
    assert content == '{"vote": "A"}'


def test_huggingface_model_keeps_output_without_thinking_delimiter():
    model = HuggingfaceModel(
        "fake",
        tokenizer=ThinkingTokenizer(),
        model_instance=ThinkingModel(),
        reasoning=True,
    )

    content, reasoning = model._decode_response([30])

    assert reasoning is None
    assert content == "plain"


def test_huggingface_model_forwards_additional_generation_kwargs():
    backend = ThinkingModel()
    model = HuggingfaceModel(
        "fake",
        tokenizer=ThinkingTokenizer(),
        model_instance=backend,
        generation_kwargs={"top_k": 20, "repetition_penalty": 1.0},
        max_tokens=81920,
        temperature=1.0,
        top_p=0.95,
    )

    response = model.respond([{"role": "user", "content": "Choose."}])

    assert response.content == "plain"
    assert backend.last_generation_kwargs["max_new_tokens"] == 81920
    assert backend.last_generation_kwargs["top_k"] == 20
    assert backend.last_generation_kwargs["repetition_penalty"] == 1.0
