from types import SimpleNamespace

import requests

import pytest

from maslab import (
    GeminiAPIModel,
    HuggingfaceModel,
    Model,
    NvidiaBuildAPIModel,
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
    assert backend.last_generation_kwargs["temperature"] == 1.0
    assert backend.last_generation_kwargs["top_p"] == 0.95
    assert backend.last_generation_kwargs["top_k"] == 20
    assert backend.last_generation_kwargs["repetition_penalty"] == 1.0


@pytest.mark.parametrize("model_class", [GeminiAPIModel, NvidiaBuildAPIModel])
def test_http_adapters_own_generation_credentials_retry_and_timeout_options(model_class, monkeypatch):
    calls = []
    sleeps = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise requests.exceptions.Timeout("simulated timeout")
        if model_class is GeminiAPIModel:
            payload = {
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "answer"}]}}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
            }
        else:
            payload = {
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(model_class.__module__ + ".time.sleep", sleeps.append)
    model = model_class(
        "test-model", api_key="test-key", reasoning=True,
        max_tokens=256, temperature=0.2, top_p=0.8,
        max_retries=1, timeout_seconds=7,
    )

    response = model.respond([{"role": "user", "content": "question"}])

    assert response.content == "answer"
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert len(calls) == 2
    assert sleeps == [1]
    assert all(kwargs["timeout"] == 7 for _, kwargs in calls)
    request = calls[-1][1]
    if model_class is GeminiAPIModel:
        assert request["headers"]["x-goog-api-key"] == "test-key"
        assert request["json"]["generationConfig"] == {
            "maxOutputTokens": 256, "temperature": 0.2, "topP": 0.8,
            "thinkingConfig": {"thinkingBudget": -1},
        }
    else:
        assert request["headers"]["Authorization"] == "Bearer test-key"
        assert request["json"]["max_tokens"] == 256
        assert request["json"]["temperature"] == 0.2
        assert request["json"]["top_p"] == 0.8
        assert request["json"]["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("model_class", [GeminiAPIModel, NvidiaBuildAPIModel, HuggingfaceModel])
def test_model_adapters_reject_removed_generation_config_argument(model_class):
    with pytest.raises(TypeError, match="generation_config"):
        model_class("fake", generation_config=object())
