"""Hugging Face causal language model adapter."""

from contextlib import nullcontext
from copy import deepcopy
import time

from ..core import Model, Response


class HuggingfaceModel(Model):
    def __init__(
        self,
        name,
        *,
        tokenizer=None,
        model_instance=None,
        tokenizer_kwargs=None,
        model_kwargs=None,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False,
        auto_model_class="causal_lm",
        generation_kwargs=None,
        reasoning=False,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
    ):
        super().__init__(name)
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs,
            )

        if model_instance is None:
            if auto_model_class == "causal_lm":
                from transformers import AutoModelForCausalLM

                model_loader = AutoModelForCausalLM
            elif auto_model_class == "multimodal_lm":
                from transformers import AutoModelForMultimodalLM

                model_loader = AutoModelForMultimodalLM
            else:
                raise ValueError(
                    "auto_model_class must be 'causal_lm' or 'multimodal_lm'"
                )
            load_kwargs = {
                "trust_remote_code": trust_remote_code,
                **model_kwargs,
            }
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype

            model_instance = model_loader.from_pretrained(name, **load_kwargs)

        self.tokenizer = tokenizer
        self.model = model_instance
        self.device = getattr(model_instance, "device", None)
        self.generation_kwargs = dict(generation_kwargs or {})

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                self.tokenizer.pad_token_id = eos_token_id

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _render_prompt(self, messages):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
                enable_thinking=self.reasoning,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _prepare_inputs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        return inputs

    def _no_grad(self):
        try:
            import torch
        except ImportError:
            return nullcontext()
        return torch.no_grad()

    @staticmethod
    def _sequence_length(sequence):
        shape = getattr(sequence, "shape", None)
        if shape is not None:
            return int(shape[-1])
        return len(sequence)

    @staticmethod
    def _token_ids(sequence):
        values = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _decode_response(self, generated_ids):
        token_ids = self._token_ids(generated_ids)
        content_ids = token_ids
        reasoning = None

        if self.reasoning:
            convert_token = getattr(self.tokenizer, "convert_tokens_to_ids", None)
            end_thinking_id = convert_token("</think>") if convert_token else None
            unknown_id = getattr(self.tokenizer, "unk_token_id", None)
            if (
                end_thinking_id is not None
                and end_thinking_id != unknown_id
                and end_thinking_id in token_ids
            ):
                split_at = len(token_ids) - 1 - token_ids[::-1].index(end_thinking_id)
                reasoning = self.tokenizer.decode(
                    token_ids[:split_at], skip_special_tokens=True
                ).strip()
                reasoning = reasoning.removeprefix("<think>").strip() or None
                content_ids = token_ids[split_at + 1 :]

        content = self.tokenizer.decode(
            content_ids, skip_special_tokens=True
        ).strip()
        return content, reasoning

    def respond(self, messages) -> Response:
        started_at = time.perf_counter()
        prompt_messages = deepcopy(messages)
        prompt = self._render_prompt(messages)
        inputs = self._prepare_inputs(prompt)
        input_tokens = self._sequence_length(inputs["input_ids"])

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        generation_kwargs.update(self.generation_kwargs)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with self._no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][input_tokens:]
        content, reasoning = self._decode_response(generated_ids)

        return Response(
            prompt=prompt_messages,
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=self._sequence_length(generated_ids),
            generation_time=time.perf_counter() - started_at,
        )
