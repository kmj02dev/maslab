"""NVIDIA Build API model adapter."""

from copy import deepcopy
import time

import requests

from ..base import Model, Response


class NvidiaBuildAPIModel(Model):
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def _post_with_retries(self, url, **kwargs):
        for retry_idx in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                if retry_idx >= self.max_retries:
                    raise
            except requests.exceptions.HTTPError as error:
                response = error.response
                if (
                    response is None
                    or response.status_code not in self.RETRYABLE_STATUS_CODES
                    or retry_idx >= self.max_retries
                ):
                    raise
            time.sleep(2 ** retry_idx)

    def respond(self, messages) -> Response:
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        prompt_messages = deepcopy(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        payload = {
            "messages": messages,
            "model": self.name,
            "chat_template_kwargs": {
                "enable_thinking": self.reasoning
            },
            "max_tokens": self.max_tokens,
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        response = self._post_with_retries(
            invoke_url,
            headers=headers,
            json=payload,
            stream=False,
        )

        data = response.json()

        finish_reason = data["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            raise RuntimeError(f"Unexpected model output termination: {finish_reason}")

        message = data["choices"][0]["message"]
        usage = data["usage"]

        return Response(
            prompt=prompt_messages,
            content=message["content"],
            reasoning=message.get("reasoning_content"),
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            generation_time=None,
        )
