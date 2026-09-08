"""Gemini Generate Content API model adapter."""

from copy import deepcopy
import time

import requests

from ..core import Model, Response
from ._http import RETRYABLE_STATUS_CODES as HTTP_RETRYABLE_STATUS_CODES


class GeminiAPIModel(Model):
    RETRYABLE_STATUS_CODES = set(HTTP_RETRYABLE_STATUS_CODES)

    def __init__(
        self,
        name,
        api_key=None,
        reasoning=False,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        max_retries=2,
        timeout_seconds=None,
    ):
        super().__init__(name)
        self.api_key = api_key
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

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
                wait_seconds = 2 ** retry_idx
            except requests.exceptions.HTTPError as error:
                response = error.response
                if (
                    response is None
                    or response.status_code not in self.RETRYABLE_STATUS_CODES
                ):
                    raise
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    quota_metrics = []
                    try:
                        details = response.json().get("error", {}).get("details", [])
                        for detail in details:
                            retry_after = detail.get("retryDelay", retry_after)
                            quota_metrics.extend(
                                violation.get("quotaMetric", "")
                                for violation in detail.get("violations", [])
                            )
                    except (TypeError, ValueError):
                        pass
                    wait_seconds = (
                        float(retry_after.removesuffix("s")) + 1
                        if retry_after else 60
                    )
                    metric_text = ", ".join(filter(None, quota_metrics)) or "quota"
                    print(
                        f"Gemini API rate-limited ({metric_text}); "
                        f"retrying in {wait_seconds:g}s",
                        flush=True,
                    )
                else:
                    wait_seconds = 2 ** retry_idx
                if retry_idx >= self.max_retries:
                    raise
            time.sleep(wait_seconds)

    def respond(self, messages) -> Response:
        started_at = time.perf_counter()
        prompt_messages = deepcopy(messages)
        system_text = "\n\n".join(
            message["content"]
            for message in messages
            if message["role"] == "system"
        )
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        generation_options = {"maxOutputTokens": self.max_tokens}
        if self.name.startswith("gemini-3"):
            generation_options["thinkingConfig"] = {
                "thinkingLevel": "medium" if self.reasoning else "minimal",
            }
        else:
            generation_options.update({
                "temperature": self.temperature,
                "topP": self.top_p,
                "thinkingConfig": {
                    "thinkingBudget": -1 if self.reasoning else 0,
                },
            })
        payload = {
            "contents": contents,
            "generationConfig": generation_options,
        }
        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_text}],
            }

        response = self._post_with_retries(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.name}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            json=payload,
        )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(
                f"Gemini returned no candidates: {data.get('promptFeedback')}"
            )
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason != "STOP":
            raise RuntimeError(f"Unexpected model output termination: {finish_reason}")
        content = "".join(
            part.get("text", "")
            for part in candidate.get("content", {}).get("parts", [])
            if not part.get("thought")
        ).strip()
        usage = data.get("usageMetadata") or {}

        return Response(
            prompt=prompt_messages,
            content=content,
            reasoning=None,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            generation_time=time.perf_counter() - started_at,
        )
