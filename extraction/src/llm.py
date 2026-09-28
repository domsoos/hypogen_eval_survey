from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError


@dataclass
class CallMeta:
    latency_sec: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class GlimmerClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 6000,
        timeout: float = 300.0,
        json_mode: bool | None = None,
    ):
        self.base_url = base_url or os.getenv("GLIMMER_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("GLIMMER_API_KEY", "EMPTY")
        self.model = model or os.getenv("GLIMMER_MODEL", "Muse-Glimmer-30B")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if json_mode is None:
            json_mode = os.getenv("GLIMMER_JSON_MODE", "0") == "1"
        self.json_mode = json_mode
        self.retries = int(os.getenv("GLIMMER_RETRIES", "1"))
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Tolerant recovery: extract outermost JSON object.
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def call(self, system: str, user: str, output_model: Type[BaseModel], seed: Optional[int] = None) -> tuple[BaseModel, CallMeta, str]:
        last_exc: Exception | None = None
        last_raw = ""
        for attempt in range(self.retries + 1):
            attempt_user = user
            if attempt:
                attempt_user += (
                    "\n\nRETRY INSTRUCTION: The previous response could not be parsed or validated. "
                    "Return ONLY one valid JSON object matching the requested schema exactly. "
                    "Do not use markdown fences or explanatory text."
                )
            kwargs = dict(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": attempt_user},
                ],
            )
            if seed is not None:
                kwargs["seed"] = seed + attempt
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            t0 = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                raw = response.choices[0].message.content or ""
                last_raw = raw
                data = self._extract_json(raw)
                parsed = output_model.model_validate(data)

                usage = getattr(response, "usage", None)
                meta = CallMeta(
                    latency_sec=latency,
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                    total_tokens=getattr(usage, "total_tokens", None) if usage else None,
                )
                return parsed, meta, raw
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break

        raise RuntimeError(
            f"Model call failed after {self.retries + 1} attempt(s): {last_exc}\n\nLAST RAW RESPONSE:\n{last_raw}"
        ) from last_exc
