"""Loopback Ollama decoder for the existing cited answer workflow."""

import json
import os

import httpx

from careops.providers.transport import ProviderFailure, choice_result, post_json

DEFAULT_MODEL = "careops-qwen"
DEFAULT_PORT = 11434


class OllamaGenerator:
    def __init__(self, model=None, port=None, transport=None):
        self.model = model or os.environ.get("CAREOPS_OLLAMA_MODEL", DEFAULT_MODEL)
        try:
            selected_port = int(port if port is not None else os.environ.get(
                "CAREOPS_OLLAMA_PORT", DEFAULT_PORT,
            ))
        except ValueError:
            raise ProviderFailure("invalid_local_port") from None
        if not 1 <= selected_port <= 65535:
            raise ProviderFailure("invalid_local_port")
        self.url = f"http://127.0.0.1:{selected_port}/api/chat"
        self.transport = transport

    def complete(self, system, data, choices):
        with httpx.Client(transport=self.transport, timeout=30, trust_env=False) as client:
            response = post_json(client, self.url, {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system + "\nReturn a JSON object with a choice field."},
                    {"role": "user", "content": data},
                ],
                "format": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": list(choices)}},
                    "required": ["choice"],
                    "additionalProperties": False,
                },
                "options": {"temperature": 0, "seed": 19, "num_predict": 24, "num_ctx": 4096},
            })
        if response.get("done") is not True or response.get("done_reason") == "length":
            return choice_result(None, choices)
        try:
            content = json.loads(response["message"]["content"])
        except (KeyError, TypeError, ValueError):
            raise ProviderFailure("invalid_response") from None
        if not isinstance(content, dict) or set(content) != {"choice"}:
            return choice_result(None, choices)
        return choice_result(content["choice"], choices, response.get("prompt_eval_count"),
                             response.get("eval_count"))

    def choose(self, system, data, choices):
        return self.complete(system, data, choices).value
