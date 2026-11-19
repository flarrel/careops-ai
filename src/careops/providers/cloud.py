"""Model-choice clients for the explicit, synthetic-only provider smoke command."""

import os
import re

import httpx

from careops.providers.transport import ProviderFailure, choice_result, error_code, post_json

PROVIDERS = ("openai", "anthropic", "gemini", "groq", "vertex")
CHAT_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}
KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "vertex": "VERTEX_ACCESS_TOKEN",
}
MAX_TOKENS = 256


class CloudChoice:
    def __init__(self, provider, model, *, transport=None):
        if provider not in PROVIDERS:
            raise ValueError("Unknown provider")
        model_pattern = r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,127}" if provider == "groq" else (
            r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}"
        )
        if not re.fullmatch(model_pattern, model):
            raise ValueError("Invalid provider model name")
        self.provider = provider
        self.model = model
        self.key = os.environ.get(KEY_NAMES[provider], "").strip()
        if not self.key:
            raise ProviderFailure("missing_credentials")
        self.transport = transport
        if provider == "vertex":
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", project) or not re.fullmatch(
                r"[a-z][a-z0-9-]{0,31}", location,
            ):
                raise ProviderFailure("missing_vertex_location_or_project")
            endpoint = "aiplatform.googleapis.com" if location == "global" else (
                location + "-aiplatform.googleapis.com"
            )
            self.vertex_url = (
                f"https://{endpoint}/v1/projects/{project}/locations/{location}"
                f"/publishers/google/models/{model}:generateContent"
            )

    def complete(self, system, data, choices):
        instructions = system + "\nReturn exactly one of " + ", ".join(choices) + "."
        if self.provider == "anthropic":
            return self.claude(instructions, data, choices)
        with httpx.Client(transport=self.transport, timeout=20, trust_env=False) as client:
            if self.provider in CHAT_URLS:
                return self.chat(client, instructions, data, choices)
            return self.google(client, instructions, data, choices)

    def chat(self, client, instructions, data, choices):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": data}],
            "max_completion_tokens": MAX_TOKENS,
        }
        if self.provider == "openai":
            payload["store"] = False
        if self.provider == "groq" and self.model.startswith("openai/gpt-oss-"):
            payload.update(reasoning_effort="low", include_reasoning=False)
        result = post_json(client, CHAT_URLS[self.provider], payload,
                           {"Authorization": "Bearer " + self.key})
        try:
            best = result["choices"][0]
            raw = best["message"].get("content") if best["finish_reason"] == "stop" else None
            usage = result.get("usage") or {}
            return choice_result(raw, choices, usage.get("prompt_tokens"),
                                 usage.get("completion_tokens"))
        except (AttributeError, KeyError, IndexError, TypeError):
            raise ProviderFailure("invalid_response") from None

    def claude(self, instructions, data, choices):
        import httpx2
        from anthropic import Anthropic, APIConnectionError, APIError, APIStatusError, APITimeoutError

        try:
            with httpx2.Client(transport=self.transport, timeout=20, trust_env=False) as client:
                vendor = Anthropic(api_key=self.key, base_url="https://api.anthropic.com",
                                   max_retries=0, timeout=20, http_client=client)
                response = vendor.messages.create(
                    model=self.model, max_tokens=MAX_TOKENS, system=instructions,
                    messages=[{"role": "user", "content": data}],
                )
            text = "".join(block.text for block in response.content if block.type == "text")
            raw = text if response.stop_reason in ("end_turn", "stop_sequence") else None
            return choice_result(raw, choices, response.usage.input_tokens,
                                 response.usage.output_tokens)
        except APITimeoutError:
            raise ProviderFailure("timeout") from None
        except APIStatusError as exc:
            raise ProviderFailure(error_code(exc.status_code)) from None
        except APIConnectionError:
            raise ProviderFailure("connection") from None
        except (APIError, AttributeError, TypeError, ValueError):
            raise ProviderFailure("invalid_response") from None

    def google(self, client, instructions, data, choices):
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.key}
        else:
            url = self.vertex_url
            headers = {"Authorization": "Bearer " + self.key}
        result = post_json(client, url, {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": [{"text": data}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_TOKENS},
        }, headers)
        try:
            candidate = (result.get("candidates") or [{}])[0]
            raw = None
            if candidate.get("finishReason") == "STOP":
                parts = candidate["content"]["parts"]
                raw = "".join(part.get("text", "") for part in parts if not part.get("thought"))
            usage = result.get("usageMetadata") or {}
            completion = usage.get("candidatesTokenCount")
            thought = usage.get("thoughtsTokenCount")
            if type(completion) is int and type(thought) is int:
                completion += thought
            return choice_result(raw, choices, usage.get("promptTokenCount"), completion)
        except (AttributeError, KeyError, IndexError, TypeError):
            raise ProviderFailure("invalid_response") from None
