"""Normalize provider choices and redact transport failures."""

from dataclasses import dataclass

import httpx


class ProviderFailure(RuntimeError):
    """A short error code safe for local client output."""


@dataclass(frozen=True)
class Choice:
    value: str | None
    input_tokens: int | None
    output_tokens: int | None


def error_code(status):
    return {
        401: "authorization",
        403: "authorization",
        404: "model_unavailable",
        429: "quota",
    }.get(status, "provider_unavailable")


def post_json(client, url, payload, headers=None):
    try:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    except httpx.TimeoutException:
        raise ProviderFailure("timeout") from None
    except httpx.HTTPStatusError as exc:
        raise ProviderFailure(error_code(exc.response.status_code)) from None
    except httpx.RequestError:
        raise ProviderFailure("connection") from None
    except ValueError:
        raise ProviderFailure("invalid_response") from None
    if not isinstance(body, dict):
        raise ProviderFailure("invalid_response")
    return body


def choice_result(text, allowed, input_tokens=None, output_tokens=None):
    value = text.strip() if isinstance(text, str) else None
    return Choice(
        value if value in allowed else None,
        input_tokens if type(input_tokens) is int and input_tokens >= 0 else None,
        output_tokens if type(output_tokens) is int and output_tokens >= 0 else None,
    )
