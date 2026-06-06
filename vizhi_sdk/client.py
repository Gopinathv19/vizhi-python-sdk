"""Synchronous client for the Vizhi chat completion API."""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .exceptions import APIError, AuthenticationError, InvalidResponseError
from .models import ChatAnswer
Message = Mapping[str, str]
Query = str | Message | Iterable[Message]


class ModelProvider:
    """A model-bound client authenticated with a Vizhi API token."""

    def __init__(
        self,
        model: str,
        token: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        call_sdk: str | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if not token or not token.strip():
            raise ValueError("token must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.model = model.strip()
        self.token = token.strip()
        self.base_url = (
            base_url  
        ).rstrip("/")
        self.timeout = timeout
        self.call_sdk = call_sdk

    def chat(
        self,
        queries: Query,
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> ChatAnswer:
        """Send one chat request and return its answer with usage metadata."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _normalize_messages(queries),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.call_sdk is not None:
            payload["call_sdk"] = self.call_sdk

        data = self._post("/v1/chat/completions", payload)
        return _parse_answer(data)

    def _post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "vizhi-python-sdk/0.1.0",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _decode_json(response.read())
        except HTTPError as exc:
            response_data = _decode_json(exc.read(), allow_invalid=True)
            message = _error_message(response_data, exc.reason)
            if exc.code in (401, 403):
                raise AuthenticationError(message) from exc
            raise APIError(
                message, status_code=exc.code, response=response_data
            ) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise APIError(f"Could not connect to Vizhi backend: {reason}") from exc


def provide_model(
    model_name: str,
    token: str,
    *,
    base_url: str | None = None,
    timeout: float = 60.0,
    call_sdk: str | None = None,
) -> ModelProvider:
    """Create a Vizhi client bound to a model and API token."""
    return ModelProvider(
        model_name,
        token,
        base_url=base_url,
        timeout=timeout,
        call_sdk=call_sdk,
    )


def _normalize_messages(queries: Query) -> list[dict[str, str]]:
    if isinstance(queries, str):
        messages: list[Message] = [{"role": "user", "content": queries}]
    elif isinstance(queries, Mapping):
        nested_messages = queries.get("messages")
        if nested_messages is not None:
            if isinstance(nested_messages, (str, bytes, Mapping)):
                raise ValueError("'messages' must be an iterable of message mappings")
            messages = list(nested_messages)
        else:
            messages = [queries]
    else:
        messages = list(queries)

    if not messages:
        raise ValueError("queries must contain at least one message")

    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("each message must have a non-empty string 'role'")
        if not isinstance(content, str) or not content:
            raise ValueError("each message must have a non-empty string 'content'")
        normalized.append({"role": role, "content": content})
    return normalized


def _decode_json(data: bytes, *, allow_invalid: bool = False) -> Mapping[str, Any]:
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if allow_invalid:
            return {"detail": data.decode("utf-8", errors="replace")}
        raise InvalidResponseError("Vizhi backend returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise InvalidResponseError("Vizhi backend returned a non-object JSON response")
    return decoded


def _error_message(response: Mapping[str, Any], fallback: object) -> str:
    detail = response.get("detail")
    return str(detail if detail is not None else fallback)


def _parse_answer(data: Mapping[str, Any]) -> ChatAnswer:
    try:
        choice = data["choices"][0]
        usage = data["usage"]
        metadata = data["vizhi_metadata"]
        return ChatAnswer(
            content=str(choice["message"]["content"]),
            input_tokens=int(usage["prompt_tokens"]),
            output_tokens=int(usage["completion_tokens"]),
            total_tokens=int(usage["total_tokens"]),
            latency_ms=int(metadata["latency_ms"]),
            model=str(data["model"]),
            provider=str(metadata["provider"]),
            query_id=str(metadata["query_id"]),
            agent_id=str(metadata["agent_id"]),
            response_id=str(data["id"]),
            finish_reason=str(choice["finish_reason"]),
            raw=data,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InvalidResponseError(
            "Vizhi backend response is missing required chat metadata"
        ) from exc
