"""Small dependency-free client for a locally running Ollama server."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class OllamaError(RuntimeError):
    """The local model server could not provide a usable response."""


def chat(
    messages: list[dict[str, str]], *, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Call Ollama's local ``/api/chat`` endpoint without streaming."""
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    if tools:
        payload["tools"] = tools
    request = Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310: configurable local host
            body = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise OllamaError("Could not reach the local Ollama server.") from error

    if not isinstance(body.get("message"), dict):
        raise OllamaError("Ollama returned an invalid chat response.")
    return body
