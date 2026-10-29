"""Small helpers shared by the reverse-engineered HTTP providers."""

from __future__ import annotations

from typing import Any

import httpx


def as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a JSON object, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a JSON array, otherwise an empty list."""
    return value if isinstance(value, list) else []


def describe_error(error: Any) -> tuple[str, str]:
    """Normalise string-or-object error payloads into ``(message, code)``."""
    if isinstance(error, str):
        return (error, "upstream_error")
    if isinstance(error, dict):
        keys = ("message", "error", "detail", "error_message")
        message = next(
            (str(error[key]) for key in keys if isinstance(error.get(key), str) and error[key]),
            "upstream error",
        )
        code = error.get("code") or error.get("type") or error.get("status") or "upstream_error"
        return (message, str(code))
    return (str(error)[:200], "upstream_error")


def error_payload(response: httpx.Response) -> dict[str, Any]:
    """Return the error object from ``response``, or a synthesised one."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return {"message": text[:200] or f"upstream returned HTTP {response.status_code}"}
    if isinstance(body, dict) and "error" in body:
        error = body["error"]
        return error if isinstance(error, dict) else {"message": str(error)[:200]}
    return body if isinstance(body, dict) else {"message": str(body)[:200]}


def sse_data(line: str) -> str | None:
    """Return the payload of an SSE ``data:`` line, or ``None`` for any other line."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip() or None
