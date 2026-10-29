"""``alyx`` -- image generation through ``image-generation.perchance.org``.

The endpoint is instruction-shaped like the text one: a generator name plus a
prompt, and a body that carries either a hosted URL or inline base64::

    import providers.alyx
    from oinyx import OinyxClient

    image = await OinyxClient().image("alyx/ai-image-generator", "a red fox in snow")
    print(image.url, image.b64)

Because the response shape varies per generator, extraction scans the JSON body
for the first plausible URL / data URI / base64 field instead of trusting one
key. Raw ``image/*`` bodies are encoded too. Like every Perchance endpoint this
one sits behind a Cloudflare managed challenge -- see
:mod:`providers.cloudflare_base` for the ``cookies`` recipe.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from oinyx import (
    BaseImageProvider,
    ImageResponse,
    ProviderError,
    RateLimitError,
    estimate_tokens,
    register_provider,
    retry,
)

from .cloudflare_base import CloudflareBase, UpstreamError

ENDPOINT = "https://image-generation.perchance.org/api/generate"
ORIGIN = "https://perchance.org"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
DEFAULT_MODEL = "ai-image-generator"

#: Generators are selected by name; override with ``config["models"]``.
KNOWN_MODELS = [
    "ai-image-generator",
    "ai-anime-image-generator",
    "ai-portrait-generator",
    "ai-logo-generator",
]

#: Keys whose string values are base64 payloads rather than URLs.
B64_KEYS = ("b64_json", "base64Image", "base64", "b64", "imageBase64")

#: Keys holding a revised prompt, reported back on the response.
PROMPT_KEYS = ("revisedPrompt", "revised_prompt", "finalPrompt")


def _is_candidate(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://", "data:image"))


@register_provider("alyx")
class AlyxProvider(BaseImageProvider, CloudflareBase):
    """oinyx image provider backed by Perchance's hosted image generators."""

    name = "alyx"
    endpoint = ENDPOINT
    origin = ORIGIN
    user_agent = DEFAULT_USER_AGENT

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        CloudflareBase.__init__(self, config)
        self.default_generator: str = str(self.config.get("generator_name", DEFAULT_MODEL))
        self.width: int = int(self.config.get("width", 512))
        self.height: int = int(self.config.get("height", 512))
        self._models: list[str] = list(self.config.get("models") or KNOWN_MODELS)
        self._post = retry(
            attempts=self.attempts,
            backoff=self.backoff,
            on=(RateLimitError, UpstreamError),
        )(self.post)

    async def generate_image(self, model: str, prompt: str, **kwargs: Any) -> ImageResponse:
        """Render ``prompt`` and return the first image in the response.

        Extra keyword arguments are merged into the request body verbatim, so
        generator-specific knobs (``seed``, ``negativePrompt``, ...) work
        without an update to this module.
        """
        payload = self.build_payload(model, prompt, kwargs)
        generator = str(payload["generatorName"])
        response = await self._post(payload, generator)
        image = self.extract_image(response, generator)
        if not image.has_image:
            raise ProviderError(
                "no image found in response body",
                provider=self.name,
                model=generator,
                status_code=response.status_code,
            )
        return image

    async def list_models(self) -> list[str]:
        """Return the configured Perchance image generator names."""
        return list(self._models)

    def build_payload(
        self,
        model: str,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the instruction-shaped request body."""
        generator = str(kwargs.pop("generatorName", None) or model or self.default_generator)
        instruction = str(kwargs.pop("instruction", None) or prompt)
        start_with = str(kwargs.pop("startWith", self.config.get("start_with", "")))
        stop_sequences = list(
            kwargs.pop("stopSequences", self.config.get("stop_sequences") or [])
        )
        payload: dict[str, Any] = {
            "instruction": instruction,
            "startWith": start_with,
            "stopSequences": stop_sequences,
            "generatorName": generator,
            "startWithTokenCount": int(kwargs.pop("startWithTokenCount", 0) or 0),
            "instructionTokenCount": int(
                kwargs.pop("instructionTokenCount", None) or estimate_tokens(instruction)
            ),
            "width": int(kwargs.pop("width", self.width)),
            "height": int(kwargs.pop("height", self.height)),
        }
        extras = self.config.get("extra_payload")
        if isinstance(extras, dict):
            payload.update(extras)
        payload.update({key: value for key, value in kwargs.items() if value is not None})
        return payload

    def extract_image(self, response: httpx.Response, model: str) -> ImageResponse:
        """Build an :class:`~oinyx.ImageResponse` from any of the known shapes."""
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        raw = {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
        }
        if content_type.startswith("image/"):
            return ImageResponse(
                id=f"alyx-{response.status_code}",
                model=model,
                b64=base64.b64encode(response.content).decode("ascii"),
                mime_type=content_type,
                raw=raw,
            )
        data: Any = response.text
        try:
            data = response.json()
        except ValueError:
            pass
        url, b64, revised = _scan(data)
        if b64.startswith("data:image"):
            mime_type, b64 = _split_data_uri(b64)
            return ImageResponse(
                id=f"alyx-{response.status_code}",
                model=model,
                b64=b64,
                mime_type=mime_type,
                revised_prompt=revised,
                raw=raw,
            )
        return ImageResponse(
            id=f"alyx-{response.status_code}",
            model=model,
            url=url or None,
            b64=b64 or None,
            mime_type="image/png" if b64 else None,
            revised_prompt=revised,
            raw=raw,
        )


def _scan(node: Any) -> tuple[str, str, str | None]:
    """Depth-first search for ``(url, base64, revised_prompt)``.

    Earlier hits win, so the outermost/first field the endpoint returns is the
    one reported.
    """
    found: dict[str, Any] = {"url": "", "b64": "", "revised": None}
    _visit(node, found)
    return (str(found["url"]), str(found["b64"]), found["revised"])


def _record(found: dict[str, Any], value: str) -> None:
    """File one candidate string under ``url`` or ``b64``."""
    if value.startswith("data:image"):
        found["b64"] = found["b64"] or value
    else:
        found["url"] = found["url"] or value


def _visit(node: Any, found: dict[str, Any]) -> None:
    if isinstance(node, dict):
        for key in PROMPT_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value and found["revised"] is None:
                found["revised"] = value
                break
        for key, value in node.items():
            if isinstance(value, str) and value:
                if key in B64_KEYS and not value.startswith("data:image"):
                    found["b64"] = found["b64"] or value
                elif _is_candidate(value):
                    _record(found, value)
            elif isinstance(value, (dict, list, tuple)):
                _visit(value, found)
    elif isinstance(node, (list, tuple)):
        for value in node:
            if isinstance(value, str) and _is_candidate(value):
                _record(found, value)
            elif isinstance(value, (dict, list, tuple)):
                _visit(value, found)
    elif isinstance(node, str) and _is_candidate(node):
        _record(found, node)


def _split_data_uri(value: str) -> tuple[str, str]:
    """``("image/png", "<base64>")`` from ``data:image/png;base64,AAAA``."""
    header, _, payload = value.partition(",")
    mime = header.split(":", 1)[1].split(";")[0] if ":" in header else ""
    return (mime or "image/png", payload)
