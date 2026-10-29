"""Offline tests for the bundled ``alyx`` image provider."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from oinyx import (
    ImageResponse,
    OinyxClient,
    ProviderError,
    ProviderRegistry,
    estimate_tokens,
)
from providers.alyx import DEFAULT_MODEL, ENDPOINT, KNOWN_MODELS, AlyxProvider
from providers.cloudflare_base import CloudflareChallengeError

PNG = b"\x89PNG\r\n\x1a\n fake png payload"


def make_provider(handler: Callable[[httpx.Request], httpx.Response], **config: Any) -> AlyxProvider:
    """Provider wired to a mock transport, with retries that do not sleep."""
    return AlyxProvider({"transport": httpx.MockTransport(handler), "backoff": 0, **config})


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", ENDPOINT),
    )


def image_response(body: bytes, content_type: str = "image/webp") -> httpx.Response:
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": content_type},
        request=httpx.Request("POST", ENDPOINT),
    )


@pytest.fixture(autouse=True)
def registered() -> None:
    """Re-register the provider after the global registry was cleared."""
    ProviderRegistry.register(AlyxProvider)


class TestRegistration:
    def test_it_is_registered_under_alyx(self) -> None:
        assert ProviderRegistry.list_providers() == ["alyx"]
        assert isinstance(ProviderRegistry.get("alyx"), AlyxProvider)

    def test_it_advertises_the_image_capability(self) -> None:
        provider = AlyxProvider()
        assert provider.supports_images is True
        assert provider.endpoint == ENDPOINT
        assert provider.default_generator == DEFAULT_MODEL
        assert repr(provider).startswith("<AlyxProvider")

    async def test_chat_and_stream_point_at_generate_image(self) -> None:
        provider = AlyxProvider()
        with pytest.raises(ProviderError, match="use generate_image"):
            await provider.chat("m", [])
        with pytest.raises(ProviderError, match="use generate_image"):
            async for _ in provider.stream("m", []):
                pass


class TestPayload:
    async def test_body_matches_the_instruction_shape(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"imageUrl": "https://cdn/x.png"})

        prompt = "a red fox in snow"
        await make_provider(handler).generate_image("ai-image-generator", prompt)
        assert captured[0] == {
            "instruction": prompt,
            "startWith": "",
            "stopSequences": [],
            "generatorName": "ai-image-generator",
            "startWithTokenCount": 0,
            "instructionTokenCount": estimate_tokens(prompt),
            "width": 512,
            "height": 512,
        }

    async def test_size_and_generator_can_be_overridden(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"imageUrl": "https://cdn/x.png"})

        provider = make_provider(handler, width=768, height=1024, generator_name="ai-logo-generator")
        await provider.generate_image("", "logo", width=256, seed=7, ignore=None)
        assert captured[0]["generatorName"] == "ai-logo-generator"
        assert captured[0]["width"] == 256
        assert captured[0]["height"] == 1024
        assert captured[0]["seed"] == 7
        assert "ignore" not in captured[0]

    async def test_configured_extra_fields_are_merged(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content.decode()))
            return json_response({"imageUrl": "https://cdn/x.png"})

        await make_provider(handler, extra_payload={"modelVersion": "v2"}).generate_image(
            "m", "prompt", width=64
        )
        assert captured[0]["modelVersion"] == "v2"
        assert captured[0]["width"] == 64


class TestImageExtraction:
    @pytest.mark.parametrize(
        "payload",
        [
            {"imageUrl": "https://cdn/a.png"},
            {"imageUrls": ["https://cdn/b.png"]},
            {"images": [{"url": "https://cdn/c.png"}]},
            {"data": [{"url": "https://cdn/d.png"}]},
            {"output": ["https://cdn/e.png"]},
            {"result": {"nested": {"href": None, "image": "https://cdn/f.png"}}},
        ],
    )
    async def test_url_shapes(self, payload: Any) -> None:
        provider = make_provider(lambda request: json_response(payload))
        image = await provider.generate_image("m", "prompt")
        assert image.url and image.url.startswith("https://cdn/")
        assert image.b64 is None
        assert image.model == "m"

    async def test_base64_field_becomes_inline_image_data(self) -> None:
        provider = make_provider(
            lambda request: json_response({"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        )
        image = await provider.generate_image("m", "prompt")
        assert image.url is None
        assert base64.b64decode(image.b64 or "") == PNG
        assert image.mime_type == "image/png"

    async def test_data_uri_is_split_into_mime_and_payload(self) -> None:
        uri = "data:image/jpeg;base64," + base64.b64encode(PNG).decode()
        provider = make_provider(lambda request: json_response({"image": uri, "revisedPrompt": "a fox"}))
        image = await provider.generate_image("m", "prompt")
        assert image.mime_type == "image/jpeg"
        assert base64.b64decode(image.b64 or "") == PNG
        assert image.revised_prompt == "a fox"

    async def test_raw_image_body_is_encoded(self) -> None:
        provider = make_provider(lambda request: image_response(PNG))
        image = await provider.generate_image("m", "prompt")
        assert base64.b64decode(image.b64 or "") == PNG
        assert image.mime_type == "image/webp"
        assert image.raw["status_code"] == 200

    async def test_bare_url_in_a_text_body_is_picked_up(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="https://cdn/only.png",
                headers={"content-type": "text/plain"},
                request=httpx.Request("POST", ENDPOINT),
            )

        image = await make_provider(handler).generate_image("m", "prompt")
        assert image.url == "https://cdn/only.png"

    async def test_a_body_without_an_image_is_an_error(self) -> None:
        provider = make_provider(lambda request: json_response({"status": "queued"}))
        with pytest.raises(ProviderError, match="no image found"):
            await provider.generate_image("m", "prompt")

    def test_scan_reports_the_first_hit_only(self) -> None:
        from providers.alyx import _scan

        url, b64, revised = _scan({"images": [{"url": "https://cdn/1.png", "revisedPrompt": "r"}]})
        assert (url, b64, revised) == ("https://cdn/1.png", "", "r")

    def test_scan_skips_unusable_entries(self) -> None:
        from providers.alyx import _scan

        url, b64, revised = _scan({"imageUrls": ["", 5, None, {"href": "https://cdn/2.png"}]})
        assert (url, b64, revised) == ("https://cdn/2.png", "", None)

    def test_scan_of_an_unrelated_string_finds_nothing(self) -> None:
        from providers.alyx import _scan

        assert _scan("generating, check back later") == ("", "", None)


class TestErrors:
    async def test_challenge_is_reported(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                text="<html>Just a moment...</html>",
                headers={"content-type": "text/html", "cf-mitigated": "challenge"},
                request=httpx.Request("POST", ENDPOINT),
            )

        with pytest.raises(CloudflareChallengeError):
            await make_provider(handler, attempts=1).generate_image("m", "prompt")

    async def test_server_errors_are_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return httpx.Response(
                    500, text="busy", request=httpx.Request("POST", ENDPOINT)
                )
            return json_response({"imageUrl": "https://cdn/x.png"})

        image = await make_provider(handler, attempts=3).generate_image("m", "prompt")
        assert image.url == "https://cdn/x.png"
        assert len(calls) == 2


class TestModelListing:
    async def test_static_catalog_is_returned(self) -> None:
        assert await AlyxProvider().list_models() == KNOWN_MODELS

    async def test_catalog_is_overridable(self) -> None:
        assert await AlyxProvider({"models": ["ai-image-generator"]}).list_models() == [
            "ai-image-generator"
        ]


class TestClientIntegration:
    @staticmethod
    def client_for(handler: Callable[[httpx.Request], httpx.Response], **extra: Any) -> OinyxClient:
        return OinyxClient(
            config={"alyx": {"transport": httpx.MockTransport(handler), "backoff": 0, **extra}}
        )

    async def test_image_routing_through_oinyx_client(self) -> None:
        client = self.client_for(lambda request: json_response({"imageUrl": "https://cdn/x.png"}))
        image = await client.image("alyx/ai-image-generator", "a red fox")
        assert isinstance(image, ImageResponse)
        assert image.url == "https://cdn/x.png"
        assert image.model == "ai-image-generator"

    def test_image_sync_outside_the_loop(self) -> None:
        client = self.client_for(lambda request: json_response({"imageUrl": "https://cdn/y.png"}))
        assert client.image_sync("alyx/m", "prompt").url == "https://cdn/y.png"
