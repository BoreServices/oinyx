# oinyx

A provider-agnostic Python client library for chat-completion AI models.

`oinyx` ships **zero built-in providers**. It provides the unified call
shape, the data models, the router, and a thread-safe plugin registry; backends
are contributed by third-party packages that register themselves at import time.
This repository keeps its own adapters in [`providers/`](providers) -- see
[Bundled adapters](#bundled-adapters).

- Pure Python, `>= 3.10`, **no runtime dependencies** in the core
- Async-first with a sync wrapper
- OpenAI-shaped dataclasses (`ChatMessage`, `ChatResponse`, `StreamChunk`,
  `Usage`, and `ImageResponse` for image providers)
- Fully typed, `mypy --strict` clean, `py.typed` marker included

## Install

```bash
pip install oinyx            # from a checkout: pip install -e ".[dev]"
```

## Quickstart

Once any provider is registered, the client routes on the `provider/` prefix:

```python
import asyncio
from oinyx import OinyxClient

async def main() -> None:
    client = OinyxClient()          # optionally provider="myprovider"

    response = await client.chat("myprovider/my-model", "Say hello.")
    print(response.content, response.usage.total_tokens)

    # Messages can be ChatMessage objects, OpenAI-shaped dicts, or bare strings.
    response = await client.chat(
        "myprovider/my-model",
        [{"role": "system", "content": "Be brief."}, "Say hello."],
        temperature=0.2,
    )

    # Streaming: pass stream=True and iterate the chunks.
    stream = await client.chat("myprovider/my-model", "Count to three.", stream=True)
    async for chunk in stream:
        print(chunk.delta, end="", flush=True)

asyncio.run(main())
```

Blocking code can use `chat_sync`, which owns its own event loop:

```python
client = OinyxClient(provider="myprovider")
response = client.chat_sync("my-model", "Say hello.")          # ChatResponse
chunks = client.chat_sync("my-model", "Say hello.", stream=True)  # list[StreamChunk]
```

`chat_sync` raises `OinyxError` if called from inside a running loop -- use
`await client.chat(...)` there. `client.collect(...)` streams and collapses the
result into one `ChatResponse`.

### Configuration

```python
# Nested per-provider config...
client = OinyxClient(config={"myprovider": {"api_key": "sk-..."}, "other": {"region": "eu"}})

# ...or flat config, handed to whichever provider the call resolves to.
client = OinyxClient(provider="myprovider", config={"api_key": "sk-"})

# Or configure the registry once, process-wide.
from oinyx import ProviderRegistry
ProviderRegistry.configure("myprovider", {"api_key": "sk-"})
```

Providers receive `config` as the sole constructor argument and may read any
keys they like.

## Image generation

Providers that render images subclass `BaseImageProvider` instead, which
implements `chat`/`stream` as clear errors so an image-only plugin can still be
registered, and requires one method:

```python
from oinyx import BaseImageProvider, ImageResponse, register_provider


@register_provider("myimagegen")
class MyImageProvider(BaseImageProvider):
    async def generate_image(self, model: str, prompt: str, **kwargs) -> ImageResponse:
        return ImageResponse(model=model, url="https://cdn/result.png", width=1024, height=1024)
```

The client routes `image()` exactly like `chat()`:

```python
image = await client.image("myimagegen/sd3", "a red fox in snow", width=1024)
image.url, image.b64, image.mime_type, image.has_image
image = client.image_sync("myimagegen/sd3", "a red fox in snow")   # blocking variant
```

`ImageResponse` carries either a hosted `url` or inline `b64` (plus
`mime_type`, `revised_prompt`, `width`, `height`, `raw`) and round-trips through
`to_dict()` / `from_dict()` like the other models. Calling `image()` on a
text-only provider raises `OinyxError`.

## Bundled adapters

`oinyx/` itself stays provider-free; the adapters live beside it in
[`providers/`](providers) and register themselves on import:

```python
import providers            # registers fable5, kai, perchance, alyx, fai, and the relays
from oinyx import OinyxClient

reply = await OinyxClient().chat("kai/auto", "hello")
image = await OinyxClient().image("alyx/ai-image-generator", "a red fox in snow")
```

| Name | Kind | Upstream | Notes |
| --- | --- | --- | --- |
| `fable5` | chat | `https://fable5.io/api/chat` | Free `openai/gpt-4o`. The endpoint answers with a **plain-text** body and reports the served model in the `x-chat-model` header; 1-32 messages, 8000 chars each. |
| `kai` | chat | `https://api.kilo.ai/api/gateway/chat/completions` | Kilo Code's free models, unauthenticated OpenAI JSON + real SSE. Short aliases (`kai/auto`, `kai/nemo3-ultra`, ...) are mapped in `KAI_MODELS`; unknown ids pass through. `python -m providers.kai` re-verifies the table against the live catalog. |
| `perchance` | chat | `https://text-generation.perchance.org/api/generate` | Instruction-shaped API: the conversation is flattened into one `instruction` string, the model name is the Perchance **generator** (`ai-chat`, `ai-code-generator`, ...). Streaming is emulated locally because the endpoint returns one body. |
| `alyx` | image | `https://image-generation.perchance.org/api/generate` | Same request shape, image results. Extraction scans the body for the first URL, data URI, `b64_json`, or raw `image/*` bytes, since the field name differs per generator. |
| `fai` | chat | `https://aifreeforever.com/api/generate-ai-answer-vertex` | Reverse-engineered site API: an `interactionProof` payload, a per-model `Referer`, and an SSE reply. Needs the browser's `cf_clearance` cookie (see [Configuration](#configuration)) and a matching `user_agent`. `python -m providers.fai` prints the catalog from `/api/chat-models` plus one probe answer. |
| `llm7` | chat | `https://api.llm7.io/v1` | OpenAI-compatible relay; see [relays](#the-openai-compatible-relays). |
| `pollinations` | chat | `https://text.pollinations.ai/openai` | OpenAI-compatible relay; see [relays](#the-openai-compatible-relays). |
| `ovh` | chat | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` | OpenAI-compatible relay; see [relays](#the-openai-compatible-relays). |
| `aihorde` | chat | `https://oai.aihorde.net/v1` | OpenAI-compatible relay over crowdsourced workers; see [relays](#the-openai-compatible-relays). |
| `freeai` | chat | `https://api.free.ai/v1` | OpenAI-compatible relay; see [relays](#the-openai-compatible-relays). |
| `keylessai` | chat | `https://keylessai.thryx.workers.dev/v1` | OpenAI-compatible relay, **unverified**; see [relays](#the-openai-compatible-relays). |

### Perchance is Cloudflare-gated

Both Perchance endpoints answer a plain HTTP client with `403` +
`cf-mitigated: challenge` and an HTML "Just a moment..." page; the challenge
needs a real browser to execute its script. The providers therefore raise
`CloudflareChallengeError` with actionable guidance rather than mis-parsing the
challenge page, and support replaying a browser session:

```python
ProviderRegistry.configure(
    "perchance",
    {
        # cf_clearance is bound to the user-agent that solved the challenge,
        # so both must come from the same browser profile.
        "cookies": {"cf_clearance": "...", "cf_bm": "..."},
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/141.0.0.0 Safari/537.36",
    },
)
```

`cookies` also accepts a full `Cookie` header string. Other knobs: `timeout`,
`attempts`, `backoff`, `headers`, `origin`, `referer`, plus `transport` /
`client` for injecting an `httpx` client in tests. Because the API is gated, the
`KNOWN_MODELS` lists in both modules are generator names to start from, not a
verified catalog -- set `config["models"]` to the generators you actually use.

## The OpenAI-compatible relays

`llm7`, `pollinations`, `ovh`, `aihorde`, `freeai`, and `keylessai` are thin
subclasses of `providers.openai_relay.OpenAiRelay`, which speaks
`POST {base_url}/chat/completions` with OpenAI JSON and real SSE. An adapter is
a docstring, a catalog, and a few class attributes:

```python
class OvhProvider(OpenAiRelay):
    name = "ovh"
    base_url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    default_model = "Meta-Llama-3_3-70B-Instruct"
    catalog = OVH_MODELS          # alias -> exact upstream id
```

What the base gives every relay: aliases resolve through `catalog` and unknown
ids pass through untouched; `temperature`, `tools`, `response_format`, and the
rest of `PASSTHROUGH_PARAMS` forward from kwargs; `stream_options.include_usage`
is opt-in with `config["include_usage"]`; a `402`/`429` -- or prose like "at its
limit" in a `503` -- raises `RateLimitError` with `retry_after`, `401`/`403`
raises `AuthenticationError`, and `5xx` raises `UpstreamError` and is retried
(`attempts` defaults to 2). `python -m providers.ovh` prints the live catalog
and probes each alias.

What differs, as verified against the live endpoints:

| Adapter | Caveat |
| --- | --- |
| `llm7` | Keyless access is gated by the catalog's `tier` field -- only `turbo` rows answer, `pro` rows return 401. `PAID_MODELS` lists what is deliberately **not** routed. |
| `pollinations` | The anonymous tier exposes exactly one chat model, `openai-fast` (GPT-OSS 20B); all five aliases collapse onto it. `flux`, `turbo`, and `gptimage` are image models. Over quota it returns a `503` JSON body. |
| `ovh` | The free limit is 2 requests/minute **per model**, so cycling models is what raises throughput. Ids are Title-Case (`Qwen3.5-9B`). Reasoning models can reply with `content: null` when the whole budget went to `reasoning_content`, which lands in `ChatResponse.raw["reasoning"]`. `Qwen3Guard-*` are classifiers, not chat. |
| `freeai` | Its ~469-row catalog is mostly a paid marketplace; only `self_hosted` rows answer, and they all resolve to one Qwen deployment. |
| `aihorde` | Requires `Authorization: Bearer 0000000000`, which `anonymous_key` supplies. Models are volunteer worker deployments, so latency is queue-bound -- `attempts=1`, `timeout=300`. `list_models()` prefers the live worker list and falls back to the aliases. |
| `keylessai` | The host did not resolve from the machine these adapters were written on, so its table is unverified rather than wrong. |

## Discovering what answers right now

Every keyless relay above throttles **per IP**, so an endpoint that worked a
minute ago can 402 or 429 from your address. `providers/stillworks.py` is the
answer to that -- it is a registry client, not a provider, which is why it is
the one module in `providers/` that registers nothing. The upstream service
live-probes the relays and publishes the model ids that are answering now, with
proved capabilities:

```python
from providers import stillworks

registry = await stillworks.fetch()
for route in stillworks.rank(registry.routes, provider="ovh", capability="tools"):
    print(route.model_string, route.success_rate, f"{route.latency_ms}ms")

# Walk the list until something answers.
route, reply = await stillworks.walk("Say hello.")
print(route.provider, route.model, reply.content)
```

`fetch()` needs a browser `User-Agent` (the registry 403s without one) and
raises `ProviderError` rather than leaking `httpx` transport errors. `rank()`
sorts by success rate then latency and drops rows that don't map to a bundled
adapter, unless `routable_only=False`. `walk()` returns `(Route, ChatResponse)`
-- a blank completion counts as no answer -- collects every candidate's failure
text instead of raising on the first, and stamps the winner's provenance into
`ChatResponse.raw["stillworks"]`. Each `Route` carries the full upstream record
in `Route.raw`. `python -m providers.stillworks` prints the table, optionally
walking it.

## Writing a provider plugin

A provider is a subclass of `BaseProvider` registered under a unique name.

```python
# oinyx_myprovider/__init__.py
from collections.abc import AsyncIterator
from typing import Any

from oinyx import BaseProvider, ChatMessage, ChatResponse, StreamChunk, Usage, register_provider


@register_provider("myprovider")            # also usable as ProviderRegistry.register(cls)
class MyProvider(BaseProvider):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key = self.config.get("api_key", "")

    async def chat(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = [m.to_dict() for m in messages]
        data = await self._post(model, payload, stream=False, **kwargs)   # your transport
        return ChatResponse(
            id=data["id"],
            model=model,
            content=data["choices"][0]["message"]["content"],
            finish_reason=data["choices"][0].get("finish_reason"),
            usage=Usage.from_dict(data.get("usage")),
            raw=data,
        )

    async def stream(self, model: str, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[StreamChunk]:
        async for event in self._sse(model, [m.to_dict() for m in messages], **kwargs):
            delta = event["choices"][0].get("delta", {})
            yield StreamChunk(
                delta=delta.get("content", "") or "",
                finish_reason=event["choices"][0].get("finish_reason"),
                raw=event,
            )

    async def list_models(self) -> list[str]:
        return ["my-model-small", "my-model-large"]
```

That is the whole contract. `chat` and `stream` are required; `list_models`
is optional and defaults to an empty list. Third-party users get routing with a
single import:

```python
import oinyx_myprovider          # registration happens on import
from oinyx import OinyxClient

response = await OinyxClient().chat("myprovider/my-model-large", "hello")
```

A complete, importable example lives in [`examples/myprovider_plugin.py`](examples/myprovider_plugin.py),
and [`examples/bot/`](examples/bot) is a discord.py bot that turns every bundled
adapter into its own family of 24 slash commands with model autocomplete.

### Guidelines for plugin authors

| Concern | Convention |
| --- | --- |
| Model names | Receive the string **without** the `provider/` prefix; the client already stripped it. |
| Errors | Raise `RateLimitError` (429), `AuthenticationError` (401/403), or `ProviderError` for anything else, always passing `provider=` and `model=`. |
| Retries | Decorate your transport call with `@retry(...)` from `oinyx.utils`, or leave it to the caller. |
| Missing usage | Either fill `Usage` yourself or leave it zeroed; `collect_stream` and `fill_usage` fall back to a `chars/4` estimate. |
| Interrupted streams | Raise `StreamInterruptedError` when the connection dies before a finish reason arrives. |
| HTTP | Depend on `httpx` in your own package (`oinyx[http]` documents the pin); the core never performs network I/O. |

## Helpers

```python
from oinyx import collect_stream, estimate_tokens, retry
from oinyx.exceptions import RateLimitError

@retry(attempts=4, backoff=0.5, on=(RateLimitError,))
async def flaky_call() -> str: ...

response = await collect_stream(await client.chat("myprovider/m", "hi", stream=True), model="m")
tokens = estimate_tokens("roughly four characters per token")
```

`retry` works on both async and sync functions, accepts a numeric base delay
(exponential: `backoff * 2 ** (attempt - 1)`) or a callable, and takes an
injectable `sleep=` for tests.

## Registry API

```python
from oinyx import ProviderRegistry

ProviderRegistry.register(MyProvider)      # class decorator or call
ProviderRegistry.get("myprovider")         # lazily instantiated, cached
ProviderRegistry.get("myprovider", {"api_key": "x"})  # one-off instance
ProviderRegistry.configure("myprovider", {"api_key": "x"})  # stored config
ProviderRegistry.list_providers()          # sorted names
ProviderRegistry.unregister("myprovider")  # drop class, instance, and config
ProviderRegistry.clear()                   # test teardown only
```

All operations are guarded by a re-entrant lock, so registration is safe from
threads and event loops alike.

## Routing and aliases

```python
from oinyx import ModelRouter, OinyxClient

client = OinyxClient(aliases={"gpt-4o": "myprovider/gpt-4o-2024"})
client.route("gpt-4o")            # RoutedModel(provider='myprovider', model='gpt-4o-2024')
client.route("bare", "myprovider") # default provider fills in the prefix
```

A `provider/model` string splits on the first `/`; aliases may chain, and
cycles raise `RoutingError`. Unknown names raise `ProviderNotFoundError`, which
lists the registered providers.

## Exceptions

`OinyxError` is the base class; every subclass carries `provider` and `model`
context and renders it in `str()`.

```
OinyxError
├── RoutingError              # model string unresolvable
├── ProviderNotFoundError     # nothing registered under that name
├── ProviderError             # backend failure (.status_code when known)
│   ├── RateLimitError        #   429 / quota exhausted
│   └── AuthenticationError   #   401 / 403 / bad key
└── StreamInterruptedError    # stream died before completion
```

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest              # suite + coverage (core is at 100%)
.venv/bin/python -m pytest -m live      # opt in to tests that hit real upstreams
.venv/bin/python -m mypy                # strict, over oinyx/ and providers/
```

The suite exercises chat, streaming, routing, retries, and every error path
against a `MockProvider` in `tests/conftest.py`, and the bundled adapters
against `httpx.MockTransport`, with no network access by default. The `-m live`
relays are opt-in probes of the upstream catalogs, so they are where table drift
shows up first.

## License

MIT.
# sources
# sources
