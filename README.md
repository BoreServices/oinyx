# oinyx

A provider-agnostic Python client library for chat-completion AI models.

`oinyx` ships **zero built-in providers**. It provides the unified call
shape, the data models, the router, and a thread-safe plugin registry; backends
are contributed by third-party packages that register themselves at import time.

- Pure Python, `>= 3.10`, **no runtime dependencies** in the core
- Async-first with a sync wrapper
- OpenAI-shaped dataclasses (`ChatMessage`, `ChatResponse`, `StreamChunk`, `Usage`)
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

A complete, importable example lives in [`examples/myprovider_plugin.py`](examples/myprovider_plugin.py).

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
.venv/bin/python -m pytest              # suite + coverage
.venv/bin/python -m mypy                # strict, over oinyx/
```

The suite exercises chat, streaming, routing, retries, and every error path
against a `MockProvider` in `tests/conftest.py`, with no network access.

## License

MIT.
