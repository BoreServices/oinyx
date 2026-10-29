"""The 24 chat verbs and 24 image verbs, implemented against :class:`Call`."""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

from oinyx import ChatMessage, ProviderError, estimate_tokens

from . import stillworks_bridge
from .catalog import endpoint_of, offline_models, provider_instance, spec_for
from .config import ALL_PROVIDERS
from .context import Call
from .rendering import (
    bullet_list,
    chunks,
    describe_response,
    fence,
    image_attachment,
    image_meta,
    image_reference,
    meta,
    size_of,
    table,
    truncate,
)
from .verbs import verbs_for

Handler = Callable[[Call], Awaitable[None]]

#: The command names each provider exposes, for ``/<provider>_info``.
VERBS: dict[str, tuple[str, ...]] = {name: tuple(verbs_for(name)) for name in ALL_PROVIDERS}


async def answer(call: Call, prompt: str, *, context: bool = True, json_mode: bool = False) -> str:
    """Run one completion, store the exchange, and return the text."""
    await call.defer()
    model = call.model()
    messages = call.messages(prompt, context=context)
    kwargs = call.knob_kwargs(json_mode=json_mode)
    response = await call.client.chat(f"{call.provider}/{model}", messages, **kwargs)
    call.store.append(call.session, prompt, response.content)
    return str(response.content)


async def ask(call: Call) -> None:
    """``/<provider>_ask`` -- one prompt, this channel's context, the whole answer."""
    prompt = str(call.arg("prompt", ""))
    model = call.model()
    started = time.perf_counter()
    response = await call.client.chat(
        f"{call.provider}/{model}", call.messages(prompt), **call.knob_kwargs()
    )
    call.store.append(call.session, prompt, response.content)
    stamp = describe_response(response, call.provider, int((time.perf_counter() - started) * 1000))
    body = response.content.strip() or "_(the model returned nothing)_"
    await call.send(f"{stamp}\n\n{body}")


async def stream(call: Call) -> None:
    """``/<provider>_stream`` -- edit one message as deltas arrive."""
    prompt = str(call.arg("prompt", ""))
    model = call.model()
    every = int(call.arg("edit_every") or call.settings.edit_every)
    await call.defer()
    message = await call.say("…")
    pieces: list[str] = []
    buffer = ""
    shown = 0
    reasoning = 0
    started = time.perf_counter()
    deltas = await call.client.chat(
        f"{call.provider}/{model}", call.messages(prompt), stream=True, **call.knob_kwargs()
    )
    async for chunk in deltas:
        if chunk.raw.get("reasoning"):
            reasoning += len(str(chunk.raw["reasoning"]))
        if not chunk.delta:
            continue
        pieces.append(chunk.delta)
        buffer += chunk.delta
        if len(buffer) >= every:
            shown = len("".join(pieces))
            await call.edit(message, "".join(pieces)[:1988] + "…")
            buffer = ""
    text = "".join(pieces)
    elapsed = int((time.perf_counter() - started) * 1000)
    stamp = meta(
        call.provider,
        model,
        elapsed_ms=elapsed,
        extra=f"{shown} chars streamed" + (f", {reasoning} chars of reasoning" if reasoning else ""),
    )
    if not text:
        await call.edit(message, f"_(no deltas arrived)_\n{stamp}")
        return
    call.store.append(call.session, prompt, text)
    parts = chunks(text)
    await call.edit(message, parts[0][:1988])
    for extra in parts[1:]:
        await call.send(extra)
    await call.send(stamp)


async def json_command(call: Call) -> None:
    """``/<provider>_json`` -- demand a JSON object and show it parsed."""
    prompt = str(call.arg("prompt", ""))
    fields = str(call.arg("fields", ""))
    shape = f" Return exactly these keys: {fields}." if fields else ""
    spec = spec_for(call.provider)
    text = await answer(
        call,
        "Respond with a single JSON object and nothing else."
        f"{shape}\nTask: {prompt}"[:6000],
        context=False,
        json_mode=spec.json_mode,
    )
    payload = first_json(text)
    if payload is None:
        await call.send(f"not JSON, said `{call.provider}`:\n{fence(text)}")
        return
    await call.send(fence(json.dumps(payload, indent=2, ensure_ascii=False), "json"))


async def code(call: Call) -> None:
    """``/<provider>_code`` -- write code, fenced."""
    prompt = str(call.arg("prompt", ""))
    suffix = str(call.arg("suffix", ""))
    text = await answer(call, f"Write code. {suffix}\nRequest: {prompt}"[:6000], context=False)
    await call.send(fence(text.strip("`\n")))


async def role(call: Call) -> None:
    """``/<provider>_role`` -- answer as a persona, without channel history."""
    persona = str(call.arg("persona", ""))
    prompt = str(call.arg("prompt", ""))
    model = call.model()
    messages = [ChatMessage("system", f"You are {persona}. Stay in character."),
                ChatMessage("user", call.store.clip(prompt))]
    response = await call.client.chat(f"{call.provider}/{model}", messages, **call.knob_kwargs())
    call.store.append(call.session, prompt, response.content)
    await call.send(response.content)


async def translate(call: Call) -> None:
    """``/<provider>_translate`` -- straight translation, no commentary."""
    text = str(call.arg("text", ""))
    target = str(call.arg("target", ""))
    answer_text = await answer(
        call,
        f"Translate the following into {target}. Output only the translation.\n\n{text}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def summarize(call: Call) -> None:
    """``/<provider>_summarize`` -- compress supplied text."""
    text = str(call.arg("text", ""))
    words = int(call.arg("words") or 120)
    answer_text = await answer(
        call,
        f"Summarize this in at most {words} words.\n\n{text}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def continue_answer(call: Call) -> None:
    """``/<provider>_continue`` -- extend the last stored answer."""
    turns = call.session.turns
    if not turns:
        await call.send("nothing to continue yet -- ask something first.", ephemeral=True)
        return
    direction = str(call.arg("direction", ""))
    tail = turns[-1].content
    prompt = (
        f"Here is what you just said:\n\n{tail[:3000]}\n\n"
        f"Continue from exactly there, without repeating it. {direction}"
    )
    answer_text = await answer(call, prompt[:6000])
    await call.send(answer_text)


async def rewrite(call: Call) -> None:
    """``/<provider>_rewrite`` -- same content, different voice."""
    text = str(call.arg("text", ""))
    style = str(call.arg("style", "clearer and more direct"))
    answer_text = await answer(
        call,
        f"Rewrite the text below so it is {style}. Keep the meaning and the facts.\n\n{text}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def explain(call: Call) -> None:
    """``/<provider>_explain`` -- explain for a chosen audience."""
    prompt = str(call.arg("prompt", ""))
    audience = str(call.arg("audience", "a curious beginner"))
    answer_text = await answer(
        call, f"Explain this to {audience}. {prompt}"[:6000], context=False
    )
    await call.send(answer_text)


async def brainstorm(call: Call) -> None:
    """``/<provider>_brainstorm`` -- a numbered list of ideas."""
    prompt = str(call.arg("prompt", ""))
    count = int(call.arg("count") or 8)
    answer_text = await answer(
        call,
        f"Give exactly {count} distinct ideas for the following. Number them, one line each.\n\n{prompt}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def review(call: Call) -> None:
    """``/<provider>_review`` -- critique a snippet."""
    code_text = str(call.arg("code", ""))
    focus = str(call.arg("focus", "correctness and security"))
    answer_text = await answer(
        call,
        f"Review this code, prioritising {focus}. Be specific and cite lines.\n\n{code_text}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def extract(call: Call) -> None:
    """``/<provider>_extract`` -- pull facts out of text."""
    text = str(call.arg("text", ""))
    fields = str(call.arg("fields", ""))
    wanted = f" Extract only these fields: {fields}." if fields else ""
    answer_text = await answer(
        call,
        f"List the facts in this text as short bullets.{wanted}\n\n{text}"[:6000],
        context=False,
    )
    await call.send(answer_text)


async def compare(call: Call) -> None:
    """``/<provider>_compare`` -- the same prompt through two models."""
    prompt = str(call.arg("prompt", ""))
    first = call.model()
    second = str(call.arg("model_b", ""))
    if not second:
        await call.send("give me a second model with `model_b`.", ephemeral=True)
        return
    await call.defer()
    results: list[tuple[str, str]] = []
    for model in (first, second):
        try:
            response = await call.client.chat(
                f"{call.provider}/{model}", call.messages(prompt, context=False),
                **call.knob_kwargs(),
            )
            results.append((model, response.content.strip() or "(empty)"))
        except Exception as exc:  # noqa: BLE001 - a comparison must show both sides
            results.append((model, call.failure(exc)))
    blocks = [f"**{call.provider}/{name}**\n\n{truncate_text(body)}" for name, body in results]
    await call.send("\n\n---\n\n".join(blocks))


async def probe(call: Call) -> None:
    """``/<provider>_probe`` -- tiny prompt per alias, reported honestly."""
    everything = bool(call.arg("all"))
    model = call.model()
    targets = [model]
    if everything:
        pairs = offline_models(call.provider, provider_instance(call.provider))
        targets = [value for value, _ in pairs][:8] or [model]
    rows: list[tuple[str, str]] = []
    await call.defer()
    for target in targets:
        started = time.perf_counter()
        try:
            response = await call.client.chat(
                f"{call.provider}/{target}",
                [ChatMessage("user", "reply with exactly: ok")],
                max_tokens=16,
            )
            rows.append((target, f"OK {int((time.perf_counter() - started) * 1000)}ms "
                                 f"{truncate(response.content, 40)!r}"))
        except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
            rows.append((target, f"FAIL {truncate(type(exc).__name__ + ' ' + str(exc), 60)}"))
    await call.send(table(rows, width=90))


async def models(call: Call) -> None:
    """``/<provider>_models`` -- what this command will accept as ``model``."""
    live = bool(call.arg("live"))
    needle = str(call.arg("filter", "")).lower()
    spec = spec_for(call.provider)
    pairs = await call.catalog.models(call.provider, live=live)
    if needle:
        pairs = [(value, label) for value, label in pairs if needle in f"{value} {label}".lower()]
    scope = "pinned + live" if live and spec.live_catalog else "pinned"
    header = meta(
        call.provider,
        "catalog",
        extra=f"{len(pairs)} entries ({scope}) at {endpoint_of(provider_instance(call.provider))}",
    )
    await call.send(f"{header}\n\n{table(pairs, width=70)}")


async def info(call: Call) -> None:
    """``/<provider>_info`` -- the endpoint, its limits, and its known traps."""
    spec = spec_for(call.provider)
    instance = provider_instance(call.provider)
    default = getattr(instance, "default_model", "") or getattr(instance, "default_generator", "")
    rows = [
        ("kind", spec.kind),
        ("endpoint", endpoint_of(instance)),
        ("default model", str(default) or "(none)"),
        ("streaming", "yes" if spec.streaming else "no"),
        ("json mode", "yes" if spec.json_mode else "prompt only"),
        ("live catalog", "yes" if spec.live_catalog else "pinned table"),
        ("timeout", f"{getattr(instance, 'timeout', call.settings.timeout)}s"),
        ("attempts", str(getattr(instance, 'attempts', call.settings.attempts))),
        ("knobs", ", ".join(sorted(_provider_options(call.provider))) or "none"),
        ("commands", f"{len(VERBS[call.provider])} registered"),
    ]
    body = (
        f"**{call.provider}** -- {spec.summary}\n\n{table(rows, width=90)}\n\n"
        "**Watch out**\n" + bullet_list(list(spec.traps))
        + f"\n\n_{spec.history}; {len(call.session.turns)} turns stored in this channel._"
    )
    await call.send(body)


async def walk(call: Call) -> None:
    """``/<provider>_walk`` -- ignore the pinned provider and take what answers."""
    prompt = str(call.arg("prompt", ""))
    limit = int(call.arg("limit") or 5)
    await call.defer(thinking=True)
    label, text, failures = await stillworks_bridge.walk(call.client, prompt, limit=limit)
    if not text:
        await call.send("no candidate answered:\n\n" + bullet_list(failures, limit=limit))
        return
    await call.send(f"**answered by** `{label}`\n\n{truncate_text(text)}")


async def session(call: Call) -> None:
    """``/<provider>_session`` -- what the bot is remembering for this channel."""
    rows = [
        ("turns", str(len(call.session.turns))),
        ("notes", str(len(call.session.notes))),
        ("defaults", json.dumps(call.session.defaults) if call.session.defaults else "(none)"),
        ("last prompt", truncate(call.session.last_prompt, 60) or "(none)"),
        ("clip", f"{call.store.max_chars} chars"),
        ("ttl", f"{int(call.store.ttl)}s"),
    ]
    await call.send(table(rows, width=70))


async def history(call: Call) -> None:
    """``/<provider>_history`` -- the stored transcript."""
    last = int(call.arg("last") or 10)
    turns = call.session.transcript(last)
    if not turns:
        await call.send("nothing stored for this channel yet.", ephemeral=True)
        return
    lines = [f"**{turn.role}** {truncate(turn.content, 300)}" for turn in turns]
    await call.send("\n".join(lines))


async def remember(call: Call) -> None:
    """``/<provider>_remember`` -- a standing instruction for this channel."""
    note = str(call.arg("note", ""))
    call.session.notes.append(call.store.clip(note))
    call.session.touch()
    await call.send(f"remembered ({len(call.session.notes)} notes).", ephemeral=True)


async def forget(call: Call) -> None:
    """``/<provider>_forget`` -- drop stored turns."""
    turns = call.arg("turns")
    removed = call.store.forget(call.session, int(turns) if turns else None)
    await call.send(f"dropped {removed} messages.", ephemeral=True)


async def defaults(call: Call) -> None:
    """``/<provider>_defaults`` -- sticky model and length for this channel."""
    changed: list[str] = []
    for key in ("model", "max_tokens", "width", "height"):
        value = call.arg(key)
        if value is not None:
            call.session.defaults[key] = value
            changed.append(f"{key}={value}")
    call.session.touch()
    if not changed:
        await call.send(
            "current defaults: " + (json.dumps(call.session.defaults) if call.session.defaults else "(none)")
        )
        return
    await call.send("set " + ", ".join(changed), ephemeral=True)


async def tokens(call: Call) -> None:
    """``/<provider>_tokens`` -- local estimate, no request sent."""
    text = str(call.arg("text", ""))
    estimate = estimate_tokens(text)
    rows = [
        ("characters", str(len(text))),
        ("words", str(len(text.split()))),
        ("estimated tokens", str(estimate)),
        ("stored turns", str(len(call.session.turns))),
        ("context estimate", str(sum(estimate_tokens(turn.content) for turn in call.session.turns))),
    ]
    body = table(rows, width=40)
    if call.arg("report"):
        body += "\n\n" + fence(text[:1500])
    await call.send(body)


# --------------------------------------------------------------------------- image


def preset(call: Call, width: int, height: int) -> dict[str, Any]:
    """Apply a size preset unless the caller passed explicit dimensions."""
    kwargs = call.knob_kwargs()
    kwargs.setdefault("width", width)
    kwargs.setdefault("height", height)
    return kwargs


async def render(call: Call, prompt: str, kwargs: dict[str, Any] | None = None) -> str:
    """Generate one image and post it; returns the caption line."""
    await call.defer()
    model = call.model()
    started = time.perf_counter()
    image = await call.client.image(f"{call.provider}/{model}", prompt, **(kwargs or call.knob_kwargs()))
    call.store.append(call.session, prompt, f"[rendered {size_of(image)}]")
    stamp = image_meta(image, int((time.perf_counter() - started) * 1000))
    await call.send(f"{prompt[:400]}\n\n{stamp}", file=image_attachment(image, f"{call.provider}-{model}"))
    if not image.b64 and image.url:
        await call.send(image_reference(image))
    return stamp


async def generate(call: Call) -> None:
    """``/alyx_generate`` -- render a prompt."""
    await render(call, str(call.arg("prompt", "")))


async def wide(call: Call) -> None:
    """``/alyx_wide`` -- landscape preset."""
    await render(call, str(call.arg("prompt", "")), preset(call, 896, 512))


async def tall(call: Call) -> None:
    """``/alyx_tall`` -- portrait-orientation preset."""
    await render(call, str(call.arg("prompt", "")), preset(call, 512, 896))


async def square(call: Call) -> None:
    """``/alyx_square`` -- square preset."""
    await render(call, str(call.arg("prompt", "")), preset(call, 768, 768))


async def hd(call: Call) -> None:
    """``/alyx_hd`` -- the largest preset that survives the generator."""
    await render(call, str(call.arg("prompt", "")), preset(call, 1024, 1024))


async def variation(call: Call) -> None:
    """``/alyx_variation`` -- the same prompt at several seeds."""
    prompt = str(call.arg("prompt", ""))
    count = min(int(call.arg("count") or 3), 4)
    base = call.knob_kwargs()
    await call.defer()
    notes: list[str] = []
    for index in range(count):
        kwargs = dict(base)
        kwargs["seed"] = int(call.arg("seed") or 1000) + index
        model = call.model()
        started = time.perf_counter()
        try:
            image = await call.client.image(f"{call.provider}/{model}", prompt, **kwargs)
        except Exception as exc:  # noqa: BLE001 - report each seed separately
            notes.append(f"seed {kwargs['seed']}: {truncate(exc, 60)}")
            continue
        notes.append(
            f"seed {kwargs['seed']}: {size_of(image)} "
            f"{int((time.perf_counter() - started) * 1000)}ms"
        )
        await call.send(f"seed `{kwargs['seed']}`", file=image_attachment(image, f"seed-{index}"))
    await call.send("\n".join(f"- {note}" for note in notes) or "no seeds answered")


async def seed_probe(call: Call) -> None:
    """``/alyx_seed_probe`` -- one seed, and what actually came back."""
    prompt = str(call.arg("prompt", ""))
    kwargs = call.knob_kwargs()
    kwargs["seed"] = int(call.arg("seed") or 42)
    await render(call, prompt, kwargs)


async def negative(call: Call) -> None:
    """``/alyx_negative`` -- explicit negative prompt."""
    prompt = str(call.arg("prompt", ""))
    kwargs = call.knob_kwargs()
    kwargs["negativePrompt"] = str(call.arg("negative_prompt", "blurry, watermark, text"))
    await render(call, prompt, kwargs)


async def _fixed_generator(call: Call, generator: str) -> None:
    """Render with a named generator, ignoring the ``model`` option."""
    prompt = str(call.arg("prompt", ""))
    kwargs = call.knob_kwargs()
    kwargs["generatorName"] = generator
    await call.defer()
    started = time.perf_counter()
    image = await call.client.image(f"{call.provider}/{generator}", prompt, **kwargs)
    call.store.append(call.session, prompt, f"[rendered {generator}]")
    await call.send(
        f"{prompt[:400]}\n\n{image_meta(image, int((time.perf_counter() - started) * 1000))}",
        file=image_attachment(image, generator),
    )


async def anime(call: Call) -> None:
    """``/alyx_anime`` -- the anime generator."""
    await _fixed_generator(call, "ai-anime-image-generator")


async def portrait(call: Call) -> None:
    """``/alyx_portrait`` -- the portrait generator."""
    await _fixed_generator(call, "ai-portrait-generator")


async def logo(call: Call) -> None:
    """``/alyx_logo`` -- the logo generator."""
    await _fixed_generator(call, "ai-logo-generator")


async def batch(call: Call) -> None:
    """``/alyx_batch`` -- several prompts in one command."""
    raw = str(call.arg("prompts", ""))
    prompts = [part.strip() for part in raw.split("|") if part.strip()][:4]
    if not prompts:
        await call.send("give me prompts separated by `|`.", ephemeral=True)
        return
    await call.defer()
    for index, prompt in enumerate(prompts):
        try:
            image = await call.client.image(
                f"{call.provider}/{call.model()}", prompt, **call.knob_kwargs()
            )
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not stop the batch
            await call.send(f"`{truncate(prompt, 50)}`: {call.failure(exc)}")
            continue
        await call.send(f"{index + 1}. {truncate(prompt, 200)}", file=image_attachment(image, f"batch-{index}"))


async def image_compare(call: Call) -> None:
    """``/alyx_compare`` -- one prompt through two generators."""
    prompt = str(call.arg("prompt", ""))
    second = str(call.arg("generator_b", ""))
    pairs = [call.model(), second]
    await call.defer()
    for generator in pairs:
        try:
            image = await call.client.image(
                f"{call.provider}/{generator}", prompt, **call.knob_kwargs()
            )
        except Exception as exc:  # noqa: BLE001 - show both outcomes
            await call.send(f"`{generator}`: {call.failure(exc)}")
            continue
        await call.send(f"`{generator}`: {size_of(image)}", file=image_attachment(image, generator))


async def size_probe(call: Call) -> None:
    """``/alyx_size_probe`` -- ask for three sizes and report what stuck."""
    prompt = str(call.arg("prompt", ""))
    await call.defer()
    rows: list[tuple[str, str]] = []
    for size in (512, 768, 1024):
        kwargs = call.knob_kwargs()
        kwargs.update({"width": size, "height": size})
        try:
            image = await call.client.image(f"{call.provider}/{call.model()}", prompt, **kwargs)
            rows.append((f"{size}x{size}", f"got {size_of(image)} {image.mime_type}"))
        except Exception as exc:  # noqa: BLE001 - this command exists to report failures
            rows.append((f"{size}x{size}", truncate(exc, 60)))
    await call.send(table(rows, width=70))


async def retry(call: Call) -> None:
    """``/alyx_retry`` -- render this channel's last prompt again."""
    prompt = call.session.last_prompt
    if not prompt or prompt.startswith("[rendered"):
        await call.send("no previous prompt to retry.", ephemeral=True)
        return
    await render(call, prompt)


async def prompt_command(call: Call) -> None:
    """``/alyx_prompt`` -- expand an idea with a text model, then render it."""
    idea = str(call.arg("idea", ""))
    await call.defer()
    try:
        drafted = await call.client.chat(
            call.settings.fallback_text_model,
            [ChatMessage(
                "user",
                "Rewrite this idea as one detailed image-generation prompt, "
                f"40 words maximum, no preamble:\n\n{idea}",
            )],
            max_tokens=120,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the raw idea
        await call.send(f"prompt writer failed: {call.failure(exc)}\n\nrendering the idea verbatim")
        drafted = None
    text = drafted.content.strip() if drafted else idea
    await render(call, text)
    await call.send(f"prompt used: {truncate(text, 300)}")


async def image_models(call: Call) -> None:
    """``/alyx_models`` -- the generator names this command accepts."""
    needle = str(call.arg("filter", "")).lower()
    pairs = offline_models("alyx", provider_instance("alyx"))
    rows = [(value, label) for value, label in pairs if not needle or needle in value.lower()]
    await call.send(table(rows, width=60))


async def image_probe(call: Call) -> None:
    """``/alyx_probe`` -- a tiny render that reports the response shape."""
    model = call.model("ai-image-generator")
    await call.defer()
    started = time.perf_counter()
    try:
        image = await call.client.image(
            f"{call.provider}/{model}", "a red circle on white", width=64, height=64
        )
    except ProviderError as exc:
        await call.send(f"`{model}`: {call.failure(exc)}")
        return
    rows = [
        ("generator", model),
        ("mime", image.mime_type or "(unknown)"),
        ("size", size_of(image)),
        ("delivery", "base64" if image.b64 else "url"),
        ("elapsed", f"{int((time.perf_counter() - started) * 1000)}ms"),
    ]
    await call.send(table(rows, width=50))


async def image_defaults(call: Call) -> None:
    """``/alyx_defaults`` -- sticky generator and size for this channel."""
    await defaults(call)


# --------------------------------------------------------------------------- shared


def truncate_text(text: str, limit: int = 1800) -> str:
    """Cut an answer for display without breaking a code fence silently."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…({len(text) - limit} more characters)"


def first_json(text: str) -> Any:
    """Pull the first JSON object out of a reply that may surround it with prose."""
    starter = text.find("{")
    if starter < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[starter:])
    except ValueError:
        return None
    return payload


#: Options that belong to the upstream call for one provider, for ``/info``.
def _provider_options(name: str) -> list[str]:
    from .params import PROVIDER_KNOBS

    return list(PROVIDER_KNOBS.get(name, ()))


CHAT: dict[str, Handler] = {
    "ask": ask,
    "stream": stream,
    "json": json_command,
    "code": code,
    "role": role,
    "translate": translate,
    "summarize": summarize,
    "continue": continue_answer,
    "rewrite": rewrite,
    "explain": explain,
    "brainstorm": brainstorm,
    "review": review,
    "extract": extract,
    "compare": compare,
    "probe": probe,
    "models": models,
    "info": info,
    "walk": walk,
    "session": session,
    "history": history,
    "remember": remember,
    "forget": forget,
    "defaults": defaults,
    "tokens": tokens,
}

IMAGES: dict[str, Handler] = {
    "generate": generate,
    "wide": wide,
    "tall": tall,
    "square": square,
    "hd": hd,
    "variation": variation,
    "seed_probe": seed_probe,
    "negative": negative,
    "anime": anime,
    "portrait": portrait,
    "logo": logo,
    "batch": batch,
    "compare": image_compare,
    "size_probe": size_probe,
    "retry": retry,
    "prompt": prompt_command,
    "models": image_models,
    "info": info,
    "probe": image_probe,
    "defaults": image_defaults,
    "session": session,
    "history": history,
    "forget": forget,
    "tokens": tokens,
}


def handler_for(provider: str, verb: str) -> Handler:
    """Return the implementation of ``/<provider>_<verb>``."""
    table_map = IMAGES if provider == "alyx" else CHAT
    return table_map[verb]
