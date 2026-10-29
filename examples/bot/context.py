"""The per-invocation context every handler receives, and its reply plumbing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import discord

from oinyx import ChatMessage, OinyxClient

from .catalog import CatalogCache, default_model
from .config import Settings
from .rendering import chunks
from .sessions import Session, SessionStore

log = logging.getLogger("oinyx.bot")


@dataclass
class Call:
    """Everything a handler needs: the interaction, the provider, and the state."""

    interaction: discord.Interaction
    provider: str
    verb: str
    args: dict[str, Any]
    settings: Settings
    client: OinyxClient
    catalog: CatalogCache
    store: SessionStore
    session: Session
    started: float = field(default_factory=time.perf_counter)
    _acked: bool = False
    _sent: bool = False

    @property
    def channel_id(self) -> int:
        """The channel this command ran in; ``0`` when unavailable."""
        return getattr(self.interaction.channel, "id", 0)

    @property
    def elapsed_ms(self) -> int:
        """Milliseconds since the command was invoked."""
        return int((time.perf_counter() - self.started) * 1000)

    def arg(self, name: str, default: Any = None) -> Any:
        """Read one resolved option."""
        value = self.args.get(name)
        return default if value is None else value

    async def defer(self, *, thinking: bool = True, ephemeral: bool = False) -> None:
        """Acknowledge the interaction once, tolerating a late caller."""
        if self._acked or self.interaction.response.is_done():
            return
        await self.interaction.response.defer(
            thinking=thinking and not ephemeral, ephemeral=ephemeral
        )
        self._acked = True

    async def send(
        self,
        content: str | Iterable[str],
        *,
        file: discord.File | None = None,
        files: list[discord.File] | None = None,
        embed: discord.Embed | None = None,
        ephemeral: bool = False,
    ) -> None:
        """Reply in one or more messages, deferring first if nothing was sent yet.

        ``content`` may be a single string, which is split on line boundaries, or
        an iterable of already-split parts. Attachments and the embed ride on the
        first message only.
        """
        await self.defer(thinking=not ephemeral, ephemeral=ephemeral)
        if isinstance(content, str):
            parts = chunks(content) or [""]
        else:
            parts = list(content)
        attachments = [item for item in (file, *(files or ())) if item is not None]
        for index, part in enumerate(parts):
            kwargs: dict[str, Any] = {"content": part[:1994] or None}
            if index == 0:
                if embed is not None:
                    kwargs["embed"] = embed
                if attachments:
                    kwargs["files"] = attachments
            if self._sent or self.interaction.response.is_done():
                await self.interaction.followup.send(**kwargs)
            else:
                await self.interaction.response.send_message(ephemeral=ephemeral, **kwargs)
            self._sent = True

    async def say(self, text: str, *, ephemeral: bool = False) -> Any:
        """Send one short message and return it, so a caller can edit it later.

        ``Any`` rather than ``discord.Message`` because discord.py's own stubs
        return a union here; the object is only ever passed back to :meth:`edit`.
        """
        await self.defer(thinking=not ephemeral, ephemeral=ephemeral)
        sent: Any
        if self._sent or self.interaction.response.is_done():
            # ``wait=True`` is not optional: without it the webhook returns None
            # and the caller has nothing to edit.
            sent = await self.interaction.followup.send(text[:1994], ephemeral=ephemeral, wait=True)
        else:
            sent = await self.interaction.response.send_message(text[:1994], ephemeral=ephemeral)
        self._sent = True
        return self._editable(sent)

    @staticmethod
    def _editable(sent: Any) -> Any:
        """Return the thing that can be edited.

        discord.py 2.7 gives ``response.send_message`` back as an
        :class:`InteractionCallbackResponse` wrapper; the editable payload lives on
        its ``resource``. A webhook message is already editable, so it passes
        through untouched.
        """
        resource = getattr(sent, "resource", None)
        return resource if resource is not None else sent

    async def edit(self, message: Any, text: str) -> None:
        """Progressively edit a previously sent message."""
        try:
            await message.edit(content=text[:1994])
        except discord.HTTPException as exc:
            log.debug("edit skipped: %s", exc)

    def messages(self, prompt: str, *, context: bool = True) -> list[ChatMessage]:
        """Return the conversation to send for ``prompt``.

        ``context`` pulls the channel's stored turns in; commands that work on
        supplied text always pass ``False`` so a paste does not become a reply to
        the last three unrelated messages.
        """
        if not context:
            return [ChatMessage("user", self.store.clip(prompt))]
        history = self.session.messages(self.store.clip(prompt))
        return fit(self.provider, history)

    def knob_kwargs(self, *, json_mode: bool = False) -> dict[str, Any]:
        """Convert resolved options into the kwargs this provider's adapter reads."""
        return build_kwargs(self.provider, self.args, json_mode=json_mode)

    def model(self, default: str = "") -> str:
        """Return the requested model: option, then channel default, then adapter default."""
        return str(
            self.arg("model")
            or self.session.defaults.get("model")
            or default
            or default_model(self.provider)
        )

    def failure(self, exc: Exception) -> str:
        """Log and render an exception for Discord."""
        from .rendering import error_text

        log.warning("%s/%s failed: %s", self.provider, self.verb, exc, exc_info=exc)
        return error_text(exc)


#: Per-provider message-count and size ceilings, enforced before the call.
LIMITS: dict[str, tuple[int, int]] = {
    "fable5": (32, 8000),
    "kai": (64, 20000),
    "perchance": (24, 12000),
    "fai": (50, 12000),
}
DEFAULT_LIMIT = (24, 16000)

#: Option renames, because each adapter spells its knobs its own way.
RENAMES: dict[str, dict[str, str]] = {
    "perchance": {"start_with": "startWith", "stop": "stopSequences"},
    "alyx": {"start_with": "startWith", "stop": "stopSequences", "negative_prompt": "negativePrompt"},
}

#: Options that shape the command rather than the upstream request.
LOCAL_ONLY = {
    "model", "model_b", "generator_b", "live", "all", "filter", "report", "edit_every",
    "turns", "last", "clear", "json", "text", "prompt", "code", "idea", "prompts", "note",
    "target", "persona", "style", "fields", "audience", "focus", "direction", "suffix",
    "count", "words", "limit",
}


def fit(provider: str, messages: list[ChatMessage]) -> list[ChatMessage]:
    """Trim a conversation to the provider's turn count and per-message size."""
    turns, size = LIMITS.get(provider, DEFAULT_LIMIT)
    clipped = [ChatMessage(message.role, message.content[:size]) for message in messages]
    if len(clipped) <= turns:
        return clipped
    head = [message for message in clipped if message.role == "system"]
    tail = [message for message in clipped if message.role != "system"][-turns:]
    return head + tail


def build_kwargs(provider: str, args: dict[str, Any], *, json_mode: bool = False) -> dict[str, Any]:
    """Map command options onto the keyword arguments the adapter forwards."""
    renames = RENAMES.get(provider, {})
    kwargs: dict[str, Any] = {}
    for name, value in args.items():
        if value is None or name in LOCAL_ONLY:
            continue
        kwargs[renames.get(name, name)] = value
    if "stop" in kwargs and isinstance(kwargs["stop"], str):
        stops = [part.strip() for part in kwargs["stop"].split(",") if part.strip()]
        kwargs["stop"] = stops
    if json_mode:
        if provider == "perchance" or provider == "fai":
            kwargs.pop("response_format", None)
        elif provider != "alyx":
            kwargs["response_format"] = {"type": "object"}
    return kwargs
