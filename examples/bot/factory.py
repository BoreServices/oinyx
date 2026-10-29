"""The command factory: 24 verbs x N providers, each with its own option set."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import discord
from discord import app_commands

from oinyx import OinyxClient, OinyxError

from .catalog import CatalogCache, choices
from .config import Settings
from .context import Call
from .handlers import Handler, handler_for
from .params import KNOBS, Param, knobs_for, model_param, param_for
from .rendering import error_text
from .sessions import SessionStore
from .verbs import Verb, verbs_for

log = logging.getLogger("oinyx.bot")

#: Discord's hard limit on slash command names.
MAX_NAME = 32


@dataclass
class Runtime:
    """Everything the generated commands share at call time."""

    settings: Settings
    client: OinyxClient
    catalog: CatalogCache
    store: SessionStore


def resolve_params(provider: str, verb: Verb) -> list[Param]:
    """Expand one verb's argument list into the options for ``provider``.

    ``knobs`` splices in the sampling flags that provider actually forwards, and
    every model-shaped option is marked for autocomplete.
    """
    resolved: list[Param] = []
    seen: set[str] = set()
    for name in verb.args:
        candidates: list[Param]
        if name == KNOBS:
            candidates = list(knobs_for(provider))
        elif name in ("model", "model_b", "generator_b"):
            candidates = [model_param(provider, name)]
        else:
            candidates = [param_for(name)]
        for candidate in candidates:
            if candidate.name not in seen:
                seen.add(candidate.name)
                resolved.append(candidate)
    return resolved


def build_callback(runtime: Runtime, provider: str, verb_name: str, params: list[Param]) -> Any:
    """Create a coroutine whose declared signature is exactly ``params``.

    Discord reads the callback's signature to build the option list, so the
    signature is synthesised here while the body stays a single dispatcher.
    """

    async def callback(interaction: discord.Interaction, **kwargs: Any) -> None:
        await dispatch(runtime, provider, verb_name, interaction, kwargs)

    declared = [
        inspect.Parameter(
            "interaction",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=discord.Interaction,
        )
    ]
    descriptions: dict[str, str] = {}
    for param in params:
        declared.append(
            inspect.Parameter(
                param.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=param.annotated(),
                default=inspect.Parameter.empty if param.required else None,
            )
        )
        descriptions[param.name] = param.description
    callback.__signature__ = inspect.Signature(parameters=declared)  # type: ignore[attr-defined]
    callback.__discord_app_commands_param_description__ = descriptions  # type: ignore[attr-defined]
    callback.__name__ = f"{provider}_{verb_name}"
    callback.__qualname__ = callback.__name__
    return callback


def autocomplete_for(runtime: Runtime, provider: str) -> Any:
    """Build the autocomplete coroutine that lists one provider's models."""

    async def complete(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        return choices(await runtime.catalog.models(provider), current)

    complete.__name__ = f"complete_{provider}"
    complete.__qualname__ = complete.__name__
    return complete


def build_commands(runtime: Runtime, provider: str) -> list[app_commands.Command[Any, Any, Any]]:
    """Return the 24 commands for ``provider``."""
    verbs = verbs_for(provider)
    built: list[app_commands.Command[Any, Any, Any]] = []
    for name, verb in verbs.items():
        command_name = f"{provider}_{name}"
        if len(command_name) > MAX_NAME:
            raise ValueError(f"{command_name!r} exceeds Discord's {MAX_NAME} character limit")
        params = resolve_params(provider, verb)
        callback = build_callback(runtime, provider, name, params)
        command: app_commands.Command[Any, Any, Any] = app_commands.Command(
            name=command_name,
            description=verb.describe(provider)[:100],
            callback=callback,
        )
        for param in params:
            if param.autocomplete:
                command.autocomplete(param.name)(autocomplete_for(runtime, provider))
        built.append(command)
    return built


async def dispatch(
    runtime: Runtime,
    provider: str,
    verb_name: str,
    interaction: discord.Interaction,
    kwargs: dict[str, Any],
) -> None:
    """Resolve state for one invocation and hand it to the verb's handler."""
    args = {key: value for key, value in kwargs.items() if value is not None}
    session = runtime.store.get(getattr(interaction.channel, "id", 0), provider)
    call = Call(
        interaction=interaction,
        provider=provider,
        verb=verb_name,
        args=args,
        settings=runtime.settings,
        client=runtime.client,
        catalog=runtime.catalog,
        store=runtime.store,
        session=session,
    )
    handler: Handler = handler_for(provider, verb_name)
    try:
        await handler(call)
    except (OinyxError, discord.DiscordException) as exc:
        await report(call, exc)
    except Exception as exc:  # noqa: BLE001 - a slash command must never die quietly
        log.exception("%s/%s crashed", provider, verb_name)
        await report(call, exc)


async def report(call: Call, exc: Exception) -> None:
    """Send a failure back to the user without masking the original error."""
    log.warning("%s/%s failed: %s", call.provider, call.verb, exc)
    try:
        await call.send(error_text(exc), ephemeral=True)
    except Exception:  # noqa: BLE001 - the interaction may already be gone
        log.warning("could not report %s to Discord", exc)
