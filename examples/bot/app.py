"""The Discord client, command attachment, sync policy, and the CLI entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from typing import Any

import discord
from discord import app_commands

from oinyx import OinyxClient

from .catalog import CatalogCache, provider_instance
from .config import ALL_PROVIDERS, COMMAND_LIMIT, Settings
from .factory import Runtime, build_commands
from .sessions import SessionStore
from .verbs import verbs_for

#: Discord's own ceiling per install target; discord.py enforces it client-side.
GUILD_COMMAND_LIMIT = COMMAND_LIMIT
GLOBAL_COMMAND_LIMIT = COMMAND_LIMIT

log = logging.getLogger("oinyx.bot")


def make_runtime(settings: Settings) -> Runtime:
    """Configure the adapters and build the objects every command shares."""
    unknown = [name for name in settings.providers if name not in ALL_PROVIDERS]
    if unknown:
        raise ValueError(
            f"unknown provider(s): {', '.join(unknown)}; known: {', '.join(ALL_PROVIDERS)}"
        )
    config: dict[str, Any] = {}
    for name in settings.providers:
        entry = dict(settings.provider_config.get(name, {}))
        entry.setdefault("timeout", settings.timeout)
        entry.setdefault("attempts", settings.attempts)
        entry.setdefault("backoff", settings.backoff)
        config[name] = entry
        provider_instance(name, entry)
    client = OinyxClient(config=config)
    catalog = CatalogCache(client, config)
    store = SessionStore(
        max_turns=settings.max_turns,
        max_chars=settings.max_chars,
        ttl=float(settings.session_ttl),
    )
    return Runtime(settings=settings, client=client, catalog=catalog, store=store)


def attach_commands(tree: app_commands.CommandTree, runtime: Runtime) -> int:
    """Install ``/<provider>_<verb>`` for every provider, spread over its targets.

    Each provider's 24 commands go to exactly one install target, because
    Discord caps a target at 100 commands; ``settings.install_plan`` decides who
    goes where.
    """
    added = 0
    for target, providers in runtime.settings.install_plan:
        guild = discord.Object(id=target) if target else None
        for provider in providers:
            commands = build_commands(runtime, provider)
            for command in commands:
                tree.add_command(command, guild=guild)
            added += len(commands)
            log.info("registered %s commands for %s -> %s", len(commands), provider,
                     f"guild {target}" if target else "global")
    return added


class OinyxBot(discord.Client):
    """A :class:`discord.Client` whose slash commands are generated per provider."""

    def __init__(self, settings: Settings, runtime: Runtime) -> None:
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.runtime = runtime
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        """Build the command tree before the gateway connection opens."""
        if not self.installed():
            attach_commands(self.tree, self.runtime)
        log.info(
            "%s slash commands across %s providers",
            len(self.installed()),
            len(self.settings.providers),
        )

    def installed(self) -> list[Any]:
        """Every command this client installed, across all targets."""
        commands: list[Any] = []
        for target, _ in self.settings.install_plan:
            guild = discord.Object(id=target) if target else None
            commands.extend(self.tree.get_commands(guild=guild) if guild else self.tree.get_commands())
        return commands

    async def on_ready(self) -> None:
        """Sync each install target the plan uses."""
        for target, providers in self.settings.install_plan:
            guild = discord.Object(id=target) if target else None
            expected = len(providers) * 24
            try:
                synced = await self.tree.sync(guild=guild)
            except discord.HTTPException as exc:
                log.error("sync for %s failed: %s", target or "global", exc)
                continue
            log.info(
                "synced %s/%s commands to %s (%s)",
                len(synced),
                expected,
                f"guild {target}" if target else "global",
                ", ".join(providers),
            )

    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        """Keep the gateway alive; every command already reports its own failures."""
        log.exception("unhandled error in %s", event_method, exc_info=sys.exc_info())


def build_bot(settings: Settings) -> OinyxBot:
    """Create the client with its runtime and command tree populated."""
    runtime = make_runtime(settings)
    bot = OinyxBot(settings, runtime)
    attach_commands(bot.tree, runtime)
    return bot


def command_names(settings: Settings) -> dict[str, list[str]]:
    """Return each provider's expected command names, for the check report."""
    return {
        provider: [f"{provider}_{verb}" for verb in verbs_for(provider)]
        for provider in settings.providers
    }


def report(settings: Settings) -> dict[str, Any]:
    """Build everything offline and describe the resulting command tree."""
    try:
        plan = settings.install_plan
    except ValueError as exc:
        # Same shape as the success branch below, so --check can print it either way.
        return {
            "providers": list(settings.providers),
            "targets": {},
            "commands_per_provider": {},
            "total": 0,
            "autocompleted_options": 0,
            "option_total": 0,
            "missing": sorted(name for group in command_names(settings).values() for name in group),
            "sample": {},
            "problems": [str(exc)],
        }
    bot = build_bot(settings)
    tree = bot.tree
    payload: list[dict[str, Any]] = []
    commands: list[Any] = []
    for target, _ in plan:
        guild = discord.Object(id=target) if target else None
        for command in tree.get_commands(guild=guild) if guild else tree.get_commands():
            commands.append(command)
            payload.append(command.to_dict(tree))

    per_provider: dict[str, int] = {}
    for command in commands:
        prefix = command.name.split("_", 1)[0]
        per_provider[prefix] = per_provider.get(prefix, 0) + 1

    problems: list[str] = []
    for command in commands:
        if len(command.name) > 32:
            problems.append(f"{command.name} is over the 32 character name limit")
        if not command.description:
            problems.append(f"{command.name} has no description")
    names = {command.name for command in commands}
    if len(names) != len(commands):
        problems.append("duplicate command names")

    sample = next((item for item in payload if item["name"] == "ovh_ask"), None)
    if sample is None:
        sample = payload[0] if payload else {}
    expected = {name for group in command_names(settings).values() for name in group}
    return {
        "providers": list(settings.providers),
        "targets": {
            (f"guild {target}" if target else "global"): list(providers) for target, providers in plan
        },
        "commands_per_provider": per_provider,
        "total": len(commands),
        "autocompleted_options": sum(
            1
            for item in payload
            for option in item.get("options", [])
            if option.get("autocomplete")
        ),
        "option_total": sum(len(item.get("options", [])) for item in payload),
        "missing": sorted(expected - names),
        "sample": sample,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m examples.bot`` -- run the bot, or validate the tree with ``--check``."""
    parser = argparse.ArgumentParser(prog="oinyx-bot", description="oinyx Discord bot")
    parser.add_argument("--token", help="Discord bot token (default: $OINXY_BOT_TOKEN)")
    parser.add_argument("--guild", type=int, action="append", default=[],
                        help="guild id to install into; repeat or comma-separate for more capacity")
    parser.add_argument("--providers", help="comma-separated provider allowlist")
    parser.add_argument("--check", action="store_true", help="build the command tree, print it, and exit")
    parser.add_argument("--json", action="store_true", help="print the check report as JSON")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.token:
        overrides["token"] = args.token
    if args.guild:
        overrides["guilds"] = tuple(args.guild)
    if args.providers:
        overrides["providers"] = tuple(
            part.strip() for part in args.providers.split(",") if part.strip()
        )
    settings = Settings.from_env()
    if overrides:
        settings = dataclasses.replace(settings, **overrides)

    logging.basicConfig(
        level="DEBUG" if args.verbose else getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.check:
        summary = report(settings)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"providers: {', '.join(summary['providers'])}")
            for target, providers in summary.get("targets", {}).items():
                print(f"  {target:<10} {len(providers) * 24:>3} commands  {', '.join(providers)}")
            for provider, count in sorted(summary["commands_per_provider"].items()):
                print(f"    {provider:<13} {count:>3}")
            print(
                f"total: {summary['total']} commands, "
                f"{summary['option_total']} options, "
                f"{summary['autocompleted_options']} autocompleted"
            )
            if summary["missing"] and summary["total"]:
                print("missing: " + ", ".join(summary["missing"]))
            for problem in summary["problems"]:
                print(f"PROBLEM: {problem}")
        return 1 if summary["problems"] or summary["missing"] else 0

    if not settings.token:
        parser.error("no token: set OINXY_BOT_TOKEN or pass --token (or use --check)")
    try:
        build_bot(settings).run(settings.token, log_handler=None)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
