"""Environment-driven settings for the bot, and the per-provider oinyx config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

#: Names registered by :mod:`providers`; the image adapter is listed separately.
CHAT_PROVIDERS = (
    "fable5",
    "kai",
    "perchance",
    "fai",
    "llm7",
    "pollinations",
    "ovh",
    "aihorde",
    "freeai",
    "keylessai",
)
IMAGE_PROVIDERS = ("alyx",)
ALL_PROVIDERS = CHAT_PROVIDERS + IMAGE_PROVIDERS

#: Discord caps an install target at 100 commands, so one target holds four
#: providers (96). Widen this by listing guilds in ``OINXY_BOT_GUILDS``: three
#: targets carry all eleven providers.
DEFAULT_PROVIDERS = ("kai", "llm7", "ovh", "pollinations")

#: Verbs are registered as ``/<provider>_<verb>``; Discord caps names at 32.
MAX_COMMAND_NAME = 32

#: Commands per install target. Discord allows 200 per guild, but discord.py
#: guards both global and per-guild installs at 100, so that is the real limit.
COMMAND_LIMIT = 100

#: Every adapter ships 24 commands, which is what the install plan budgets for.
COMMANDS_PER_PROVIDER = 24


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


def _num(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def _text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _guilds() -> tuple[int, ...]:
    """Read ``OINXY_BOT_GUILD`` and ``OINXY_BOT_GUILDS``, deduplicated in order."""
    raw = ",".join([_text("OINXY_BOT_GUILD"), _text("OINXY_BOT_GUILDS")])
    seen: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if value.isdigit() and int(value) not in seen:
            seen.append(int(value))
    return tuple(seen)


@dataclass(frozen=True)
class Settings:
    """Everything the bot needs, read once from the environment."""

    token: str = ""
    guilds: tuple[int, ...] = ()
    providers: tuple[str, ...] = DEFAULT_PROVIDERS
    max_turns: int = 12
    max_chars: int = 6000
    timeout: float = 90.0
    attempts: int = 2
    backoff: float = 1.0
    session_ttl: int = 3600
    fallback_text_model: str = "pollinations/fast"
    edit_every: int = 1200
    log_level: str = "INFO"
    provider_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from ``OINXY_BOT_*`` and per-provider ``*_USER_AGENT`` vars."""
        return cls(
            token=_text("OINXY_BOT_TOKEN"),
            guilds=_guilds(),
            providers=_csv("OINXY_BOT_PROVIDERS", DEFAULT_PROVIDERS),
            max_turns=_num("OINXY_BOT_MAX_TURNS", 12),
            max_chars=_num("OINXY_BOT_MAX_CHARS", 6000),
            timeout=float(_num("OINXY_BOT_TIMEOUT", 90)),
            attempts=_num("OINXY_BOT_ATTEMPTS", 2),
            backoff=float(_num("OINXY_BOT_BACKOFF", 1)),
            session_ttl=_num("OINXY_BOT_SESSION_TTL", 3600),
            fallback_text_model=_text("OINXY_BOT_TEXT_FALLBACK", "pollinations/fast"),
            edit_every=_num("OINXY_BOT_EDIT_EVERY", 1200),
            log_level=_text("OINXY_BOT_LOG_LEVEL", "INFO"),
            provider_config=provider_config_from_env(),
        )

    @property
    def command_count(self) -> int:
        """How many slash commands this configuration registers."""
        return len(self.providers) * COMMANDS_PER_PROVIDER

    @property
    def install_plan(self) -> list[tuple[int | None, tuple[str, ...]]]:
        """Spread the providers over guilds so no install target is over budget.

        One entry per guild, in order; ``(None, providers)`` when the bot is told
        to install globally. Raises when the configured set cannot fit, because
        the failure is a configuration mistake rather than something to recover
        from at runtime.
        """
        capacity = max(1, COMMAND_LIMIT // COMMANDS_PER_PROVIDER)
        remaining = list(self.providers)
        plan: list[tuple[int | None, tuple[str, ...]]] = []
        targets: list[int | None] = list(self.guilds) or [None]
        for target in targets:
            block, remaining = tuple(remaining[:capacity]), remaining[capacity:]
            if block:
                plan.append((target, block))
        if remaining:
            raise ValueError(
                f"{len(self.providers)} providers need {len(self.providers) * COMMANDS_PER_PROVIDER} "
                f"commands but {len(targets)} install target(s) fit "
                f"{len(targets) * capacity}; add guilds to OINXY_BOT_GUILDS or trim providers"
            )
        return plan


def provider_config_from_env() -> dict[str, dict[str, Any]]:
    """Merge the shared HTTP limits with the browser session each site demands.

    Cookies are only ever read from the environment; nothing here belongs in the
    repository.
    """
    per_provider: dict[str, dict[str, Any]] = {}
    for name in ALL_PROVIDERS:
        entry: dict[str, Any] = {"attempts": 2, "backoff": 1.0}
        prefix = name.upper()
        agent = _text(f"{prefix}_USER_AGENT")
        if agent:
            entry["user_agent"] = agent
        cookies = _text(f"{prefix}_COOKIES")
        if cookies:
            entry["cookies"] = cookies
        api_key = _text(f"{prefix}_API_KEY")
        if api_key:
            entry["api_key"] = api_key
        per_provider[name] = entry
    return per_provider
