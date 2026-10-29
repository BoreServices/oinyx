"""Per-channel conversation state: stored turns, standing notes, and defaults."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from oinyx import ChatMessage


@dataclass
class Session:
    """Everything the bot remembers about one channel and one provider."""

    channel_id: int
    provider: str
    turns: list[ChatMessage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    last_prompt: str = ""
    updated: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        """Mark the session as used now."""
        self.updated = time.monotonic()

    def stale(self, ttl: float) -> bool:
        """Whether this session has been idle for longer than ``ttl`` seconds."""
        return time.monotonic() - self.updated > ttl

    def messages(self, prompt: str | None = None) -> list[ChatMessage]:
        """Return the conversation to send: notes, history, then ``prompt``.

        Notes become one system turn so every request carries them, and the
        trailing prompt is appended without being stored yet.
        """
        history = list(self.turns)
        if self.notes:
            system = ChatMessage(
                "system", "Standing instructions for this channel:\n- " + "\n- ".join(self.notes)
            )
            history = [system, *history]
        if prompt:
            history.append(ChatMessage("user", prompt))
        return history

    def record(self, user: str, answer: str) -> None:
        """Store one exchange and remember it as the latest prompt."""
        self.turns.append(ChatMessage("user", user))
        self.turns.append(ChatMessage("assistant", answer))
        self.last_prompt = user
        self.touch()

    def transcript(self, last: int = 10) -> list[ChatMessage]:
        """Return the most recent ``last`` stored turns."""
        return self.turns[-last:] if last else list(self.turns)


class SessionStore:
    """Bounded, expiring session state keyed by channel and provider."""

    def __init__(self, max_turns: int = 12, max_chars: int = 6000, ttl: float = 3600.0) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.ttl = ttl
        self._sessions: dict[tuple[int, str], Session] = {}

    def get(self, channel_id: int, provider: str) -> Session:
        """Return the live session for this channel and provider, pruning idle ones."""
        self.prune()
        key = (channel_id, provider)
        session = self._sessions.get(key)
        if session is None:
            session = Session(channel_id=channel_id, provider=provider)
            self._sessions[key] = session
        return session

    def append(self, session: Session, user: str, answer: str) -> None:
        """Store an exchange and keep the history inside the configured budget."""
        session.record(self.clip(user), self.clip(answer))
        overflow = len(session.turns) - self.max_turns * 2
        if overflow > 0:
            del session.turns[:overflow]

    def clip(self, text: str) -> str:
        """Cut ``text`` down to the per-message character budget."""
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + "\n[truncated by the bot]"

    def forget(self, session: Session, turns: int | None) -> int:
        """Drop the last ``turns`` messages, or everything when omitted."""
        removed = len(session.turns)
        if not turns or turns >= removed:
            session.turns.clear()
        else:
            del session.turns[-turns:]
        session.touch()
        return removed - len(session.turns)

    def prune(self) -> int:
        """Expire idle sessions and report how many went away."""
        stale = [key for key, value in self._sessions.items() if value.stale(self.ttl)]
        for key in stale:
            del self._sessions[key]
        return len(stale)

    def count(self) -> int:
        """Number of live sessions."""
        return len(self._sessions)
