"""``python -m examples.bot`` -- run the bot, or validate the command tree with ``--check``."""

from __future__ import annotations

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
