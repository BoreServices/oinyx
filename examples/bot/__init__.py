"""A discord.py front end for the adapters in :mod:`providers`.

One command set per provider, generated from a declarative spec: every verb is
registered as ``/<provider>_<verb>`` with a ``model`` parameter backed by that
provider's own catalog autocomplete, plus the sampling knobs the adapter
actually reads.
"""

from __future__ import annotations

from .app import OinyxBot, build_bot, main

__all__ = ["OinyxBot", "build_bot", "main"]
