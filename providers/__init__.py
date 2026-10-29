"""Local oinyx provider plugins.

Importing this package registers every bundled provider::

    import providers                     # registers "fable5"
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("fable5/openai/gpt-4o", "hello")
"""

from __future__ import annotations

from .fable5io import Fable5Provider

__all__ = ["Fable5Provider"]
