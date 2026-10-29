"""Local oinyx provider plugins.

Importing this package registers every bundled provider::

    import providers                     # registers "fable5" and "kai"
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("kai/auto", "hello")
"""

from __future__ import annotations

from .fable5io import Fable5Provider
from .kai import KaiProvider

__all__ = ["Fable5Provider", "KaiProvider"]
