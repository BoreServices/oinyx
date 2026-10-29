"""Local oinyx provider plugins.

Importing this package registers every bundled provider::

    import providers              # fable5, kai, perchance, alyx, fai
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("kai/auto", "hello")
    image = await OinyxClient().image("alyx/ai-image-generator", "a red fox")
"""

from __future__ import annotations

from .alyx import AlyxProvider
from .fable5io import Fable5Provider
from .fai import FaiProvider
from .kai import KaiProvider
from .perchance import PerchanceProvider

__all__ = ["AlyxProvider", "Fable5Provider", "FaiProvider", "KaiProvider", "PerchanceProvider"]
