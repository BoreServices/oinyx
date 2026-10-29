"""Local oinyx provider plugins.

Importing this package registers every bundled provider::

    import providers              # fable5, kai, perchance, alyx, fai, + the relays
    from oinyx import OinyxClient

    reply = await OinyxClient().chat("kai/auto", "hello")
    image = await OinyxClient().image("alyx/ai-image-generator", "a red fox")

The relay adapters (``llm7``, ``ovh``, ``pollinations``, ``freeai``,
``aihorde``, ``keylessai``) all subclass :class:`providers.OpenAiRelay`;
``stillworks`` is not a provider but a discovery registry that says which of
them is answering right now.
"""

from __future__ import annotations

from .aihorde import AIHordeProvider
from .alyx import AlyxProvider
from .fable5io import Fable5Provider
from .fai import FaiProvider
from .freeai import FreeAiProvider
from .kai import KaiProvider
from .keylessai import KeylessAiProvider
from .llm7 import Llm7Provider
from .openai_relay import OpenAiRelay
from .ovh import OvhProvider
from .perchance import PerchanceProvider
from .pollinations import PollinationsProvider

__all__ = [
    "AIHordeProvider",
    "AlyxProvider",
    "Fable5Provider",
    "FaiProvider",
    "FreeAiProvider",
    "KaiProvider",
    "KeylessAiProvider",
    "Llm7Provider",
    "OpenAiRelay",
    "OvhProvider",
    "PerchanceProvider",
    "PollinationsProvider",
]
