"""Parameter descriptors: one definition per knob, assembled per provider.

``model`` is always autocompleted from the adapter's own catalog, and the
sampling knobs a command exposes are exactly the ones that adapter forwards --
``fable5`` never advertises ``presence_penalty``, Perchance never advertises
``top_p``, and the OpenAI relays get the whole passthrough set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from discord.app_commands import Range

#: Marker inside a verb's argument list that splices in the provider's knobs.
KNOBS = "knobs"


@dataclass(frozen=True)
class Param:
    """One slash-command option: name, Discord type, bounds, and help text."""

    name: str
    kind: str = "str"
    description: str = "value"
    required: bool = False
    lo: float | None = None
    hi: float | None = None
    default: Any = None
    autocomplete: bool = False

    def annotated(self) -> Any:
        """Return the annotation Discord's param compiler expects."""
        base: Any
        if self.kind == "int":
            base = self._range(int)
        elif self.kind == "number":
            base = self._range(float)
        elif self.kind == "bool":
            base = bool
        else:
            base = str
        return base if self.required else Optional[base]

    def _range(self, kind: type) -> Any:
        if self.lo is None and self.hi is None:
            return kind
        return Range[kind, self.lo, self.hi]

    def described(self, text: str) -> Param:
        """Return a copy carrying a provider-specific help string."""
        return replace(self, description=text)


def _p(name: str, kind: str = "str", required: bool = False, **kw: Any) -> Param:
    return Param(name=name, kind=kind, required=required, **kw)


#: Every knob and verb-local argument the bot knows how to render.
PARAMS: dict[str, Param] = {
    # -- model selection -----------------------------------------------------
    "model": _p("model", "str", description="alias or upstream id; blank uses the default"),
    "model_b": _p("model_b", "str", required=True, description="second model to compare against"),
    "generator_b": _p(
        "generator_b", "str", required=True, description="second generator to compare against"
    ),
    # -- prompts -------------------------------------------------------------
    "prompt": _p("prompt", required=True, description="what to ask", lo=None, hi=None),
    "text": _p("text", required=True, description="the text to work on"),
    "code": _p("code", required=True, description="the code to review"),
    "idea": _p("idea", required=True, description="a short idea to expand"),
    "prompts": _p("prompts", required=True, description="prompt|prompt|prompt"),
    "note": _p("note", required=True, description="a fact to keep in this channel's context"),
    "target": _p("target", required=True, description="language to translate into"),
    "persona": _p("persona", required=True, description="who the model should answer as"),
    "style": _p("style", description="the tone to rewrite into"),
    "fields": _p("fields", description="comma-separated keys to extract"),
    "audience": _p("audience", description="who the explanation is for"),
    "focus": _p("focus", description="what to prioritise in the review"),
    "direction": _p("direction", description="how the continuation should go"),
    "suffix": _p("suffix", description="extra instruction appended to the prompt"),
    # -- sampling knobs ------------------------------------------------------
    "temperature": _p(
        "temperature", "number", description="sampling temperature", lo=0.0, hi=2.0
    ),
    "top_p": _p("top_p", "number", description="nucleus sampling", lo=0.0, hi=1.0),
    "max_tokens": _p("max_tokens", "int", description="answer length cap", lo=1, hi=8192),
    "stop": _p("stop", "str", description="stop sequences, comma-separated"),
    "seed": _p("seed", "int", description="deterministic sampling seed", lo=-1, hi=2**31 - 1),
    "presence_penalty": _p(
        "presence_penalty", "number", description="presence penalty", lo=-2.0, hi=2.0
    ),
    "frequency_penalty": _p(
        "frequency_penalty", "number", description="frequency penalty", lo=-2.0, hi=2.0
    ),
    "user": _p("user", "str", description="opaque user id the relay may log-balance on"),
    "json_mode": _p("json", "bool", description="ask for machine-readable JSON"),
    # -- Perchance / alyx knobs ---------------------------------------------
    "start_with": _p("start_with", "str", description="text the generator must begin with"),
    "width": _p("width", "int", description="image width in pixels", lo=64, hi=1536),
    "height": _p("height", "int", description="image height in pixels", lo=64, hi=1536),
    "negative_prompt": _p("negative_prompt", "str", description="what to keep out of the image"),
    # -- freeaiforever knobs -------------------------------------------------
    "tone": _p("tone", "str", description="site tone option, e.g. friendly or professional"),
    "output_format": _p("format", "str", description="site format option, e.g. paragraph or bullets"),
    "time_zone": _p("time_zone", "str", description="IANA zone the site stamps answers with"),
    "follow_ups": _p("follow_ups", "bool", description="ask the site for suggested follow-ups"),
    "resend": _p("resend", "bool", description="replay the previous answer instead of asking anew"),
    # -- verb-local options --------------------------------------------------
    "count": _p("count", "int", description="how many results to ask for", lo=1, hi=12),
    "words": _p("words", "int", description="rough word budget for the summary", lo=10, hi=800),
    "limit": _p("limit", "int", description="how many candidates to try", lo=1, hi=12),
    "last": _p("last", "int", description="how many turns to show", lo=1, hi=50),
    "turns": _p("turns", "int", description="how many turns to drop; blank clears everything", lo=0, hi=50),
    "live": _p("live", "bool", description="ask the upstream catalog instead of the local table"),
    "all": _p("all", "bool", description="probe every alias, not just the default"),
    "filter": _p("filter", "str", description="case-insensitive name filter"),
    "report": _p("report", "bool", description="include the per-message breakdown"),
    "edit_every": _p(
        "edit_every", "int", description="characters between message edits", lo=200, hi=4000
    ),
    "clear": _p("clear", "bool", description="reset the stored context first"),
}

#: The knobs each adapter provably forwards, in display order.
PROVIDER_KNOBS: dict[str, tuple[str, ...]] = {
    "fable5": ("temperature", "top_p", "max_tokens"),
    "kai": ("temperature", "top_p", "max_tokens", "stop", "seed"),
    "perchance": ("start_with", "stop", "temperature"),
    "fai": ("tone", "output_format", "time_zone", "follow_ups", "resend", "max_tokens"),
    "llm7": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "pollinations": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "ovh": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "aihorde": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "freeai": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "keylessai": (
        "temperature", "top_p", "max_tokens", "stop", "seed",
        "presence_penalty", "frequency_penalty", "user", "json_mode",
    ),
    "alyx": ("width", "height", "seed", "negative_prompt", "start_with"),
}

#: Help text that differs per provider, because the endpoints differ.
KNOBS_NOTE: dict[str, dict[str, str]] = {
    "fable5": {
        "temperature": "forwarded verbatim; the endpoint replies in plain text",
        "max_tokens": "forwarded verbatim; fable5 may ignore sampling knobs entirely",
        "model": "fable5 serves openai/gpt-4o and reports it in x-chat-model",
    },
    "kai": {
        "model": "KAI_MODELS alias such as auto or nemo3-ultra; unknown ids pass through",
        "max_tokens": "gateway answer cap; reasoning models eat it first",
    },
    "perchance": {
        "model": "Perchance generator name, e.g. ai-chat or ai-code-generator",
        "temperature": "forwarded to the generator, which usually ignores it",
        "stop": "stopSequences for the generator, comma-separated",
    },
    "fai": {
        "model": "aifreeforever slug, e.g. gemini-3-5; the Referer header follows it",
        "max_tokens": "site option max_tokens; the page decides the real cap",
    },
    "ovh": {"model": "OVH_MODELS alias or the exact Title-Case upstream id"},
    "llm7": {"model": "LLM7_MODELS alias; pro-tier ids are answered with 401"},
    "pollinations": {
        "model": "POLLINATIONS_MODELS alias; the anonymous tier is one model deep"
    },
    "aihorde": {
        "model": "AIHORDE_MODELS alias or a full worker id like koboldcpp/Name",
        "seed": "crowdsourced workers may ignore seeds entirely",
    },
    "freeai": {"model": "FREEAI_MODELS alias; only self_hosted rows answer"},
    "keylessai": {"model": "KEYLESSAI_MODELS alias; this host was never reachable here"},
    "alyx": {
        "model": "Perchance image generator, e.g. ai-anime-image-generator",
        "seed": "forwarded to the generator; support varies per generator",
    },
}


def param_for(name: str) -> Param:
    """Look up a parameter definition by the name used in a verb's argument list."""
    return PARAMS[name]


def knobs_for(provider: str) -> tuple[Param, ...]:
    """Return the provider's knob set with provider-specific help applied."""
    notes = KNOBS_NOTE.get(provider, {})
    return tuple(
        PARAMS[key].described(notes[key]) if key in notes else PARAMS[key]
        for key in PROVIDER_KNOBS[provider]
    )


def model_param(provider: str, name: str = "model") -> Param:
    """Build the autocompleted model parameter for ``provider``."""
    notes = KNOBS_NOTE.get(provider, {})
    base = PARAMS["model"] if name == "model" else PARAMS[name]
    description = notes.get("model", base.description)
    return replace(base, name=name, description=description, autocomplete=True)
