"""The 24 verbs per provider, declared as arguments plus the handler that runs them."""

from __future__ import annotations

from dataclasses import dataclass, field

from .params import KNOBS

#: Only the image adapter uses the image verb set.
IMAGE_VERBS_PROVIDER = ("alyx",)


@dataclass(frozen=True)
class Verb:
    """One command: what it says about itself, and the options it takes."""

    name: str
    description: str
    args: tuple[str, ...] = field(default_factory=tuple)
    #: Providers that get a different description, because the result differs.
    notes: dict[str, str] = field(default_factory=dict)

    def describe(self, provider: str) -> str:
        """Return the command description for ``provider``."""
        return self.notes.get(provider, self.description)


def _verb(name: str, description: str, *args: str, **notes: str) -> Verb:
    return Verb(name=name, description=description, args=args, notes=notes)


#: The chat command set. ``knobs`` splices in the provider's own sampling flags.
CHAT: dict[str, Verb] = {
    verb.name: verb
    for verb in (
        _verb("ask", "One-shot completion", "prompt", "model", KNOBS),
        _verb(
            "stream",
            "Watch the answer arrive by editing one message",
            "prompt", "model", KNOBS, "edit_every",
            perchance="Perchance returns one body; the edits are local",
            aihorde="Crowdsourced workers stream in bursts, and queue first",
        ),
        _verb(
            "json",
            "Force a JSON object out of the model",
            "prompt", "model", "fields", KNOBS,
        ),
        _verb(
            "code",
            "Write code, fenced",
            "prompt", "model", "suffix", KNOBS,
            fable5="fable5 replies in plain text, so fences arrive unlabelled",
        ),
        _verb(
            "role",
            "Answer as a persona for one turn",
            "persona", "prompt", "model", KNOBS,
        ),
        _verb(
            "translate",
            "Translate text",
            "text", "target", "model", KNOBS,
        ),
        _verb(
            "summarize",
            "Summarize text",
            "text", "model", "words", KNOBS,
        ),
        _verb(
            "continue",
            "Continue this channel's last answer",
            "model", "direction", KNOBS,
        ),
        _verb("rewrite", "Rewrite text in a different voice", "text", "style", "model", KNOBS),
        _verb(
            "explain",
            "Explain something to a chosen audience",
            "prompt", "audience", "model", KNOBS,
        ),
        _verb(
            "brainstorm",
            "List ideas",
            "prompt", "count", "model", KNOBS,
        ),
        _verb(
            "review",
            "Review a code snippet",
            "code", "focus", "model", KNOBS,
        ),
        _verb(
            "extract",
            "Pull facts out of text",
            "text", "fields", "model", KNOBS,
        ),
        _verb(
            "compare",
            "Ask two models the same question",
            "prompt", "model", "model_b", KNOBS,
        ),
        _verb(
            "probe",
            "Ping the upstream and report what it says",
            "model", "all",
        ),
        _verb(
            "models",
            "List the models this command will accept",
            "live", "filter",
            kai="Kilo's free tier moves; live reads the leaderboard gateway",
            fable5="fable5 publishes no catalog; the list is the table in the adapter",
            perchance="Generator names come from config, not a catalog endpoint",
            alyx="Generator names come from config, not a catalog endpoint",
        ),
        _verb(
            "info",
            "Endpoint, default model, limits, and the known traps",
        ),
        _verb(
            "walk",
            "Try the stillworks registry until some endpoint answers",
            "prompt", "limit",
        ),
        _verb(
            "session",
            "Show this channel's stored context",
        ),
        _verb(
            "history",
            "Print the stored transcript",
            "last",
        ),
        _verb(
            "remember",
            "Add a standing instruction to this channel",
            "note",
        ),
        _verb(
            "forget",
            "Drop stored turns",
            "turns",
        ),
        _verb(
            "defaults",
            "Set this channel's default model and length cap",
            "model", "max_tokens",
        ),
        _verb(
            "tokens",
            "Estimate tokens locally, no network call",
            "text", "report",
        ),
    )
}

#: The image command set for ``alyx``.
IMAGE: dict[str, Verb] = {
    verb.name: verb
    for verb in (
        _verb("generate", "Generate an image", "prompt", "model", KNOBS),
        _verb("wide", "Generate a landscape image", "prompt", "model", KNOBS),
        _verb("tall", "Generate a portrait-orientation image", "prompt", "model", KNOBS),
        _verb("square", "Generate a square image", "prompt", "model", KNOBS),
        _verb("hd", "Generate at the largest preset", "prompt", "model", KNOBS),
        _verb("variation", "Generate the same prompt at several seeds", "prompt", "count", "model", KNOBS),
        _verb("seed_probe", "Generate at one seed and report what stuck", "prompt", "seed", "model", KNOBS),
        _verb("negative", "Generate with a negative prompt", "prompt", "negative_prompt", "model", KNOBS),
        _verb("anime", "Generate with the anime generator", "prompt", KNOBS),
        _verb("portrait", "Generate with the portrait generator", "prompt", KNOBS),
        _verb("logo", "Generate with the logo generator", "prompt", KNOBS),
        _verb("batch", "Generate several prompts at once", "prompts", "model", KNOBS),
        _verb("compare", "Render one prompt on two generators", "prompt", "generator_b", KNOBS),
        _verb("size_probe", "Render at three sizes to see what the generator honours", "prompt", "model", KNOBS),
        _verb("retry", "Regenerate this channel's last prompt", "model", KNOBS),
        _verb("prompt", "Expand an idea into a prompt, then render it", "idea", "model", KNOBS),
        _verb("models", "List the generators this command will accept", "filter"),
        _verb("info", "Endpoint, default generator, size, and the Cloudflare caveat"),
        _verb("probe", "Render a test image and report the response shape", "model"),
        _verb("defaults", "Set this channel's default generator and size", "model", "width", "height"),
        _verb("session", "Show this channel's stored image state"),
        _verb("history", "Print the stored generation transcript", "last"),
        _verb("forget", "Drop stored generations", "turns"),
        _verb("tokens", "Estimate the prompt's token count, no network call", "text", "report"),
    )
}


def verbs_for(provider: str) -> dict[str, Verb]:
    """Return the 24 command definitions for ``provider``."""
    return IMAGE if provider in IMAGE_VERBS_PROVIDER else CHAT


#: The two verb sets, by name, for reporting and validation.
CHAT_VERBS: tuple[str, ...] = tuple(CHAT)
IMAGE_VERBS: tuple[str, ...] = tuple(IMAGE)
