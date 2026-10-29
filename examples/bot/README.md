# oinyx Discord bot

A discord.py bot that exposes **every bundled adapter as its own command family**:
24 slash commands per provider, named `/<provider>_<verb>`, each with a `model`
option backed by autocomplete and only the options that provider actually
honours.

```
/kai_ask  /kai_stream  /kai_json ... /kai_tokens      (24)
/ovh_ask  /ovh_stream  /ovh_json ... /ovh_tokens      (24)
/alyx_generate /alyx_wide ... /alyx_tokens            (24)
```

Nothing here is installed with the library — `examples/` is a consumer of
`oinyx` and `providers/`, the same way a third-party plugin package would be.

## Install and run

```bash
uv pip install --python .venv/bin/python -r examples/bot/requirements.txt
export OINXY_BOT_TOKEN=...           # from the Discord developer portal
export OINXY_BOT_GUILDS=123456789    # one guild id per install target, see below
.venv/bin/python -m examples.bot
```

Validate the whole tree without a token or a network call first:

```bash
.venv/bin/python -m examples.bot --check                 # default provider set
.venv/bin/python -m examples.bot --check --json          # machine-readable
.venv/bin/python -m examples.bot --check --providers alyx,kai --guild 1 --guild 2
```

`--check` builds the command tree exactly as the bot would, serialises every
command with `Command.to_dict`, and exits non-zero when a name is too long, a
description is missing, a command is not registered, or the set does not fit the
configured install targets.

## The command budget decides how many providers you may run

Discord caps an install target at 100 slash commands, and discord.py enforces it
client-side for global *and* per-guild installs. Twenty-four commands per
provider therefore means **four providers per target**. `Settings.install_plan`
spreads the configured providers over `OINXY_BOT_GUILDS` in order and raises a
readable error when they cannot fit:

| Providers | Targets needed |
| --- | --- |
| 1–4 | 1 (or global) |
| 5–8 | 2 |
| 9–11 | 3 |

Add test guilds to `OINXY_BOT_GUILDS` (comma-separated) to widen capacity; a
provider is never split across two targets, so `/ovh_*` always lives in one place.
The bot syncs only the targets it planned, and guild sync is near-instant — keep
it that way rather than installing globally.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `OINXY_BOT_TOKEN` | -- | required to run; not required for `--check` |
| `OINXY_BOT_GUILD`, `OINXY_BOT_GUILDS` | empty | install targets, empty means global |
| `OINXY_BOT_PROVIDERS` | `kai,llm7,ovh,pollinations` | which adapters get commands |
| `OINXY_BOT_MAX_TURNS` | 12 | stored turns per channel and provider |
| `OINXY_BOT_MAX_CHARS` | 6000 | characters of history kept before clipping |
| `OINXY_BOT_SESSION_TTL` | 3600 | seconds before a channel's session expires |
| `OINXY_BOT_TIMEOUT` | 90 | per-request timeout, seconds |
| `OINXY_BOT_ATTEMPTS` / `OINXY_BOT_BACKOFF` | 2 / 1.0 | retry policy handed to each adapter |
| `OINXY_BOT_TEXT_FALLBACK` | `pollinations/fast` | text model used by image commands that polish a prompt first |
| `OINXY_BOT_EDIT_EVERY` | 1200 | characters streamed between edits of the placeholder message |
| `OINXY_BOT_LOG_LEVEL` | `INFO` | logging level |

Per adapter, `<PROVIDER>_API_KEY`, `<PROVIDER>_COOKIES` and
`<PROVIDER>_USER_AGENT` are read straight into that adapter's config — for
example `PERCHANCE_COOKIES`, `FAI_USER_AGENT`, `LLM7_API_KEY`.

**Cookies are only ever read from the environment.** A `cf_clearance` value is a
live session for your browser profile; it must never be committed, pasted into a
command, or written to a file in this repository.

## What the commands do

Twenty-four chat verbs, identical across the ten text adapters:

`ask stream json code role translate summarize continue rewrite explain
brainstorm review extract compare probe models info walk session history
remember forget defaults tokens`

`/alyx` gets twenty-four image verbs instead: `generate wide tall square hd
variation seed_probe negative anime portrait logo batch compare size_probe
retry prompt models info probe defaults session history forget tokens`.

Shared behaviour:

- **Context** — `ask`, `stream`, `json`, `code`, `role` and friends keep a
  per-channel, per-provider transcript (`/x_session`, `/x_history`,
  `/x_remember` for standing notes, `/x_forget` to drop them). Verbs that work on
  text you supply (`summarize`, `translate`, `extract`, ...) deliberately ignore
  history so a paste is not answered as a reply to last week's chat.
- **Provenance** — every answer is stamped with the provider, the model asked
  for, the model that actually served it when they differ, latency, token counts,
  and how much of the reply was reasoning rather than answer.
- **Limits** — messages are trimmed to what the endpoint accepts (32 turns for
  `fable5`, 64 for `kai`, ...) and long answers are split at line boundaries with
  code fences closed and reopened, so a truncated fence cannot swallow the rest.
- **`walk`** — hands the prompt to `providers.stillworks`, which tries models in
  rank order until one answers. That is the escape hatch for the per-IP throttling
  these free relays all practise.
- **`probe`** — fires the same short prompt at several models on one host and
  reports which answered, useful the morning a table goes stale.

## Autocomplete

The `model` option (and `model_b` for `compare`, `generator_b` for alyx's
second generator) is autocompleted from the same table the adapters ship:

- pinned aliases always, so completion works with no network and no quota;
- for `kai` the aliases are `KAI_MODELS` minus the `kai/` prefix;
- for the relays the aliases render as `alias -> upstream id`;
- `/<provider>_models --live` merges the upstream catalog into the cached view, so
  the live ids show up in completion for the rest of the cache TTL (300s).

A failed live read is swallowed: autocomplete must never surface an upstream
outage or block on one.

## Provider-specific notes the commands encode

| Provider | Options it gets | Why only those |
| --- | --- | --- |
| `fable5` | `temperature top_p max_tokens` | the endpoint may ignore sampling knobs entirely; no penalties exist on it |
| `kai` | `+ stop seed` | real SSE streaming, OpenAI-shaped body |
| `perchance` | `start_with stop temperature` | the model name is a *generator*; `startWith`/`stopSequences` are camelCase; streaming is emulated locally |
| `fai` | `tone format time_zone follow_ups resend max_tokens` | instruction-shaped API; `json` refuses `response_format` because it cannot honour it |
| `llm7` | full OpenAI set incl. `json`, `user` | `tier=turbo` models need a key; the anonymous tier answers the rest |
| `pollinations` | full OpenAI set | the anonymous tier is one model thin; 503s arrive as JSON |
| `ovh` | full OpenAI set | 2 requests/minute **per model**, ids are Title-Case, reasoning arrives as `content: null` |
| `aihorde` | full OpenAI set | crowdsourced workers, queueing, so the adapter uses one attempt with a 300s timeout |
| `freeai` | full OpenAI set | most of its ~469-row catalog is a paid marketplace; only `self_hosted` ids answer |
| `keylessai` | full OpenAI set | the host never resolved from this machine, so its table is unverified |
| `alyx` | `width height seed negative_prompt start_with` | image-only; results carry a url or inline base64, never text; Cloudflare-gated like `perchance` |

`perchance`, `alyx` and `fai` sit behind a Cloudflare managed challenge: without a
browser-supplied `cf_clearance` plus the matching user agent they raise
`CloudflareChallengeError`, and the bot shows that guidance as an ephemeral reply
instead of pretending the answer was empty.

## Layout

| File | Role |
| --- | --- |
| `config.py` | env-driven `Settings`, and `install_plan` (the 100-command arithmetic) |
| `params.py` | every option, its bounds, its help string, and which provider gets which |
| `verbs.py` | the 24 chat verbs and 24 image verbs, declaratively |
| `factory.py` | turns `verb × provider` into a real `app_commands.Command` |
| `catalog.py` | provider specs, the never-raising `CatalogCache`, choice building |
| `context.py` | the `Call` object handlers receive, plus provider limits and renames |
| `handlers.py` | the 48 verb implementations |
| `rendering.py` | chunking, provenance stamps, tables, image attachments |
| `sessions.py` | per-channel transcripts, notes, sticky defaults |
| `stillworks_bridge.py` | `walk` and the registry table |
| `app.py` | client, sync policy, CLI (`--check`) |

`factory.py` is the interesting one: discord.py derives a command's parameters
from `inspect.signature(callback)`, so each generated command gets a synthesized
signature declaring exactly its own options, and one shared dispatcher implements
all of them.
