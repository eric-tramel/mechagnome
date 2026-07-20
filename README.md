# Mechagnome

> [!WARNING]
> **Mechagnome is experimental research software. Run it only inside a
> disposable, tightly sandboxed environment such as an isolated VM or
> container.** Its agents create and execute arbitrary Python with filesystem,
> process, and network access. Do not run it directly on a workstation that
> contains credentials, sensitive files, or access to systems you care about.

A small proof of a metaprogrammable agent toolbox. The model sees only five
operations:

- `help`
- `search_tools`
- `read_tool_source`
- `write_tool`
- `call_tool`

It begins with no user-authored tools. During a session, an agent can write a
Python tool, call it immediately, find it again later, compose it from other
tools, and inspect current or historical sessions. The five core operations are
also stored as readable, immutable source versions and can be replaced through
`write_tool`.

The fixed part is deliberately small: SQLite storage, version resolution,
execution, event append, recursion limits, binding changes, and host rollback.
The behavior above that kernel is source the agent can inspect and rewrite.

## Run the agent

Set the same OpenRouter credential convention used by Qwen Code, then run the
program with no subcommand:

```bash
export OPENROUTER_API_KEY=...
uv run mechagnome
```

That opens the terminal UI with `z-ai/glm-5.2` as the default model. Responses
appear incrementally as OpenRouter streams them; fragmented tool calls are
assembled before execution. The main pane is the saved multi-turn conversation,
and the sidebar updates as the agent creates or replaces tools. State defaults to
`~/.local/share/mechagnome/toolbox.db`, so new TUI sessions share the same
toolbox and can inspect earlier sessions.

TUI commands:

- `Esc` stops the active model stream or tool rollout.
- `/new` starts a fresh saved conversation without clearing tools.
- `/tools` or `Ctrl+T` opens tool management. Select a tool and immutable
  version to inspect syntax-highlighted source, its diff from the prior version,
  creation provenance, per-version call outcomes, and calling sessions. User
  tools can be deleted from the active toolbox after confirmation; source,
  versions, usage, and session history remain available for audit.
- `/sessions` lists saved sessions.
- `/help` shows shortcuts; `/quit` exits.

Override the defaults with flags or environment variables:

```bash
uv run mechagnome \
  --model z-ai/glm-5.2 \
  --db .toolbox/toolbox.db
```

The recognized environment variables are `OPENROUTER_API_KEY`,
`MECHAGNOME_MODEL`, and `MECHAGNOME_DB`. The provider endpoint and
credential name are deliberately pinned to OpenRouter. OpenRouter uses its
normalized Chat Completions tool-calling interface; the request is identified
as `mechagnome` through `X-OpenRouter-Title`.

## Deterministic proof

The network-free demo remains available for inspecting the core mechanism:

```bash
uv run mechagnome --db .toolbox/demo.db demo
```

The deterministic demo writes and reuses an `add` tool, builds `double` by
calling `add`, reads its still-running session, replaces `search_tools`, rolls
that replacement back through the host kernel, and reopens the database to
show that state survived.

Inspect or repair bindings without relying on editable core code:

```bash
uv run mechagnome --db .toolbox/demo.db bindings
uv run mechagnome --db .toolbox/demo.db rollback search_tools 1
uv run mechagnome --db .toolbox/demo.db rollback call_tool 1
```

## Tool ABI

Every authored source program is a module containing one synchronous entry
point:

```python
def main(input, ctx):
    return {"result": input["value"] * 2}
```

`input` is a dictionary and the return value must be JSON-serializable. The
schema stored beside a user tool is descriptive metadata in this prototype; it
is not a complete JSON Schema enforcement engine.

Tools can compose through the active dispatcher:

```python
def main(input, ctx):
    return ctx.call_tool(
        "add",
        {"a": input["value"], "b": input["value"]},
    )
```

That call intentionally routes through the current source of `call_tool`.
Replacing `call_tool` can therefore change nested routing, caching, tracing, or
retry policy for the whole toolbox.

Tools can read durable sessions:

```python
def main(input, ctx):
    current = ctx.sessions.current(after=0, limit=50)
    previous = ctx.sessions.list(limit=20, cursor=0)
    older = ctx.sessions.read(input["session_id"], after=0, limit=50)
    return {"current": current, "previous": previous, "older": older}
```

A `call_started` event is committed before the source runs, so a tool reading
the current session sees its own in-progress call. Events use a stable
per-session sequence and include nested parent call IDs and resolved versions.

## Model adapter boundary

The TUI uses the bundled streaming `OpenRouterModel`, while `Harness` itself
remains provider-neutral. An adapter receives the accumulated messages and the
same five operation definitions on every turn, then returns a `ModelTurn`:

```python
from mechagnome import Harness, Kernel, ModelTurn, ToolCall


class MyModel:
    def respond(self, messages, tools):
        # Translate one provider response into ModelTurn.
        return ModelTurn(
            calls=(ToolCall("help", {"topic": "quickstart"}, "call-1"),)
        )


result = Harness(Kernel(".toolbox/toolbox.db")).run(
    MyModel(),
    "Build what you need to solve this task.",
)
```

For persistent interactive use, `Harness.start(model)` returns a `Conversation`;
each `send()` reuses its message history and durable session ID.

Adapters may additionally implement `stream(messages, tools)` and yield
`ModelStreamEvent(text_delta=...)` values followed by exactly one
`ModelStreamEvent(turn=...)`. The TUI displays transient deltas as they arrive,
then saves the canonical completed model turn; adapters with only `respond()`
continue to work unchanged.

The harness rejects model calls outside the five-operation surface. Dynamic
tools never need to be registered with the inference provider; they are reached
through the stable `call_tool(name, args, version=None)` envelope.

## Metacircular core

The outer names and schemas of the five operations are pinned so a provider can
keep one stable tool surface. Their descriptions, source, and behavior are
versioned like any other tool. Privilege comes from the logical core slot being
invoked:

| Slot | Low-level capability |
| --- | --- |
| `help` | none; its documentation lives in editable source |
| `search_tools` | enumerate active tool metadata |
| `read_tool_source` | read stored versions |
| `write_tool` | compile, store, and bind versions |
| `call_tool` | resolve and execute a version |

Copying the source of `search_tools` into a user tool does not give that copy
the catalog capability. Activating the same source in the `search_tools` slot
does. This is an API invariant, not an adversarial security boundary.

Every successful write creates and activates an immutable integer version in
one transaction, while `base_version` rejects stale binding updates. An
invocation resolves its version before executing, so a core tool can replace
itself: the running frame finishes on the old source and its next invocation
sees the new binding.

## Safety boundary

Mechagnome should be treated as code-execution research infrastructure, not as
a secured agent runtime. Use a disposable sandbox with a dedicated unprivileged
user, an empty or explicitly mounted working directory, outbound network limits,
and credentials scoped to the experiment. Destroy the environment after use.

Authored tools are arbitrary Python. Each model-requested call tree runs in a
fresh worker process with a small environment allowlist. This prevents direct
inheritance through the worker's `os.environ`; it is **not credential
separation**. The provider client and generated code run as the same OS user,
so on permissive systems a tool may inspect the parent process or its memory and
recover the OpenRouter key. Treat the experiment key as expendable and assume
agent-authored code can spend or exfiltrate it.

Nested tool calls remain together in the worker, and the host terminates the
process group after 120 seconds. Committed events are relayed to the TUI from
SQLite while the worker runs. Tools still retain ordinary filesystem, process,
and network access; they can corrupt the database, inspect files available to
the current OS user, consume resources, or attack other local processes through
facilities the OS permits. Context capabilities prevent accidental
architectural confusion, not hostile source. Nested depth and call-count bounds
supplement the worker timeout.

Run experiments in a disposable environment, use a tightly capped OpenRouter
key created only for that experiment, restrict its spend externally, and expose
only data you are prepared for generated code to access. Stronger credential
isolation requires an OS boundary such as a separate UID/namespace or an
external credential-holding broker; Mechagnome does not provide one.

The SQLite file is created owner-readable/writable only (`0600`). That protects
ordinary local privacy, but it does not defend against generated code running
under the same user account.

## Prior art

The toolbox loop is informed by systems that create, retrieve, validate, and
compose tools or skills online: [DynaSaur](https://arxiv.org/abs/2411.01747),
[Voyager](https://arxiv.org/abs/2305.16291),
[SkillWeaver](https://arxiv.org/abs/2504.07079),
[LLMs As Tool Makers](https://arxiv.org/abs/2305.17126),
[CREATOR](https://arxiv.org/abs/2305.14318),
[CRAFT](https://arxiv.org/abs/2309.17428),
[UCT](https://arxiv.org/abs/2602.01983), and
[MetaForge](https://arxiv.org/abs/2606.01801).

This prototype focuses specifically on the less explored combination of
editable core operations, slot-derived capabilities, live write-ahead session
access, and host-recoverable self-replacement.

## Development

```bash
uv run --group dev ruff format --check src tests
uv run --group dev ruff check src tests
uv run --group dev pytest
```
