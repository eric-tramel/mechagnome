# Authoring tools

An authored tool is a Python module stored with a name, description, input
schema, and immutable version. Its source must define an asynchronous `main`
entry point with exactly two positional parameters:

```python
async def main(input, ctx):
    return {"result": input["value"] * 2}
```

Helper functions, classes, constants, and imports may live in the same source
module. Mechagnome executes the module in a fresh namespace for every call, so
module globals do not persist between invocations.

## Inputs, schemas, and results

`input` is the JSON object supplied as the tool's `args`. Validate values whose
type, range, or combination matters to the implementation:

```python
async def main(input, ctx):
    values = input.get("values")
    if not isinstance(values, list) or not values:
        return {"error": "values must be a nonempty array"}
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return {"error": "every value must be a number"}
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
    }
```

The `input_schema` saved by `write_tool` is descriptive metadata in this
prototype, not a complete runtime validator. Make it accurate because
`search_tools` and callers use it to discover and construct calls. For the
example above, a useful schema is:

```json
{
  "type": "object",
  "properties": {
    "values": {
      "type": "array",
      "items": {"type": "number"},
      "minItems": 1
    }
  },
  "required": ["values"],
  "additionalProperties": false
}
```

`main` may return any JSON-serializable value: an object, array, string, finite
number, boolean, or `None`. Do not return Python-only values such as `Path`,
`set`, bytes, generators, exceptions, or a context object. An unhandled
exception becomes a failed tool-call event; use a JSON error object when a
domain-level failure is an expected result callers should inspect.

`main` must use `async def`. Await every async context operation before using or
returning its result; an un-awaited coroutine is not JSON-serializable. Ordinary
synchronous Python is still valid inside `main`, but blocking I/O prevents that
tool from doing other async work while it waits. Prefer async libraries for I/O.
Tools persisted by older Mechagnome versions with a synchronous `main` continue
to run through a compatibility context, but `write_tool` rejects new synchronous
definitions. Do not copy that legacy form into new or replacement tools.

## The `ctx` context object

Every invocation receives its own `ctx` (`ToolContext`). It is a runtime handle,
not data to save or return. Its public interface is:

- `await ctx.call_tool(name, args, version=None)` — invoke another tool.
- `ctx.caller_session_id` — the durable session ID for this call tree.
- `ctx.sessions` — a bounded, read-only `SessionAccess` object.
- `ctx.model_provider` — a bounded text-completion capability supplied by the
  host.
- `ctx.kernel` — a narrow capability available only to the matching core slot.

Treat names beginning with `_` as private implementation details. A nested tool
gets a new context object for its own invocation. It shares the durable session
and snapshotted toolbox selection with its caller, while receiving its own call
identity and nesting depth.

### Calling another tool

Pass a JSON object as the nested tool's arguments:

```python
async def main(input, ctx):
    doubled = await ctx.call_tool(
        "add",
        {"a": input["value"], "b": input["value"]},
    )
    return {"doubled": doubled}
```

Omitting `version` resolves the active binding in the current toolbox scope.
Pass an integer to require one immutable version:

```python
async def main(input, ctx):
    return await ctx.call_tool("normalize", input, version=3)
```

Calls route through the currently active `call_tool` implementation. This lets
that core tool provide shared routing, tracing, caching, or retry behavior. All
nested calls count toward the call tree's depth and call-count limits, so avoid
unbounded recursion and large fan-out loops. See `help(topic="composition")`
for versioning, write-then-call, and dispatch details.

### Requesting a model completion

`ctx.model_provider` lets authored tools request text from the model provider
configured by the host. Supply the complete message sequence for the nested
request; the outer conversation, agent system prompt, and model-facing tools are
not added automatically.

```python
async def main(input, ctx):
    answer = await ctx.model_provider.complete([
        {
            "role": "system",
            "content": "Answer accurately in one short sentence.",
        },
        {
            "role": "user",
            "content": input["question"],
        },
    ])
    return {"answer": answer}
```

`complete(messages)` is asynchronous and returns a string when awaited. It
accepts between 1 and 64 messages. Every message must contain exactly two
string fields:

- `role`, which must be `system`, `user`, or `assistant`.
- `content`, which contains the text for that message.

The interface does not accept model names, sampling parameters, tools, images,
or streaming callbacks. Those policies belong to the host provider. The bundled
OpenRouter provider requests text-only output and caps it at 2,048 tokens.

The complete message request is limited to 256 KiB of encoded JSON. Provider
results have a 1 MiB UTF-8 limit and also cross a bounded JSON transport frame,
so heavily escaped text can reach the transport limit earlier. One top-level
tool call tree may make at most eight model-provider attempts. Nested tools get
the same provider capability and share that call-count limit:

```python
async def main(input, ctx):
    drafts = []
    for audience in input["audiences"]:
        drafts.append(await ctx.model_provider.complete([
            {
                "role": "user",
                "content": (
                    f"Explain {input['topic']} for this audience: {audience}"
                ),
            },
        ]))
    return {"drafts": drafts}
```

Keep fan-out within the shared eight-attempt limit. This is not a monetary,
input-token, or total-output-token budget; a request may incur provider charges
even if it is later cancelled or its response is rejected. Enforce model and
spend limits with the host provider as well.

Every accepted `complete()` call creates a distinct durable `completion`
session. Its `parent_session_id` is the current agent session and its
`origin_call_id` identifies the exact tool invocation that requested it. The
child records `model_input`, `model`, and `final` events (or a sanitized
`model_failed` event), appears in the normal session list, and can be read by
its own ID. Invalid requests rejected before dispatch consume the shared attempt
budget but do not create a child session.

The capability can raise
`invalid_model_request` when request validation rejects message shape or size,
`model_provider_limit` when the call or response-size limit is exceeded,
`model_provider_failed` when the host provider fails, and
`model_provider_unavailable` when the harness did not configure a provider.
`model_provider_protocol` reports a broken worker-to-host connection. Provider
failures are deliberately sanitized, so authored source should not depend on
provider-specific exception details.

### Running a child agent

Use `run_agent(prompt)` when the delegated work may need tools or further
delegation:

```python
async def main(input, ctx):
    answer = await ctx.model_provider.run_agent(
        f"Investigate this task and return a concise answer: {input['task']}"
    )
    return {"answer": answer}
```

The call is asynchronous and returns the child agent's final text when awaited.
Before any model traffic, the provider creates a durable `conversation` session
parented to the caller's session and attributed to the current tool call. Child
agents receive the same bounded capability, so a child calling `run_agent()`
creates a grandchild with the child session as its direct parent. All model and
tool events are recorded under the session in which they actually occur.

These failures raise `mechagnome.ToolboxError`; inspect its stable `code` only
when the tool has a meaningful fallback, and re-raise everything else:

```python
from mechagnome import ToolboxError


async def main(input, ctx):
    try:
        answer = await ctx.model_provider.complete([
            {"role": "user", "content": input["question"]},
        ])
    except ToolboxError as error:
        if error.code == "model_provider_unavailable":
            return {
                "answer": None,
                "reason": "no model provider is configured",
            }
        raise
    return {"answer": answer}
```

The isolated worker receives a socket-backed proxy, not the concrete provider,
endpoint, or credential. This keeps those details out of the normal context API,
but it grants authored tools authority to spend the configured model account. It
is credential opacity, not credential separation: authored code still runs as
the provider client’s OS user and must be treated as trusted code.

Tool model access must be authorized explicitly by the host. Passing only a raw
root transport to `Harness` does not expose a coincidental `complete()` method
to authored tools; pass a completion transport explicitly or construct a
`ModelProvider` intentionally.

The proxy is also an authenticated data-egress capability: every supplied
message is sent to the configured provider, and authored code can place data
read from files or saved sessions into those messages. Do not pass sensitive
data to `complete()` unless disclosure to that provider is explicitly intended.

### Reading the current session

`ctx.sessions` is a `SessionAccess` object scoped to the caller's durable
session. `ctx.sessions.id` and `ctx.caller_session_id` are the same ID.
`ctx.sessions.metadata()` returns that session's kind, direct parent, derived
root, origin call, cwd, and creation time. Pass another saved ID to inspect its
lineage.

```python
async def main(input, ctx):
    page = ctx.sessions.current(after=0, limit=100)
    failures = [
        {
            "seq": event["seq"],
            "tool": event.get("tool_name"),
            "error": event["payload"],
        }
        for event in page["events"]
        if event["kind"] == "call_failed"
    ]
    return {
        "session_id": ctx.caller_session_id,
        "failures": failures,
        "next_after": page["next_after"],
    }
```

A `call_started` event is committed before source begins, so
`ctx.sessions.current()` can see the tool's own in-progress call. Its matching
`call_succeeded` or `call_failed` event does not exist until after `main`
returns or raises.

### Listing and paging through saved sessions

The session context exposes three read-only methods:

- `ctx.sessions.current(after=0, limit=50)` reads this call's session.
- `ctx.sessions.read(session_id, after=0, limit=50)` reads a saved session.
- `ctx.sessions.list(limit=20, cursor=0)` lists saved sessions, newest first.

`current` and `read` return `session_id`, `events`, and `next_after`. Pass the
returned `next_after` as the next page's `after`; a value of `None` means the
page is complete. `list` returns `sessions` and `next_cursor` and uses the same
pattern with `cursor`.

```python
def read_all_events(session_access, session_id):
    events = []
    after = 0
    while True:
        page = session_access.read(session_id, after=after, limit=100)
        events.extend(page["events"])
        after = page["next_after"]
        if after is None:
            return events


async def main(input, ctx):
    sessions = ctx.sessions.list(limit=10, cursor=0)["sessions"]
    if not sessions:
        return {"events": []}
    session_id = input.get("session_id", sessions[0]["id"])
    return {"events": read_all_events(ctx.sessions, session_id)}
```

Event dictionaries contain `seq`, `kind`, `created_at`, `payload`, and tracing
fields such as `call_id`, `parent_call_id`, `tool_name`, `tool_version`,
`toolbox_id`, and `tool_version_id`. Fields that do not apply to an event may be
`None`. See `help(topic="sessions")` for response shapes and event semantics.

### Core-only kernel capabilities

Ordinary tools must not use `ctx.kernel`. Access raises `capability_denied`.
Instead, call a core operation through `ctx.call_tool`, just as the following
ordinary tool does:

```python
async def main(input, ctx):
    return await ctx.call_tool("search_tools", {
        "query": input["query"],
        "include_core": False,
    })
```

Editable core implementations receive one narrow capability according to the
logical slot in which they run:

| Core slot | Allowed context method |
| --- | --- |
| `help` | none |
| `search_tools` | `ctx.kernel.catalog(include_core=True)` |
| `view_tool` | `ctx.kernel.view_tool(name, version=None)` |
| `write_tool` | `ctx.kernel.write_tool(...)` |
| `call_tool` | `await ctx.kernel.execute(name, args, version=None)` |

For example, a replacement `search_tools` implementation can inspect the
effective catalog:

```python
async def main(input, ctx):
    query = input["query"].lower()
    matches = []
    for tool in ctx.kernel.catalog(
        include_core=input.get("include_core", True)
    ):
        text = f'{tool["name"]} {tool["description"]}'.lower()
        if query in text:
            matches.append({
                "name": tool["name"],
                "description": tool["description"],
            })
    return {"items": matches}
```

Copying this source into an ordinary tool does not grant catalog access, and a
core slot cannot use another slot's capability. Privilege follows the active
logical slot, not the source text.

## Imports, state, and execution safety

Imports, filesystem access, network clients, and subprocesses are ordinary
Python capabilities. Relative paths resolve from the durable session's launch
working directory; selecting another toolbox namespace does not change it.
Persist durable tool state explicitly in a suitable file or service rather than
in module globals, and account for concurrent calls when updating shared state.

Model-requested call trees run in a filtered-environment subprocess with depth,
call-count, and time bounds. Required Git configuration and SSH authentication
variables are preserved so authored tools can use the harness's Git access.
This is operational containment, not credential separation or a hostile-code
sandbox. Authored code still has the OS access of the provider client, and
`ctx.model_provider` intentionally grants model-spend authority. Run Mechagnome
only in a disposable, tightly isolated environment with expendable credentials.

## Authoring checklist

Before writing or replacing a tool:

1. Search for an existing reusable tool with `search_tools`.
2. Give the tool a specific name and a description that states when to use it.
3. Provide an accurate object-shaped input schema.
4. Define `async def main(input, ctx)`, await async context calls, and return
   only JSON data.
5. Use the public context APIs; never depend on private `ctx._...` attributes.
6. Prefer small composed tools and pin nested versions only when reproducibility
   matters more than following the active binding.
7. Read the active source and pass `base_version` when updating an existing
   tool, then call the new version immediately with a representative input.
