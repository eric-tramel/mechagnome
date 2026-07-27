# Sessions

Every tool receives bounded access to durable session history and conversation
prompting plus revisioned annotations through `ctx.sessions` (`SessionAccess`).
The context is scoped to the session that started the current call tree:

```python
async def main(input, ctx):
    return {
        "caller_session_id": ctx.caller_session_id,
        "session_access_id": ctx.sessions.id,
    }
```

Those two IDs are equal. Nested tools receive new `ToolContext` objects but keep
the same durable session ID. A model completion requested by one of those tools
is different: it becomes its own durable child session. A model session bound
under that child can create grandchildren in the same way.

Sessions have immutable lineage and context metadata. `parent_session_id`
identifies the session tree, while `origin_session_id` and `origin_call_id`
identify the tool call that created a child. Forks additionally record
`context_source_session_id` and `context_through_seq`, an immutable snapshot
boundary. Kinds are `generic` for host/tool-only history, `conversation` for
every root or recursively launched agent, and `completion` for one accepted text-only
`await ctx.model_provider.complete(...)` call. `root_session_id` is derived by
walking the parent chain rather than stored separately.

## Session API

- `ctx.sessions.current(after=0, limit=50)` reads the current session.
- `ctx.sessions.read(session_id, after=0, limit=50)` reads any saved session.
- `ctx.sessions.list(limit=20, cursor=0)` lists saved sessions in reverse
  creation order.
- `ctx.sessions.metadata(session_id=None)` returns identity and lineage for the
  current or named session.
- `ctx.sessions.set_title(title, *, session_id=None, expected_revision=None)`
  sets or clears a short title. Use no more than four words.
- `ctx.sessions.set_description(description, *, session_id=None,
  expected_revision=None)` sets or clears a description.
- `ctx.sessions.get(session_id=None)` returns a `ToolSession` handle.
- `await ctx.sessions.inspect(job_id)` inspects a detached prompt job.

A `ToolSession` exposes `id`, `kind`, `parent_id`, `root_id`, `title`,
`description`, `origin_session_id`, `origin_call_id`, `metadata`, `read()`,
`update_metadata()`, and async `prompt()`. Prompting has three explicit modes:

```python
async def main(input, ctx):
    session = ctx.sessions.get(input.get("session_id"))
    return await session.prompt(
        input["prompt"],
        mode=input.get("mode", "continue"),
        detach=input.get("detach", False),
    )
```

- `continue` appends the userspace prompt to the same idle conversation.
- `spawn` creates a fresh child conversation with inherited lineage and toolbox
  scope but no inherited transcript.
- `fork` creates a child whose initial context is the source conversation
  through its latest completed turn.

Pair the returned session identity with metadata updates to make spawned or
forked work easy to recognize later:

```python
async def main(input, ctx):
    source = ctx.sessions.get(input.get("session_id"))
    outcome = await source.prompt(input["prompt"], mode="spawn")
    spawned = ctx.sessions.get(outcome["session_id"])
    metadata = spawned.update_metadata(
        title="Dependency investigation",
        description="Checks whether the proposed upgrade breaks callers.",
    )
    return {"outcome": outcome, "metadata": metadata}
```

`update_metadata()` accepts `title` and `description`; either may be cleared
with `None`, and `expected_revision=` provides optimistic concurrency. Updates
are restricted to the caller's session tree. The returned metadata and the
handle's `title`, `description`, and `metadata` snapshot reflect the update
immediately. Keep titles to no more than four words so they remain scannable;
this is usage guidance, not an enforced word limit.

When the labels are known before work starts, apply them atomically with the
prompt so invalid metadata cannot launch an unlabeled session:

```python
outcome = await source.prompt(
    input["prompt"],
    mode="spawn",
    metadata={"title": "Dependency investigation", "description": input["goal"]},
)
```

Foreground calls return `session_id`, `status`, and `result`. Detached calls
return `job_id`, `session_id`, and `status`; inspect the job later through
`ctx.sessions.inspect(job_id)`. A session cannot continue itself from one of
its active tool calls and returns `conversation_busy`. Spawn remains valid from
an active tool call because it prompts a different child session. Forks exclude
an unfinished active turn and snapshot the latest completed boundary.

`current` and `read` return this shape:

```json
{
  "session_id": "...",
  "session": {
    "id": "...",
    "parent_session_id": null,
    "root_session_id": "...",
    "kind": "conversation",
    "origin_session_id": null,
    "origin_call_id": null,
    "context_source_session_id": null,
    "context_through_seq": null,
    "title": "Dependency investigation",
    "description": "Checks whether the proposed upgrade breaks callers.",
    "annotation_revision": 2,
    "cwd": "/workspace",
    "created_at": "..."
  },
  "events": [],
  "next_after": null
}
```

Events are ordered by their stable, per-session `seq`. When `next_after` is an
integer, pass it as the next call's `after`; `None` means there are no more
events in that page. Limits are clamped to the range 1 through 100.

`list` returns:

```json
{
  "sessions": [
    {
      "id": "...",
      "cwd": "/workspace",
      "parent_session_id": null,
      "root_session_id": "...",
      "kind": "conversation",
      "origin_session_id": null,
      "origin_call_id": null,
      "context_source_session_id": null,
      "context_through_seq": null,
      "title": "Dependency investigation",
      "description": "Checks whether the proposed upgrade breaks callers.",
      "annotation_revision": 2,
      "created_at": "...",
      "event_count": 12
    }
  ],
  "next_cursor": null
}
```

Pass a non-null `next_cursor` back as `cursor` to continue listing. List limits
are also clamped to 1 through 100.

## Writing annotations

Both setters default to `ctx.sessions.id`; use `session_id=` for another session
in the same tree. `ToolSession.update_metadata()` can update one or both fields
through a handle. These APIs return the complete updated session metadata.
Strings are stored verbatim, including blank strings, and `None` clears the
selected field. A named target must already exist. Titles are limited to 256
UTF-8 bytes and descriptions to 4096 UTF-8 bytes. Titles should also contain no
more than four words, but that recommendation is intentionally not validated.

Each actual change atomically updates one field, increments
`annotation_revision`, and appends one `session_annotation_changed` event to the
target. Repeating the current value is a no-op. The event payload records the
field, old and new values, revision, actor session, and actor call; its structural
`call_id` fields remain `None` because a cross-session actor does not belong to
the target's call tree. Annotation events count toward `event_count`.

For optimistic concurrency, read `annotation_revision` and pass it back as
`expected_revision`. A stale revision raises `session_annotation_conflict` with
the expected and actual revisions so the caller can refetch and retry. The
revision check happens before no-op detection. Without a guard, concurrent
writes are serialized: different fields compose and same-field writes are
last-commit-wins. Exhausted database contention raises the retryable
`session_annotation_busy` error instead of a raw SQLite lock error.

An annotation is its own committed mutation. It remains durable if the authored
tool later fails or is cancelled.

## Reading context from events

A tool can inspect earlier calls in its current session without receiving the
whole history as input:

```python
def read_current_events(session_access):
    events = []
    after = 0
    while True:
        page = session_access.current(after=after, limit=100)
        events.extend(page["events"])
        after = page["next_after"]
        if after is None:
            return events


async def main(input, ctx):
    events = read_current_events(ctx.sessions)
    recent_tools = [
        event["tool_name"]
        for event in events
        if event["kind"] == "call_succeeded"
        and event.get("tool_name") is not None
    ]
    return {
        "session_id": ctx.caller_session_id,
        "recent_tools": recent_tools[-10:],
    }
```

To read every event from a known session, follow `next_after`:

```python
def read_all(session_access, session_id):
    events = []
    after = 0
    while True:
        page = session_access.read(session_id, after=after, limit=100)
        events.extend(page["events"])
        after = page["next_after"]
        if after is None:
            return events


async def main(input, ctx):
    return {
        "events": read_all(ctx.sessions, input["session_id"]),
    }
```

## Event fields and timing

Each event contains:

- `seq`, `kind`, `created_at`, and a JSON `payload`.
- `call_id` and `parent_call_id` for the nested call tree.
- `tool_name`, `tool_version`, and stable `tool_version_id` when applicable.
- `toolbox_id` when the event belongs to a toolbox.

Fields that do not apply to an event may be `None`. Common tool-call kinds are
`call_started`, `call_succeeded`, and `call_failed`; sessions also record model,
binding, invocation-scope, and toolbox-selection events.

User-initiated stopped rollouts record a `cancelled` event with `mode`,
`user_initiated`, and any incomplete visible model text in `partial_text`.
Partial text remains inspectable session data but is not replayed as a complete
assistant message. The next user prompt is stored in its raw form and transmitted
once with a `<user cancelled previous turn>` header.

Accepted one-shot model completions record `model_input`, then `model` and
`final`; provider failures record a sanitized `model_failed` terminal event.
Invalid requests rejected before provider dispatch do not create a session.
Each child inherits the exact cwd and ordered toolbox selection snapshotted for
its parent invocation.

A `call_started` event is committed before authored source begins. A tool that
calls `ctx.sessions.current()` can therefore see its own in-progress call and
its `parent_call_id`. The corresponding completion or failure event is committed
only after the source returns or raises. Session reads observe committed state at
the moment of the call; the context object does not push live updates.

Normal `call_succeeded` and `call_failed` payloads include `duration_ms`, measured
with a monotonic clock from after `call_started` commits through source loading,
`main`, nested calls, and result validation. The event's `tool_version_id`
attributes that sample to the resolved version. Nested durations are inclusive,
so parent and child durations must not be summed. Calls interrupted by timeout,
cancellation, or worker failure may have no terminal event or duration sample.

Session reads are not an authorization boundary: a tool can list saved sessions
and read one when it knows the ID. Prompt and annotation mutations are
restricted to the current session tree. Do not store secrets in session
payloads that authored code should not inspect or send to the configured model
provider.
