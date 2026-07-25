# Sessions

Every tool receives bounded access to durable session history and fixed mutable
annotations through `ctx.sessions` (`SessionAccess`). The context is scoped to
the session that started the current call tree:

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

Sessions have immutable `kind`, `parent_session_id`, and `origin_call_id`
metadata. Kinds are `generic` for host/tool-only history, `conversation` for
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
  sets or clears a title.
- `ctx.sessions.set_description(description, *, session_id=None,
  expected_revision=None)` sets or clears a description.

`current` and `read` return this shape:

```json
{
  "session_id": "...",
  "session": {
    "id": "...",
    "parent_session_id": null,
    "root_session_id": "...",
    "kind": "conversation",
    "origin_call_id": null,
    "title": "Investigate dependencies",
    "description": null,
    "annotation_revision": 1,
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
      "origin_call_id": null,
      "title": "Investigate dependencies",
      "description": null,
      "annotation_revision": 1,
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

Both setters default to `ctx.sessions.id`; use `session_id=` to target any other
existing saved session. They return the complete updated session metadata.
Strings are stored verbatim, including blank strings, and `None` clears the
selected field. A named target must already exist.

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

Session access is not an authorization boundary: trusted authored tools can
list, read, and annotate any saved session whose ID they know. Do not store
secrets in session payloads that authored code should not inspect.
