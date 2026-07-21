# Sessions

Every tool receives bounded, read-only access to durable session history through
`ctx.sessions` (`SessionAccess`). The context is scoped to the session that
started the current call tree:

```python
def main(input, ctx):
    return {
        "caller_session_id": ctx.caller_session_id,
        "session_access_id": ctx.sessions.id,
    }
```

Those two IDs are equal. Nested tools receive new `ToolContext` objects but keep
the same durable session ID.

## Session API

- `ctx.sessions.current(after=0, limit=50)` reads the current session.
- `ctx.sessions.read(session_id, after=0, limit=50)` reads any saved session.
- `ctx.sessions.list(limit=20, cursor=0)` lists saved sessions in reverse
  creation order.

`current` and `read` return this shape:

```json
{
  "session_id": "...",
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
      "created_at": "...",
      "event_count": 12
    }
  ],
  "next_cursor": null
}
```

Pass a non-null `next_cursor` back as `cursor` to continue listing. List limits
are also clamped to 1 through 100.

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


def main(input, ctx):
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


def main(input, ctx):
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

A `call_started` event is committed before authored source begins. A tool that
calls `ctx.sessions.current()` can therefore see its own in-progress call and
its `parent_call_id`. The corresponding completion or failure event is committed
only after the source returns or raises. Session reads observe committed state at
the moment of the call; the context object does not push live updates.

Session access is read-only but not an authorization boundary: a tool can list
saved sessions and read one when it knows the ID. Do not store secrets in
session payloads that authored code should not inspect.
