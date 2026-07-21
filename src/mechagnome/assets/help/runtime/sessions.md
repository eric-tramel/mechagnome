# Sessions

Every tool receives bounded, read-only access to saved session events:

```python
current = ctx.sessions.current(after=0, limit=50)
saved = ctx.sessions.list(limit=20, cursor=0)
older = ctx.sessions.read(session_id, after=0, limit=50)
```

`ctx.sessions.current` reads the live session. `ctx.sessions.list` enumerates
saved sessions, and `ctx.sessions.read` reads any saved session by ID.

A `call_started` event is committed before tool source begins, so a tool reading
the current session can see its own in-progress call.
