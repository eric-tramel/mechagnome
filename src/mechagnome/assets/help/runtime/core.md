# Core operations

The five base operations are ordinary readable tool versions:

- `help`
- `search_tools`
- `view_tool`
- `write_tool`
- `call_tool`

Their names and outer schemas are fixed, but their descriptions, source, and
behavior are editable. Privilege follows the active logical core slot, not
copied source text. Copying `search_tools` source into an ordinary user tool,
for example, does not grant catalog access.

A host-only rollback command can recover a broken core binding.

## Detached `call_tool` jobs

The model-facing `call_tool` operation can return before a long-running tool
finishes. Start a process-lifetime background job with:

```json
{"name": "slow_report", "args": {"path": "data.csv"}, "detach": true}
```

The immediate result is `{"job_id": "...", "status": "running"}`. The agent
can continue with other operations and later inspect it through the same core
operation:

```json
{"job_id": "..."}
```

Inspection returns `status`, a bounded merged stdout/stderr `output_tail`, and
`truncated`. A successful terminal response also contains `result` (including
an explicit `null`); a failed response contains a structured `error`.

Detached jobs are owned by the current host process. They do not survive an app
restart, at most four run concurrently, and app shutdown stops unfinished jobs.
The latest 64 completed handles remain inspectable; older completed handles are
evicted. Each result is limited to 1 MiB; an oversized result becomes a
structured `detached_result_too_large` failure. There is no public per-job
cancel operation. Escape stops only the
foreground rollout, while clearing or ending a TUI session hides its detached
rows without cancelling the underlying jobs as long as the app remains open.
Ending the final tab exits the app and triggers shutdown. Programmatic `Harness`
owners must call `close()` to stop background work.

The bounded output tail may contain sensitive data written by the tool.
Inspecting a handle supplies that tail to the model and therefore records it in
the conversation transcript.

Detach start and inspection are host controls rather than calls through the
editable dispatcher, so dispatcher-specific logging or policy does not wrap
those control actions. The background job itself executes one ordinary,
providerless call through the active `call_tool` implementation in an inherited
toolbox and working-directory scope. It receives the ordinary filesystem/toolbox
environment but not `ctx.model_provider`. Detach is available only on the
model-facing top-level operation, not on authored `ctx.call_tool` calls.

The subprocess boundary cleans up the worker's process group. It is not a
hostile-code sandbox and cannot control a tool that deliberately creates a new,
independent process session.
