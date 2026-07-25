# Core tools

The nine base tools are ordinary readable tool versions:

- `help`
- `list_tools`
- `list_tool_namespaces`
- `search_tools`
- `view_tool`
- `write_tool`
- `call_tool`
- `delete_tool`
- `run_agent`

Their names and outer schemas are fixed, but their descriptions, source, and
behavior are editable. Privilege follows the active logical core slot, not
copied source text. Copying `search_tools` source into an ordinary user tool,
for example, does not grant catalog access.

When an upgrade introduces a new fixed core name, an existing user-authored
tool with that name is preserved under a collision-free `_legacy` suffix before
the shipped core slot is installed.

A host-only rollback command can recover a broken core binding.

`delete_tool` removes a tool's binding from the active toolbox, so it
disappears from catalog/search/listings, while retaining its lineage and
version history. Core tools cannot be deleted.

The optional `version` parameter targets a specific version row instead of
the whole binding. Deleting the active version rolls the binding back to the
next-highest remaining version. The last remaining version of a lineage
cannot be removed.

## Agent tool

`run_agent` is an editable core tool like the others above. Its shipped source
is an ordinary wrapper over the same `ctx.sessions` handles available to every
authored tool; it receives no slot-specific agent capability. Every root, child,
and grandchild conversation receives the active tool version. Spawn a fresh
child agent in the foreground with:

```json
{
  "prompt": "Investigate the failing tests and summarize the cause.",
  "title": "Test failure investigation",
  "description": "Delegated analysis of the current failing test suite."
}
```

`title` and `description` are optional. The ordinary `run_agent` source passes
them through the generic session prompt request, which validates and applies
them before the rollout starts. Every authored tool can also retrieve a prompt
result's `session_id` and call
`ctx.sessions.get(session_id).update_metadata(...)` afterward.

Target a saved conversation and choose an explicit prompting mode:

```json
{"session_id": "...", "prompt": "Follow up.", "mode": "continue"}
```

```json
{"session_id": "...", "prompt": "Try another approach.", "mode": "fork"}
```

The default mode is `spawn`. `continue` appends to the same idle conversation;
`spawn` creates a fresh child without transcript inheritance; and `fork`
creates a child with context snapshotted through the source's latest completed
turn.

Set `detach` when the agent should continue independently:

```json
{"prompt": "Run the long analysis and return the result.", "detach": true}
```

The immediate response contains `job_id`, the prompted `session_id`, and
`status`. Job and session identities are distinct so the same conversation may
be continued by multiple detached prompts over its lifetime. Inspect the job
later through the same tool with `{"job_id": "..."}`. Successful terminal
snapshots include `result`; failures include a structured `error`.

Foreground agents share a 16-active limit. Each top-level rollout and all of
its recursive descendants also share a cumulative 64-agent launch budget.
Detached agents use a separate four-job pool, retain the latest 64 terminal
handles, and retain at most 1 MiB per answer. The creator or any ancestor
conversation may inspect a handle. Foreground cancellation does not stop a
detached agent; application or `Harness` shutdown does. Detached mode requires
a provider that can isolate cancellation per conversation session.

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
