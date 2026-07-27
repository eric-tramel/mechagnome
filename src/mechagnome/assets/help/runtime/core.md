# Core tools

The twelve base tools are ordinary readable tool versions:

- `help`
- `list_tools`
- `list_tool_namespaces`
- `search_tools`
- `view_tool`
- `write_tool`
- `call_tool`
- `get_tool_run`
- `wait_tool_run`
- `cancel_tool_run`
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

`title` and `description` are optional. Keep titles to no more than four words
so the session tree stays easy to scan; this is guidance, not a runtime word
limit. The ordinary `run_agent` source passes the metadata through the generic
session prompt request, which validates and applies it before the rollout
starts. Every authored tool can also retrieve a prompt result's `session_id`
and call `ctx.sessions.get(session_id).update_metadata(...)` afterward.

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

`run_agent` is an ordinary tool: its foreground result contains the durable
`session_id` and final `result`. Detach it through `call_tool` when it should
continue independently:

```json
{
  "name": "run_agent",
  "args": {"prompt": "Run the long analysis and return the result."},
  "detach": true
}
```

The immediate response contains `run_id`, `tool_name`, and `status`. The
terminal ToolRun result contains the agent's distinct durable `session_id` and
answer text.

Foreground agents share a 16-active limit. Each top-level rollout and all of
its recursive descendants also share a cumulative 64-agent launch budget.
Detached agents use the same four-run pool and 64-terminal retention as other
tools. Foreground cancellation does not stop a detached agent; its ToolRun
cancellation domain is independent.

## Detached ToolRuns

`detach` is an invocation mode, not a second operation hidden inside
`call_tool`. Start a process-lifetime run with:

```json
{"name": "slow_report", "args": {"path": "data.csv"}, "detach": true}
```

The immediate result contains `run_id`, `tool_name`, and `status`. The deprecated
`job_id` alias is also returned during schema version 14. Check status without
copying retained output into the conversation:

```json
{"run_id": "..."}
```

Call `get_tool_run` with that input for lightweight status. Call
`wait_tool_run` with optional `timeout_ms` (0–30000) to wait for completion; a
nonterminal timeout adds `timed_out=true`, while a terminal response includes
the bounded merged stdout/stderr `output_tail` and `truncated`. Success includes
`result` (including explicit `null`); failure or cancellation includes a typed
`error`. Polling is repeated `get_tool_run`.

Request cancellation with:

```json
{"run_id": "..."}
```

`cancel_tool_run` moves an active run through `cancelling` to `cancelled` and
reports whether this call newly requested cancellation. Repeated cancellation
is idempotent. The canonical terminal error code is `tool_run_cancelled`.

ToolRuns are owned by the current host process. They do not survive an app
restart, at most four run concurrently, and app shutdown stops unfinished jobs.
The latest 64 completed handles remain inspectable; older completed handles are
evicted. Each result is limited to 1 MiB; an oversized result becomes a
structured `detached_result_too_large` failure. The first Escape stops only the foreground rollout after its
current model response and requested tools finish. A second Escape while it is
stopping immediately cancels the model stream, foreground child agents, and
foreground tool processes. Detached jobs remain unaffected. Clearing or ending
a TUI session hides its detached rows without cancelling the underlying jobs as
long as the app remains open.
Ending the final tab exits the app and triggers shutdown. Programmatic `Harness`
owners must call `close()` to stop background work.

The bounded output tail may contain sensitive data written by the tool.
Inspecting a handle supplies that tail to the model and therefore records it in
the conversation transcript.

The background run executes one ordinary call through the active `call_tool`
implementation in an inherited toolbox, working-directory, model-provider, and
agent capability scope. Authored tools can create the same runs with
`await ctx.call_tool(name, args, version=None, detach=True)`. The creator or any
ancestor conversation may get, wait for, or cancel a run; siblings, unrelated
sessions, unknown IDs, and evicted IDs all return `unknown_tool_run`.

The subprocess boundary cleans up the worker's process group. It is not a
hostile-code sandbox and cannot control a tool that deliberately creates a new,
independent process session.
