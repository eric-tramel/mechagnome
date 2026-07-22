# Composing tools

Authored tools compose through
`await ctx.call_tool(name, args, version=None)`. The arguments must be a JSON
object and the nested result is returned directly to the caller.

```python
async def main(input, ctx):
    total = await ctx.call_tool("add", {
        "a": input["value"],
        "b": input["value"],
    })
    return {"doubled": total}
```

Each nested invocation receives its own context object. The caller and callee
share the durable session and the toolbox selection snapshotted for the top-level
call, while event tracing records distinct call IDs and their parent-child
relationship. They also share the host-bound `ctx.model_provider` capability and
its limit of eight delegated-model attempts per top-level call tree. Each
accepted provider completion is logged in a new child session whose origin is
the nested tool that requested it. `run_agent()` similarly creates a full
`conversation` child, and recursive delegation creates grandchildren under the
agent that made each call. Ordinary nested tool calls themselves remain in the
shared parent session.

## Active and pinned calls

Omitting `version` resolves the current active binding within the snapshotted
toolbox scope:

```python
async def main(input, ctx):
    return await ctx.call_tool("normalize", input)
```

Supplying `version` selects that immutable version from the winning tool
lineage:

```python
async def main(input, ctx):
    return await ctx.call_tool("normalize", input, version=3)
```

Prefer active calls when a composed workflow should pick up fixes and upgrades.
Pin a version when reproducibility or compatibility with one exact contract is
more important. A version number does not select a shadowed toolbox; namespace
precedence first chooses the lineage, and the version is resolved within it.

## Dispatch and mutation

`ctx.call_tool` intentionally routes through the active editable `call_tool`
source. Replacing that core implementation can change nested routing, tracing,
caching, retries, or policy across the toolbox. The low-level
`await ctx.kernel.execute(...)` method belongs only to the logical `call_tool`
core slot and is what bottoms out dispatch without recursively calling itself.

A tool can call `write_tool` and then invoke the newly active version before it
returns:

```python
async def main(input, ctx):
    written = await ctx.call_tool("write_tool", {
        "name": "constant_value",
        "description": "Return the configured constant value.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
        },
        "source": (
            "async def main(input, ctx):\n"
            f"    return {input['value']!r}\n"
        ),
    })
    result = await ctx.call_tool("constant_value", {})
    return {"written": written, "result": result}
```

Writes are durable and are not rolled back if a later nested call fails. Use
`base_version` when replacing an existing tool to reject a stale update.

## Limits and design guidance

Every nested call counts against the call tree's depth and total-call limits.
Avoid cycles, unbounded recursion, and large fan-out loops. Keep tools small,
give them discoverable descriptions and accurate schemas, and return structured
JSON so other tools can compose results without parsing prose.

Use `asyncio.gather` when nested calls are independent and can safely run at the
same time; await dependent calls in order:

```python
import asyncio


async def main(input, ctx):
    left, right = await asyncio.gather(
        ctx.call_tool("lookup", {"id": input["left_id"]}),
        ctx.call_tool("lookup", {"id": input["right_id"]}),
    )
    return {"left": left, "right": right}
```

Nested failures propagate to the caller unless authored source catches the
exception. Catch only failures the tool can handle meaningfully; otherwise let
the failed call and its trace remain visible.

Long-running work requested directly by the model may use the host's detached
`call_tool` mode described in `help({"topic": "core"})`. Authored
`ctx.call_tool` composition remains an awaited foreground call: a tool cannot
detach one of its own nested calls.
