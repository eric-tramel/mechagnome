# Composing tools

Call another tool from authored source with:

```python
def main(input, ctx):
    return ctx.call_tool("add", {"a": input["x"], "b": input["x"]})
```

`ctx.call_tool(name, args, version=None)` routes nested calls through the
currently active `call_tool` source. Supplying a version pins the nested call to
that immutable version; omitting it resolves the active binding.

Prefer small tools whose descriptions make composition and reuse discoverable.
