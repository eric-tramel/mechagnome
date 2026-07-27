# Hierarchical tool namespaces

Namespaces organize tools for discovery inside their owning toolbox. They are
case-sensitive slash-delimited paths such as `development/python` or
`quality/formatting`. A tool may belong to several namespaces and still has one
flat callable name, one lineage, and one immutable version history.

New core tools start in `core`; new user-authored tools start in
`uncategorized`. Supply `namespaces` with a normal `write_tool` call to choose
the initial or replacement assignments:

```json
{
  "name": "format_python",
  "description": "Format Python source.",
  "input_schema": {"type": "object"},
  "source": "async def main(input, ctx):\n    return input['source']\n",
  "namespaces": ["development/python", "quality/formatting"]
}
```

For an existing active tool, call `write_tool` with only `name` and a nonempty
`namespaces` array to replace its assignments without creating a new tool
version. Memberships are mutable metadata for the complete lineage, so viewing
an older version shows the lineage's current namespaces. Assignment updates are
last-write-wins. `base_version`, when supplied, checks the active source version
but does not create an independent namespace revision.

`list_tools` pages through tools and accepts a `namespace` filter. Filtering
`development` includes exact `development` assignments and descendants such as
`development/python`, but not `device`. `list_tool_namespaces` pages through
namespace paths with recursive, de-duplicated tool counts; intermediate parent
paths are included even when no tool is assigned directly to them. `search_tools`
returns each tool's sorted namespace paths and also indexes namespace text for
keyword ranking.

The TUI sidebar presents these paths as a collapsible tree. Tool leaves show only
their callable names; selecting one opens its source, history, and usage details.

Namespaces do not affect `call_tool` resolution, the fixed twelve core tool
slots, capabilities, or filesystem access. They are organization metadata, not
security boundaries. Use `help(topic="toolboxes")` for the separate ordered
toolbox-stack and working-directory routing behavior.

Persisted custom replacements for `search_tools` or `write_tool` remain
callable, but they must be updated to forward the new namespace arguments before
they provide the shipped namespace-aware behavior.
