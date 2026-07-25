# Mechagnome help

Mechagnome begins with seven editable core operations, the host-owned
`run_agent` action, and no domain-specific user tools. Every recursively launched
agent receives the same actions. Use these topics as you grow and reuse the
toolbox:

- `quickstart` — the write/call loop
- `authoring` — the Python tool ABI
- `composition` — tools calling tools
- `sessions` — current and historical event access
- `namespaces` — hierarchical tool discovery and organization
- `toolboxes` — ordered session toolbox composition and cwd routing
- `versioning` — immutable versions, activation, and rollback
- `core` — editing the seven base operations

Call `help` again with one of those names as the `topic` to read the complete
Markdown document.

## Project source

Mechagnome is typically installed as a vendored tool (e.g. via `uv tool install`),
which means the package on disk is a deployed copy, not a working source tree.
The full source for the harness lives on GitHub:

  **https://github.com/eric-tramel/mechagnome**

Visit the repository to browse the source, open issues, or contribute changes
to the harness itself (core operations, asset files, the runtime, etc.).
Modifying a vendored installation in-place will not persist across reinstalls or
upgrades — work against the GitHub source and reinstall from your fork instead.
