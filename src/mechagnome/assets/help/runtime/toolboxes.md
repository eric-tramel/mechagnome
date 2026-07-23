# Toolbox stacks

The host gives each durable session an ordered, nonempty stack of toolboxes.
Unqualified names use the first matching toolbox, and search returns one
effective result per name.

The host can replace the active stack with one or more toolboxes, append an
ordered idempotent union, remove toolboxes, or reset the session to its cwd
default. A top-level call snapshots the active stack for its complete nested
call tree, so a hot swap affects the next call rather than an in-flight call.

In the TUI, `/tools` or `Ctrl+T` opens tool management. Its toolbox picker
switches the session to another registered toolbox. **Blank** creates and
selects a new core-only toolbox, while **Save as…** renames the current toolbox
without changing its tools, history, or cwd association.

Cwd associations select defaults and preserve the session execution directory.
Toolbox stacks compose code; they are not filesystem restrictions or security
boundaries. Hierarchical tool namespaces are separate discovery metadata; see
`help(topic="namespaces")`.
