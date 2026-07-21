# Toolbox namespaces

The host gives each durable session an ordered, nonempty set of toolbox
namespaces. Unqualified names use the first matching toolbox, and search returns
one effective result per name.

The host can replace the active order with one or more namespaces, append an
ordered idempotent union, remove namespaces, or reset the session to its cwd
default. A top-level call snapshots the active order for its complete nested
call tree, so a hot swap affects the next call rather than an in-flight call.

Cwd associations select defaults and preserve the session execution directory.
They organize tools; they are not filesystem restrictions or security
boundaries.
