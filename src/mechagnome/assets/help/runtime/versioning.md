# Versioning

Every source-authoring `write_tool` call creates an immutable integer version
and atomically moves the active binding to it. A namespace-only `write_tool`
call updates current lineage metadata without creating or activating a version.

Core version 1 is the exception: it is the code-shipped default and refreshes
from the installed Mechagnome library at startup. Persisted core version 2 and
later implementations remain immutable and override version 1 while active.

Pass `base_version` when replacing a tool to reject a stale activation. Calls
resolve and pin their version before execution, so a tool can replace itself:
the running call finishes on the old source and its next call uses the new
binding.

The host-only rollback command can reactivate a known version without relying
on editable core code. Rolling a core tool back to version 1 selects the default
shipped by the currently running library.
