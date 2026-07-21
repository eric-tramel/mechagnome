# Versioning

Every `write_tool` call creates an immutable integer version. A successful write
atomically moves the active binding to the new version.

Pass `base_version` when replacing a tool to reject a stale activation. Calls
resolve and pin their version before execution, so a tool can replace itself:
the running call finishes on the old source and its next call uses the new
binding.

The host-only rollback command can reactivate a known version without relying
on editable core code.
