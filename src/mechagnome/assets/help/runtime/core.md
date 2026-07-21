# Core operations

The five base operations are ordinary readable tool versions:

- `help`
- `search_tools`
- `read_tool_source`
- `write_tool`
- `call_tool`

Their names and outer schemas are fixed, but their descriptions, source, and
behavior are editable. Privilege follows the active logical core slot, not
copied source text. Copying `search_tools` source into an ordinary user tool,
for example, does not grant catalog access.

A host-only rollback command can recover a broken core binding.
