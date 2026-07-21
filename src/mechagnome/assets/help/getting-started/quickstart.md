# Quickstart

1. Call `search_tools` before creating a duplicate.
2. Call `write_tool` with a small `def main(input, ctx)` implementation.
3. Call the new tool immediately through `call_tool`; failures are observations
   you can repair.
4. Call `read_tool_source` before changing an existing tool.

Prefer small tools with descriptions that make later reuse easy to discover.
