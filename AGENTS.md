# Agent Instructions

This is a small, local proof of a metaprogrammable agent toolbox.

## Development

- Use `uv` for Python commands.
- Keep runtime dependencies limited to the TUI and provider transport surface.
- Prefer explicit code over framework abstractions.
- Run `uv run --group dev ruff format --check src tests`,
  `uv run --group dev ruff check src tests`, and
  `uv run --group dev pytest` before finishing.
- Model-requested tool call trees run in a filtered-environment subprocess.
  This is not credential separation or a hostile-code sandbox; tools retain
  ordinary OS access under the same user as the provider client.
