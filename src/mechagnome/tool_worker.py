"""Private subprocess entry point for one isolated editable tool call."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from mechagnome.kernel import InvocationScope, Kernel, ToolboxError
from mechagnome.model_provider import _ModelProviderProxy


def _write_response(path: Path, response: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(response, stream, allow_nan=False, separators=(",", ":"))


def main() -> int:
    """Read one request, invoke the kernel, and write one JSON response."""
    if len(sys.argv) != 3:
        return 2
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    provider_connection: socket.socket | None = None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        provider = None
        provider_fd = request.get("model_provider_fd")
        if provider_fd is not None:
            if not isinstance(provider_fd, int) or provider_fd < 0:
                raise ToolboxError(
                    "tool_worker_failed", "invalid model provider descriptor"
                )
            os.set_inheritable(provider_fd, False)
            provider_connection = socket.socket(fileno=provider_fd)
            provider = _ModelProviderProxy(provider_connection)
        kernel = Kernel(
            request["db_path"],
            max_depth=request["max_depth"],
            max_calls=request["max_calls"],
            cwd=request["cwd"],
        )
        scope = InvocationScope(
            session_id=request["session_id"],
            toolbox_ids=tuple(request["toolbox_ids"]),
            cwd=request["cwd"],
        )
        result = kernel.call(
            request["name"],
            request["args"],
            session_id=request["session_id"],
            scope=scope,
            model_provider=provider,
        )
        response = {"ok": True, "result": result}
    except BaseException as error:
        if isinstance(error, ToolboxError):
            failure = error.to_dict()["error"]
        else:
            failure = {"code": "tool_failed", "message": str(error), "details": {}}
        response = {"ok": False, "error": failure}
    finally:
        if provider_connection is not None:
            provider_connection.close()
    _write_response(response_path, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
