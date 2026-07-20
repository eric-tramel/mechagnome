"""Private subprocess entry point for one isolated editable tool call."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mechagnome.kernel import Kernel, ToolboxError


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
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        kernel = Kernel(
            request["db_path"],
            max_depth=request["max_depth"],
            max_calls=request["max_calls"],
        )
        result = kernel.call(
            request["name"],
            request["args"],
            session_id=request["session_id"],
        )
        response = {"ok": True, "result": result}
    except BaseException as error:
        if isinstance(error, ToolboxError):
            failure = error.to_dict()["error"]
        else:
            failure = {"code": "tool_failed", "message": str(error), "details": {}}
        response = {"ok": False, "error": failure}
    _write_response(response_path, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
