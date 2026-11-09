"""POST /api/search — Vercel file-based function."""
import os
import sys

_file = globals().get("__file__", "")
_paths = [os.path.dirname(os.path.abspath(_file))] if _file else []
_paths.append(os.path.join(os.getcwd(), "api"))
for _path in _paths:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from _serverless import make_handler  # noqa: E402

handler = make_handler("search")
