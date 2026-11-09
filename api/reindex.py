"""POST /api/reindex — Vercel file-based function.

The index is baked into api/_index_data.py at build time, so this route
explains how to regenerate it instead of pretending to rebuild in a read-only
function filesystem.
"""
import os
import sys

_file = globals().get("__file__", "")
_paths = [os.path.dirname(os.path.abspath(_file))] if _file else []
_paths.append(os.path.join(os.getcwd(), "api"))
for _path in _paths:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from _serverless import make_handler  # noqa: E402

handler = make_handler("reindex")
