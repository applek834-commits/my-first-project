"""GET /api — Vercel file-based function.

Every /api/*.py file maps to one route. The heavy lifting lives in the
underscore-prefixed helpers, which Vercel treats as regular modules rather
than as additional functions.
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

handler = make_handler("index")
