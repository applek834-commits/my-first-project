"""GET /api/status — Vercel file-based function."""
import os
import sys

_file = globals().get("__file__", "")
_paths = [os.path.dirname(os.path.abspath(_file))] if _file else []
_paths.append(os.path.join(os.getcwd(), "api"))
for _path in _paths:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from _serverless import make_handler  # noqa: E402

_Base = make_handler("status")


class handler(_Base):
    """Vercel entrypoint — MUST be a top-level class named `handler`.

    Vercel's Python builder only creates a function for an api/*.py file when
    it statically finds a top-level `app` or a top-level `class handler`
    (vercel/vercel#14493). A plain `handler = make_handler(...)` assignment is
    classified as a helper module and silently skipped, which makes the whole
    deployment fail with `unused_function`.
    """
