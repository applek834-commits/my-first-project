"""Offline smoke test for the Vercel serverless backend — no server required.

    python scripts/test_serverless.py

It calls the route dispatcher directly and also drives the real
BaseHTTPRequestHandler over an in-memory socket pair, so a broken Vercel
wiring is caught before deploying. Groq is never contacted.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "api"
sys.path.insert(0, str(API))

import _serverless as rag  # noqa: E402

PASS = 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global PASS
    if not condition:
        raise AssertionError(f"FAILED: {label} {extra}")
    PASS += 1
    print(f"  ok  {label}{(' — ' + extra) if extra else ''}")


def http_probe(route_file: str, method: str, path: str, body: dict | None = None):
    """Run one real HTTP request through the handler over a socket pair."""
    spec = importlib.util.spec_from_file_location(
        f"_probe_{route_file}", API / f"{route_file}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    check(f"{route_file}.py exposes a handler class", issubclass(module.handler, BaseHTTPRequestHandler))

    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    request = (
        f"{method} {path} HTTP/1.0\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "\r\n"
    ).encode("ascii") + payload

    client, server = socket.socketpair()
    client.sendall(request)
    instance = module.handler(server, ("127.0.0.1", 0), None)
    del instance
    client.settimeout(2)
    chunks = []
    try:
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except socket.timeout:
        pass
    finally:
        client.close()
        server.close()
    head, _, raw_body = b"".join(chunks).partition(b"\r\n\r\n")
    return head.decode("iso-8859-1"), json.loads(raw_body or b"{}")


def main() -> None:
    print("--- route dispatcher ---")
    status, data = rag.handle("status", "GET", {}, {})
    check("GET /api/status 200", status == 200)
    check("status is ready", data["status"] == "ready")
    check("14 documents", data["documents"] == 14)
    # Fragment merging makes the serverless chunk count slightly lower than the
    # local MiniLM backend's 1,355; assert it is in the same ballpark instead.
    check(
        "precomputed index has ~1.3k chunks",
        1200 < data["chunks"] <= 1355,
        str(data["chunks"]),
    )
    check(
        "no tiny fragment chunks",
        all(len(c[1]) >= 100 for c in rag.index()["chunks"]),
    )
    check("defaults exposed", data["defaults"]["max_k"] == 20)
    check("runtime flagged", data["runtime"] == "vercel-serverless")

    status, data = rag.handle("health", "GET", {}, {})
    check("GET /api/health 200", status == 200 and data["ok"] is True)

    status, data = rag.handle("documents", "GET", {}, {})
    check("GET /api/documents lists files", status == 200 and data["count"] == 14)
    check("file shape", set(data["files"][0]) == {"name", "characters"})

    status, data = rag.handle("models", "GET", {}, {})
    check("GET /api/models shape", set(data) == {"models", "default", "error"})

    status, data = rag.handle(
        "search", "POST", {}, {"question": "What is precision?", "k": 5, "mode": "hybrid"}
    )
    check("POST /api/search 200", status == 200)
    check("five ranked sources", len(data["sources"]) == 5)
    check(
        "source shape matches FastAPI",
        set(data["sources"][0])
        == {"rank", "score", "source", "chunk", "semantic_score", "keyword_score"},
    )
    check("scores are floats", isinstance(data["top_score"], float))

    status, data = rag.handle(
        "ask", "POST", {}, {"question": "What is precision?", "use_llm": False}
    )
    check("POST /api/ask retrieval-only 200", status == 200)
    check("no Groq call when use_llm=false", data["model"] is None)
    check("confident question not abstained", data["low_confidence"] is False)
    check(
        "AskResponse shape matches FastAPI",
        set(data)
        == {
            "answer", "question", "sources", "model", "mode", "k",
            "top_score", "low_confidence", "elapsed_ms", "usage",
        },
    )

    status, data = rag.handle(
        "ask", "POST", {}, {"question": "What is LangChain?", "use_llm": True}
    )
    check("out-of-corpus question abstains", data["low_confidence"] is True)
    check("abstain skips Groq", data["model"] is None and "I do not know" in data["answer"])

    status, data = rag.handle("reindex", "POST", {}, {})
    check("POST /api/reindex 202", status == 202 and data["accepted"] is True)

    try:
        rag.handle("ask", "POST", {}, {"question": ""})
        raise AssertionError("empty question should be rejected")
    except rag.ApiError as exc:
        check("empty question -> 400", exc.status == 400)

    try:
        rag.handle("status", "POST", {}, {})
        raise AssertionError("wrong method should be rejected")
    except rag.ApiError as exc:
        check("GET-only route rejects POST -> 405", exc.status == 405)

    print("\n--- real HTTP handler ---")
    head, data = http_probe("status", "GET", "/api/status")
    check("status over HTTP", "200" in head and data["status"] == "ready")
    check("CORS header present", "Access-Control-Allow-Origin" in head)
    check("JSON content type", "application/json" in head)

    head, data = http_probe(
        "ask", "POST", "/api/ask", {"question": "What is precision?", "use_llm": False}
    )
    check("ask over HTTP", "200" in head and len(data["sources"]) == 5)

    head, _ = http_probe("models", "OPTIONS", "/api/models")
    check("OPTIONS preflight 204", "204" in head)

    head, data = http_probe("ask", "POST", "/api/ask", {"question": ""})
    check("validation error is JSON detail", "400" in head and "detail" in data)

    print(f"\nAll {PASS} checks passed.")


if __name__ == "__main__":
    main()
