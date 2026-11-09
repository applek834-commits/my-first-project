"""Vercel serverless implementation of the RAG Course Assistant API.

The local FastAPI service (``backend/``) embeds the corpus with MiniLM and
searches a FAISS index. A Vercel function has no persistent disk, no model
weights and a small cold-start budget, so this module uses a different but
API-compatible retrieval strategy:

  * the corpus is chunked, TF-IDF weighted and inverted-indexed at build time
    (``scripts/build_serverless_index.py``) into ``api/_index_data.py``;
  * queries are scored with sublinear TF-IDF cosine + BM25 in pure Python;
  * generation still comes from Groq, called with ``urllib.request``.

Everything here is standard library only, so Vercel needs no requirements.txt,
no pip install and no model download. The response bodies match
``backend/main.py`` exactly, so the same React frontend works unchanged.
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler

# The function's directory is not guaranteed to be on sys.path in every
# Vercel runtime layout; make sibling helper modules importable explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _index_data import PAYLOAD  # noqa: E402
from _text import (  # noqa: E402
    SYSTEM_PROMPT,
    build_prompt,
    merge_fragments,
    recursive_chunk,
    tokenize,
)

MODES = ("hybrid", "semantic", "keyword")
INDEX_BACKEND = "precomputed TF-IDF + BM25"
EMBEDDING_MODEL = "TF-IDF (precomputed, no neural embeddings)"

BM25_K1 = 1.5
BM25_B = 0.75
MODEL_CACHE_TTL = 300.0

# Limits for user-supplied documents (the "My notes" feature). Vercel
# function payloads are capped at 4.5 MB by the platform; we stay well under
# that while still allowing several substantial text files.
MAX_EXTRA_DOCS = 10
MAX_EXTRA_DOC_CHARS = 50_000
MAX_EXTRA_TOTAL_CHARS = 200_000


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TEMPERATURE = _float_env("TEMPERATURE", 0.2)
MAX_TOKENS = _int_env("MAX_TOKENS", 800)
DEFAULT_K = _int_env("DEFAULT_K", 5)
MAX_K = _int_env("MAX_K", 20)
ABSTAIN_GUARD = _bool_env("ABSTAIN_GUARD", True)
# TF-IDF cosine similarities are naturally lower than MiniLM cosine ones, so
# the serverless build uses its own calibrated abstain threshold. See
# scripts/calibrate_serverless.py for the measurements behind this value.
MIN_SCORE = _float_env("MIN_SCORE", 0.12)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Precomputed index
# ---------------------------------------------------------------------------
_INDEX: dict | None = None


def index() -> dict:
    """Decode once per warm function instance (a few milliseconds)."""
    global _INDEX
    if _INDEX is None:
        raw = zlib.decompress(base64.b64decode(PAYLOAD)).decode("utf-8")
        _INDEX = json.loads(raw)
    return _INDEX


def _validate_extra_docs(raw):
    """Validate and normalise the extra_docs field from the request body.

    Returns a list of (source, text) tuples, or raises ApiError.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ApiError(400, "extra_docs must be a list of objects.")
    if len(raw) > MAX_EXTRA_DOCS:
        raise ApiError(400, f"extra_docs may contain at most {MAX_EXTRA_DOCS} documents.")
    total = 0
    docs = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ApiError(400, f"extra_docs[{i}] must be an object.")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, f"extra_docs[{i}].text must be a non-empty string.")
        if len(text) > MAX_EXTRA_DOC_CHARS:
            raise ApiError(
                400,
                f"extra_docs[{i}].text exceeds the {MAX_EXTRA_DOC_CHARS:,}-character limit.",
            )
        total += len(text)
        if total > MAX_EXTRA_TOTAL_CHARS:
            raise ApiError(
                400,
                f"extra_docs total exceeds the {MAX_EXTRA_TOTAL_CHARS:,}-character limit.",
            )
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            source = f"my-notes-{i + 1}"
        docs.append((source.strip()[:120], text))
    return docs


def _score_extra_docs(question, extra_docs, k):
    """Chunk and TF-IDF-score user-supplied docs at request time.

    Returns a list of hit dicts with the same shape as baked-index hits,
    plus their raw TF-IDF cosine score. These are merged with the baked
    index hits so a user's own notes can outrank (or be outranked by)
    the course corpus.
    """
    if not extra_docs:
        return []

    # Build a mini TF-IDF index over only the user's chunks, using the same
    # sublinear-tf formula as the baked index. idf is computed from this
    # small collection, which is fine — we just need relative ranking.
    chunks = []
    sources = []
    for source, text in extra_docs:
        pieces = merge_fragments(recursive_chunk(text))
        for piece in pieces:
            chunks.append(piece)
            sources.append(source)
    if not chunks:
        return []

    n = len(chunks)
    doc_tokens = [tokenize(c) for c in chunks]
    df = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    idf = {term: math.log(1.0 + n / (df_t + 1)) for term, df_t in df.items()}

    # Document weights and norms
    doc_weights = []
    doc_norms = []
    for tokens in doc_tokens:
        counts = Counter(tokens)
        weights = {t: (1.0 + math.log(tf)) * idf.get(t, 0.0) for t, tf in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        doc_weights.append(weights)
        doc_norms.append(norm)

    # Query weights
    q_counts = Counter(tokenize(question))
    q_weights = {t: (1.0 + math.log(tf)) * idf.get(t, 0.0) for t, tf in q_counts.items()}
    q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

    scores = []
    for i, weights in enumerate(doc_weights):
        dot = sum(w * weights.get(t, 0.0) for t, w in q_weights.items())
        cos = dot / (q_norm * doc_norms[i]) if doc_norms[i] > 0 else 0.0
        if cos > 0:
            scores.append((i, cos))

    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[:k]

    max_score = max((s for _, s in scores), default=0.0) or 1.0
    hits = []
    for rank, (i, score) in enumerate(scores, start=1):
        hits.append(
            {
                "rank": rank,
                "score": round(score / max_score, 4),
                "source": sources[i],
                "chunk": chunks[i],
                "semantic_score": round(score, 4),
                "keyword_score": round(score / max_score, 4),
                "_extra": True,
            }
        )
    return hits


def retrieve(
    question: str,
    k: int = 5,
    mode: str = "hybrid",
    extra_docs: list = None,
) -> tuple[list[dict], float]:
    """Return (source dicts, calibrated confidence) using TF-IDF + BM25.

    A single rare word can score highly on cosine alone even for a nonsense
    question, so confidence also accounts for query-term coverage: the share
    of the query's TF-IDF mass made up of terms that actually exist in the
    corpus. That keeps "What is precision?" high while "Who won the 2022
    football world cup?" correctly abstains.

    If ``extra_docs`` is provided (a list of (source, text) tuples), those
    documents are chunked and TF-IDF-scored at request time, then merged
    with the baked index hits so the user's own notes participate in
    retrieval alongside the course corpus.
    """
    idx = index()
    n = idx["n"]
    idf = idx["idf"]
    postings = idx["postings"]
    norms = idx["norm"]
    doc_lengths = idx["dl"]
    avgdl = idx["avgdl"] or 1.0

    q_counts = Counter(tokenize(question))
    q_weights = {
        t: (1.0 + math.log(tf)) * idf[t]
        for t, tf in q_counts.items()
        if t in idf
    }
    q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
    # Out-of-vocabulary terms still count against coverage, at the largest idf.
    oov_idf = max(idf.values(), default=1.0)
    total_mass = sum(
        (1.0 + math.log(tf)) * idf.get(t, oov_idf) for t, tf in q_counts.items()
    )
    matched_mass = sum(q_weights.values())
    coverage = matched_mass / total_mass if total_mass else 0.0

    dots: dict[int, float] = defaultdict(float)
    bm25: dict[int, float] = defaultdict(float)
    for term, q_w in q_weights.items():
        entries = postings.get(term, ())
        df = len(entries)
        bm_idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for chunk_id, tf in entries:
            d_w = (1.0 + math.log(tf)) * idf[term]
            dots[chunk_id] += q_w * d_w
            length_ratio = doc_lengths[chunk_id] / avgdl
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio)
            bm25[chunk_id] += bm_idf * (tf * (BM25_K1 + 1.0)) / denom

    semantic = {
        i: dot / (q_norm * norms[i])
        for i, dot in dots.items()
        if norms[i] > 0
    }
    max_bm = max(bm25.values(), default=0.0) or 1.0
    keyword = {i: score / max_bm for i, score in bm25.items()}
    top_semantic = max(semantic.values(), default=0.0)

    k = max(1, min(int(k), MAX_K, n))
    if mode == "semantic":
        ranked = sorted(semantic.items(), key=lambda x: x[1], reverse=True)[:k]
        scores = dict(ranked)
    elif mode == "keyword":
        ranked = sorted(keyword.items(), key=lambda x: x[1], reverse=True)[:k]
        scores = dict(ranked)
    else:
        combined = defaultdict(float)
        for i, score in semantic.items():
            combined[i] += score
        for i, score in keyword.items():
            combined[i] += score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        scores = dict(ranked)

    docs = idx["docs"]
    chunks = idx["chunks"]
    hits = [
        {
            "rank": rank,
            "score": round(scores.get(i, 0.0), 4),
            "source": docs[chunks[i][0]][0],
            "chunk": chunks[i][1],
            "semantic_score": round(semantic.get(i, 0.0), 4),
            "keyword_score": round(keyword.get(i, 0.0), 4),
        }
        for rank, (i, _) in enumerate(ranked, start=1)
    ]

    # Merge user-supplied notes, if any. Both sets are normalised to [0, 1]
    # before merging so neither dominates simply because its raw scale is
    # higher. The top k by combined score wins.
    if extra_docs:
        extra_hits = _score_extra_docs(question, extra_docs, k)
        if extra_hits:
            # Normalise baked-index scores to [0, 1] so the two sets are
            # comparable. The baked index's top hit defines the scale.
            baked_max = max((h["score"] for h in hits), default=0.0) or 1.0
            for h in hits:
                h["_norm"] = h["score"] / baked_max
            for h in extra_hits:
                h["_norm"] = h["score"]  # already 0-1 from _score_extra_docs
            merged = hits + extra_hits
            merged.sort(key=lambda h: h["_norm"], reverse=True)
            merged = merged[:k]
            for rank, h in enumerate(merged, start=1):
                h["rank"] = rank
                h.pop("_norm", None)
            hits = merged
            # Extra docs can raise confidence — use the best extra score if
            # it beats the baked coverage-weighted score.
            extra_top = max((h["semantic_score"] for h in extra_hits), default=0.0)
            if extra_top > top_semantic * coverage:
                return hits, round(extra_top, 4)

    return hits, round(top_semantic * coverage, 4)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
_GROQ_BASE = "https://api.groq.com/openai/v1/"
_model_cache: dict = {"models": [], "fetched_at": 0.0}


def _api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def _groq(path: str, payload: dict | None = None, method: str = "GET", timeout: int = 30) -> dict:
    key = _api_key()
    if not key:
        raise ApiError(
            400,
            "GROQ_API_KEY is not set. Add it as an Environment Variable in your "
            "Vercel project settings, or turn off 'Generate an answer' to use "
            "retrieval-only mode, which needs no key.",
        )
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        _GROQ_BASE + path,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "rag-course-assistant/1.0 (Vercel serverless)",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            message = json.loads(raw).get("error", {}).get("message", raw)
        except (ValueError, AttributeError):
            message = raw
        raise ApiError(
            502,
            f"Groq request failed (HTTP {exc.code}): {message} — check that "
            "GROQ_MODEL is an active model on your account.",
        ) from exc
    except Exception as exc:
        raise ApiError(502, f"Groq request failed: {exc}") from exc


def list_models(refresh: bool = False) -> tuple[list[dict], str | None]:
    now = time.time()
    if (
        not refresh
        and _model_cache["models"]
        and now - _model_cache["fetched_at"] < MODEL_CACHE_TTL
    ):
        return list(_model_cache["models"]), None
    if not _api_key():
        return [], "GROQ_API_KEY is not set."
    try:
        raw = _groq("models", timeout=15).get("data", [])
    except ApiError as exc:
        return list(_model_cache["models"]), exc.message

    skip = ("whisper", "tts", "orpheus", "guard", "transcribe", "translate", "embed")
    items = []
    for entry in raw:
        model_id = str(entry.get("id", ""))
        if not model_id or entry.get("active") is False:
            continue
        if any(token in model_id.lower() for token in skip):
            continue
        items.append(
            {
                "id": model_id,
                "context_window": entry.get("context_window"),
                "owned_by": entry.get("owned_by"),
            }
        )
    items.sort(key=lambda item: item["id"])
    _model_cache.update({"models": items, "fetched_at": now})
    return items, None


def generate(prompt: str, model: str | None) -> tuple[str, dict | None]:
    chosen = (model or GROQ_MODEL).strip()
    completion = _groq(
        "chat/completions",
        {
            "model": chosen,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            # Current Groq parameter name; some newer models reject max_tokens.
            "max_completion_tokens": MAX_TOKENS,
        },
        method="POST",
        timeout=60,
    )
    choices = completion.get("choices") or [{}]
    answer = ((choices[0].get("message") or {}).get("content") or "").strip()
    if not answer:
        answer = (
            "The model returned no final answer within the token budget. "
            "Raise MAX_TOKENS in your Vercel environment variables and try again."
        )
    usage = None
    if completion.get("usage"):
        u = completion["usage"]
        details = u.get("completion_tokens_details") or {}
        usage = {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        }
    return answer, usage


# ---------------------------------------------------------------------------
# Request validation + route handlers
# ---------------------------------------------------------------------------
def _query_payload(body: dict) -> tuple[str, int, str, list]:
    question = body.get("question")
    if not isinstance(question, str) or not (1 <= len(question.strip()) <= 2000):
        raise ApiError(400, "question must be a non-empty string of at most 2000 characters.")
    try:
        k = int(body.get("k", DEFAULT_K))
    except (TypeError, ValueError):
        raise ApiError(400, "k must be an integer.") from None
    if not (1 <= k <= MAX_K):
        raise ApiError(400, f"k must be between 1 and {MAX_K}.")
    mode = body.get("mode", "hybrid")
    if mode not in MODES:
        raise ApiError(400, f"mode must be one of {', '.join(MODES)}.")
    extra_docs = _validate_extra_docs(body.get("extra_docs"))
    return question.strip(), k, mode, extra_docs


def handle(route: str, method: str, query: dict, body: dict) -> tuple[int, dict]:
    expected = {
        "index": "GET",
        "health": "GET",
        "status": "GET",
        "documents": "GET",
        "models": "GET",
        "ask": "POST",
        "search": "POST",
        "reindex": "POST",
    }.get(route)
    if expected is None:
        raise ApiError(404, f"Unknown API route: {route}")
    if method != expected:
        raise ApiError(405, f"{route} expects {expected}, got {method}.")

    idx = index()

    if route == "index":
        return 200, {
            "name": "RAG Course Assistant API (Vercel serverless)",
            "runtime": "python stdlib, precomputed TF-IDF + BM25 index",
            "status": "/api/status",
            "endpoints": sorted(f"/api/{r}" for r in expected if r != "index") + ["/api/index"],
        }

    if route == "health":
        return 200, {"ok": True, "status": "ready", "uptime_s": 0}

    if route == "status":
        return 200, {
            "status": "ready",
            "detail": (
                f"Precomputed index ready — {idx['n']} chunks from "
                f"{len(idx['docs'])} documents ({INDEX_BACKEND}). "
                f"Built {idx['built_at']}."
            ),
            "documents": len(idx["docs"]),
            "chunks": idx["n"],
            "document_names": [d[0] for d in idx["docs"]],
            "vector_backend": INDEX_BACKEND,
            "embedding_model": EMBEDDING_MODEL,
            "groq_model": GROQ_MODEL,
            "groq_configured": bool(_api_key()),
            "corpus_dir": "ragg/ (baked into api/_index_data.py at build time)",
            "defaults": {
                "k": DEFAULT_K,
                "max_k": MAX_K,
                "min_score": MIN_SCORE,
                "modes": list(MODES),
                "chunk_size": 300,
                "chunk_overlap": 50,
                "max_tokens": MAX_TOKENS,
                "max_extra_docs": MAX_EXTRA_DOCS,
                "max_extra_doc_chars": MAX_EXTRA_DOC_CHARS,
            },
            "runtime": "vercel-serverless",
        }

    if route == "documents":
        return 200, {
            "count": len(idx["docs"]),
            "files": [
                {"name": name, "characters": characters}
                for name, characters in idx["docs"]
            ],
        }

    if route == "models":
        refresh = query.get("refresh", [""])[0].lower() in {"1", "true", "yes"}
        items, error = list_models(refresh=refresh)
        return 200, {"models": items, "default": GROQ_MODEL, "error": error}

    if route == "reindex":
        # Serverless filesystems are read-only; the index is baked at build time.
        return 202, {
            "accepted": True,
            "detail": (
                "The serverless index is precomputed, so nothing needs rebuilding "
                "at runtime. To change the corpus, edit ragg/*.txt, run "
                "`python scripts/build_serverless_index.py`, commit the updated "
                "api/_index_data.py and redeploy."
            ),
        }

    question, k, mode, extra_docs = _query_payload(body)
    started = time.perf_counter()
    hits, top_score = retrieve(question, k=k, mode=mode, extra_docs=extra_docs)
    elapsed = round((time.perf_counter() - started) * 1000)

    # Strip internal keys from extra-docs hits before serialising.
    for h in hits:
        h.pop("_extra", None)

    if route == "search":
        return 200, {
            "question": question,
            "sources": hits,
            "mode": mode,
            "k": k,
            "top_score": top_score,
            "elapsed_ms": elapsed,
        }

    # /api/ask
    low_confidence = ABSTAIN_GUARD and top_score < MIN_SCORE
    model = None
    usage = None
    if low_confidence:
        answer = (
            "I do not know. The retrieved passages do not seem to contain the "
            f"answer (best similarity {top_score:.3f} is below the "
            f"{MIN_SCORE:.2f} threshold). Try rephrasing, raising top-k, "
            "or switching to hybrid search."
        )
    elif not body.get("use_llm", True):
        answer = (
            "Retrieval only — generation was skipped. Expand the sources below "
            "to inspect the chunks that would have been sent to the model."
        )
    else:
        override = body.get("model")
        if override is not None:
            if not isinstance(override, str) or len(override) > 120:
                raise ApiError(400, "model must be a string of at most 120 characters.")
            override = override.strip() or None
        model = override or GROQ_MODEL
        answer, usage = generate(build_prompt(question, hits), model)

    return 200, {
        "answer": answer,
        "question": question,
        "sources": hits,
        "model": model,
        "mode": mode,
        "k": k,
        "top_score": top_score,
        "low_confidence": low_confidence,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# HTTP adapter — one BaseHTTPRequestHandler class per Vercel /api/*.py file
# ---------------------------------------------------------------------------
def make_handler(route: str):
    """Return a `handler` class Vercel can load for a single file-based route."""

    class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel convention
        server_version = "RAGServerless/1.0"

        def log_message(self, *_args) -> None:  # keep invocation logs clean
            return

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "86400")

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length <= 0:
                return {}
            if length > 256 * 1024:
                raise ApiError(413, "Request body is too large.")
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ApiError(400, "Invalid JSON body.") from None
            if not isinstance(parsed, dict):
                raise ApiError(400, "JSON body must be an object.")
            return parsed

        def _dispatch(self, method: str) -> None:
            try:
                parsed = urllib.parse.urlsplit(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                body = self._body() if method == "POST" else {}
                status, payload = handle(route, method, query, body)
            except ApiError as exc:
                status, payload = exc.status, {"detail": exc.message}
            except Exception as exc:  # never leak a stack trace to the browser
                traceback.print_exc()
                status, payload = 500, {"detail": f"{type(exc).__name__}: {exc}"}
            self._json(status, payload)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

    return handler
