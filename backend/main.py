"""FastAPI backend for the RAG Course Assistant.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /api/status     index build state + corpus stats
    GET  /api/documents  files in the corpus
    GET  /api/models     active Groq models for this key (cached 5 min)
    POST /api/ask        retrieve + grounded answer via Groq
    POST /api/search     retrieval only (no API key required)
    POST /api/reindex    force a rebuild of the FAISS index
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # supports `uvicorn backend.main:app` and `cd backend && uvicorn main:app`
    from .config import settings
    from .rag_engine import SYSTEM_PROMPT, RAGEngine, RetrievalMode
except ImportError:  # pragma: no cover
    from config import settings
    from rag_engine import SYSTEM_PROMPT, RAGEngine, RetrievalMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("rag-api")


class AppState:
    def __init__(self) -> None:
        self.engine: RAGEngine | None = None
        self.status: str = "loading"
        self.detail: str = "Starting up..."
        self.rebuilding = False
        self.lock = threading.Lock()
        self.started_at = time.time()


state = AppState()

# Model IDs on Groq change often. Cache the live list so the UI can offer a
# picker without hitting the API on every render.
MODEL_CACHE_TTL = 300.0
_model_lock = threading.Lock()
_model_cache: dict = {"models": [], "fetched_at": 0.0}


def _start_build(force: bool = False) -> None:
    """Build the index on a background thread so the server stays responsive."""
    with state.lock:
        if state.rebuilding:
            return
        state.rebuilding = True
        state.status = "loading"
        state.detail = "Loading embeddings and building the vector index..."

    def worker() -> None:
        try:
            engine = RAGEngine(settings, force_rebuild=force)
            state.engine = engine
            state.status = "ready"
            state.detail = (
                f"Indexed {engine.stats['chunks']} chunks from "
                f"{engine.stats['documents']} documents "
                f"({engine.stats['vector_backend']} index)."
            )
        except Exception as exc:  # surface a readable error to the UI
            logger.exception("Failed to build the RAG index")
            state.status = "error"
            state.detail = f"{type(exc).__name__}: {exc}"
            state.engine = None
        finally:
            state.rebuilding = False

    threading.Thread(target=worker, name="rag-index-builder", daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _start_build(force=False)
    yield


app = FastAPI(
    title="RAG Course Assistant",
    version="1.0.0",
    description="FastAPI + FAISS + Groq RAG service built from the Week 8 notebook.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ExtraDoc(BaseModel):
    source: str = Field(default="", max_length=120)
    text: str = Field(..., min_length=1, max_length=50000)


class QueryBase(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=settings.default_k, ge=1, le=settings.max_k)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    extra_docs: list[ExtraDoc] = Field(default_factory=list, max_length=10)


class AskRequest(QueryBase):
    use_llm: bool = True
    # Lets the UI switch models without editing .env / restarting the server.
    model: str | None = Field(default=None, max_length=120)


class SearchRequest(QueryBase):
    pass


class SourceOut(BaseModel):
    rank: int
    score: float
    source: str
    chunk: str
    semantic_score: float | None = None
    keyword_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    question: str
    sources: list[SourceOut]
    model: str | None
    mode: str
    k: int
    top_score: float
    low_confidence: bool
    elapsed_ms: int
    usage: dict | None = None


class SearchResponse(BaseModel):
    question: str
    sources: list[SourceOut]
    mode: str
    k: int
    top_score: float
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_ready() -> RAGEngine:
    if state.status == "loading" or state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Index is still building: {state.detail}",
        )
    if state.status == "error":
        raise HTTPException(status_code=500, detail=state.detail)
    return state.engine


def _sources(result) -> list[SourceOut]:
    return [
        SourceOut(
            rank=hit.rank,
            score=hit.score,
            source=hit.source,
            chunk=hit.chunk,
            semantic_score=hit.semantic_score,
            keyword_score=hit.keyword_score,
        )
        for hit in result.hits
    ]


def _fetch_model_list(force: bool = False) -> tuple[list[dict], str | None]:
    """Return chat-capable models active for this key, plus an error string."""
    with _model_lock:
        now = time.time()
        if (
            not force
            and _model_cache["models"]
            and now - _model_cache["fetched_at"] < MODEL_CACHE_TTL
        ):
            return list(_model_cache["models"]), None
        if not settings.groq_configured:
            return [], "GROQ_API_KEY is not set."

        try:
            import httpx

            response = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=15.0,
            )
            response.raise_for_status()
            raw = response.json().get("data", [])
        except Exception as exc:
            logger.warning("Could not list Groq models: %s", exc)
            return list(_model_cache["models"]), str(exc)

        # Audio, guardrail and embedding models cannot serve chat completions.
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


def _generate_with_groq(
    prompt: str, model: str | None = None
) -> tuple[str, dict | None]:
    """Step 6 of the notebook, swapped from a local Qwen pipeline to Groq."""
    if not settings.groq_configured:
        raise HTTPException(
            status_code=400,
            detail=(
                "GROQ_API_KEY is not set. Put it in a .env file at the project "
                "root, or use 'Retrieval only' mode which needs no key."
            ),
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise HTTPException(
            status_code=500, detail="The 'groq' package is not installed."
        ) from exc

    chosen_model = (model or settings.groq_model).strip()
    client = Groq(api_key=settings.groq_api_key)
    try:
        completion = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=settings.temperature,
            # max_completion_tokens is the current parameter name on Groq;
            # the older max_tokens is rejected by some newer models.
            max_completion_tokens=settings.max_tokens,
        )
    except Exception as exc:
        logger.exception("Groq request failed")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Groq request failed: {exc} — check that GROQ_MODEL "
                "is an active model on your account "
                "(https://console.groq.com/docs/models)."
            ),
        ) from exc

    message = completion.choices[0].message
    answer = (getattr(message, "content", None) or "").strip()
    if not answer:
        # Reasoning models can spend the whole token budget on thinking.
        answer = (
            "The model returned no final answer within the token budget. "
            "Raise MAX_TOKENS in .env and try again."
        )

    usage = None
    if getattr(completion, "usage", None):
        details = getattr(completion.usage, "completion_tokens_details", None)
        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        }
    return answer, usage


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "name": "RAG Course Assistant API",
        "docs": "/docs",
        "status": "/api/status",
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "status": state.status, "uptime_s": round(time.time() - state.started_at)}


@app.get("/api/status")
def status() -> dict:
    engine = state.engine
    return {
        "status": state.status,
        "detail": state.detail,
        "documents": engine.stats["documents"] if engine else 0,
        "chunks": engine.stats["chunks"] if engine else 0,
        "document_names": engine.stats["document_names"] if engine else [],
        "vector_backend": engine.stats["vector_backend"] if engine else None,
        "embedding_model": settings.embedding_model,
        "groq_model": settings.groq_model,
        "groq_configured": settings.groq_configured,
        "corpus_dir": str(settings.corpus_dir),
        "defaults": {
            "k": settings.default_k,
            "max_k": settings.max_k,
            "min_score": settings.min_score,
            "modes": ["hybrid", "semantic", "keyword"],
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "max_tokens": settings.max_tokens,
        },
    }


@app.get("/api/documents")
def documents() -> dict:
    engine = _require_ready()
    return {
        "count": engine.stats["documents"],
        "files": [
            {"name": name, "characters": len(engine.docs[name])}
            for name in engine.stats["document_names"]
        ],
    }


@app.get("/api/models")
def models(refresh: bool = False) -> dict:
    """Active chat models for this Groq key. `?refresh=true` bypasses the cache."""
    items, error = _fetch_model_list(force=refresh)
    return {
        "models": items,
        "default": settings.groq_model,
        "error": error,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    engine = _require_ready()
    started = time.perf_counter()
    extra = [(d.source or f"my-notes-{i+1}", d.text) for i, d in enumerate(payload.extra_docs)]
    result = engine.retrieve(payload.question, k=payload.k, mode=payload.mode, extra_docs=extra)
    sources = _sources(result)

    # Notebook Step "Test honestly": a low top score means the corpus likely
    # does not contain the answer, so abstain instead of burning an API call.
    low_confidence = (
        settings.abstain_guard
        and result.top_semantic_score < settings.min_score
    )
    usage = None
    model = None

    if low_confidence:
        answer = (
            "I do not know. The retrieved passages do not seem to contain the "
            f"answer (best similarity {result.top_semantic_score:.3f} is below the "
            f"{settings.min_score:.2f} threshold). Try rephrasing, raising top-k, "
            "or switching to hybrid search."
        )
    elif not payload.use_llm:
        answer = (
            "Retrieval only — generation was skipped. Expand the sources below "
            "to inspect the chunks that would have been sent to the model."
        )
    else:
        chosen_model = (payload.model or settings.groq_model).strip()
        prompt = engine.build_prompt(payload.question, result.hits)
        answer, usage = _generate_with_groq(prompt, model=chosen_model)
        model = chosen_model

    return AskResponse(
        answer=answer,
        question=payload.question,
        sources=sources,
        model=model,
        mode=payload.mode,
        k=payload.k,
        top_score=result.top_semantic_score,
        low_confidence=low_confidence,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        usage=usage,
    )


@app.post("/api/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    """Retrieval debugging endpoint — works without a Groq API key."""
    engine = _require_ready()
    started = time.perf_counter()
    extra = [(d.source or f"my-notes-{i+1}", d.text) for i, d in enumerate(payload.extra_docs)]
    result = engine.retrieve(payload.question, k=payload.k, mode=payload.mode, extra_docs=extra)
    return SearchResponse(
        question=payload.question,
        sources=_sources(result),
        mode=payload.mode,
        k=payload.k,
        top_score=result.top_semantic_score,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


@app.post("/api/reindex", status_code=202)
def reindex() -> dict:
    if state.rebuilding:
        raise HTTPException(status_code=409, detail="A build is already in progress.")
    _start_build(force=True)
    return {"accepted": True, "detail": "Index rebuild started."}
