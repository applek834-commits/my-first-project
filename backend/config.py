"""Configuration for the RAG backend.

Every value comes from an environment variable or a ``.env`` file at the
project root. Secrets are never written into the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent

# A root-level .env is the normal place; backend/.env may override it.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env", override=True)


def _resolve(raw: str) -> Path:
    """Relative paths are resolved against the project root, not the cwd."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Generation (Groq free tier) -------------------------------------
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature: float = _float("TEMPERATURE", 0.2)
    # 800 stays under the free tier's per-model output-token-per-minute cap
    # (qwen/qwen3.8-27b allows 1000) while leaving room for reasoning tokens.
    max_tokens: int = _int("MAX_TOKENS", 800)

    # --- Corpus and chunking (same defaults as the notebook) ------------
    corpus_dir: Path = _resolve(os.getenv("CORPUS_DIR", "./ragg"))
    chunk_size: int = _int("CHUNK_SIZE", 300)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 50)

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 32)
    max_length: int = _int("MAX_LENGTH", 256)

    # --- Retrieval -------------------------------------------------------
    default_k: int = _int("DEFAULT_K", 5)
    max_k: int = _int("MAX_K", 20)
    hybrid_candidates: int = _int("HYBRID_CANDIDATES", 20)
    min_score: float = _float("MIN_SCORE", 0.4)
    abstain_guard: bool = _bool("ABSTAIN_GUARD", True)

    # --- Index cache -----------------------------------------------------
    storage_dir: Path = _resolve(os.getenv("STORAGE_DIR", "./backend/storage"))
    index_path: Path = _resolve(os.getenv("INDEX_PATH", "./backend/storage/rag.index"))
    vectors_path: Path = _resolve(
        os.getenv("VECTORS_PATH", "./backend/storage/rag_vectors.npy")
    )
    meta_path: Path = _resolve(os.getenv("META_PATH", "./backend/storage/rag_meta.json"))

    # --- Server ----------------------------------------------------------
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())


settings = Settings()
