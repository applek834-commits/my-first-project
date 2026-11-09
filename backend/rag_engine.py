"""RAG engine — a service-oriented port of the Week 8 Day 2 notebook.

Notebook cells that live here:
  * Step 0b  -> load_corpus()
  * Step 1a  -> recursive_chunk()
  * Step 1b  -> RAGEngine._chunk_corpus()
  * Step 2   -> MiniLMEmbedder.embed()  (mean-pooled all-MiniLM-L6-v2)
  * Step 3a  -> VectorIndex (IndexFlatIP + L2 normalisation = cosine)
  * Step 3b  -> VectorIndex.save()/load() + metadata cache
  * Step 4a  -> RAGEngine.retrieve()
  * Step 5   -> RAGEngine.build_prompt()
  * Step 8   -> RAGEngine.retrieve(mode="hybrid") (semantic + TF-IDF)

Generation (Step 6) is deliberately NOT here: it lives in main.py and uses
the Groq API with a free model instead of a local Qwen/DeepSeek pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # FAISS is preferred; NumPy exact search keeps the app usable without it.
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on the local install
    faiss = None
    HAS_FAISS = False

try:
    from .config import Settings
except ImportError:  # pragma: no cover - allows `cd backend && uvicorn main:app`
    from config import Settings

logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "semantic", "keyword"]
SEPARATORS = ["\n\n", "\n", ". ", " "]

SYSTEM_PROMPT = (
    "You are a course assistant. You answer questions strictly from the "
    "context supplied with each question. If the context does not contain the "
    "answer, say exactly \"I do not know\". Never guess, never use outside "
    "knowledge, and cite the source file for each fact you use."
)


# ---------------------------------------------------------------------------
# Step 0b: corpus
# ---------------------------------------------------------------------------
def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Read every .txt file in the corpus folder, keyed by filename."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus folder not found: {corpus_dir}")
    docs: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs[path.name] = text
    if not docs:
        raise ValueError(f"No non-empty .txt documents found in {corpus_dir}")
    return docs


def corpus_signature(corpus_dir: Path) -> str:
    """Hash of file names + contents, used to invalidate the saved index."""
    digest = hashlib.sha256()
    for path in sorted(corpus_dir.glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.md5(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Step 1a: recursive chunking (unchanged from the notebook)
# ---------------------------------------------------------------------------
def recursive_chunk(
    text: str, chunk_size: int = 300, chunk_overlap: int = 50
) -> list[str]:
    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        if not seps:
            step = max(1, chunk_size - chunk_overlap)
            return [t[i : i + chunk_size] for i in range(0, len(t), step)]
        sep = seps[0]
        parts = t.split(sep)
        # keep the separator attached to each piece, except the last one
        pieces = [
            p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)
        ]
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current)
                if len(piece) > chunk_size:
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    chunks = [c for c in _split(text, SEPARATORS) if c.strip()]
    # Each chunk carries the first `chunk_overlap` characters of the next one,
    # so a sentence cut at a boundary still appears whole somewhere.
    out = []
    for i, c in enumerate(chunks):
        if i < len(chunks) - 1:
            out.append(c + chunks[i + 1][:chunk_overlap])
        else:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Step 2: embeddings
# ---------------------------------------------------------------------------
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Equivalent to faiss.normalize_L2, but works with or without FAISS."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class MiniLMEmbedder:
    """Mean-pooled sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)."""

    def __init__(self, settings: Settings):
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading embedding model %s", settings.embedding_model)
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        self.model = AutoModel.from_pretrained(settings.embedding_model)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)
        logger.info("Embedding model ready (dim=%s)", self.dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch

        vectors = []
        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.settings.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            mean = (output * mask).sum(1) / mask.sum(1)
            vectors.append(mean.numpy())
        return np.vstack(vectors).astype("float32")


# ---------------------------------------------------------------------------
# Step 3: vector index
# ---------------------------------------------------------------------------
class VectorIndex:
    """FAISS IndexFlatIP, with an exact NumPy cosine fallback."""

    def __init__(self, dim: int, use_faiss: bool = HAS_FAISS):
        self.dim = dim
        self.use_faiss = use_faiss and HAS_FAISS
        self._index = faiss.IndexFlatIP(dim) if self.use_faiss else None
        self._matrix: np.ndarray | None = None

    @property
    def backend(self) -> str:
        return "faiss" if self.use_faiss else "numpy"

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal) if self._index is not None else (
            0 if self._matrix is None else self._matrix.shape[0]
        )

    def add(self, vectors: np.ndarray) -> None:
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._matrix = np.ascontiguousarray(vectors, dtype="float32")

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), both shaped (1, k), like FAISS does."""
        if self._index is not None:
            return self._index.search(query, k)
        assert self._matrix is not None
        sims = self._matrix @ query[0]
        ids = np.argsort(sims)[::-1][:k].astype("int64")
        return sims[ids].reshape(1, -1), ids.reshape(1, -1)

    def save(self, index_path: Path, vectors_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(index_path))
        else:
            assert self._matrix is not None
            np.save(str(vectors_path), self._matrix)

    @classmethod
    def load(
        cls,
        index_path: Path,
        vectors_path: Path,
        dim: int,
        backend: str,
    ) -> "VectorIndex":
        if backend == "faiss" and HAS_FAISS and index_path.exists():
            obj = cls(dim, use_faiss=True)
            obj._index = faiss.read_index(str(index_path))
            return obj
        if not vectors_path.exists():
            raise FileNotFoundError("Saved vectors are missing; reindex required.")
        obj = cls(dim, use_faiss=False)
        obj._matrix = np.load(str(vectors_path)).astype("float32")
        return obj


# ---------------------------------------------------------------------------
# Retrieval result types
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    rank: int
    score: float
    source: str
    chunk: str
    semantic_score: float | None = None
    keyword_score: float | None = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    mode: RetrievalMode
    k: int
    top_semantic_score: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, settings: Settings, force_rebuild: bool = False):
        self.settings = settings
        self.docs = load_corpus(settings.corpus_dir)
        self.signature = corpus_signature(settings.corpus_dir)
        self.embedder = MiniLMEmbedder(settings)
        self._load_or_build(force_rebuild)

    # -- Step 1b + Step 2 + Step 3 ----------------------------------------
    def _chunk_corpus(self) -> tuple[list[str], list[str]]:
        chunks: list[str] = []
        sources: list[str] = []
        for name, text in self.docs.items():
            for piece in recursive_chunk(
                text, self.settings.chunk_size, self.settings.chunk_overlap
            ):
                chunks.append(piece)
                sources.append(name)
        return chunks, sources

    def _read_meta(self) -> dict | None:
        if not self.settings.meta_path.exists():
            return None
        try:
            return json.loads(self.settings.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_is_valid(self, meta: dict | None) -> bool:
        if not meta:
            return False
        expected = {
            "signature": self.signature,
            "embedding_model": self.settings.embedding_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "dimension": self.embedder.dim,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            return False
        path = (
            self.settings.index_path
            if meta.get("vector_backend") == "faiss" and HAS_FAISS
            else self.settings.vectors_path
        )
        return path.exists()

    def _load_or_build(self, force: bool) -> None:
        meta = None if force else self._read_meta()
        if self._cache_is_valid(meta):
            assert meta is not None
            logger.info("Loading saved index from %s", self.settings.storage_dir)
            self.chunks: list[str] = meta["chunks"]
            self.sources: list[str] = meta["sources"]
            self.index = VectorIndex.load(
                self.settings.index_path,
                self.settings.vectors_path,
                self.embedder.dim,
                meta.get("vector_backend", "faiss"),
            )
            if self.index.ntotal != len(self.chunks):
                logger.warning("Index size mismatch; rebuilding.")
                self._build()
                return
        else:
            self._build()

        # Step 8's keyword retriever is cheap to fit at startup.
        self.tfidf = TfidfVectorizer().fit(self.chunks)
        self.tfidf_matrix = self.tfidf.transform(self.chunks)
        logger.info(
            "RAG ready: %s documents, %s chunks, %s index",
            len(self.docs),
            len(self.chunks),
            self.index.backend,
        )

    def _build(self) -> None:
        logger.info("Building index from %s", self.settings.corpus_dir)
        self.chunks, self.sources = self._chunk_corpus()
        vectors = l2_normalize(self.embedder.embed(self.chunks))
        self.index = VectorIndex(self.embedder.dim)
        self.index.add(vectors)
        self.index.save(self.settings.index_path, self.settings.vectors_path)

        meta = {
            "signature": self.signature,
            "embedding_model": self.settings.embedding_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "dimension": self.embedder.dim,
            "vector_backend": self.index.backend,
            "chunks": self.chunks,
            "sources": self.sources,
            "documents": sorted(self.docs.keys()),
        }
        self.settings.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    # -- Step 4a + Step 8 -------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
        extra_docs: list = None,
    ) -> RetrievalResult:
        total = len(self.chunks)
        if total == 0:
            return RetrievalResult(hits=[], mode=mode, k=0)

        k = max(1, min(int(k), self.settings.max_k, total))
        candidates = max(k, min(self.settings.hybrid_candidates, total))

        # Semantic side: same embedding model, normalised, IndexFlatIP.
        query = l2_normalize(self.embedder.embed([question]))
        scores, ids = self.index.search(query, candidates)
        semantic = {
            int(ids[0][i]): float(scores[0][i])
            for i in range(candidates)
            if int(ids[0][i]) >= 0
        }
        top_semantic = max(semantic.values(), default=0.0)

        # Keyword side: TF-IDF from the notebook's stretch step.
        keyword_scores = (
            self.tfidf.transform([question]) @ self.tfidf_matrix.T
        ).toarray()[0]
        keyword_top = keyword_scores.argsort()[::-1][:candidates]
        keyword = {
            int(i): float(keyword_scores[i])
            for i in keyword_top
            if float(keyword_scores[i]) > 0
        }
        max_keyword = max(keyword.values(), default=1.0) or 1.0

        if mode == "semantic":
            ranked = sorted(semantic.items(), key=lambda x: x[1], reverse=True)[:k]
            combined_scores = dict(ranked)
        elif mode == "keyword":
            ranked = sorted(keyword.items(), key=lambda x: x[1], reverse=True)[:k]
            combined_scores = {i: s / max_keyword for i, s in ranked}
        else:
            combined: dict[int, float] = {}
            for i, score in semantic.items():
                combined[i] = combined.get(i, 0.0) + score
            for i, score in keyword.items():
                combined[i] = combined.get(i, 0.0) + (score / max_keyword)
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
            combined_scores = dict(ranked)

        hits = [
            Hit(
                rank=rank,
                score=round(combined_scores.get(idx, 0.0), 4),
                source=self.sources[idx],
                chunk=self.chunks[idx],
                semantic_score=round(semantic.get(idx, 0.0), 4),
                keyword_score=round(keyword.get(idx, 0.0), 4),
            )
            for rank, (idx, _) in enumerate(ranked, start=1)
        ]

        # Merge user-supplied notes, if any. The notes are chunked and
        # embedded with the same MiniLM model, then their cosine scores are
        # normalised and merged with the corpus hits so a user's own notes
        # can participate in retrieval.
        top_extra = 0.0
        if extra_docs:
            extra_hits, top_extra = self._retrieve_extra(question, extra_docs, k)
            if extra_hits:
                baked_max = max((h.score for h in hits), default=0.0) or 1.0
                all_hits = [
                    {"rank": 0, "score": h.score / baked_max, "source": h.source,
                     "chunk": h.chunk, "semantic_score": h.semantic_score,
                     "keyword_score": h.keyword_score}
                    for h in hits
                ] + [
                    {"rank": 0, "score": h.score, "source": h.source,
                     "chunk": h.chunk, "semantic_score": h.semantic_score,
                     "keyword_score": h.keyword_score}
                    for h in extra_hits
                ]
                all_hits.sort(key=lambda h: h["score"], reverse=True)
                all_hits = all_hits[:k]
                hits = [
                    Hit(
                        rank=rank,
                        score=round(h["score"], 4),
                        source=h["source"],
                        chunk=h["chunk"],
                        semantic_score=round(h["semantic_score"], 4),
                        keyword_score=round(h["keyword_score"], 4),
                    )
                    for rank, h in enumerate(all_hits, start=1)
                ]

        final_top = max(top_semantic, top_extra)
        return RetrievalResult(
            hits=hits,
            mode=mode,
            k=k,
            top_semantic_score=round(final_top, 4),
        )

    def _retrieve_extra(self, question, extra_docs, k):
        """Embed and search user-supplied documents at request time."""
        chunks = []
        sources = []
        for source, text in extra_docs:
            for piece in recursive_chunk(text, self.settings.chunk_size, self.settings.chunk_overlap):
                chunks.append(piece)
                sources.append(source)
        if not chunks:
            return [], 0.0

        vectors = l2_normalize(self.embedder.embed(chunks))
        query = l2_normalize(self.embedder.embed([question]))
        # NumPy cosine via dot product (vectors are already L2-normalised)
        sims = vectors @ query[0]
        top_ids = np.argsort(sims)[::-1][:k]
        top_extra = float(sims[top_ids[0]]) if len(top_ids) > 0 else 0.0

        hits = []
        max_sim = max(float(s) for s in sims) or 1.0
        for rank, idx in enumerate(top_ids, start=1):
            sim = float(sims[idx])
            if sim <= 0:
                continue
            hits.append(Hit(
                rank=rank,
                score=round(sim / max_sim, 4),
                source=sources[idx],
                chunk=chunks[idx],
                semantic_score=round(sim, 4),
                keyword_score=round(sim / max_sim, 4),
            ))
        return hits, top_extra

    # -- Step 5: augmented, grounded prompt ------------------------------
    def build_prompt(self, question: str, hits: list[Hit]) -> str:
        context = ""
        for i, hit in enumerate(hits, start=1):
            context += f"\n[Source {i}: {hit.source}]\n{hit.chunk}\n"
        return (
            "Answer the question using ONLY the context below. "
            'If the answer is not in the context, say "I do not know". '
            "Do not guess. Cite the source file for each fact you use.\n"
            f"\nContext:{context}"
            f"\nQuestion: {question}\nAnswer:"
        )

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.docs),
            "document_names": sorted(self.docs.keys()),
            "chunks": len(self.chunks),
            "dimension": self.embedder.dim,
            "vector_backend": self.index.backend,
            "embedding_model": self.settings.embedding_model,
        }
