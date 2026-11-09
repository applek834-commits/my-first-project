"""Pure-stdlib text helpers for the Vercel serverless RAG backend.

This module is shared by ``scripts/build_serverless_index.py`` (which runs on a
developer machine) and ``api/_serverless.py`` (which runs inside a Vercel
Python function). Keeping one tokenizer here guarantees the precomputed index
and live queries speak exactly the same vocabulary.

No third-party imports: the function bundle must work without torch,
transformers, numpy, faiss or a requirements.txt install step.
"""
from __future__ import annotations

import re
from pathlib import Path

# Step 1a of the notebook — identical to backend/rag_engine.recursive_chunk.
SEPARATORS = ["\n\n", "\n", ". ", " "]

SYSTEM_PROMPT = (
    "You are a course assistant. You answer questions strictly from the "
    "context supplied with each question. If the context does not contain the "
    'answer, say exactly "I do not know". Never guess, never use outside '
    "knowledge, and cite the source file for each fact you use."
)


def recursive_chunk(
    text: str, chunk_size: int = 300, chunk_overlap: int = 50
) -> list[str]:
    """Split on paragraph, line, sentence, then word, in that order."""

    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        if not seps:
            step = max(1, chunk_size - chunk_overlap)
            return [t[i : i + chunk_size] for i in range(0, len(t), step)]
        sep = seps[0]
        parts = t.split(sep)
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


MIN_CHUNK_CHARS = 100


def merge_fragments(chunks: list[str], min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Fold sub-100-char trailing fragments into the next passage.

    The notebook's overlap sometimes leaves a fragment like "### Average
    Precision (AP) and mAP". Those tiny chunks dominate L2-normalised TF-IDF
    scores without containing an answer, so both the offline index builder and
    live user-supplied documents go through this same cleanup.
    """
    merged: list[str] = []
    carry = ""
    for chunk in chunks:
        piece = f"{carry} {chunk}".strip() if carry else chunk
        if len(piece) < min_chars:
            carry = piece
            continue
        merged.append(piece)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}".strip()
        else:
            merged.append(carry)
    return merged


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Read every non-empty .txt file, keyed by filename (notebook Step 0b)."""
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


# --- Sparse-retrieval vocabulary -------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")

_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves tell explain define describe give list show name say
    means meaning mean use used using example examples please question answer
    """.split()
)


def _stem(word: str) -> str:
    """Conservative suffix folding so 'embeddings' matches 'embedding'.

    It is deliberately mild: the same function runs at index time and query
    time, so aggressive stemming that collapses unrelated words is worse than
    missing a plural.
    """
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed") and not word.endswith("eed"):
        return word[:-2]
    if (
        len(word) > 4
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is", "as"))
    ):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, drop stopwords and fold simple suffixes."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        word = raw.strip("-'")
        if len(word) < 2 or word in _STOPWORDS:
            continue
        out.append(_stem(word))
    return out


def build_prompt(question: str, hits: list[dict]) -> str:
    """Step 5 of the notebook — grounded, source-labelled context."""
    context = ""
    for i, hit in enumerate(hits, start=1):
        context += f"\n[Source {i}: {hit['source']}]\n{hit['chunk']}\n"
    return (
        "Answer the question using ONLY the context below. "
        'If the answer is not in the context, say "I do not know". '
        "Do not guess. Cite the source file for each fact you use.\n"
        f"\nContext:{context}"
        f"\nQuestion: {question}\nAnswer:"
    )
