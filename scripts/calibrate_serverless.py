"""Measure TF-IDF cosine scores so MIN_SCORE is a real, measured threshold.

The local MiniLM backend abstains below 0.4 cosine. Sparse TF-IDF cosine is on
a different (lower) scale, so the serverless backend needs its own value.
This script prints the top confidence for the UI's example questions plus
deliberately out-of-corpus ones, and shows which would be abstained.

    python scripts/calibrate_serverless.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import _serverless as rag  # noqa: E402

IN_CORPUS = [
    "What is precision?",
    "What does top-k mean in retrieval?",
    "How do I handle missing values in pandas?",
    "What is Retrieval-Augmented Generation?",
    "What is F1 score?",
    "Explain fine-tuning",
]
OUT_OF_CORPUS = [
    "What is LangChain?",
    "Who won the 2022 football world cup?",
    "How do I bake sourdough bread?",
    "Explain quantum chromodynamics",
]


def main() -> None:
    threshold = rag.MIN_SCORE
    print(f"MIN_SCORE = {threshold}\n")
    for label, questions in (("IN CORPUS", IN_CORPUS), ("OUT OF CORPUS", OUT_OF_CORPUS)):
        print(f"--- {label} ---")
        for question in questions:
            hits, top = rag.retrieve(question, k=5, mode="hybrid")
            best = hits[0]["source"] if hits else "-"
            verdict = "ABSTAIN" if top < threshold else "answer"
            print(f"  {top:6.3f}  {verdict:7s}  {best:34s}  {question}")
        print()
    print("Adjust MIN_SCORE (Vercel env var) if the two groups overlap.")


if __name__ == "__main__":
    main()
