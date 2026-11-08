# FastAPI backend image. Deploy this on a Python/container host (Render,
# Railway, Fly.io, Hugging Face Spaces) -- NOT on Vercel serverless functions:
# torch + transformers + faiss-cpu, the MiniLM weights and the in-memory FAISS
# index do not fit a serverless function's size, memory or startup budget.
# 3.11 has the widest prebuilt-wheel coverage for torch / faiss-cpu / scipy,
# so the image builds without compiling anything from source.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000

WORKDIR /app

# Only needed if a dependency has to be built from source on this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first: it avoids the multi-GB CUDA wheel, and pip will then
# treat the `torch>=2.2.0` line in requirements.txt as already satisfied.
RUN pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend ./backend
COPY ragg ./ragg

# Bake the ~90 MB MiniLM weights into the image so boot is not blocked on a
# Hugging Face download. The FAISS index is still built on first startup.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# Hosts expose different ports (HF Spaces uses 7860), so honour $PORT.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
