# Deploying

## Short answer

Yes — a **`vercel.json`** at the repo root is the missing file. But one important
caveat: **Vercel can host only the React frontend of this project.** Vercel does
support Python/FastAPI serverless functions in general, but this backend cannot
run inside one:

- `torch` + `transformers` + `faiss-cpu` are far larger than a deployed
  function's bundle limit;
- it loads ~90 MB of MiniLM weights and builds a 1,355-chunk FAISS index in
  memory on startup, which needs more RAM and time than a function invocation
  allows;
- the index cache needs a writable, persistent disk.

So this app deploys as **two services**:

```
Browser ──► Vercel (static React build)
                │  VITE_API_BASE_URL
                ▼
        FastAPI backend on a Python/container host
        (Render · Railway · Fly.io · Hugging Face Spaces)
```

## Why the Vercel import failed

The repo root had no `package.json` and no `vercel.json`. The only manifest was
`frontend/package.json`, so Vercel could not detect a framework, a build command
or an output directory. The new root `vercel.json` supplies all three, and a thin
root `package.json` is included as a fallback for build systems that expect one.

---

## 1 · Frontend on Vercel

1. In Vercel, **Add New… → Project → Import** this GitHub repo.
2. **Root Directory:** leave it at the repo root (`./`). `vercel.json` handles
   the `frontend/` subdirectory.
   - *Alternative:* set Root Directory to `frontend` instead; Vercel then
     auto-detects Vite (`npm run build` → `dist`) and ignores `vercel.json`.
     Either approach works — pick one, not both.
3. **Environment Variables** — add **before** the first build:

   | Name | Value |
   | --- | --- |
   | `VITE_API_BASE_URL` | `https://YOUR-BACKEND-HOST/api` |

   Must include the `/api` suffix and use **HTTPS**, or browsers will block it
   as mixed content. Vite inlines `VITE_*` values at **build time**, so you must
   **redeploy** after changing this value.

4. Click **Deploy**. The build runs
   `npm --prefix frontend ci` → `npm --prefix frontend run build` and publishes
   `frontend/dist`.

> Deploying before the backend exists is fine: the UI builds and renders, but
> shows a "Backend offline" banner and a disabled composer. That is expected,
> not a broken build.

---

## 2 · Backend on a Python host

The root **`Dockerfile`** builds the API. Use a host with **at least 2 GB RAM** —
the 512 MB free tiers will be killed during the embedding/index build.

Good options:

| Host | Notes |
| --- | --- |
| Hugging Face Spaces (Docker SDK) | Free CPU tier has enough RAM; already suited to ML images. Set `PORT=7860`. |
| Render / Railway / Fly.io | Use the `Dockerfile`; choose a Starter plan or bigger. |

Set these environment variables on the backend host (never commit them):

| Name | Value |
| --- | --- |
| `GROQ_API_KEY` | your free Groq key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `MAX_TOKENS` | `800` |
| `ALLOWED_ORIGINS` | `https://YOUR-PROJECT.vercel.app` (exact origin, no trailing slash) |
| `PORT` | the port your host expects (defaults to `8000`; HF Spaces uses `7860`) |

`ALLOWED_ORIGINS` defaults to `http://localhost:5173,http://127.0.0.1:5173`, so
**a deployed frontend will get CORS errors until you add its Vercel origin**.
Comma-separate multiple origins, e.g. localhost plus your production domain.

Notes:

- Health check path: `GET /api/health`.
- The first boot downloads/loads MiniLM and builds the index — allow a few
  minutes; poll `GET /api/status` until it reports `ready`.
- Container filesystems are ephemeral, so the index rebuilds after every
  restart. Mount a persistent volume at `/app/backend/storage` (set
  `STORAGE_DIR` accordingly) to skip repeat builds.
- If `faiss-cpu` has no wheel for your platform, remove it from
  `backend/requirements.txt`; `VectorIndex` falls back to exact NumPy cosine
  search automatically.

---

## 3 · Verify

```bash
curl https://YOUR-BACKEND-HOST/api/health
curl https://YOUR-BACKEND-HOST/api/status
```

Then open the Vercel URL. The topbar should show **Ready**, the sidebar should
list 14 documents / 1,355 chunks, and "What is precision?" should return a
grounded answer with source cards.

---

## 4 · Alternative: proxy instead of CORS

If you'd rather keep the browser on a single origin, leave
`VITE_API_BASE_URL` empty and add a rewrite to `vercel.json`, replacing the
placeholder with your real backend URL:

```json
"rewrites": [
  { "source": "/api/:path*", "destination": "https://YOUR-BACKEND-HOST/api/:path*" }
]
```

Vercel then forwards `/api/*` to FastAPI, so no `ALLOWED_ORIGINS` change is
needed. This URL is static config, so it must be edited in the repo and pushed.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "No framework detected" / "Missing build script" | Use repo root with `vercel.json`, **or** set Root Directory = `frontend`. |
| Build succeeds but `/api/status` 404s | `VITE_API_BASE_URL` was empty or set after the build. Set it and redeploy. |
| "No API at /api …" banner | Same as above — the static host is returning `index.html` for `/api/*`. |
| CORS error in the console | Add the exact Vercel origin to `ALLOWED_ORIGINS` on the backend and restart it. |
| Mixed-content / blocked request | Backend must be HTTPS. |
| Backend container exits or is OOM-killed | Needs ≥2 GB RAM; move off the 512 MB free tier. |
| Status stuck on "Indexing" | First index build takes minutes; read the backend logs. |
| `model_not_found` from Groq | Query `GET /openai/v1/models` and set `GROQ_MODEL` to a currently active ID. |
