# Deploying

## Short answer

**One Vercel project now hosts the whole app.** The React frontend is a static
Vite build, and the backend runs as Vercel Python Functions in the root `api/`
directory. There is no second server, no Docker host, no CORS configuration and
no `VITE_API_BASE_URL` to set. The only thing Vercel needs from you is the
`GROQ_API_KEY` environment variable.

```
Browser ──► https://YOUR-PROJECT.vercel.app
              ├── /*          static React build (frontend/dist)
              └── /api/*      Python serverless functions (api/*.py)
                                 ├── precomputed index (api/_index_data.py)
                                 └── Groq chat completions (HTTPS)
```

The `vercel.json` at the repo root supplies the framework, install command,
build command and output directory. Vercel additionally maps every `.py` file
in `api/` to a route (`api/ask.py` → `/api/ask`). Files beginning with `_`
(`_serverless.py`, `_text.py`, `_index_data.py`) are helper modules, not routes.

> **Important — why every route file contains `class handler(...)`.**
> Vercel only creates a Python function when it can **statically** see a
> top-level `app` or a top-level `class handler` in the file
> ([vercel/vercel#14493](https://github.com/vercel/vercel/pull/14493)). A
> dynamic `handler = make_handler("ask")` assignment is classified as a helper
> module and silently skipped; then a `functions` rule in `vercel.json` fails
> the whole build with `unused_function`. Each route therefore defines a real
> `class handler` that subclasses the generated handler. `scripts/test_serverless.py`
> replicates Vercel's AST check so this cannot regress. There is deliberately
> **no `functions` block** in `vercel.json` — `.vercelignore` keeps the function
> bundles small instead, without any pattern that Vercel has to validate.

> **Why the earlier guidance changed:** the original FastAPI backend needs
> `torch`, `transformers`, `faiss-cpu`, ~90 MB of MiniLM weights and a writable
> index cache. That cannot fit a serverless function. Instead of hosting it
> elsewhere, there is now a second, API-compatible backend written with the
> Python standard library only and a **precomputed** retrieval index.

---

## 1 · Deploy to Vercel

1. **Add New… → Project → Import** this GitHub repository.
2. **Root Directory:** leave at `./` (the repo root). `vercel.json` handles the
   `frontend/` subdirectory.
3. **Framework / Build settings:** leave them alone — everything is pinned in
   `vercel.json`:

   | Setting | Value |
   | --- | --- |
   | Framework Preset | Vite |
   | Install Command | `npm --prefix frontend ci` |
   | Build Command | `npm --prefix frontend run build` |
   | Output Directory | `frontend/dist` |

4. **Environment Variables** — add before the first deploy:

   | Name | Value | Required |
   | --- | --- | --- |
   | `GROQ_API_KEY` | your free key from <https://console.groq.com/keys> | **Yes**, for generated answers |
   | `GROQ_MODEL` | `openai/gpt-oss-20b` | No (this is the default) |
   | `MAX_TOKENS` | `800` | No (default; keep ≤1000 on the free tier) |
   | `MIN_SCORE` | `0.12` | No (serverless abstain threshold) |
   | `TEMPERATURE` | `0.2` | No (default) |

   **Do NOT set `VITE_API_BASE_URL`.** It must stay empty so the browser calls
   the same-origin `/api` functions. Setting it to a placeholder is the most
   common cause of a broken deploy.

5. Click **Deploy**.

> Without `GROQ_API_KEY` the site still builds and renders: retrieval and
> `/api/search` work, but generated answers return a clear "key is not set"
> message. Turn off **Generate an answer** in the sidebar to use the app with
> no key at all.

---

## 2 · Verify

```bash
curl https://YOUR-PROJECT.vercel.app/api/health
curl https://YOUR-PROJECT.vercel.app/api/status
curl https://YOUR-PROJECT.vercel.app/api/documents
curl -X POST https://YOUR-PROJECT.vercel.app/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is precision?","k":5,"mode":"hybrid"}'
```

`/api/status` should report `status: "ready"`, 14 documents, ~1,277 chunks and
`runtime: "vercel-serverless"` — with no warm-up delay, because the index is
already baked in. In the UI the topbar shows **Ready** immediately and the
sidebar footer reads *Vercel serverless · TF-IDF + BM25 · Groq*.

---

## 3 · What is different from the local FastAPI backend

Both backends answer the exact same REST contract, so the React app is
unchanged. Only the retrieval internals differ.

| | Local (`backend/`) | Vercel (`api/`) |
| --- | --- | --- |
| Runtime | Uvicorn + FastAPI | Python 3.12 serverless, stdlib only |
| Dependencies | torch, transformers, faiss-cpu, scikit-learn, groq | none (no `requirements.txt`) |
| Embeddings | `all-MiniLM-L6-v2`, 384-d, downloaded on first boot | none — TF-IDF weights precomputed |
| Index | FAISS `IndexFlatIP`, built in RAM, cached on disk | inverted index in `api/_index_data.py` (≈294 KiB) |
| Retrieval | MiniLM cosine + TF-IDF | sublinear TF-IDF cosine + BM25 + query-coverage confidence |
| Chunks | 1,355 | 1,277 (sub-100-char fragments merged) |
| Abstain threshold | `MIN_SCORE=0.4` (cosine) | `MIN_SCORE=0.12` (calibrated, see below) |
| Cold start | minutes for the first index build | milliseconds to decode the blob |
| `/api/reindex` | rebuilds FAISS in the background | returns build instructions (read-only FS) |

The serverless confidence score multiplies the best TF-IDF cosine by the share
of the query's vocabulary that exists in the corpus. That prevents a single
rare out-of-corpus word from looking like a confident match. Measured with
`scripts/calibrate_serverless.py`: in-corpus example questions score
**0.39–0.55**, out-of-corpus ones **0.00–0.07**, so `0.12` sits in the gap.

The trade-off is honest: this is **sparse lexical retrieval, not neural
semantic search**. It handles exact terms, numbers and near-morphological
variants well, but it will not match a synonym the corpus never uses. For the
full MiniLM + FAISS experience, run the local backend (section 5).

---

## 4 · Changing the corpus

The Vercel filesystem is read-only, so the index is generated on your machine
and committed. After editing anything in `ragg/`:

```bash
python scripts/build_serverless_index.py     # rewrites api/_index_data.py
python scripts/calibrate_serverless.py      # optional: re-check MIN_SCORE
python scripts/test_serverless.py           # optional: 55 offline checks
git add api/_index_data.py ragg/
git commit -m "Refresh serverless index" && git push
```

Vercel redeploys automatically on the push. `scripts/` needs only the standard
library — no virtualenv, no GPU, no model download.

`api/_index_data.py` is a single Python module holding a zlib-compressed,
base64-encoded JSON blob. Embedding the data in a `.py` module (rather than a
`.json` file) guarantees Vercel's Python bundler ships it with every function.

### Adding your own notes from the app

The sidebar includes a **My notes** panel where you can add `.txt`/`.md` files
or paste text directly. Notes are stored in `localStorage` (per-browser) and
sent with each question as the `extra_docs` field in the `/api/ask` and
`/api/search` request body. The serverless function chunks and TF-IDF-scores
them alongside the baked corpus at request time, so a matching note can
surface in the top-k sources. No storage, credentials, or redeploy required.
Limits: 10 documents, 50 KB each, 200 KB total.

---

## 5 · Local development (full MiniLM + FAISS)

Nothing about local development changed:

```bash
cp .env.example .env          # paste your Groq key
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
# second terminal
cd frontend && npm install && npm run dev
```

Vite proxies `/api` to port 8000. The UI detects the local backend from
`/api/status` and labels the pipeline *MiniLM · 384d / FAISS + TF-IDF*.

The root `Dockerfile` is still available if you ever want to host that full
FastAPI service separately (Render, Railway, Fly.io, HF Spaces). In that case
set `VITE_API_BASE_URL=https://YOUR-HOST/api` at build time and add the Vercel
origin to `ALLOWED_ORIGINS` on the container. This is optional — not needed
for the normal Vercel deployment.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/api/status` returns 404 or HTML | The `api/` functions were not built. Keep Root Directory = `./` (not `frontend/`), and do not set `VITE_API_BASE_URL`. Check the Vercel **Functions** tab. |
| "No framework detected" | Import from the repo root; `vercel.json` + root `package.json` supply the Vite preset. |
| "No JSON API at /api …" banner | Same as the 404 above — a stale `VITE_API_BASE_URL` build variable also causes it. Remove it and redeploy. |
| `GROQ_API_KEY is not set` | Add it in **Project Settings → Environment Variables**, then redeploy (env changes do not apply to old deployments). |
| Status shows "Indexing" forever | Not possible in serverless mode; if you see it, the browser is hitting the local FastAPI backend. Clear `VITE_API_BASE_URL`. |
| `model_not_found` / 404 from Groq | The model ID was retired. Open `GET /api/models` or the sidebar model picker and choose an active one, or set `GROQ_MODEL`. |
| 429 "output tokens per minute" | Lower `MAX_TOKENS` to ~600, or pick a model with a higher free-tier cap. |
| "I do not know" on a reasonable question | Raise **Top-k**, switch to `hybrid`, or lower `MIN_SCORE`. The serverless index is lexical, so rephrase using words from the notes. |
| CORS error | Should not happen on Vercel (single origin, `Access-Control-Allow-Origin: *`). If using a separate backend, add the Vercel origin to `ALLOWED_ORIGINS`. |
| `Error: [unused_function] The pattern … doesn't match any Serverless Functions` | Vercel created **zero** Python functions. Almost always because an `api/*.py` file has no top-level `class handler` / `app` (see the note above). Fix the class, or remove the `functions` block entirely — this repo no longer has one. |
| Function bundle too large | `.vercelignore` excludes `backend/`, `ragg/`, `docs/`, `scripts/`, the notebook, images and `frontend/node_modules/` from the deployment source. Add more patterns there if needed. |
| Chunk count differs from the README | Expected: local FAISS = 1,355, serverless = 1,277 after fragment merging. |
