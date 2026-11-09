// Thin client for the RAG API. In dev, Vite proxies /api -> :8000 (FastAPI).
// On Vercel the same-origin /api routes are Python serverless functions.
// Set VITE_API_BASE_URL only if you host a separate backend instead.
const BASE = import.meta.env.VITE_API_BASE_URL || '/api'

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    throw new Error(
      BASE === '/api'
        ? 'Cannot reach /api. Locally, start FastAPI with: uvicorn backend.main:app --reload --port 8000. On Vercel, check that the Python functions in api/ were deployed.'
        : `Cannot reach the backend at ${BASE}. Check the URL, that the service is running, and ALLOWED_ORIGINS on the backend.`,
    )
  }

  const raw = await response.text()
  let data = null
  try {
    data = raw ? JSON.parse(raw) : null
  } catch {
    data = null
  }

  // A static host returns index.html for unknown paths. Give a useful hint
  // instead of dumping HTML into the error banner.
  if (!response.ok && data === null && raw) {
    throw new Error(
      BASE === '/api'
        ? `No JSON API at /api (HTTP ${response.status}). On Vercel, confirm the Python functions in api/ were built; for a separate backend, set VITE_API_BASE_URL and rebuild.`
        : `HTTP ${response.status} from ${BASE} was not JSON. Is this the backend URL (ending in /api)?`,
    )
  }

  if (!response.ok) {
    let detail = data?.detail ?? raw ?? `HTTP ${response.status}`
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || String(d)).join('; ')
    if (typeof detail !== 'string') detail = JSON.stringify(detail)
    throw new Error(detail || `HTTP ${response.status}`)
  }
  return data
}

const post = (path, body) =>
  request(path, { method: 'POST', body: JSON.stringify(body) })

export const getStatus = () => request('/status')
export const getDocuments = () => request('/documents')
export const getModels = (refresh = false) =>
  request(`/models${refresh ? '?refresh=true' : ''}`)
export const askQuestion = (payload) => post('/ask', payload)
export const searchOnly = (payload) =>
  post('/search', { question: payload.question, k: payload.k, mode: payload.mode })
export const reindex = () => post('/reindex', {})
