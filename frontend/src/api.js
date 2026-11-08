// Thin client for the FastAPI backend. In dev, Vite proxies /api -> :8000.
// Set VITE_API_BASE_URL to point at a deployed backend instead.
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
      'Cannot reach the backend. Start FastAPI with: uvicorn backend.main:app --reload --port 8000',
    )
  }

  const raw = await response.text()
  let data = null
  try {
    data = raw ? JSON.parse(raw) : null
  } catch {
    data = null
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
