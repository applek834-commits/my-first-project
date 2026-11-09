import { useState } from 'react'
import {
  IconAlert,
  IconBook,
  IconChevron,
  IconCpu,
  IconDatabase,
  IconKey,
  IconMoon,
  IconRefresh,
  IconSliders,
  IconSparkle,
  IconSun,
  IconTrash,
  IconZap,
} from './Icons'
import MyNotes from './MyNotes'

const MODES = [
  { id: 'hybrid', label: 'Hybrid', hint: 'FAISS + TF-IDF · best for exact numbers' },
  { id: 'semantic', label: 'Semantic', hint: 'FAISS cosine similarity only' },
  { id: 'keyword', label: 'Keyword', hint: 'TF-IDF exact-word matching only' },
]

const SERVERLESS_HINTS = {
  hybrid: 'TF-IDF cosine + BM25 · best all-rounder',
  semantic: 'TF-IDF cosine similarity only',
  keyword: 'BM25 exact-word matching only',
}

export default function Sidebar({
  status,
  controls,
  setControls,
  onReindex,
  onClear,
  onExample,
  reindexing,
  serverless = false,
  exampleQuestions,
  onClose,
  models,
  modelsError,
  onRefreshModels,
  refreshingModels,
  theme,
  setTheme,
  myNotes,
  setMyNotes,
}) {
  const [showFiles, setShowFiles] = useState(false)
  const patch = (change) => setControls((prev) => ({ ...prev, ...change }))
  const defaults = status?.defaults || {}
  const state = status?.status || 'loading'
  const files = status?.document_names || []
  const selectedModel = models.find((m) => m.id === controls.model)
  // The Groq model-list call can be slow or fail. Until it answers (or if it
  // never does), show the backend's configured default instead of a disabled
  // "Loading models…" box, and leave the control usable.
  const fallbackModel = controls.model || status?.groq_model || ''

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <IconBook size={20} />
        </div>
        <div className="brand-text">
          <h1>RAG Course Assistant</h1>
          <p>Grounded answers, with receipts</p>
        </div>
        <button
          type="button"
          className="icon-btn sidebar-close"
          onClick={onClose}
          title="Close panel"
        >
          ×
        </button>
      </div>

      <div className={`status-card status-${state}`}>
        {state === 'loading' && <div className="status-shimmer" />}
        <div className="status-row">
          <span className="status-dot" />
          <strong>
            {state === 'ready'
              ? 'Index ready'
              : state === 'error'
                ? 'Index error'
                : 'Building index…'}
          </strong>
          {state === 'ready' && <span className="status-live">LIVE</span>}
        </div>
        <p className="status-detail">
          {status?.detail || 'Connecting to the backend…'}
        </p>
        {state === 'ready' && (
          <div className="status-stats">
            <span>{status.documents} files</span>
            <span>{status.chunks.toLocaleString()} chunks</span>
            <span>{status.vector_backend}</span>
          </div>
        )}
        {state === 'error' && <IconAlert size={16} className="status-alert" />}
      </div>

      {status && !status.groq_configured && (
        <div className="key-warning">
          <IconKey size={15} />
          <span>
            <strong>No Groq key found.</strong> Add <code>GROQ_API_KEY</code> to{' '}
            <code>.env</code> for generated answers. Retrieval-only mode still works.
          </span>
        </div>
      )}

      <div className="panel">
        <h2 className="panel-title">
          <IconSliders size={13} /> Retrieval
        </h2>

        <label className="field-label">Search mode</label>
        <div className="segmented">
          {MODES.map((mode) => (
            <button
              type="button"
              key={mode.id}
              className={controls.mode === mode.id ? 'active' : ''}
              onClick={() => patch({ mode: mode.id })}
              title={mode.hint}
            >
              {mode.label}
            </button>
          ))}
        </div>
        <p className="field-hint">
          {serverless
            ? SERVERLESS_HINTS[controls.mode]
            : MODES.find((m) => m.id === controls.mode)?.hint}
        </p>

        <div className="field-row">
          <label className="field-label" htmlFor="topk">
            Top-k passages
          </label>
          <span className="field-value">{controls.k}</span>
        </div>
        <input
          id="topk"
          type="range"
          min="1"
          max={defaults.max_k || 10}
          value={controls.k}
          onChange={(event) => patch({ k: Number(event.target.value) })}
        />
        <p className="field-hint">
          Abstains below similarity {defaults.min_score ?? 0.4}
        </p>

        <label className="switch-row">
          <input
            type="checkbox"
            checked={controls.showChunks}
            onChange={(event) => patch({ showChunks: event.target.checked })}
          />
          <span>Expand retrieved chunks</span>
        </label>
      </div>

      <div className="panel">
        <h2 className="panel-title">
          <IconSparkle size={13} /> Generation
        </h2>
        <label className="field-label" htmlFor="model-select">
          Groq model
        </label>
        <div className="select-row">
          <select
            id="model-select"
            value={controls.model}
            onChange={(event) => patch({ model: event.target.value })}
            disabled={!models.length && !fallbackModel}
          >
            {!models.length && (
              <option value={controls.model}>
                {fallbackModel || 'Server default'}
              </option>
            )}
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="icon-btn select-refresh"
            onClick={onRefreshModels}
            title="Refresh the active model list from Groq"
            disabled={refreshingModels}
          >
            <IconRefresh size={15} className={refreshingModels ? 'spin' : ''} />
          </button>
        </div>
        <p className="field-hint">
          {selectedModel?.context_window
            ? `${Number(selectedModel.context_window).toLocaleString()} token context · `
            : ''}
          live from your Groq account
        </p>
        {modelsError && <p className="field-error">{modelsError}</p>}

        <label className="switch-row">
          <input
            type="checkbox"
            checked={controls.useLlm}
            onChange={(event) => patch({ useLlm: event.target.checked })}
          />
          <span>Generate an answer</span>
        </label>
      </div>

      <div className="panel">
        <h2 className="panel-title">
          <IconZap size={13} /> Try a question
        </h2>
        <div className="examples">
          {exampleQuestions.map((question) => (
            <button type="button" key={question} onClick={() => onExample(question)}>
              <span>{question}</span>
              <IconChevron size={13} />
            </button>
          ))}
        </div>
      </div>

      <div className="panel corpus-panel">
        <button
          type="button"
          className="corpus-head"
          onClick={() => setShowFiles((prev) => !prev)}
        >
          <IconDatabase size={15} />
          <span>Corpus · {files.length} files</span>
          <IconChevron size={14} className={showFiles ? 'chevron-open' : ''} />
        </button>
        {showFiles && (
          <ul className="corpus-files">
            {files.map((name) => (
              <li key={name} title={name}>
                <IconBook size={11} /> {name}
              </li>
            ))}
          </ul>
        )}
        <p className="corpus-meta">
          {defaults.chunk_size ?? 300} char chunks · {defaults.chunk_overlap ?? 50}{' '}
          overlap
          <br />
          <IconCpu size={11} /> {status?.embedding_model || 'all-MiniLM-L6-v2'}
        </p>
      </div>

      <MyNotes notes={myNotes} setMyNotes={setMyNotes} />

      <div className="panel appearance-panel">
        <h2 className="panel-title">Appearance</h2>
        <div className="segmented">
          <button
            type="button"
            className={theme === 'light' ? 'active' : ''}
            onClick={() => setTheme('light')}
          >
            <IconSun size={13} /> Light
          </button>
          <button
            type="button"
            className={theme === 'dark' ? 'active' : ''}
            onClick={() => setTheme('dark')}
          >
            <IconMoon size={13} /> Dark
          </button>
        </div>
      </div>

      <div className="sidebar-footer">
        <button
          type="button"
          className="ghost-btn"
          onClick={onReindex}
          disabled={reindexing}
        >
          <IconRefresh size={15} className={reindexing ? 'spin' : ''} />
          {reindexing ? 'Rebuilding…' : serverless ? 'Index build info' : 'Rebuild index'}
        </button>
        <button type="button" className="ghost-btn" onClick={onClear}>
          <IconTrash size={15} /> Clear chat
        </button>
        <p className="footer-model">
          {serverless
            ? 'Vercel serverless · TF-IDF + BM25 · Groq'
            : 'FastAPI · FAISS · Groq free tier'}
        </p>
      </div>
    </aside>
  )
}
