import { useCallback, useEffect, useRef, useState } from 'react'
import Message from './components/Message'
import Pipeline from './components/Pipeline'
import Sidebar from './components/Sidebar'
import {
  IconArrow,
  IconMenu,
  IconMoon,
  IconSend,
  IconSparkle,
  IconSun,
} from './components/Icons'
import { askQuestion, getModels, getStatus, reindex } from './api'

const EXAMPLES = [
  'What is precision?',
  'What does top-k mean in retrieval?',
  'How do I handle missing values in pandas?',
  'What is Retrieval-Augmented Generation?',
  'What is LangChain?',
]

const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('rag-theme') || 'light',
  )
  const [status, setStatus] = useState(null)
  const [models, setModels] = useState([])
  const [modelsError, setModelsError] = useState(null)
  const [refreshingModels, setRefreshingModels] = useState(false)
  const [controls, setControls] = useState({
    mode: 'hybrid',
    k: 5,
    useLlm: true,
    showChunks: false,
    model: '',
  })
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [banner, setBanner] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const inputRef = useRef(null)

  // Theme lives on <html> so CSS variables can be swapped globally.
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('rag-theme', theme)
    } catch {
      /* storage blocked — theme still applies for this session */
    }
  }, [theme])

  // Keep the sidebar status fresh while the index builds or rebuilds.
  useEffect(() => {
    let active = true
    let timer
    const load = async () => {
      try {
        const next = await getStatus()
        if (!active) return
        setStatus(next)
        setBanner((prev) =>
          prev && prev.startsWith('Backend offline') ? null : prev,
        )
        // The serverless index is baked in, so status is always "ready".
        // Stop polling instead of generating ~900 function calls per hour
        // for a value that can never change during a deployment.
        if (next.runtime === 'vercel-serverless' && next.status === 'ready') {
          return
        }
      } catch (err) {
        if (active) setBanner(`Backend offline — ${err.message}`)
      }
      if (active) timer = setTimeout(load, 4000)
    }
    load()
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [])

  const loadModels = useCallback(async (refresh = false) => {
    if (refresh) setRefreshingModels(true)
    try {
      const data = await getModels(refresh)
      const list = data.models || []
      setModels(list)
      setModelsError(data.error || null)
      setControls((prev) => {
        if (list.some((model) => model.id === prev.model)) return prev
        const preferred = list.some((model) => model.id === data.default)
          ? data.default
          : list[0]?.id || prev.model
        return { ...prev, model: preferred }
      })
    } catch (err) {
      setModelsError(err.message)
    } finally {
      if (refresh) setRefreshingModels(false)
    }
  }, [])

  useEffect(() => {
    loadModels(false)
  }, [loadModels])

  useEffect(() => {
    if (status?.status === 'ready') setReindexing(false)
  }, [status?.status])

  // Land on the start of the newest exchange, not the very bottom: long
  // answers with five source cards otherwise hide their own opening line.
  useEffect(() => {
    const lastUser = [...messages].reverse().find((msg) => msg.role === 'user')
    if (!lastUser) return
    const node = document.querySelector(
      `[data-message-id="${lastUser.id}"]`,
    )
    node?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [messages])

  // Cmd/Ctrl+K jumps to the composer.
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') setSidebarOpen(false)
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const ready = status?.status === 'ready'

  const send = useCallback(
    async (text) => {
      const question = (text ?? input).trim()
      if (!question || busy) return

      setInput('')
      setBanner(null)
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: 'user', content: question },
        {
          id: uid(),
          role: 'assistant',
          content: '',
          pending: true,
          question,
          docs: status?.documents || 14,
        },
      ])
      setBusy(true)

      const payload = {
        question,
        k: controls.k,
        mode: controls.mode,
        use_llm: controls.useLlm,
        model: controls.model || undefined,
      }

      try {
        const data = await askQuestion(payload)
        setMessages((prev) =>
          prev.map((msg) =>
            msg.pending
              ? {
                  ...msg,
                  pending: false,
                  content: data.answer || 'No answer returned.',
                  sources: data.sources,
                  mode: data.mode,
                  model: data.model,
                  elapsedMs: data.elapsed_ms,
                  topScore: data.top_score,
                  lowConfidence: data.low_confidence,
                  retrievalOnly: !controls.useLlm,
                  usage: data.usage,
                }
              : msg,
          ),
        )
      } catch (err) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.pending
              ? { ...msg, pending: false, error: true, content: err.message }
              : msg,
          ),
        )
      } finally {
        setBusy(false)
        inputRef.current?.focus()
      }
    },
    [busy, controls, input, status?.documents],
  )

  const handleReindex = async () => {
    setReindexing(true)
    setBanner(null)
    try {
      await reindex()
      setBanner(
        serverless
          ? 'This deployment uses a precomputed index — see DEPLOYMENT.md to rebuild it from ragg/.'
          : 'Index rebuild started. This can take a minute on first run.',
      )
      // In serverless mode status never changes, so clear the spinner here too.
      if (serverless) setReindexing(false)
    } catch (err) {
      setBanner(err.message)
      setReindexing(false)
    }
  }

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  const activeModel = controls.model || status?.groq_model || 'Groq'
  // /api/status reports which backend is answering. The Vercel build uses a
  // precomputed index, so the UI labels itself accordingly.
  const serverless = status?.runtime === 'vercel-serverless'

  return (
    <div className="app">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />
      <div className="grid-overlay" />

      {sidebarOpen && (
        <div className="scrim" onClick={() => setSidebarOpen(false)} />
      )}
      <div className={`sidebar-wrap ${sidebarOpen ? 'sidebar-wrap-open' : ''}`}>
        <Sidebar
          status={status}
          controls={controls}
          setControls={setControls}
          onReindex={handleReindex}
          onClear={() => setMessages([])}
          onExample={(question) => {
            setSidebarOpen(false)
            send(question)
          }}
          reindexing={reindexing}
          serverless={serverless}
          exampleQuestions={EXAMPLES}
          onClose={() => setSidebarOpen(false)}
          models={models}
          modelsError={modelsError}
          onRefreshModels={() => loadModels(true)}
          refreshingModels={refreshingModels}
          theme={theme}
          setTheme={setTheme}
        />
      </div>

      <main className="main">
        <header className="topbar">
          <button
            type="button"
            className="icon-btn menu-btn"
            onClick={() => setSidebarOpen(true)}
            title="Open settings"
          >
            <IconMenu size={19} />
          </button>
          <div className="topbar-title">
            <span className="topbar-mark">
              <IconSparkle size={15} />
            </span>
            <span>Ask your course notes</span>
          </div>
          <div className="topbar-right">
            <span className="pill pill-model" title="Active Groq model">
              <IconSparkle size={11} /> {activeModel}
            </span>
            <span className={`pill pill-${status?.status || 'loading'}`}>
              <span className="pill-dot" />
              {ready ? 'Ready' : status?.status === 'error' ? 'Error' : 'Indexing'}
            </span>
            <button
              type="button"
              className="icon-btn theme-btn"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? <IconMoon size={17} /> : <IconSun size={17} />}
            </button>
          </div>
        </header>

        {banner && (
          <div
            className={`banner ${
              banner.startsWith('Backend offline') ? 'banner-error' : ''
            }`}
          >
            {banner}
            <button type="button" onClick={() => setBanner(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        <div className="conversation">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-eyebrow">
                <span className="live-dot" /> Retrieval-Augmented Generation · local corpus
              </div>
              <h2>
                Ask your notes.
                <br />
                <span>See the evidence.</span>
              </h2>
              <p>
                Every answer comes from passages in your own notes, written by a
                free Groq model — with sources, scores, and an honest “I do not
                know”.
              </p>

              <div className="welcome-pipeline-card">
                <div className="wp-head">
                  <span>THE PIPELINE</span>
                  <span className="wp-head-right">5 stages · ~2 s</span>
                </div>
                <Pipeline
                  active={busy}
                  docs={status?.documents || 14}
                  serverless={serverless}
                />
                <div className="wp-foot">
                  <IconArrow size={13} /> From a raw <code>.txt</code> file to a
                  cited answer
                </div>
              </div>

              <div className="welcome-examples">
                {EXAMPLES.slice(0, 4).map((question) => (
                  <button type="button" key={question} onClick={() => send(question)}>
                    <IconArrow size={13} /> {question}
                  </button>
                ))}
              </div>

              <div className="welcome-stats">
                <div>
                  <strong>{status?.documents ?? '—'}</strong>
                  <span>documents</span>
                </div>
                <div>
                  <strong>{status?.chunks?.toLocaleString() ?? '—'}</strong>
                  <span>chunks indexed</span>
                </div>
                <div>
                  <strong>384</strong>
                  <span>dimensions / vector</span>
                </div>
              </div>

              {!ready && (
                <p className="welcome-note">
                  Waiting for the backend to finish indexing the corpus…
                </p>
              )}
            </div>
          ) : (
            <div className="messages">
              {messages.map((message) => (
                <Message
                  key={message.id}
                  message={{ ...message, showChunks: controls.showChunks }}
                  onRetry={(question) => send(question)}
                />
              ))}
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className={`composer ${busy ? 'composer-busy' : ''}`}>
            <div className="composer-accent" />
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                ready
                  ? 'Ask a question about your course notes…'
                  : 'Waiting for the index to be ready…'
              }
              rows={1}
              disabled={!ready || busy}
            />
            <button
              type="button"
              className="send-btn"
              onClick={() => send()}
              disabled={!ready || busy || !input.trim()}
              title="Send (Enter)"
            >
              <IconSend size={17} />
            </button>
          </div>
          <p className="composer-hint">
            <kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline ·{' '}
            <kbd>⌘/Ctrl</kbd>+<kbd>K</kbd> focus · {controls.mode}, k={controls.k}
            {!controls.useLlm && ' · retrieval only'}
          </p>
        </div>
      </main>
    </div>
  )
}
