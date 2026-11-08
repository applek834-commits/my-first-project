import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Pipeline from './Pipeline'
import Sources from './Sources'
import {
  IconAlert,
  IconBot,
  IconCheck,
  IconCopy,
  IconRefresh,
  IconSparkle,
  IconUser,
  IconZap,
} from './Icons'

const cleanMath = (text) =>
  text
    .replace(/\\text\{([^}]*)\}/g, '$1')
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1) / ($2)')
    .replace(/\\times/g, '×')
    .replace(/\\cdot/g, '·')
    .replace(/\\left|\\right/g, '')
    .replace(/\\[a-zA-Z]+/g, '')
    .replace(/[{}]/g, '')
    .trim()

// Groq's reasoning models like LaTeX. Turn it into readable code blocks so a
// math plugin (and its extra bundle weight) is not needed.
function normalizeAnswer(text) {
  if (!text) return ''
  return text
    .replace(
      /\\\[([\s\S]*?)\\\]/g,
      (_, math) => `\n\n\`\`\`\n${cleanMath(math)}\n\`\`\`\n\n`,
    )
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => `\`${cleanMath(math)}\``)
    .replace(/【/g, '[')
    .replace(/】/g, ']')
    // Groq sometimes glues a citation to the previous word: "emails).[Source 3]".
    .replace(/([^\s])\[(Source\s*\d+[^\]]*)\]/gi, '$1 [$2]')
}

const markdownComponents = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
}

export default function Message({ message, onRetry }) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'
  const isError = Boolean(message.error)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard unavailable on insecure origins — ignore */
    }
  }

  return (
    <article
      data-message-id={message.id}
      className={`message ${isUser ? 'message-user' : 'message-bot'} ${
        isError ? 'message-error' : ''
      }`}
    >
      <div className="avatar">
        {isUser ? (
          <IconUser size={17} />
        ) : isError ? (
          <IconAlert size={17} />
        ) : (
          <IconBot size={17} />
        )}
      </div>

      <div className="message-body">
        <div className="message-meta">
          <strong>{isUser ? 'You' : 'Assistant'}</strong>
          {message.model && (
            <span className="meta-chip meta-chip-accent">
              <IconSparkle size={11} /> {message.model}
            </span>
          )}
          {message.elapsedMs != null && (
            <span className="meta-chip">
              <IconZap size={11} /> {message.elapsedMs} ms
            </span>
          )}
          {message.usage?.total_tokens != null && (
            <span className="meta-chip">{message.usage.total_tokens} tokens</span>
          )}
          {message.lowConfidence && (
            <span className="meta-chip meta-warn">low confidence</span>
          )}
          {message.retrievalOnly && (
            <span className="meta-chip">retrieval only</span>
          )}
        </div>

        {message.pending ? (
          <div className="thinking">
            <Pipeline active compact docs={message.docs} />
            <span className="thinking-label">
              Searching the corpus and composing a grounded answer…
            </span>
          </div>
        ) : isUser || isError ? (
          <div className="message-text">{message.content}</div>
        ) : (
          <div className="markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {normalizeAnswer(message.content)}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && !isError && !message.pending && (
          <>
            <Sources
              sources={message.sources}
              mode={message.mode}
              showChunks={message.showChunks}
            />
            <div className="message-actions">
              <button type="button" onClick={copy} title="Copy answer">
                {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
              {message.topScore != null && (
                <span className="top-score">
                  best match {Number(message.topScore).toFixed(3)}
                </span>
              )}
            </div>
          </>
        )}

        {isError && onRetry && message.question && (
          <button
            type="button"
            className="retry-btn"
            onClick={() => onRetry(message.question)}
          >
            <IconRefresh size={14} /> Try again
          </button>
        )}
      </div>
    </article>
  )
}
