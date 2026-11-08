import { useState } from 'react'
import { IconBook, IconChevron, IconFile } from './Icons'

const fmt = (value) => Number(value ?? 0).toFixed(3)
const pct = (value, max) =>
  `${Math.max(5, Math.min(100, (Number(value || 0) / (max || 1)) * 100))}%`

export default function Sources({ sources, mode, showChunks }) {
  const [opened, setOpened] = useState({})

  if (!sources || sources.length === 0) return null

  const maxScore = Math.max(...sources.map((s) => Number(s.score) || 0), 0.001)
  const toggle = (rank) =>
    setOpened((prev) => ({ ...prev, [rank]: !prev[rank] }))

  return (
    <section className="sources">
      <div className="sources-head">
        <IconBook size={14} />
        <span>
          {sources.length} retrieved passage{sources.length === 1 ? '' : 's'}
        </span>
        <span className="sources-mode">{mode}</span>
      </div>

      <div className="source-list">
        {sources.map((source) => {
          const expanded = showChunks || opened[source.rank]
          const width = pct(source.score, maxScore)
          return (
            <div className="source-item" key={source.rank}>
              <button
                type="button"
                className="source-toggle"
                onClick={() => toggle(source.rank)}
                aria-expanded={Boolean(expanded)}
              >
                <span className="source-rank">{source.rank}</span>
                <IconFile size={13} className="source-file-icon" />
                <span className="source-file" title={source.source}>
                  {source.source}
                </span>
                <span className="source-score" title="Combined retrieval score">
                  {fmt(source.score)}
                </span>
                <IconChevron
                  size={13}
                  className={`chevron ${expanded ? 'chevron-open' : ''}`}
                />
              </button>

              <div className="score-track">
                <span style={{ width }} />
              </div>

              {expanded && (
                <div className="source-chunk">
                  <p>{source.chunk}</p>
                  <div className="source-metrics">
                    {source.semantic_score != null && (
                      <span>
                        <i className="dot dot-semantic" /> semantic{' '}
                        {fmt(source.semantic_score)}
                      </span>
                    )}
                    {source.keyword_score != null && (
                      <span>
                        <i className="dot dot-keyword" /> keyword{' '}
                        {fmt(source.keyword_score)}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
