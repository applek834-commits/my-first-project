import { Fragment } from 'react'
import {
  IconBook,
  IconCpu,
  IconLayers,
  IconSearch,
  IconSparkle,
} from './Icons'

const LOCAL_STEPS = [
  { id: 'corpus', label: 'Corpus', detail: '14 notes', Icon: IconBook },
  { id: 'chunk', label: 'Chunk', detail: '300 / 50', Icon: IconLayers },
  { id: 'embed', label: 'Embed', detail: 'MiniLM · 384d', Icon: IconCpu },
  { id: 'retrieve', label: 'Retrieve', detail: 'FAISS + TF-IDF', Icon: IconSearch },
  { id: 'generate', label: 'Generate', detail: 'Groq free', Icon: IconSparkle },
]

// The Vercel deployment has no model weights at runtime: the index is baked
// into api/_index_data.py at build time. Show the real pipeline, not a lie.
const SERVERLESS_STEPS = [
  { id: 'corpus', label: 'Corpus', detail: '14 notes', Icon: IconBook },
  { id: 'chunk', label: 'Chunk', detail: '300 / 50', Icon: IconLayers },
  { id: 'embed', label: 'Index', detail: 'TF-IDF · baked', Icon: IconCpu },
  { id: 'retrieve', label: 'Retrieve', detail: 'TF-IDF + BM25', Icon: IconSearch },
  { id: 'generate', label: 'Generate', detail: 'Groq free', Icon: IconSparkle },
]

/**
 * The five stages of the notebook's RAG system, drawn as a small flow.
 * `active` makes a pulse travel along it while a question is being answered.
 */
export default function Pipeline({
  active = false,
  compact = false,
  docs = 14,
  serverless = false,
}) {
  const template = serverless ? SERVERLESS_STEPS : LOCAL_STEPS
  const steps = template.map((step) =>
    step.id === 'corpus' ? { ...step, detail: `${docs} notes` } : step,
  )

  return (
    <div
      className={`pipeline ${compact ? 'pipeline-compact' : ''} ${
        active ? 'pipeline-active' : ''
      }`}
      aria-label="RAG pipeline"
    >
      {steps.map(({ id, label, detail, Icon }, index) => (
        <Fragment key={id}>
          <div className="pipeline-step" style={{ '--i': index }}>
            <div className="pipeline-node">
              <Icon size={compact ? 14 : 18} />
            </div>
            {!compact && (
              <div className="pipeline-copy">
                <strong>{label}</strong>
                <span>{detail}</span>
              </div>
            )}
          </div>
          {index < steps.length - 1 && (
            <div className="pipeline-link" style={{ '--i': index }}>
              <span />
            </div>
          )}
        </Fragment>
      ))}
    </div>
  )
}
