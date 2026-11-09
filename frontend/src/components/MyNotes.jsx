import { useRef, useState } from 'react'
import {
  IconBook,
  IconChevron,
  IconFile,
  IconTrash,
} from './Icons'

const STORAGE_KEY = 'rag-my-notes'

/** Load saved notes from localStorage. */
export function loadMyNotes() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (n) =>
        n &&
        typeof n.source === 'string' &&
        typeof n.text === 'string' &&
        n.id,
    )
  } catch {
    return []
  }
}

/** Persist notes to localStorage. Returns the same list for chaining. */
export function saveMyNotes(notes) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes))
  } catch {
    /* storage full or blocked — notes live for this session only */
  }
  return notes
}

/**
 * Convert the notes list into the `extra_docs` payload expected by the
 * /api/ask and /api/search endpoints.
 */
export function notesToExtraDocs(notes) {
  return notes.map((n) => ({ source: n.source, text: n.text }))
}

export default function MyNotes({ notes, setNotes }) {
  const [open, setOpen] = useState(false)
  const [pasteText, setPasteText] = useState('')
  const [pasteName, setPasteName] = useState('')
  const fileRef = useRef(null)

  const remove = (id) => {
    const next = notes.filter((n) => n.id !== id)
    setNotes(saveMyNotes(next))
  }

  const addFromText = (source, text) => {
    const trimmed = text.trim()
    if (!trimmed) return
    const name = source.trim() || `my-notes-${notes.length + 1}.txt`
    const note = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      source: name.endsWith('.txt') ? name : `${name}.txt`,
      text: trimmed,
    }
    setNotes(saveMyNotes([...notes, note]))
  }

  const handleFiles = async (files) => {
    const valid = Array.from(files).filter((f) => {
      const isText =
        f.type.startsWith('text/') ||
        f.name.endsWith('.txt') ||
        f.name.endsWith('.md')
      return isText
    })
    for (const file of valid) {
      const text = await file.text()
      addFromText(file.name, text)
    }
    if (fileRef.current) fileRef.current.value = ''
  }

  const handlePaste = () => {
    if (!pasteText.trim()) return
    addFromText(pasteName || 'pasted-note', pasteText)
    setPasteText('')
    setPasteName('')
  }

  const totalChars = notes.reduce((sum, n) => sum + n.text.length, 0)

  return (
    <div className="panel my-notes-panel">
      <button
        type="button"
        className="corpus-head"
        onClick={() => setOpen((prev) => !prev)}
      >
        <IconFile size={15} />
        <span>
          My notes · {notes.length} {notes.length === 1 ? 'file' : 'files'}
        </span>
        <IconChevron size={14} className={open ? 'chevron-open' : ''} />
      </button>

      {open && (
        <div className="my-notes-body">
          {notes.length > 0 && (
            <ul className="corpus-files my-notes-list">
              {notes.map((note) => (
                <li key={note.id} title={note.source}>
                  <IconBook size={11} />
                  <span className="my-notes-name">{note.source}</span>
                  <span className="my-notes-size">
                    {note.text.length.toLocaleString()} ch
                  </span>
                  <button
                    type="button"
                    className="my-notes-remove"
                    onClick={() => remove(note.id)}
                    title="Remove this note"
                  >
                    <IconTrash size={11} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {notes.length > 0 && (
            <button
              type="button"
              className="ghost-btn my-notes-clear"
              onClick={() => setNotes(saveMyNotes([]))}
            >
              <IconTrash size={13} /> Remove all
            </button>
          )}

          <div className="my-notes-upload">
            <button
              type="button"
              className="ghost-btn"
              onClick={() => fileRef.current?.click()}
            >
              <IconFile size={14} /> Add .txt / .md files
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,text/plain,text/markdown"
              multiple
              onChange={(e) => handleFiles(e.target.files)}
              style={{ display: 'none' }}
            />
          </div>

          <div className="my-notes-paste">
            <input
              type="text"
              className="my-notes-name-input"
              placeholder="Note name (optional)"
              value={pasteName}
              onChange={(e) => setPasteName(e.target.value)}
            />
            <textarea
              className="my-notes-textarea"
              placeholder="Or paste text here…"
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={2}
            />
            <button
              type="button"
              className="ghost-btn"
              onClick={handlePaste}
              disabled={!pasteText.trim()}
            >
              Add note
            </button>
          </div>

          <p className="corpus-meta">
            {totalChars.toLocaleString()} chars total · stored in your browser
            <br />
            Notes are sent with each question and scored alongside the corpus
          </p>
        </div>
      )}
    </div>
  )
}
