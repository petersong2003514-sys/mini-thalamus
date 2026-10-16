import { useState, useRef, useEffect } from 'react'

const DOC_EXTS = ['pdf','docx','xlsx','xls','txt','md','csv']
const IMG_EXTS = ['png','jpg','jpeg','webp','bmp']

export default function ChatInput({ onSend, loading, onAbort }) {
  const [text, setText] = useState('')
  const [files, setFiles] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const [requireConfirm, setRequireConfirm] = useState(false)
  const [truncated, setTruncated] = useState(false)
  const fileRef = useRef(null)
  const previewUrls = useRef(new Map())

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed && files.length === 0) return
    if (loading) return
    onSend(trimmed, files, { requireConfirm })
    setText('')
    setFiles([])
  }

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isImage = (f) => {
    if (f.type && f.type.startsWith('image/')) return true
    const ext = f.name.split('.').pop().toLowerCase()
    return IMG_EXTS.includes(ext)
  }

  const addFiles = (newFiles) => {
    const valid = [...newFiles].filter(f => {
      const ext = f.name.split('.').pop().toLowerCase()
      const ok = [...DOC_EXTS, ...IMG_EXTS].includes(ext)
      if (!ok) alert(`${f.name}: 不支持的格式 (支持: ${[...DOC_EXTS, ...IMG_EXTS].join(', ')})`)
      return ok
    })
    setFiles(prev => {
      const combined = [...prev, ...valid]
      setTruncated(combined.length > 5)
      return combined.slice(0, 5)
    })
  }

  const removeFile = (idx) => {
    const f = files[idx]
    if (f && previewUrls.current.has(f)) {
      URL.revokeObjectURL(previewUrls.current.get(f))
      previewUrls.current.delete(f)
    }
    setFiles(prev => prev.filter((_, i) => i !== idx))
  }

  const handleDrop = e => {
    e.preventDefault()
    setDragOver(false)
    addFiles(e.dataTransfer.files)
  }

  const handleDragOver = e => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => setDragOver(false)

  // 管理预览 URL 生命周期
  useEffect(() => {
    const map = previewUrls.current
    for (const f of files) {
      if (isImage(f) && !map.has(f)) {
        map.set(f, URL.createObjectURL(f))
      }
    }
    const currentFiles = new Set(files)
    for (const [f, url] of map) {
      if (!currentFiles.has(f)) {
        URL.revokeObjectURL(url)
        map.delete(f)
      }
    }
  }, [files])

  // 组件卸载时清理所有 URL
  useEffect(() => {
    const map = previewUrls.current
    return () => {
      for (const url of map.values()) {
        URL.revokeObjectURL(url)
      }
      map.clear()
    }
  }, [])

  return (
    <div className="chat-input-area">
      <div className="input-wrapper">
        {/* 文件预览区 */}
        {files.length > 0 && (
          <div className="file-list">
            {files.map((f, i) => {
              const img = isImage(f)
              const previewUrl = img ? previewUrls.current.get(f) || '' : ''
              return (
                <span key={i} className={`file-tag ${img ? 'file-tag-img' : ''}`}>
                  {img && previewUrl ? (
                    <img src={previewUrl} alt={f.name}
                         className="file-thumb"
                         style={{width:40,height:40,objectFit:'cover',borderRadius:4}} />
                  ) : (
                    <span className="file-icon">📎</span>
                  )}
                  <span className="file-name">{f.name}</span>
                  <span className="file-size">({(f.size/1024).toFixed(0)}KB)</span>
                  <button className="file-rm" onClick={() => removeFile(i)} disabled={loading}>✕</button>
                </span>
              )
            })}
          </div>
        )}
        {truncated && (
          <div className="truncation-warning" style={{fontSize:'12px',color:'#e67e22',marginTop:4}}>
            ⚠ 最多只能上传 5 个文件，多余文件已被忽略
          </div>
        )}

        {/* 输入框 */}
        <div className={`input-row ${dragOver ? 'drag-over' : ''}`}
             onDrop={handleDrop}
             onDragOver={handleDragOver}
             onDragLeave={handleDragLeave}>
          <button className="upload-btn" onClick={() => fileRef.current?.click()}
                  disabled={loading} title="上传文件（文档/图片）" aria-label="上传文件">
            📎
          </button>
          <input type="file" ref={fileRef} style={{display:'none'}}
                 accept=".pdf,.docx,.xlsx,.xls,.txt,.md,.csv,.png,.jpg,.jpeg,.webp,.bmp"
                 multiple
                 onChange={e => { addFiles(e.target.files); e.target.value = '' }} />
          <input
            type="text"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={handleKey}
            placeholder={files.length > 0 ? "对文件提问，或留空让 AI 自动分析..." : "输入你的需求，系统会自动选择合适的 Agent..."}
            disabled={loading}
          />
          {loading ? (
            <button className="btn-abort" onClick={onAbort}>停止</button>
          ) : (
            <button onClick={handleSend} disabled={!text.trim() && files.length === 0} aria-label="发送消息">发送</button>
          )}
        </div>
        {/* 逐步确认复选框 */}
        <label className="require-confirm-label">
          <input
            type="checkbox"
            checked={requireConfirm}
            onChange={e => setRequireConfirm(e.target.checked)}
            disabled={loading}
          />
          需要逐步确认（关闭自动链式执行）
        </label>
      </div>
    </div>
  )
}
