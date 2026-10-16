import React, { useEffect, useRef, useState } from 'react'

// Step 标签从 agents API 动态获取，本地只做缓存
let _agentCache = null
if (typeof window !== 'undefined') {
  window.addEventListener('refresh-sidebar', () => { _agentCache = null })
}

async function getAgentMeta() {
  if (_agentCache) return _agentCache
  try {
    const res = await fetch('/api/agents')
    const data = await res.json()
    _agentCache = {}
    for (const a of data.agents) {
      _agentCache[a.id] = { name: a.name, icon: a.icon || '🔧' }
    }
  } catch { _agentCache = {} }
  return _agentCache
}

function StepLabel({ agent, step }) {
  const [meta, setMeta] = useState(null)
  useEffect(() => {
    getAgentMeta().then(m => setMeta(m?.[agent] || { name: agent, icon: '🔧' }))
  }, [agent])
  if (!meta) return <div className="route-detail">Step {step}: {agent}</div>
  return <div className="route-detail">Step {step}: {meta.icon} {meta.name}</div>
}

export default function ChatMessages({ messages, loading }) {
  const bottomRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    // 仅当用户已处于底部附近（100px 内）时才自动滚动，避免干扰阅读历史消息
    const container = containerRef.current
    if (container) {
      const { scrollTop, scrollHeight, clientHeight } = container
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      if (isNearBottom) {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }, [messages])

  return (
    <div className="chat-messages" id="chat-messages-container" ref={containerRef}
         aria-live="polite" role="log">
      {messages.map((msg, i) => (
        <Message key={msg._id || `msg-${i}-${msg.text?.slice(0,20)}`} msg={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`msg ${isUser ? 'user' : 'assistant'}`}>
      <div className="avatar">{isUser ? '👤' : '🤖'}</div>
      <div className="bubble">
        {/* 用户上传的文件附件 */}
        {msg.files?.length > 0 && (
          <div className="msg-files">
            {msg.files.map((f, i) => {
              const name = typeof f === 'string' ? f : f.name || f
              const isImg = /\.(png|jpg|jpeg|webp|bmp)$/i.test(name)
              return (
                <span key={i} className={`msg-file-tag ${isImg ? 'msg-file-img' : ''}`}>
                  {isImg ? '🖼️' : '📎'} {name}
                </span>
              )
            })}
          </div>
        )}
        {/* 用户上传的图片预览 */}
        {msg.imagePreviews?.length > 0 && (
          <div className="msg-image-previews">
            {msg.imagePreviews.map((url, i) => (
              <img key={i} src={url} alt={`preview-${i}`} className="msg-image-thumb"
                   style={{maxWidth:120, maxHeight:120, borderRadius:8, marginRight:6, marginBottom:6}} />
            ))}
          </div>
        )}
        {msg.routeTag && <RouteTag tag={msg.routeTag} />}
        {msg.stepLabels?.map((sl, i) => (
          <StepLabel key={i} agent={sl.agent} step={sl.step} />
        ))}
        {msg.statusLines?.map((s, i) => (
          <div key={i} className="route-detail">{s}</div>
        ))}
        {msg.approval && <ApprovalCard approval={msg.approval} />}
        {msg.loading ? (
          <div className="thinking"><span /><span /><span /></div>
        ) : (
          <>
            <RichText text={msg.text} />
            {/* 结构化数据渲染 */}
            {msg.structured && <StructuredView s={msg.structured} />}
            {/* 检查点 UI */}
            {msg.checkpoint && <CheckpointCard cp={msg.checkpoint} />}
          </>
        )}
      </div>
    </div>
  )
}

// ─── 结构化渲染 ───
function StructuredView({ s }) {
  const fmt = s.format || 'text'
  let content = null

  if (fmt === 'weather' && s.data) {
    content = (
      <div className="weather-card">
        <div className="weather-temp">{s.data.temp || '?'}°C</div>
        <div>
          <div className="weather-info">🌤 {s.data.condition || ''} · {s.data.city || ''}</div>
          <div className="weather-info">{(s.data.details || []).join(' · ')}</div>
          {s.data.advice && <div className="weather-advice">👔 {s.data.advice}</div>}
        </div>
      </div>
    )
  } else if (fmt === 'comparison' && s.data) {
    content = (
      <div className="structured-card">
        <table className="compare-table">
          <thead>
            <tr>
              <th>维度</th>
              <th>{(s.data.subjects || ['A','B'])[0]}</th>
              <th>{(s.data.subjects || ['A','B'])[1]}</th>
            </tr>
          </thead>
          <tbody>
            {(s.data.criteria || []).map((c, i) => (
              <tr key={i}>
                <td>{c.name || ''}</td><td>{c.A || ''}</td><td>{c.B || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {s.data.verdict && <div className="verdict">📋 {s.data.verdict}</div>}
      </div>
    )
  } else if (fmt === 'candidates' && s.data) {
    content = (
      <div>
        {(s.data.candidates || []).map((c, i) => (
          <div key={i} className="candidate-card">
            <div className="c-name">
              {c.name || ''} <span className="c-score">{c.score || ''}</span>
            </div>
            <div className="c-tags">
              {(c.pros || []).map((p, j) => <span key={j} className="pro">✓ {p}</span>)}
              {(c.cons || []).map((p, j) => <span key={j} className="con">✗ {p}</span>)}
            </div>
            {c.verdict && <div className="c-verdict">→ {c.verdict}</div>}
          </div>
        ))}
      </div>
    )
  } else if (fmt === 'list' && s.data) {
    content = (
      <div className="structured-card">
        {(s.data.items || []).map((item, i) => (
          <div key={i} className="list-item">
            <b>{item.title || ''}</b>: {item.detail || ''}
          </div>
        ))}
      </div>
    )
  } else if (fmt === 'table' && s.data) {
    content = (
      <div className="structured-card">
        <table className="compare-table">
          <thead>
            <tr>{(s.data.headers || []).map((h, i) => <th key={i}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {(s.data.rows || []).map((row, i) => (
              <tr key={i}>{row.map((c, j) => <td key={j}>{c}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  } else if (fmt === 'url_content' && s.data) {
    const d = s.data
    const [expanded, setExpanded] = useState(false)
    content = (
      <div className="url-card">
        <div className="url-card-header">
          <span className="url-card-type">{d.content_type || 'text/html'}</span>
          {d.cached && <span className="url-card-cached">cached</span>}
          <a className="url-card-link" href={d.url} target="_blank" rel="noopener noreferrer"
             title={d.url}>{d.title || d.url}</a>
        </div>
        {d.text && (
          <>
            <div className={`url-card-text ${expanded ? 'expanded' : ''}`}>{d.text}</div>
            {d.text.length > 200 && (
              <button className="url-card-toggle" onClick={() => setExpanded(e => !e)}>
                {expanded ? '收起' : '展开全文'}
              </button>
            )}
          </>
        )}
      </div>
    )
  } else if (fmt === 'code' && s.data) {
    content = (
      <div className="structured-card">
        {s.data.code && (
          <pre className="code-block"><code>{s.data.code.slice(0, 2000)}</code></pre>
        )}
        {s.data.output && (
          <div className="code-output">{s.data.output}</div>
        )}
        {s.data.images && s.data.images.length > 0 && s.data.images.map((img, i) => (
          <div key={i} className="chart-container" style={{marginTop:8}}>
            <img src={img} className="chart-img" alt="chart"
                 onError={e => { e.target.style.display = 'none' }} />
            <a href={img} download={img.split('/').pop()} className="chart-download">⬇ 下载</a>
          </div>
        ))}
      </div>
    )
  }

  // 置信度标签
  const confLevels = { high: '高置信', medium: '中等', low: '低置信' }
  const conf = s.confidence || 'medium'

  // 来源链接
  const sources = s.sources || []

  return (
    <div>
      {content}
      <span className={`confidence-tag ${conf}`}>
        {confLevels[conf] || conf}
      </span>
      {sources.length > 0 && (
        <div className="source-links">
          {sources.map((src, i) => (
            src.url ? (
              <a key={i} href={src.url} target="_blank" rel="noopener noreferrer">
                [{i + 1}] {src.title || src.url}
              </a>
            ) : (
              <span key={i}>[{i + 1}] {src.title || ''}</span>
            )
          ))}
        </div>
      )}
    </div>
  )
}

// ─── 检查点 UI（断点续传）───
function CheckpointCard({ cp }) {
  const { step, stepTotal, label, answer, data, sessionId } = cp
  const [editAnswer, setEditAnswer] = useState(answer || '')

  useEffect(() => { setEditAnswer(answer || '') }, [answer])

  const handleContinue = () => {
    const cpEl = document.getElementById(`cp-${step}`)
    if (cpEl) cpEl.style.opacity = '0.5'
    continueCheckpoint(step, sessionId, editAnswer || '')
  }

  return (
    <div className="checkpoint-card" id={`cp-${step}`}>
      <div className="cp-header">
        ⏸ 检查点 {step + 1}/{stepTotal}: {label}
      </div>
      <div className="cp-hint">✏ 双击字段可编辑，修改后点击"继续"</div>
      <div className="cp-field">
        <label>回答</label>
        <textarea id={`cp-answer-${step}`} value={editAnswer}
                  onChange={e => setEditAnswer(e.target.value)} />
      </div>
      <div className="cp-actions">
        <button className="cp-btn-cancel" onClick={() => {
          document.querySelector(`#cp-${step}`)?.remove()
        }}>✖ 取消</button>
        <button className="cp-btn-skip" onClick={() => continueCheckpoint(step, sessionId, '')}>
          ⏭ 跳过检查
        </button>
        <button className="cp-btn-continue" onClick={handleContinue}>▶ 继续</button>
      </div>
    </div>
  )
}

function continueCheckpoint(step, sid, message) {
  // 隐藏当前检查点卡片
  const cp = document.getElementById(`cp-${step}`)
  if (cp) cp.style.display = 'none'

  // 在检查点卡片后创建新的回答气泡
  const container = document.getElementById('chat-messages-container')
  if (!container) return

  const bubble = document.createElement('div')
  bubble.className = 'msg assistant'
  bubble.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><div class="thinking"><span /><span /><span /></div></div>`
  container.appendChild(bubble)
  const bubbleContent = bubble.querySelector('.bubble')
  bubble.scrollIntoView({ behavior: 'smooth' })

  fetch('/api/chat/continue', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sid, step, message })
  }).then(r => {
    if (!r.ok) throw new Error('HTTP ' + r.status)
    return r.body.getReader()
  }).then(reader => {
    const decoder = new TextDecoder()
    let buffer = ''
    let hasContent = false
    function pump() {
      reader.read().then(({ done, value }) => {
        if (done) return
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const evt = JSON.parse(line.slice(6))
            if (evt.type === 'status') {
              // 显示进度（不替换已有内容）
              const st = document.createElement('div')
              st.className = 'route-detail'
              st.textContent = evt.text
              if (!hasContent) { hasContent = true; bubbleContent.innerHTML = '' }
              bubbleContent.appendChild(st)
            } else if (evt.type === 'answer' && evt.text) {
              if (!hasContent) {
                hasContent = true
                bubbleContent.innerHTML = ''
              }
              const div = document.createElement('div')
              div.textContent = evt.text || ''
              bubbleContent.appendChild(div)
            } else if (evt.type === 'checkpoint') {
              // 统一走 React CheckpointCard — 派发事件给 useChat 处理
              window.dispatchEvent(new CustomEvent('mini_thalamus:checkpoint', { detail: evt }))
              hasContent = true  // 避免最终显示"无返回内容"
            } else if (evt.type === 'structured') {
              // 结构化内容也走统一渲染：添加到 bubble，同时可以通过 React 渲染
              const card = document.createElement('div')
              card.className = 'structured-card'
              const conf = escapeHtml(evt.data?.confidence || 'medium')
              const labels = {high:'高置信',medium:'中等',low:'低置信'}
              card.innerHTML = `<span class="confidence-tag ${conf}">${escapeHtml(labels[evt.data?.confidence] || '中等')}</span>`
              bubbleContent.appendChild(card)
            } else if (evt.type === 'error') {
              const err = document.createElement('div')
              err.className = 'error-msg'
              err.innerHTML = `<span class="err-icon">⚠️</span> ${escapeHtml(evt.text)}`
              bubbleContent.appendChild(err)
            } else if (evt.type === 'done') {
              // 完成，不刷新——自然结束
              if (!hasContent) {
                bubbleContent.innerHTML = '<span style="color:var(--text-dim)">(无返回内容)</span>'
              }
            }
          } catch (e) { /* skip */ }
        }
        bubble.scrollIntoView({ behavior: 'smooth' })
        pump()
      })
    }
    pump()
  }).catch(e => {
    bubbleContent.innerHTML = `<span class="error-msg">⚠️ 继续执行失败: ${escapeHtml(e.message)}</span>`
  })
}

function escapeHtml(text) {
  const d = document.createElement('div')
  d.textContent = text || ''
  return d.innerHTML
}

// ─── 简易 Markdown 渲染 ───
function renderSimpleMD(text) {
  // 处理 Markdown 表格: | a | b |\n|--|--|\n| 1 | 2 |
  const lines = text.split('\n')
  const result = []
  let tableLines = []
  let inTable = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    const isTableLine = /^\|.+\|$/.test(line)
    const isTableSep = /^\|[-:| ]+\|$/.test(line)

    if (isTableLine || isTableSep) {
      if (!inTable) {
        inTable = true
      }
      if (!isTableSep) {
        tableLines.push(line)
      }
    } else {
      if (inTable && tableLines.length > 0) {
        result.push({ type: 'table', rows: tableLines.map(r =>
          r.split('|').slice(1, -1).map(c => c.trim())
        )})
        tableLines = []
        inTable = false
      }
      // 标题: # / ## / ###
      const h1 = line.match(/^# (.+)/)
      const h2 = line.match(/^## (.+)/)
      const h3 = line.match(/^### (.+)/)
      if (h1) { result.push({ type: 'h1', value: h1[1] }); continue }
      if (h2) { result.push({ type: 'h2', value: h2[1] }); continue }
      if (h3) { result.push({ type: 'h3', value: h3[1] }); continue }
      // 粗体: **text**
      let escapedLine = escapeHtml(line)
      const boldText = escapedLine.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      result.push({ type: 'text', value: boldText })
    }
  }
  // 末尾残留表格
  if (inTable && tableLines.length > 0) {
    result.push({ type: 'table', rows: tableLines.map(r =>
      r.split('|').slice(1, -1).map(c => c.trim())
    )})
  }

  return result.length > 0 ? result : [{ type: 'text', value: text }]
}

// ─── 富文本解析 ───
function RichText({ text }) {
  if (!text) return null
  const codeParts = text.split(/(```\w*\n[\s\S]*?```)/g)
  return (
    <div className="msg-text">
      {codeParts.map((part, i) => {
        if (part.startsWith('```')) {
          const langMatch = part.match(/```(\w*)\n/)
          const code = part.replace(/```\w*\n/, '').replace(/```$/, '')
          return <pre key={i} className="code-block"><code>{code}</code></pre>
        }
        const imgParts = part.split(/(!\[[^\]]*\]\([^)]+\))/g)
        return imgParts.map((p, j) => {
          const m = p.match(/!\[[^\]]*\]\(([^)]+)\)/)
          if (m) {
            const url = m[1]; const filename = url.split('/').pop()
            return (
              <div key={`${i}-${j}`} className="chart-container">
                <img src={url} alt="chart" className="chart-img"
                     onError={e => { e.target.closest('.chart-container').style.display = 'none' }} />
                <a href={url} download={filename} className="chart-download">⬇ 下载</a>
              </div>
            )
          }
          // 简易 Markdown 渲染
          const mdBlocks = renderSimpleMD(p)
          return mdBlocks.map((block, k) => {
            const key = `${i}-${j}-${k}`
            if (block.type === 'h1') return <h1 key={key} className="md-h1">{block.value}</h1>
            if (block.type === 'h2') return <h2 key={key} className="md-h2">{block.value}</h2>
            if (block.type === 'h3') return <h3 key={key} className="md-h3">{block.value}</h3>
            if (block.type === 'table') {
              const [header, ...rows] = block.rows
              return (
                <table key={key} className="md-table">
                  {header && <thead><tr>{header.map((c, ci) => <th key={ci}>{c}</th>)}</tr></thead>}
                  <tbody>{rows.map((row, ri) => (
                    <tr key={ri}>{row.map((c, ci) => <td key={ci}>{c}</td>)}</tr>
                  ))}</tbody>
                </table>
              )
            }
            const parts = block.value.split(/(<strong>.*?<\/strong>)/g)
            return <span key={key}>
              {parts.map((part, pi) => {
                if (part.startsWith('<strong>') && part.endsWith('</strong>')) {
                  return <strong key={pi}>{part.slice(8, -9)}</strong>
                }
                const lines = part.split('\n')
                return lines.map((l, li) => (
                  <React.Fragment key={`${pi}-${li}`}>
                    {li > 0 && <br/>}
                    {l}
                  </React.Fragment>
                ))
              })}
            </span>
          })
        })
      })}
    </div>
  )
}

function ApprovalCard({ approval }) {
  const [visible, setVisible] = useState(true)
  if (!visible) return null
  const approve = () => {
    window.dispatchEvent(new CustomEvent('mini_thalamus:approve', {
      detail: { callIds: [approval.callId] },
    }))
    setVisible(false)
  }
  return (
    <div className="checkpoint-card">
      <div className="cp-header">🔐 需要批准：步骤 {approval.callId}</div>
      <div className="cp-hint">{approval.answer || '该步骤需要批准后才能执行'}</div>
      <div className="cp-actions">
        <button className="cp-btn-cancel" onClick={() => setVisible(false)}>✕ 取消</button>
        <button className="cp-btn-continue" onClick={approve}>✓ 批准并继续</button>
      </div>
    </div>
  )
}

function RouteTag({ tag }) {
  if (tag.type === 'multi') {
    return (
      <div>
        <span className="route-tag auto">🔗 {tag.label}</span>
        <div className="route-detail">{tag.detail} ({tag.reason})</div>
      </div>
    )
  }
  return (
    <div>
      <span className={`route-tag ${tag.type}`}>{tag.label}</span>
      {tag.reason && <div className="route-detail">理由: {tag.reason}</div>}
    </div>
  )
}
