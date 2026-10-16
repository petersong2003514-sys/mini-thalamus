import { useState, useEffect, useCallback } from 'react'

export default function Sidebar({ selected, onSelect, sessionId, fileCache, sessions, activeSessionId, onSelectSession, onDeleteSession }) {
  const [agents, setAgents] = useState([])
  const [skills, setSkills] = useState([])
  const [mcpStatus, setMcpStatus] = useState(null)
  const [loadError, setLoadError] = useState(false)

  const refreshAll = useCallback(() => {
    setLoadError(false)
    let hasError = false
    // 加载 Agent 列表
    fetch('/api/agents')
      .then(r => r.json())
      .then(data => setAgents(data.agents || []))
      .catch(() => { hasError = true })
    // 加载 Skill 列表
    fetch('/api/skills')
      .then(r => r.json())
      .then(data => setSkills(data.skills || []))
      .catch(() => { hasError = true })
    // 加载 MCP 状态
    fetch('/api/mcp/status')
      .then(r => r.json())
      .then(data => setMcpStatus(data))
      .catch(() => { hasError = true })
      .finally(() => {
        // 等待所有请求完成后再判断（简单延迟，大多数请求在此时间内完成）
        setTimeout(() => {
          if (hasError) setLoadError(true)
        }, 500)
      })
  }, [])

  useEffect(() => { refreshAll() }, [refreshAll])

  // 监听 crystallize 后的刷新事件
  useEffect(() => {
    const handler = () => refreshAll()
    window.addEventListener('refresh-sidebar', handler)
    return () => window.removeEventListener('refresh-sidebar', handler)
  }, [refreshAll])

  const deleteSkill = async (id) => {
    if (!confirm('确定删除此 Skill？')) return
    await fetch('/api/skills/' + id, { method: 'DELETE' })
    refreshAll() // 同时刷新 agents 和 skills
  }

  return (
    <aside className="sidebar">
      {/* ── 加载错误提示 ── */}
      {loadError && (
        <div className="error-msg" style={{display:'flex',alignItems:'center',gap:8,padding:'8px 12px',marginBottom:4}}>
          <span className="err-icon">⚠️</span> 加载失败
          <button onClick={refreshAll} style={{marginLeft:'auto',padding:'4px 12px',background:'var(--accent-light)',color:'#c4b5fd',border:'1px solid rgba(124,92,240,0.25)',borderRadius:'var(--radius-xs)',cursor:'pointer',fontSize:12}}>重试</button>
        </div>
      )}
      {/* ── Agent 列表 ── */}
      <h3>🧠 Agent 选择</h3>

      <div className={`agent-card auto-option ${selected === '' ? 'active' : ''}`}
           role="button" tabIndex={0}
           onClick={() => onSelect('')}
           onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect('') } }}>
        <div className="name">🤖 自动路由</div>
        <div className="desc">系统自动判断最合适的 Agent</div>
        <span className="badge auto-badge">推荐</span>
      </div>

      <div className="agent-list">
        {agents.filter(a => a.id !== 'general').map(a => (
          <div key={a.id}
               className={`agent-card ${selected === a.id ? 'active' : ''}`}
               role="button" tabIndex={0}
               onClick={() => onSelect(a.id)}
               onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(a.id) } }}>
            <div className="name">{a.icon || '🔧'} {a.name}</div>
            <div className="desc">{a.description}</div>
            <span className={`badge ${a.source === 'skill' ? 'skill-badge' : 'manual-badge'}`}>
              {a.source === 'skill' ? '固化' : a.source === 'mcp' ? 'MCP' : '手动选择'}
            </span>
          </div>
        ))}
      </div>

      {/* ── 历史会话 ── */}
      <h3>🕘 历史会话</h3>
      {sessions.length === 0 ? (
        <div className="empty-hint">暂无历史会话，开启新对话后会自动保存</div>
      ) : (
        <div className="session-list">
          {sessions.map(s => (
            <div key={s.session_id}
                 className={`session-item ${activeSessionId === s.session_id ? 'active' : ''}`}
                 role="button" tabIndex={0}
                 onClick={() => onSelectSession(s.session_id)}
                 onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelectSession(s.session_id) } }}
                 style={{display:'flex',alignItems:'center',gap:6,padding:'8px 10px',borderRadius:6,cursor:'pointer',border:'1px solid rgba(124,92,240,0.15)',marginBottom:6,background:'rgba(124,92,240,0.05)'}}>
              <div style={{flex:1,minWidth:0}}>
                <div style={{fontSize:13,fontWeight:600,whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>
                  {s.summary || s.last_query || '未命名对话'}
                </div>
                <div style={{fontSize:11,color:'#9aa0b5',marginTop:2}}>
                  {s.turn_count} 轮 · {new Date((s.updated_at || 0) * 1000).toLocaleString()}
                </div>
              </div>
              <button
                className="session-del"
                onClick={e => { e.stopPropagation(); onDeleteSession(s.session_id) }}
                title="删除该会话"
                style={{background:'transparent',border:'none',cursor:'pointer',color:'#f87171',fontSize:14}}
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ── 文件缓存 ── */}
      {fileCache && fileCache.length > 0 && (
        <>
          <h3>📂 会话文件</h3>
          <div className="file-cache">
            {fileCache.map((name, i) => (
              <span key={i} className="file-cache-chip" title={name}>
                📄 {name.length > 20 ? name.slice(0, 18) + '...' : name}
              </span>
            ))}
          </div>
        </>
      )}

      {/* ── Skill 管理 ── */}
      <h3>⚡ 固化 Skill</h3>
      {skills.length === 0 ? (
        <div className="empty-hint">
          💡 完成对话后可在下方固化按钮将工作流保存为可复用的 Skill，一键重复执行。
        </div>
      ) : (
        <div className="skill-list">
          {skills.map(s => (
            <div key={s.id} className="skill-item">
              <div className="skill-info">
                <b>{s.icon || '⚡'} {s.name}</b>
                <div className="skill-desc">{s.description}</div>
              </div>
              <button className="skill-del" onClick={() => deleteSkill(s.id)}
                      title="删除此 Skill">🗑</button>
            </div>
          ))}
        </div>
      )}

      {/* ── MCP 状态 ── */}
      <h3>🔌 MCP 工具</h3>
      {!mcpStatus || !mcpStatus.config_loaded ? (
        <div className="empty-hint">
          💡 MCP 可接入外部工具（如浏览器、数据库等）。编辑 <code>data/.mcp.json</code> 配置服务器即可自动加载。
        </div>
      ) : (
        <div className="mcp-list">
          <div className="mcp-summary">
            已连接 · {mcpStatus.tool_count || 0} 个工具可用
          </div>
          {Object.entries(mcpStatus.servers || {}).map(([name, srv]) => (
            <div key={name} className="mcp-item">
              <span className={`mcp-dot ${srv.connected ? 'connected' : 'disconnected'}`}>
                {srv.connected ? '✓' : '✗'}
              </span>
              <div className="mcp-info">
                <b>{name}</b>
                <div className="mcp-cmd">{srv.config?.command} {(srv.config?.args || []).join(' ')}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── 状态栏 ── */}
      <div className="status-bar">
        <span className="connected">●</span> 已就绪 · {agents.length} 个 Agent
      </div>
    </aside>
  )
}
