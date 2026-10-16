import React, { useState, useCallback, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatMessages from './components/ChatMessages'
import ChatInput from './components/ChatInput'
import useChat from './hooks/useChat'
import './App.css'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  componentDidCatch(error, errorInfo) {
    console.error('App crashed:', error, errorInfo)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, textAlign: 'center' }}>
          <h2>⚠️ 应用发生错误</h2>
          <p style={{ color: '#888' }}>{this.state.error?.message}</p>
          <button onClick={() => {
            this.setState({ hasError: false, error: null })
            window.location.reload()
          }}>重新加载</button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const [selectedAgent, setSelectedAgent] = useState('')
  const [sessions, setSessions] = useState([])
  const { messages, loading, send, abort, sessionId, fileCache, startNewChat, clearSession, loadSession } = useChat()
  const [crystallizing, setCrystallizing] = useState(false)
  const crystalAbortRef = useRef(null)

  // 清理 crystallize 请求（组件卸载时）
  useEffect(() => {
    return () => { crystalAbortRef.current?.abort() }
  }, [])

  const handleSend = (text, files, opts) => send(text, selectedAgent, files, opts)

  const refreshSessions = useCallback(async () => {
    try {
      const res = await fetch('/api/sessions')
      const data = await res.json()
      setSessions(data.sessions || [])
    } catch (e) {}
  }, [])

  useEffect(() => { refreshSessions() }, [refreshSessions, sessionId])

  const handleSelectSession = useCallback((sid) => {
    setSelectedAgent('')
    loadSession(sid)
  }, [loadSession])

  const handleDeleteSession = useCallback(async (sid) => {
    if (!confirm('确定删除该会话？')) return
    try {
      await fetch('/api/session/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      })
    } catch (e) {}
    if (sid === sessionId) startNewChat()
    refreshSessions()
  }, [sessionId, startNewChat, refreshSessions])

  const handleCrystallize = useCallback(async () => {
    if (!sessionId || crystallizing) return
    const name = prompt('为这个 Skill 命名（可选，留空自动命名）:')
    if (name === null) return // 用户取消
    setCrystallizing(true)
    const controller = new AbortController()
    crystalAbortRef.current = controller
    const timeoutId = setTimeout(() => controller.abort(), 30000)
    try {
      const res = await fetch('/api/skill/crystallize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: name || '' }),
        signal: controller.signal
      })
      clearTimeout(timeoutId)
      const data = await res.json()
      if (data.ok) {
        alert(`Skill 已固化: ${data.meta?.name || '?'}\n文件: ${data.skill_file}\n\n${data.summary || ''}`)
        // 刷新 sidebar 中的 agent 和 skill 列表
        window.dispatchEvent(new CustomEvent('refresh-sidebar'))
      } else {
        alert('固化失败: ' + (data.error || data.reason || '未知错误'))
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        alert('固化请求已取消或超时')
      } else {
        alert('请求失败: ' + e.message)
      }
    } finally {
      clearTimeout(timeoutId)
      setCrystallizing(false)
      crystalAbortRef.current = null
    }
  }, [sessionId, crystallizing])

  const handleCancelCrystallize = useCallback(() => {
    crystalAbortRef.current?.abort()
  }, [])

  return (
    <ErrorBoundary>
    <div className="app">
      <div className="chat-panel">
        <div className="chat-header">
          <div className="dot" />
          <h1>Mini Thalamus</h1>
          <div style={{ flex: 1 }} />
          <button
            className="crystallize-btn"
            onClick={startNewChat}
            title="开启新对话，保留旧会话"
          >
            ＋ 新对话
          </button>
          {sessionId && (
            <>
              <button
                className="crystallize-btn"
                onClick={clearSession}
                style={{ background: 'rgba(248,113,113,0.15)', borderColor: 'rgba(248,113,113,0.35)', color: '#f87171' }}
                title="清空当前会话（不可恢复）"
              >
                清空当前
              </button>
              <button
                className="crystallize-btn"
                onClick={handleCrystallize}
                disabled={crystallizing}
                title="将本次对话工作流固化为可复用的 Skill"
              >
                {crystallizing ? '分析中...' : '⚡ 固化为 Skill'}
              </button>
              {crystallizing && (
                <button
                  className="crystallize-btn"
                  onClick={handleCancelCrystallize}
                  style={{ background: 'rgba(248,113,113,0.15)', borderColor: 'rgba(248,113,113,0.35)', color: '#f87171' }}
                  title="取消固化"
                >
                  取消
                </button>
              )}
            </>
          )}
        </div>
        <ChatMessages messages={messages} loading={loading} />
        <ChatInput onSend={handleSend} loading={loading} onAbort={abort} />
      </div>
      <Sidebar selected={selectedAgent} onSelect={setSelectedAgent} sessionId={sessionId} fileCache={fileCache} sessions={sessions} activeSessionId={sessionId} onSelectSession={handleSelectSession} onDeleteSession={handleDeleteSession} />
    </div>
    </ErrorBoundary>
  )
}
