import { useState, useCallback, useRef, useEffect } from 'react'

const WELCOME = { role: 'assistant', text: '你好！我是 MiniCode，有什么可以帮你的？', routeTag: null }

export default function useChat() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: '你好！我是 MiniCode，有什么可以帮你的？\n\n试试看：\n• 「查一下最近的销售数据」\n• 「帮我分析这个CSV文件画个趋势图」\n• 「上传图片提取文字信息」\n\n👈 左侧可以手动选择 Agent，或让系统自动判断', routeTag: null }
  ])
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('minicode_session_id') || '')
  const [fileCache, setFileCache] = useState([])
  const initialSessionRef = useRef(sessionId)

  useEffect(() => {
    if (sessionId) localStorage.setItem('minicode_session_id', sessionId)
  }, [sessionId])

  useEffect(() => {
    const sid = initialSessionRef.current
    if (!sid) return
    let cancelled = false
    fetch(`/api/session/history?session_id=${encodeURIComponent(sid)}`)
      .then(r => r.json())
      .then(data => {
        if (cancelled || !data || !Array.isArray(data.turns) || !data.turns.length) return
        const restored = []
        for (const turn of data.turns) {
          restored.push({ role: 'user', text: turn.query, routeTag: null })
          restored.push({ role: 'assistant', text: turn.answer || '', routeTag: null,
                          stepLabels: turn.agent_id ? [{ step: 1, agent: turn.agent_id }] : [] })
        }
        setMessages(restored)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const abortRef = useRef(null)
  const currentStepRef = useRef(0)
  const pendingQueryRef = useRef('')
  const pendingAgentRef = useRef('')

  const send = useCallback(async (text, agentId, files = [], opts = {}) => {
    // 终止上一次未完成的请求，避免两个 SSE 流冲突
    abortRef.current?.abort()
    const requireConfirm = opts.requireConfirm

    // 为图片文件生成预览 URL（发送前）
    const imagePreviews = files
      .filter(f => f.type?.startsWith('image/') || /\.(png|jpg|jpeg|webp|bmp)$/i.test(f.name))
      .map(f => {
        if (f._previewUrl) return f._previewUrl
        return URL.createObjectURL(f)
      })

    const userMsg = {
      role: 'user',
      text: text || '(上传文件)',
      routeTag: null,
      files: files.map(f => f.name),
      imagePreviews: imagePreviews.length > 0 ? imagePreviews : undefined
    }
    const assistantMsg = { role: 'assistant', text: '', routeTag: null, loading: true }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setLoading(true)

    pendingQueryRef.current = text
    pendingAgentRef.current = agentId
    const controller = new AbortController()
    abortRef.current = controller

    try {
      let res
      if (files.length > 0) {
        const fd = new FormData()
        fd.append('message', text)
        fd.append('agent', agentId || '')
        fd.append('session_id', sessionId)
        for (const f of files) fd.append('files', f)
        res = await fetch('/api/chat/stream/form', {
          method: 'POST', body: fd, signal: controller.signal,
        })
      } else {
        res = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, agent: agentId || null, session_id: sessionId, require_confirm: requireConfirm || null, approved_call_ids: opts.approvedCallIds || null }),
          signal: controller.signal,
        })
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      setMessages(prev => {
        const next = [...prev]
        next[next.length - 1] = { ...next[next.length - 1], loading: false, text: '', routeTag: null, steps: [] }
        return next
      })

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            handleEvent(event, setMessages, setSessionId, setFileCache, currentStepRef)
          } catch (e) { if (process.env.NODE_ENV === 'development') console.error('SSE parse error:', e, line.slice(0,100)) }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError' && controller === abortRef.current) {
        setMessages(prev => {
          const next = [...prev]
          next[next.length - 1] = { ...next[next.length - 1], loading: false, text: `网络错误: ${e.message}` }
          return next
        })
      }
    } finally {
      // 只有当前请求的 controller 还是活跃的才关 loading（避免旧请求覆盖新请求的状态）
      if (controller === abortRef.current) {
        setLoading(false)
      }
    }
  }, [sessionId])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setLoading(false)
  }, [])

  const resetToWelcome = useCallback(() => {
    abortRef.current?.abort()
    pendingQueryRef.current = ''
    pendingAgentRef.current = ''
    currentStepRef.current = 0
    localStorage.removeItem('minicode_session_id')
    setSessionId('')
    setFileCache([])
    setMessages([WELCOME])
    setLoading(false)
  }, [])

  const startNewChat = useCallback(() => {
    resetToWelcome()
  }, [resetToWelcome])

  const clearSession = useCallback(async () => {
    const sid = sessionId
    resetToWelcome()
    if (!sid) return
    try {
      await fetch('/api/session/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      })
    } catch (e) {
      // 忽略清理失败，界面仍重置为新对话
    }
  }, [resetToWelcome, sessionId])

  const loadSession = useCallback(async (sid) => {
    if (!sid) return
    abortRef.current?.abort()
    setLoading(true)
    try {
      const res = await fetch(`/api/session/history?session_id=${encodeURIComponent(sid)}`)
      const data = await res.json()
      const restored = []
      for (const turn of (data.turns || [])) {
        restored.push({ role: 'user', text: turn.query, routeTag: null })
        restored.push({ role: 'assistant', text: turn.answer || '', routeTag: null,
                        stepLabels: turn.agent_id ? [{ step: 1, agent: turn.agent_id }] : [] })
      }
      setMessages(restored.length ? restored : [WELCOME])
      localStorage.setItem('minicode_session_id', sid)
      setSessionId(sid)
    } catch (e) {
      setMessages([WELCOME])
    } finally {
      setLoading(false)
    }
  }, [])

  // 组件卸载时取消进行中的请求
  useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  // 监听续传流中的检查点事件（统一走 React CheckpointCard）
  useEffect(() => {
    const handler = (e) => {
      const evt = e.detail
      if (!evt) return
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: evt.answer || '',
        routeTag: null,
        loading: false,
        checkpoint: {
          step: evt.step || 0,
          stepTotal: evt.step_total || 1,
          label: evt.label || '',
          answer: evt.answer || '',
          data: evt.data || {},
          sessionId: evt.session_id || ''
        }
      }])
    }
    window.addEventListener('minicode:checkpoint', handler)
    return () => window.removeEventListener('minicode:checkpoint', handler)
  }, [])

  useEffect(() => {
    const handler = (e) => {
      const detail = e.detail || {}
      const callIds = detail.callIds || []
      if (!pendingQueryRef.current || callIds.length === 0) return
      send(pendingQueryRef.current, pendingAgentRef.current, [], {
        approvedCallIds: callIds,
        requireConfirm: false,
      })
    }
    window.addEventListener('minicode:approve', handler)
    return () => window.removeEventListener('minicode:approve', handler)
  }, [send])

  return { messages, loading, send, abort, sessionId, fileCache, startNewChat, clearSession, loadSession }
}

// ─── SSE event handler ───

function handleEvent(event, setMessages, setSessionId, setFileCache, currentStepRef) {
  switch (event.type) {
    case 'route':
      currentStepRef.current = 0
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.routeTag = buildRouteTag(event)
        last.steps = event.multi_step || [event.agent]
        last.text = ''
        next[next.length - 1] = last
        return next
      })
      break

    case 'approval_checkpoint':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.approval = {
          callId: event.call_id,
          answer: event.answer || '',
          sessionId: event.session_id || ''
        }
        next[next.length - 1] = last
        return next
      })
      break

    case 'status': {
      const icon = event.subtype === 'agent_start' ? '▶' : event.subtype === 'agent_end' ? '✓' : ''
      const line = icon ? `${icon} ${event.text}` : event.text
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        if (!last.statusLines) last.statusLines = []
        last.statusLines = [...last.statusLines, line]
        next[next.length - 1] = last
        return next
      })
      break
    }

    case 'answer': {
      const step = event.step || 1
      setMessages(prev => {
        const next = [...prev]
        // 多步时，每步新开一个独立气泡
        if (step > currentStepRef.current && currentStepRef.current > 0) {
          const newBubble = { role: 'assistant', text: event.text, routeTag: null, loading: false,
                              stepLabels: [{ step, agent: event.agent }], statusLines: [] }
          next.push(newBubble)
        } else {
          const last = { ...next[next.length - 1] }
          if (event.step) {
            if (!last.stepLabels) last.stepLabels = []
            last.stepLabels = last.stepLabels.filter(s => s.step !== step)
            last.stepLabels.push({ step, agent: event.agent })
          }
          last.text = event.text
          next[next.length - 1] = last
        }
        currentStepRef.current = step
        return next
      })
      break
    }

    // ── Auto-Chain 自动步骤（无需用户确认）──
    case 'auto_step':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        if (!last.statusLines) last.statusLines = []
        last.statusLines = [...last.statusLines, `⚡ Step ${event.step} [${event.agent}] 自动完成`]
        next[next.length - 1] = last
        return next
      })
      break

    // ── 结构化输出事件 ──
    case 'structured':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.structured = event.data || event
        next[next.length - 1] = last
        return next
      })
      break

    // ── 检查点事件（断点续传）──
    case 'checkpoint':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.checkpoint = {
          step: event.step || 0,
          stepTotal: event.step_total || 1,
          label: event.label || '',
          answer: event.answer || '',
          data: event.data || {},
          sessionId: event.session_id || ''
        }
        next[next.length - 1] = last
        return next
      })
      break

    case 'meta_recovery':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        if (!last.statusLines) last.statusLines = []
        last.statusLines = [...last.statusLines, `🔄 [${event.from}] 失败，切换到 [${event.to}] 重试`]
        next[next.length - 1] = last
        return next
      })
      break

    case 'error':
      setMessages(prev => {
        const next = [...prev]
        const last = { ...next[next.length - 1] }
        last.text = (last.text ? last.text + '\n' : '') + `❌ ${event.text}`
        next[next.length - 1] = last
        return next
      })
      break

    case 'session_info':
      // 更新文件缓存（显示在侧边栏）
      if (event.files && setFileCache) {
        setFileCache(event.files.map(f => typeof f === 'string' ? f : (f.name || f)))
      }
      break

    case 'done':
      currentStepRef.current = 0
      if (event.session_id) setSessionId(event.session_id)
      break
  }
}

async function _getAgentNames() {
  try {
    const res = await fetch('/api/agents')
    const data = await res.json()
    const names = {}
    for (const a of data.agents) names[a.id] = a.name
    return names
  } catch { return {} }
}

// 缓存 agent 名称映射
let _nameCache = null
function _nameCachePromise() {
  if (!_nameCache) _nameCache = _getAgentNames()
  return _nameCache
}

function buildRouteTag(event) {
  if (event.multi_step) {
    const chain = event.multi_step.join(' → ')
    return { type: 'multi', label: '🔗 混合意图', detail: chain, reason: event.reason }
  }
  if (event.manual) {
    return { type: 'manual', label: '手动指定' }
  }
  const pct = ((event.confidence ?? 0) * 100).toFixed(0)
  return { type: 'auto', label: `自动路由 · ${pct}%`, reason: event.reason }
}
