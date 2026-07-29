import { useState, useEffect } from 'react'

const S = {
  section: { marginTop: 12 },
  row: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, fontSize: 13 },
  name: { fontWeight: 600, minWidth: 80 },
  value: { color: '#888', fontFamily: 'monospace', fontSize: 12 },
  badge: { fontSize: 11, padding: '1px 6px', borderRadius: 4, background: '#444', color: '#ccc' },
  stale: { background: '#6b4700', color: '#ffc107' },
  delBtn: { marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: '#f87171', fontSize: 14 },
  form: { display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 },
  input: { background: '#2a2a2a', border: '1px solid #444', borderRadius: 4, color: '#eee', padding: '4px 8px', fontSize: 12 },
  addBtn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 4, padding: '5px 12px', cursor: 'pointer', fontSize: 12 },
  testBtn: { background: '#374151', color: '#eee', border: 'none', borderRadius: 4, padding: '3px 8px', cursor: 'pointer', fontSize: 11 },
}

const EMPTY = { name: '', type: 'api_key', value: '', scope: 'session', agents: '' }

export default function CredentialPanel() {
  const [creds, setCreds] = useState([])
  const [form, setForm] = useState(EMPTY)
  const [showForm, setShowForm] = useState(false)
  const [testResult, setTestResult] = useState({})

  const load = () =>
    fetch('/api/credentials').then(r => r.json()).then(d => setCreds(d.credentials || [])).catch(() => {})

  useEffect(() => { load() }, [])

  const handleAdd = async (e) => {
    e.preventDefault()
    let valueObj
    try { valueObj = JSON.parse(form.value) } catch { valueObj = form.value }
    await fetch('/api/credentials', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: form.name, type: form.type, value: valueObj,
        scope: form.scope,
        agents: form.agents ? form.agents.split(',').map(s => s.trim()).filter(Boolean) : [],
      }),
    })
    setForm(EMPTY)
    setShowForm(false)
    load()
  }

  const handleDelete = async (name) => {
    if (!confirm(`删除凭证 "${name}"？`)) return
    await fetch(`/api/credentials/${encodeURIComponent(name)}`, { method: 'DELETE' })
    load()
  }

  const handleTest = async (name) => {
    const r = await fetch('/api/credentials/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(r => r.json()).catch(e => ({ ok: false, error: e.message }))
    setTestResult(prev => ({ ...prev, [name]: r.ok ? '✓ 可用' : '✗ ' + (r.error || '失败') }))
  }

  return (
    <div style={S.section}>
      <h3>🔑 凭证管理</h3>
      {creds.length === 0 ? (
        <div className="empty-hint">暂无凭证，点击下方添加。</div>
      ) : (
        creds.map(c => (
          <div key={c.name} style={S.row}>
            <span style={S.name}>{c.name}</span>
            <span style={S.value}>{'•'.repeat(8)}</span>
            <span style={{ ...S.badge, ...(c.stale ? S.stale : {}) }}>{c.stale ? '过期' : c.scope || 'session'}</span>
            {c.agents?.length > 0 && <span style={S.badge}>{c.agents.join(',')}</span>}
            <button style={S.testBtn} onClick={() => handleTest(c.name)}>测试</button>
            {testResult[c.name] && <span style={{ fontSize: 11, color: testResult[c.name].startsWith('✓') ? '#4ade80' : '#f87171' }}>{testResult[c.name]}</span>}
            <button style={S.delBtn} onClick={() => handleDelete(c.name)}>🗑</button>
          </div>
        ))
      )}
      {showForm ? (
        <form style={S.form} onSubmit={handleAdd}>
          <input style={S.input} placeholder="名称 (如 GITHUB_TOKEN)" value={form.name}
            onChange={e => setForm(p => ({ ...p, name: e.target.value }))} required />
          <input style={S.input} placeholder="类型 (api_key/token/...)" value={form.type}
            onChange={e => setForm(p => ({ ...p, type: e.target.value }))} />
          <input style={S.input} placeholder='值 (字符串或 JSON，如 "sk-xxx")' value={form.value}
            onChange={e => setForm(p => ({ ...p, value: e.target.value }))} required />
          <input style={S.input} placeholder="作用域: session | permanent" value={form.scope}
            onChange={e => setForm(p => ({ ...p, scope: e.target.value }))} />
          <input style={S.input} placeholder="绑定 Agent (逗号分隔，可空)" value={form.agents}
            onChange={e => setForm(p => ({ ...p, agents: e.target.value }))} />
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" style={S.addBtn}>保存</button>
            <button type="button" style={{ ...S.addBtn, background: '#374151' }} onClick={() => setShowForm(false)}>取消</button>
          </div>
        </form>
      ) : (
        <button style={{ ...S.addBtn, marginTop: 6 }} onClick={() => setShowForm(true)}>+ 添加凭证</button>
      )}
    </div>
  )
}
