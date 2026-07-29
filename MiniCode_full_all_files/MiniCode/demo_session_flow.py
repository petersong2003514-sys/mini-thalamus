"""多轮对话全链路诊断演示（非自动化测试）。"""
import sys, os, json, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_store import get_store
store = get_store()
from agents.docparse_agent import DocParseAgent, get_session_files

# ── 模拟第1轮: 上传文件 ──
print("=" * 60)
print("第1轮: 上传文件 + 提问")

test_file = os.path.join(os.path.dirname(__file__), "test_doc.txt")
with open(test_file, "rb") as f:
    content = f.read()

sid = str(uuid.uuid4())[:8]
files = [{"name": "test_doc.txt", "content": content, "size": len(content)}]

agent = DocParseAgent()

print(f"  session_id: {sid}")
print(f"  files: {[f['name'] for f in files]}")

result1 = agent.run("违约金怎么规定的", context={"files": files, "session_id": sid})

print(f"  status: {result1['status']}")
print(f"  answer: {result1.get('answer','')[:200]}")
cached = get_session_files(sid)
print(f"  MemoryStore files: {cached}")

# ── 模拟第2轮: 追问(不传文件) ──
print("=" * 60)
print("第2轮: 追问(无文件) — 应该复用缓存")

result2 = agent.run("付款方式呢", context={"files": [], "session_id": sid})

print(f"  status: {result2['status']}")
print(f"  answer: {result2.get('answer','')[:200]}")

# ── 模拟第3轮: 上传新文件(应该追加不覆盖) ──
print("=" * 60)
print("第3轮: 上传新文件 — 应该追加到缓存")

files2 = [{"name": "new_doc.txt", "content": b"hello world", "size": 11}]
result3 = agent.run("这是什么", context={"files": files2, "session_id": sid})

print(f"  status: {result3['status']}")
print(f"  answer: {result3.get('answer','')[:200]}")
cached = get_session_files(sid)
print(f"  MemoryStore files: {cached} (应该是2)")

# ── 检查文件存储 ──
print("=" * 60)
print("MemoryStore 文件检查:")
files_dir = os.path.join(store.root, "sessions", sid, "files")
if os.path.isdir(files_dir):
    for fname in os.listdir(files_dir):
        fpath = os.path.join(files_dir, fname)
        print(f"  {fname}: {os.path.getsize(fpath)} bytes")
else:
    print(f"  目录不存在: {files_dir}")

# ── 检查历史 ──
print("=" * 60)
print("MemoryStore 历史检查:")
history_path = os.path.join(store.root, "sessions", sid, "history.json")
if os.path.exists(history_path):
    history = json.loads(open(history_path, encoding="utf-8").read())
    print(f"  turns: {len(history.get('turns', []))}")
else:
    print("  history.json 尚未创建（Web 端才记录，CLI 直调不记录）")

# 清理
store.cleanup_session(sid)
print("=" * 60)
print("全链路诊断完成")
